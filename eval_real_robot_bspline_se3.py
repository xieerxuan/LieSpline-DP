"""Evaluate a real-robot LieSpline-DP policy in offline, shadow, or live mode.

See docs/real_robot_deployment.md for model inputs and hardware setup.
"""

import pathlib
import sys
from contextlib import nullcontext

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf

from liespline_dp.real_world import (
    bspline_se3_offline_eval,
    bspline_se3_rollout,
)
from liespline_dp.real_world.real_robot_env import (
    RealRobotEnv,
    wait_for_heartbeat_watchdog_disarm,
    wait_for_service_queue_drain,
)
from liespline_dp.real_world.real_robot_trajectory_recorder import (
    RealRobotTrajectoryRecorder,
    require_zero_hold_last_action_steps,
    serialize_checkpoint_config,
)
from liespline_dp.real_world.rollout_logging import emit_event

OmegaConf.register_new_resolver("eval", eval, replace=True)


_DEFAULT_BACKEND_CONFIG = (
    pathlib.Path(__file__).resolve().parent
    / "liespline_dp/config/real_world/robot_service_backend.yaml"
)


def _load_mapping(path):
    if path is None:
        return {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _load_policy(checkpoint_path, device):
    checkpoint_path = pathlib.Path(checkpoint_path).expanduser()
    with checkpoint_path.open("rb") as stream:
        payload = torch.load(
            stream,
            pickle_module=dill,
            map_location="cpu",
        )
    cfg = payload["cfg"]
    if not hasattr(cfg.task, "bspline"):
        raise click.ClickException("Checkpoint task is not a B-spline SE(3) task")
    task_gripper_representation = str(cfg.task.se3.get("gripper_representation", "waypoint"))
    policy_gripper_representation = str(cfg.policy.get("gripper_representation", "waypoint"))
    if task_gripper_representation != policy_gripper_representation:
        raise click.ClickException(
            "Checkpoint task and policy disagree on gripper representation: "
            f"{task_gripper_representation} != "
            f"{policy_gripper_representation}"
        )
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(torch.device(device))
    policy.reset()
    if int(policy.n_action_steps) != int(cfg.n_action_steps):
        raise click.ClickException("Checkpoint policy and config disagree on n_action_steps")
    if str(policy.gripper_representation) != task_gripper_representation:
        raise click.ClickException(
            "Instantiated policy gripper representation does not match the checkpoint task"
        )
    return cfg, workspace, policy, checkpoint_path.resolve()


@click.command()
@click.option("--mode", type=click.Choice(["offline", "shadow", "live"]), required=True)
@click.option("--input", "-i", required=True, help="Training checkpoint.")
@click.option("--output-dir", default=None)
@click.option(
    "--backend-config",
    default=str(_DEFAULT_BACKEND_CONFIG),
    show_default="liespline_dp/config/real_world/robot_service_backend.yaml",
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--max-batches", default=0, type=int)
@click.option("--max-cycles", default=0, type=int)
@click.option("--warmup-cycles", default=3, type=int, show_default=True)
@click.option("--action-lead-time", default=0.08, type=float, show_default=True)
@click.option("--min-join-lead-time", default=0.02, type=float, show_default=True)
@click.option(
    "--require-manual-arm/--no-require-manual-arm",
    default=True,
    show_default=True,
)
def main(
    mode,
    input,
    output_dir,
    backend_config,
    device,
    max_batches,
    max_cycles,
    warmup_cycles,
    action_lead_time,
    min_join_lead_time,
    require_manual_arm,
):
    if max_batches < 0 or max_cycles < 0 or warmup_cycles < 0:
        raise click.ClickException("Batch/cycle counts cannot be negative")
    if action_lead_time <= 0 or min_join_lead_time <= 0:
        raise click.ClickException("Action lead times must be positive")

    cfg, _, policy, checkpoint_path = _load_policy(input, device)
    rollout_device = torch.device(device)
    output_path = None
    trajectory_path = None
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == "live":
            trajectory_path = output_dir / "trajectory.npz"
            if trajectory_path.exists():
                raise click.ClickException(f"Trajectory output already exists: {trajectory_path}")
        output_path = output_dir / f"{mode}_metrics.jsonl"
    stream_context = (
        output_path.open("w", encoding="utf-8") if output_path is not None else nullcontext(None)
    )

    with stream_context as output_stream:
        emit_event(
            {
                "event": "run_start",
                "mode": mode,
                "checkpoint": str(pathlib.Path(input).expanduser()),
                "n_action_steps": int(policy.n_action_steps),
                "n_infer_steps": int(cfg.n_infer_steps),
                "control_frequency_hz": float(cfg.task.bspline.control_frequency_hz),
            },
            output_stream,
        )
        if mode == "offline":
            summary = bspline_se3_offline_eval.run_offline(
                cfg,
                policy,
                rollout_device,
                max_batches,
            )
            emit_event(summary, output_stream)
            return

        from liespline_dp.real_world.robot_service_hardware_backend import (
            RobotServiceHardwareBackend,
        )

        backend_kwargs = _load_mapping(backend_config)
        backend_kwargs["observation_frequency_hz"] = float(cfg.task.bspline.control_frequency_hz)
        backend = RobotServiceHardwareBackend(**backend_kwargs)
        recorder = None
        if mode == "live" and output_dir is not None:
            recorder = RealRobotTrajectoryRecorder(
                trajectory_path,
                checkpoint=str(checkpoint_path),
                checkpoint_config=serialize_checkpoint_config(cfg),
                rpc_address=backend.rpc_address,
                rpc_timeout_s=backend.rpc_timeout,
                observation_frequency_hz=backend.observation_frequency_hz,
                observation_max_age_s=backend.observation_max_age,
            )
        with RealRobotEnv(
            observation_backend=backend,
            command_backend=backend,
            accepted_action_callback=(
                None if recorder is None else recorder.record_accepted_actions
            ),
        ) as env:
            if recorder is not None:
                require_zero_hold_last_action_steps(backend)
            bspline_se3_rollout.run_warmup(
                env,
                policy,
                cfg,
                rollout_device,
                warmup_cycles,
                output_stream,
            )
            if mode == "shadow":
                click.echo("Shadow mode active; hardware remains disarmed.")
                bspline_se3_rollout.run_shadow(
                    env,
                    policy,
                    cfg,
                    rollout_device,
                    max_cycles,
                    output_stream,
                )
                return

            if require_manual_arm and not click.confirm(
                "B-spline checks complete. Arm the hardware service now?", default=False
            ):
                raise click.Abort()
            if not backend.arm():
                raise click.ClickException("Hardware service refused to arm")
            recorder_started = False
            rollout_completed = False
            try:
                if recorder is not None:
                    recorder.start(expected_service_instance_id=backend.service_instance_id)
                    recorder_started = True
                policy.reset()
                bspline_se3_rollout.run_live(
                    env,
                    backend,
                    policy,
                    cfg,
                    rollout_device,
                    max_cycles,
                    action_lead_time,
                    min_join_lead_time,
                    output_stream,
                )
                rollout_completed = True
            except KeyboardInterrupt:
                emit_event({"event": "operator_interrupt"}, output_stream)
            except Exception as exc:
                emit_event(
                    {
                        "event": "live_fault",
                        "reason": str(exc),
                    },
                    output_stream,
                )
                raise
            finally:
                cleanup_error = None
                if (
                    recorder_started
                    and rollout_completed
                    and env.last_accepted_queue_tail is not None
                ):
                    try:
                        wait_for_service_queue_drain(
                            env.last_accepted_queue_tail,
                            backend,
                        )
                    except BaseException as exc:
                        cleanup_error = exc
                if recorder is not None:
                    try:
                        disarmed = backend.disarm(
                            "trajectory recording complete"
                            if rollout_completed
                            else "trajectory recording interrupted"
                        )
                        if not disarmed:
                            raise RuntimeError("Hardware service did not confirm disarm")
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                if recorder is not None:
                    try:
                        env.stop()
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                if recorder_started:
                    try:
                        wait_for_heartbeat_watchdog_disarm(backend)
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                if recorder_started:
                    try:
                        recorder.close(
                            require_complete_coverage=(cleanup_error is None and rollout_completed)
                        )
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                try:
                    policy.reset()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                if cleanup_error is not None:
                    if sys.exc_info()[0] is None:
                        raise cleanup_error
                    click.echo(
                        f"Trajectory cleanup failed: {cleanup_error}",
                        err=True,
                    )


if __name__ == "__main__":
    main()
