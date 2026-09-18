"""Evaluate a real-robot DP baseline in offline, shadow, or live mode.

See docs/real_robot_deployment.md for model inputs and hardware setup.
"""

import pathlib
import sys

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf

from liespline_dp.common.pose_action_codec import PoseActionCodec
from liespline_dp.real_world import baseline_rollout
from liespline_dp.real_world.baseline_offline_eval import run_offline
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


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["offline", "shadow", "live"]),
    required=True,
)
@click.option("--input", "-i", required=True, help="Training checkpoint.")
@click.option("--output-dir", default=None)
@click.option(
    "--backend-config",
    default=str(_DEFAULT_BACKEND_CONFIG),
    show_default="liespline_dp/config/real_world/robot_service_backend.yaml",
    help="Backend kwargs YAML.",
)
@click.option("--num-inference-steps", default=None, type=int)
@click.option("--warmup-cycles", default=3, type=int, show_default=True)
@click.option("--action-lead-time", default=0.12, type=float, show_default=True)
@click.option("--min-join-lead-time", default=0.02, type=float, show_default=True)
@click.option("--rpc-timeout", default=None, type=float)
@click.option(
    "--require-manual-arm/--no-require-manual-arm",
    default=True,
    show_default=True,
)
@click.option(
    "--max-batches", default=0, type=click.IntRange(min=0), help="Offline batches; 0 evaluates all."
)
@click.option("--max-cycles", default=0, type=int, help="0 runs until Ctrl-C.")
@click.option("--device", default="cuda:0", show_default=True)
def main(
    mode,
    input,
    output_dir,
    backend_config,
    num_inference_steps,
    warmup_cycles,
    action_lead_time,
    min_join_lead_time,
    rpc_timeout,
    require_manual_arm,
    max_cycles,
    max_batches,
    device,
):
    # Validate CLI timing overrides.
    if warmup_cycles < 0 or action_lead_time <= 0.0 or min_join_lead_time <= 0.0:
        raise click.ClickException("Invalid rollout timing option")
    if num_inference_steps is not None and num_inference_steps < 1:
        raise click.ClickException("num-inference-steps must be positive")
    if rpc_timeout is not None and rpc_timeout <= 0:
        raise click.ClickException("rpc-timeout must be positive")
    shadow_mode = mode == "shadow"

    # Load checkpoint payload on CPU.
    checkpoint_path = pathlib.Path(input).expanduser().resolve()
    payload = torch.load(
        checkpoint_path.open("rb"),
        pickle_module=dill,
        map_location="cpu",
    )
    cfg = payload["cfg"]

    # Rebuild the workspace and select the rollout model.
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    rollout_device = torch.device(device)
    policy.eval().to(rollout_device)
    policy.reset()
    if num_inference_steps is not None:
        if not hasattr(policy, "num_inference_steps"):
            raise click.ClickException("This policy does not expose num_inference_steps")
        policy.num_inference_steps = int(num_inference_steps)

    output_stream = None
    if output_dir is not None:
        output_path = pathlib.Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        output_stream = (output_path / f"{mode}_metrics.jsonl").open("w", encoding="utf-8")
        click.get_current_context().call_on_close(output_stream.close)
    emit_event(
        {"event": "run_start", "mode": mode, "checkpoint": str(checkpoint_path)}, output_stream
    )
    if mode == "offline":
        emit_event(run_offline(cfg, policy, rollout_device, max_batches), output_stream)
        return

    from liespline_dp.real_world.robot_service_hardware_backend import (
        RobotServiceHardwareBackend,
    )

    # Check the policy/action-queue timing contract.
    scheduler, frequency = baseline_rollout.validate_rollout_contract(cfg, policy, shadow_mode)

    # Match robot-service observations to the policy control rate.
    backend_kwargs = _load_mapping(backend_config)
    backend_kwargs["observation_frequency_hz"] = float(frequency)
    if rpc_timeout is not None:
        backend_kwargs["rpc_timeout"] = float(rpc_timeout)
    backend = RobotServiceHardwareBackend(**backend_kwargs)

    codec = PoseActionCodec(action_format=cfg.task.get("action_format", "pose_gripper"))
    recorder = None
    if not shadow_mode and output_dir is not None:
        output_dir = pathlib.Path(output_dir).expanduser()
        trajectory_path = output_dir / "trajectory.npz"
        if trajectory_path.exists():
            raise click.ClickException(f"Trajectory output already exists: {trajectory_path}")
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
        accepted_action_callback=(None if recorder is None else recorder.record_accepted_actions),
    ) as env:
        if recorder is not None:
            require_zero_hold_last_action_steps(backend)

        # Respect backend minimum lead times.
        backend_minimum_lead = float(getattr(backend, "minimum_submission_lead_s", 0.0) or 0.0)
        effective_bootstrap_lead = max(float(action_lead_time), backend_minimum_lead)
        effective_join_lead = max(float(min_join_lead_time), backend_minimum_lead)

        # Warm up inference before arming.
        baseline_rollout.run_warmup(
            env=env,
            policy=policy,
            cfg=cfg,
            device=rollout_device,
            scheduler=scheduler,
            frequency=frequency,
            warmup_cycles=warmup_cycles,
            shadow_mode=shadow_mode,
            output_stream=output_stream,
        )

        if shadow_mode:
            click.echo("Shadow mode active; hardware remains disarmed.")
        else:
            # Gate live control on manual arm; action safety is enforced by
            # the robot service.
            if require_manual_arm and not click.confirm(
                "Baseline checks complete. Arm robot control now?", default=False
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
            rollout_completed = bool(
                baseline_rollout.run_control_loop(
                    env=env,
                    backend=backend,
                    policy=policy,
                    cfg=cfg,
                    device=rollout_device,
                    codec=codec,
                    scheduler=scheduler,
                    frequency=frequency,
                    max_cycles=max_cycles,
                    shadow_mode=shadow_mode,
                    effective_bootstrap_lead=effective_bootstrap_lead,
                    effective_join_lead=effective_join_lead,
                    output_stream=output_stream,
                )
            )
        finally:
            cleanup_error = None
            if recorder_started and rollout_completed and env.last_accepted_queue_tail is not None:
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
            if cleanup_error is not None:
                if sys.exc_info()[0] is None:
                    raise cleanup_error
                click.echo(
                    f"Trajectory cleanup failed: {cleanup_error}",
                    err=True,
                )


if __name__ == "__main__":
    main()
