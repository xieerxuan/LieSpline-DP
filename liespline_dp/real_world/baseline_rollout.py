"""Runtime helpers for native real-robot diffusion baseline rollout."""

import time

import click
import numpy as np
import torch

from liespline_dp.common.inference_scheduler import (
    EarlyInferenceScheduler,
    PredictionAlignment,
    build_boundary_offsets,
    compute_inference_trigger_monotonic,
)
from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.real_world.real_robot_env import (
    ActionQueueTail,
    continuation_prediction_target_monotonic,
    prepare_real_robot_policy_obs,
)
from liespline_dp.real_world.rollout_logging import emit_event

_LIVE_INFERENCE_TRIGGER_ADVANCE_S = 0.02


def _policy_action(policy, obs, shape_meta, device):
    policy_obs_np = prepare_real_robot_policy_obs(obs, shape_meta)
    policy_obs = dict_apply(
        policy_obs_np,
        lambda value: torch.from_numpy(value).unsqueeze(0).to(device),
    )
    started_at = time.perf_counter()
    with torch.no_grad():
        prediction = policy.predict_action(policy_obs)["action"][0]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    latency = time.perf_counter() - started_at
    return prediction.detach().cpu().numpy(), latency


def _is_cuda_device(device):
    return str(device).startswith("cuda")


def _wait_until(target_monotonic, backend=None):
    while True:
        if backend is not None:
            backend.raise_if_faulted()
        remaining = float(target_monotonic) - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.02))


def _observation_age(obs):
    diagnostics = obs.get("diagnostics", {})
    return float(diagnostics["observation_age_s"]) + max(
        0.0,
        time.monotonic() - float(diagnostics["observation_rpc_completed_monotonic"]),
    )


def _hold_gripper_plan(obs, accepted_queue_tail, n_steps):
    if accepted_queue_tail is None:
        value = float(np.asarray(obs["gripper_q"][-1]).reshape(-1)[0])
    else:
        value = float(accepted_queue_tail.gripper_target_q)
    return np.full((int(n_steps), 1), value, dtype=np.float64)


def validate_rollout_contract(cfg, policy, shadow_mode):
    n_action_steps = int(cfg.task.n_action_steps)
    n_infer_steps = int(cfg.task.n_infer_steps)
    n_latency_steps = int(cfg.task.n_latency_steps)
    if n_action_steps < 1:
        raise click.ClickException("n_action_steps must be positive")
    if n_infer_steps != n_action_steps - n_latency_steps:
        raise click.ClickException(
            "Checkpoint timing must satisfy n_infer_steps = n_action_steps - n_latency_steps"
        )
    if not shadow_mode and n_latency_steps < 1:
        raise click.ClickException(
            "Live baseline control requires n_latency_steps >= 1 so inference "
            "can finish before the active trajectory segment ends"
        )
    if cfg.task.get("prediction_alignment") != "plan_boundary":
        raise click.ClickException(
            "Checkpoint was not trained with plan-boundary action labels; "
            "retrain it with the current real-robot baseline dataset"
        )
    label_offset = cfg.task.get("action_label_offset_steps")
    if label_offset is None or int(label_offset) != n_latency_steps:
        raise click.ClickException(
            "Checkpoint action_label_offset_steps must equal n_latency_steps"
        )
    if "control_frequency_hz" not in cfg.task:
        if not shadow_mode:
            raise click.ClickException(
                "Live baseline checkpoint must declare task.control_frequency_hz"
            )
        frequency = 15.0
    else:
        frequency = float(cfg.task.control_frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise click.ClickException("control_frequency_hz must be positive")

    scheduler = EarlyInferenceScheduler(
        n_action_steps=n_action_steps,
        n_infer_steps=n_infer_steps,
        prediction_alignment=PredictionAlignment.PLAN_BOUNDARY,
    )
    policy_action_steps = int(policy.n_action_steps)
    if policy_action_steps != n_action_steps:
        raise click.ClickException(
            "Checkpoint policy.n_action_steps must equal the executable segment "
            f"length ({policy_action_steps} != {n_action_steps})"
        )
    return scheduler, frequency


def run_warmup(
    env,
    policy,
    cfg,
    device,
    scheduler,
    frequency,
    warmup_cycles,
    shadow_mode,
    output_stream=None,
):
    requested_warmup_cycles = int(warmup_cycles)
    effective_warmup_cycles = requested_warmup_cycles
    if not shadow_mode and _is_cuda_device(device):
        effective_warmup_cycles = max(effective_warmup_cycles, 2)
    if effective_warmup_cycles != requested_warmup_cycles:
        emit_event(
            {
                "event": "warmup_adjusted",
                "requested_warmup_cycles": requested_warmup_cycles,
                "effective_warmup_cycles": effective_warmup_cycles,
                "reason": (
                    "live CUDA control requires pre-arm warmup; the first "
                    "CUDA forward is a cold-start measurement"
                ),
            },
            output_stream,
        )

    warmup_latencies = []
    for warmup_index in range(effective_warmup_cycles):
        obs = env.get_obs(int(cfg.task.n_obs_steps))
        prediction, latency = _policy_action(policy, obs, cfg.task.shape_meta, device)
        if len(prediction) != scheduler.n_action_steps:
            raise click.ClickException(
                "Policy warmup output must contain exactly one executable "
                f"segment ({len(prediction)} != "
                f"{scheduler.n_action_steps})"
            )
        warmup_latencies.append(latency)
        emit_event(
            {
                "event": "warmup",
                "cycle": warmup_index,
                "inference_latency": latency,
                "observation_age": _observation_age(obs),
            },
            output_stream,
        )
    latency_check_values = warmup_latencies
    if not shadow_mode and _is_cuda_device(device) and len(warmup_latencies) > 1:
        latency_check_values = warmup_latencies[1:]
    if (
        not shadow_mode
        and latency_check_values
        and max(latency_check_values) >= (scheduler.n_latency_steps / frequency)
    ):
        raise click.ClickException(
            "Warmup inference alone exceeds the configured latency overlap: "
            f"max={max(latency_check_values):.4f}s, budget="
            f"{scheduler.n_latency_steps / frequency:.4f}s"
        )
    policy.reset()


def run_control_loop(
    env,
    backend,
    policy,
    cfg,
    device,
    codec,
    scheduler,
    frequency,
    max_cycles,
    shadow_mode,
    effective_bootstrap_lead,
    effective_join_lead,
    output_stream=None,
):
    action_offsets_s = build_boundary_offsets(scheduler.n_action_steps, frequency)
    n_obs_steps = int(cfg.task.n_obs_steps)
    accepted_queue_tail = None
    local_queue_deadline = None
    cycle = 0

    try:
        while max_cycles <= 0 or cycle < max_cycles:
            previous_local_deadline = local_queue_deadline
            if previous_local_deadline is not None:
                trigger_time = compute_inference_trigger_monotonic(
                    previous_local_deadline,
                    scheduler.n_action_steps,
                    scheduler.n_infer_steps,
                    frequency,
                    trigger_advance_s=_LIVE_INFERENCE_TRIGGER_ADVANCE_S,
                )
                _wait_until(
                    trigger_time,
                    backend=None if shadow_mode else backend,
                )

            prediction_target_monotonic = None
            if accepted_queue_tail is not None and not shadow_mode:
                prediction_target_monotonic = continuation_prediction_target_monotonic(
                    accepted_queue_tail,
                    scheduler.n_latency_steps,
                    frequency,
                )
            prediction_obs = env.get_obs(
                n_obs_steps,
                target_monotonic=prediction_target_monotonic,
            )
            prediction_observation_timestamp = float(prediction_obs["timestamp"][-1])
            diagnostics = prediction_obs.get("diagnostics", {})
            raw_model_action, inference_latency = _policy_action(
                policy, prediction_obs, cfg.task.shape_meta, device
            )
            if len(raw_model_action) != scheduler.n_action_steps:
                raise RuntimeError(
                    "Policy prediction must contain exactly one executable "
                    f"segment: need {scheduler.n_action_steps}, "
                    f"got {len(raw_model_action)}"
                )
            model_action = scheduler.prepare_pending(raw_model_action[None])[0]
            phase = "bootstrap" if accepted_queue_tail is None else "queue_continuation"
            tcp_pose, gripper_target_q = codec.decode_model_action(model_action)
            if gripper_target_q is None:
                gripper_target_q = _hold_gripper_plan(
                    prediction_obs,
                    accepted_queue_tail,
                    len(tcp_pose),
                )

            if accepted_queue_tail is None:
                actual_lead = max(float(action_offsets_s[0]), effective_bootstrap_lead)
            else:
                submission_time = time.monotonic()
                actual_lead = (
                    accepted_queue_tail.local_deadline_monotonic
                    + float(action_offsets_s[0])
                    - submission_time
                )
                if actual_lead < effective_join_lead:
                    raise RuntimeError(
                        "Baseline queue continuation missed its minimum action lead: "
                        f"lead={actual_lead:.4f}s, "
                        f"required={effective_join_lead:.4f}s"
                    )
                if submission_time >= accepted_queue_tail.local_deadline_monotonic:
                    raise RuntimeError(
                        "Baseline queue continuation missed queue exhaustion: "
                        f"late_by={submission_time - accepted_queue_tail.local_deadline_monotonic:.4f}s"
                    )

            if shadow_mode:
                submitted_gripper = np.asarray(gripper_target_q).reshape(-1, 1)
                now_monotonic = time.monotonic()
                if accepted_queue_tail is None:
                    schedule_shift = max(
                        0.0,
                        effective_bootstrap_lead - float(action_offsets_s[0]),
                    )
                    local_queue_deadline = (
                        now_monotonic + float(action_offsets_s[-1]) + schedule_shift
                    )
                    queue_tail_monotonic = local_queue_deadline
                else:
                    schedule_shift = 0.0
                    local_queue_deadline = accepted_queue_tail.local_deadline_monotonic + float(
                        action_offsets_s[-1]
                    )
                    queue_tail_monotonic = local_queue_deadline
                accepted_queue_tail = ActionQueueTail(
                    tcp_pose=tcp_pose[-1],
                    gripper_target_q=float(submitted_gripper[-1, 0]),
                    service_instance_id="shadow",
                    queue_token=f"shadow-{cycle}",
                    queue_tail_monotonic=queue_tail_monotonic,
                    local_deadline_monotonic=local_queue_deadline,
                )
                accepted_steps = 0
            else:
                result = env.exec_actions(
                    tcp_pose=tcp_pose,
                    gripper_target_q=gripper_target_q,
                    action_offsets_s=action_offsets_s,
                    prediction_observation_monotonic=(prediction_observation_timestamp),
                    expected_queue_tail=accepted_queue_tail,
                    interpolation="linear_se3",
                    requested_min_lead_s=(
                        effective_bootstrap_lead if accepted_queue_tail is None else None
                    ),
                )
                if result.accepted_tail is None:
                    raise RuntimeError("Hardware ACK did not return the accepted queue tail")
                accepted_queue_tail = result.accepted_tail
                local_queue_deadline = accepted_queue_tail.local_deadline_monotonic
                accepted_steps = result.accepted_steps
                schedule_shift = result.bootstrap_shift_s

            estimated_queue_remaining = max(0.0, local_queue_deadline - time.monotonic())
            emit_event(
                {
                    "event": "rollout_cycle",
                    "cycle": cycle,
                    "phase": phase,
                    "shadow_mode": shadow_mode,
                    "n_action_steps": scheduler.n_action_steps,
                    "n_infer_steps": scheduler.n_infer_steps,
                    "n_latency_steps": scheduler.n_latency_steps,
                    "observation_age": _observation_age(prediction_obs),
                    "prediction_target_monotonic": prediction_target_monotonic,
                    "prediction_alignment_error_s": diagnostics.get("alignment_error_s"),
                    "sensor_skew": diagnostics.get("max_sensor_skew"),
                    "ipc_rtt": getattr(backend, "last_ipc_rtt", None),
                    "inference_latency": inference_latency,
                    "first_action_offset_s": float(action_offsets_s[0]),
                    "estimated_action_lead_s": actual_lead,
                    "inference_trigger_advance_s": (_LIVE_INFERENCE_TRIGGER_ADVANCE_S),
                    "bootstrap_shift_s": schedule_shift,
                    "previous_local_queue_deadline_monotonic": (previous_local_deadline),
                    "local_queue_deadline_monotonic": local_queue_deadline,
                    "estimated_queue_remaining_s": estimated_queue_remaining,
                    "queue_token": accepted_queue_tail.queue_token,
                    "accepted_steps": accepted_steps,
                    "tcp_error": diagnostics.get("tcp_error"),
                    "gripper_error": diagnostics.get("gripper_error"),
                },
                output_stream,
            )
            cycle += 1

        if not shadow_mode and local_queue_deadline is not None:
            _wait_until(local_queue_deadline, backend=backend)
        return True
    except KeyboardInterrupt:
        return False
    except Exception as exc:
        if not shadow_mode:
            emit_event(
                {
                    "event": "live_fault",
                    "cycle": cycle,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                output_stream,
            )
        raise
