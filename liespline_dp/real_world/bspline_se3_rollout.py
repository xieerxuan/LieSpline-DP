"""Online rollout helpers for real-robot B-spline SE(3) policies."""

import time

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
    continuation_prediction_target_monotonic,
    prepare_real_robot_policy_obs,
)
from liespline_dp.real_world.rollout_logging import emit_event

_LIVE_INFERENCE_TRIGGER_ADVANCE_S = 0.02


def _policy_input(obs, shape_meta, device):
    policy_obs_np = prepare_real_robot_policy_obs(obs, shape_meta)
    return dict_apply(
        policy_obs_np,
        lambda value: torch.from_numpy(value).unsqueeze(0).to(device),
    )


def _hold_gripper_plan(obs, n_steps):
    value = float(np.asarray(obs["gripper_q"][-1]).reshape(-1)[0])
    return np.full((int(n_steps), 1), value, dtype=np.float64)


def _prediction_to_native_action(policy, result, obs):
    tcp_pose, gripper_target_q = policy.action_pose_adapter.pose_grip_to_tcp_command(
        result["waypoint_pose"],
        result.get("waypoint_grip", None),
    )
    tcp_pose = tcp_pose[0].detach().cpu().numpy()
    if gripper_target_q is None:
        gripper_target_q = _hold_gripper_plan(obs, len(tcp_pose))
    else:
        gripper_target_q = gripper_target_q[0].detach().cpu().numpy().reshape(-1, 1)
    native_action = np.concatenate([tcp_pose, gripper_target_q], axis=-1)
    expected_shape = (int(policy.n_action_steps), 7)
    if native_action.shape != expected_shape:
        raise RuntimeError(
            "B-spline policy must produce the complete hardware plan "
            f"{expected_shape}, got {native_action.shape}"
        )
    if not np.all(np.isfinite(native_action)):
        raise RuntimeError("B-spline policy returned non-finite hardware actions")
    return native_action


def _predict_live_plan(policy, obs, shape_meta, device):
    policy_obs = _policy_input(obs, shape_meta, device)
    started_at = time.perf_counter()
    with torch.no_grad():
        result = policy.predict_action(policy_obs)
        native_action = _prediction_to_native_action(policy, result, obs)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    latency = time.perf_counter() - started_at
    return native_action, latency


def _gripper_plan_diagnostics(
    gripper_target_q,
    reference_gripper,
    action_offsets_s,
):
    gripper = np.asarray(gripper_target_q, dtype=np.float64).reshape(-1)
    offsets = np.asarray(action_offsets_s, dtype=np.float64).reshape(-1)
    if len(gripper) != len(offsets):
        raise ValueError("Gripper plan and action offsets have different lengths")
    previous_gripper = float(reference_gripper)
    previous_offset = 0.0
    steps = []
    max_abs_speed = 0.0
    max_abs_delta = 0.0
    max_index = -1
    for index, (target_gripper, target_offset) in enumerate(zip(gripper, offsets)):
        dt = float(target_offset) - float(previous_offset)
        if dt <= 0.0:
            speed = float("inf")
        else:
            speed = (float(target_gripper) - previous_gripper) / dt
        delta = float(target_gripper) - previous_gripper
        abs_speed = abs(speed)
        abs_delta = abs(delta)
        if abs_speed > max_abs_speed:
            max_abs_speed = abs_speed
            max_abs_delta = abs_delta
            max_index = int(index)
        steps.append(
            {
                "index": int(index),
                "previous_gripper_q": previous_gripper,
                "target_gripper_q": float(target_gripper),
                "delta": delta,
                "dt_s": dt,
                "speed_q_s": speed,
            }
        )
        previous_gripper = float(target_gripper)
        previous_offset = float(target_offset)
    return {
        "reference_gripper_q": float(reference_gripper),
        "target_gripper_q": gripper.tolist(),
        "action_offsets_s": offsets.tolist(),
        "max_abs_gripper_delta": max_abs_delta,
        "max_abs_gripper_speed_q_s": max_abs_speed,
        "max_speed_step_index": max_index,
        "steps": steps,
    }


def _wait_until(target_monotonic, backend=None):
    while True:
        if backend is not None:
            raise_if_faulted = getattr(backend, "raise_if_faulted", None)
            if callable(raise_if_faulted):
                raise_if_faulted()
            else:
                fault = _server_fault(backend)
                if fault:
                    raise RuntimeError(f"Hardware fault while waiting: {fault}")
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


def _observation_timing_summary(obs, target_monotonic=None):
    diagnostics = dict(obs.get("diagnostics", {}) or {})
    keys = (
        "alignment_mode",
        "target_monotonic",
        "alignment_error_s",
        "max_sensor_skew_s",
        "max_sensor_skew",
        "observation_age_s",
        "observation_client_call_s",
        "observation_client_request_timing",
        "service_get_observation_timing",
        "service_observation_store_timing",
        "service_observation_payload_bytes",
        "source_status",
    )
    summary = {key: diagnostics[key] for key in keys if key in diagnostics}
    if target_monotonic is not None:
        summary["requested_target_monotonic"] = float(target_monotonic)
        try:
            summary["returned_timestamp_error_s"] = float(target_monotonic) - float(
                obs["timestamp"][-1]
            )
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    if "timestamp" in obs:
        timestamps = np.asarray(obs["timestamp"], dtype=np.float64).reshape(-1)
        if len(timestamps):
            summary["latest_observation_monotonic"] = float(timestamps[-1])
            summary["timestamp_span_s"] = (
                float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
            )
    return summary


def _server_fault(backend):
    status = dict(getattr(backend, "server_status", {}) or {})
    if getattr(backend, "last_heartbeat_error", None):
        return f"heartbeat: {backend.last_heartbeat_error}"
    if status.get("scheduler_error"):
        return f"scheduler: {status['scheduler_error']}"
    if status.get("last_fault"):
        return f"service: {status['last_fault']}"
    if status and status.get("armed") is False:
        return "hardware service became disarmed"
    return ""


def run_warmup(
    env,
    policy,
    cfg,
    device,
    warmup_cycles,
    output_stream,
):
    for cycle in range(int(warmup_cycles)):
        obs = env.get_obs(int(cfg.n_obs_steps))
        _, latency = _predict_live_plan(policy, obs, cfg.task.shape_meta, device)
        emit_event(
            {
                "event": "warmup",
                "cycle": cycle,
                "inference_latency": latency,
                "observation_age": _observation_age(obs),
            },
            output_stream,
        )
    policy.reset()


def run_shadow(
    env,
    policy,
    cfg,
    device,
    max_cycles,
    output_stream,
):
    cycle = 0
    frequency = float(cfg.task.bspline.control_frequency_hz)
    action_offsets_s = build_boundary_offsets(int(policy.n_action_steps), frequency)
    while max_cycles <= 0 or cycle < max_cycles:
        policy.reset()
        obs = env.get_obs(int(cfg.n_obs_steps))
        native_action, latency = _predict_live_plan(policy, obs, cfg.task.shape_meta, device)
        gripper_target_q = native_action[:, 6:7]
        reference_gripper = float(np.asarray(obs["gripper_q"][-1]).reshape(-1)[0])
        gripper_diagnostics = _gripper_plan_diagnostics(
            gripper_target_q,
            reference_gripper=reference_gripper,
            action_offsets_s=action_offsets_s,
        )
        diagnostics = obs.get("diagnostics", {})
        event = {
            "event": "shadow_cycle",
            "cycle": cycle,
            "action_steps": len(native_action),
            "inference_latency": latency,
            "observation_age": _observation_age(obs),
            "sensor_skew": diagnostics.get("max_sensor_skew"),
            "gripper_plan_diagnostics": gripper_diagnostics,
        }
        emit_event(event, output_stream)
        cycle += 1
    policy.reset()


def _submit_live_plan(
    env,
    native_action,
    action_offsets_s,
    prediction_observation_monotonic,
    submission_obs,
    min_action_lead_time,
    expected_queue_tail=None,
    interpolation="linear_se3",
    output_stream=None,
    preview_context=None,
):
    submit_entry_monotonic = time.monotonic()
    timing = {}
    preview_context = dict(preview_context or {})

    def _tail_remaining(now):
        if expected_queue_tail is None:
            return None
        return float(expected_queue_tail.local_deadline_monotonic - now)

    def _emit_submit_timing(outcome, extra=None):
        emitted_at = time.monotonic()
        event = {
            "event": "live_submit_timing",
            "outcome": str(outcome),
            "submit_total_s": emitted_at - submit_entry_monotonic,
            "timing": timing,
            "tail_remaining_at_entry_s": _tail_remaining(submit_entry_monotonic),
            "tail_remaining_at_emit_s": _tail_remaining(emitted_at),
        }
        if preview_context:
            event.update(preview_context)
        if extra:
            event.update(extra)
        emit_event(event, output_stream)

    native_action = np.asarray(native_action, dtype=np.float64)
    if native_action.ndim != 2 or native_action.shape[1] != 7:
        raise ValueError(
            "Native real-robot action must have shape [T,7] as [xyz,rotvec,gripper_target_q]"
        )
    tcp_pose = native_action[:, :6]
    gripper_target_q = native_action[:, 6:7]
    action_offsets_s = np.asarray(action_offsets_s, dtype=np.float64)
    if expected_queue_tail is None:
        actual_lead_time = max(float(action_offsets_s[0]), float(min_action_lead_time))
    else:
        submission_time = time.monotonic()
        actual_lead_time = (
            expected_queue_tail.local_deadline_monotonic
            + float(action_offsets_s[0])
            - submission_time
        )
        if actual_lead_time < float(min_action_lead_time):
            raise RuntimeError(
                "B-spline plan missed its minimum action lead time: "
                f"lead={actual_lead_time:.4f}s, "
                f"required={float(min_action_lead_time):.4f}s"
            )
        if submission_time >= expected_queue_tail.local_deadline_monotonic:
            raise RuntimeError(
                "B-spline queue continuation missed queue exhaustion: "
                "accepted queue tail has already elapsed"
            )
    after_deadline_check = time.monotonic()
    timing["deadline_check_s"] = after_deadline_check - submit_entry_monotonic

    reference_gripper = (
        float(np.asarray(submission_obs["gripper_q"][-1]).reshape(-1)[0])
        if expected_queue_tail is None
        else float(expected_queue_tail.gripper_target_q)
    )
    gripper_diagnostics = _gripper_plan_diagnostics(
        gripper_target_q,
        reference_gripper=reference_gripper,
        action_offsets_s=action_offsets_s,
    )
    after_gripper_diagnostics = time.monotonic()
    timing["gripper_diagnostics_s"] = after_gripper_diagnostics - after_deadline_check
    preview_event = {
        "event": "live_plan_preview",
        "gripper_plan_diagnostics": gripper_diagnostics,
        "submit_entry_tail_remaining_s": _tail_remaining(submit_entry_monotonic),
        "preview_complete_tail_remaining_s": _tail_remaining(after_gripper_diagnostics),
    }
    preview_event.update(preview_context)
    before_preview_emit = time.monotonic()
    emit_event(preview_event, output_stream)
    after_preview_emit = time.monotonic()
    timing["preview_emit_s"] = after_preview_emit - before_preview_emit

    before_exec_actions = time.monotonic()
    timing["pre_exec_actions_s"] = before_exec_actions - submit_entry_monotonic
    exec_extra = {
        "tail_remaining_before_exec_actions_s": _tail_remaining(before_exec_actions),
    }
    if expected_queue_tail is not None:
        exec_extra["first_action_lead_before_exec_actions_s"] = (
            expected_queue_tail.local_deadline_monotonic
            + float(action_offsets_s[0])
            - before_exec_actions
        )
    try:
        result = env.exec_actions(
            tcp_pose=tcp_pose,
            gripper_target_q=gripper_target_q,
            action_offsets_s=action_offsets_s,
            prediction_observation_monotonic=(prediction_observation_monotonic),
            expected_queue_tail=expected_queue_tail,
            interpolation=interpolation,
            requested_min_lead_s=(
                float(min_action_lead_time) if expected_queue_tail is None else None
            ),
        )
    except Exception as exc:
        after_exec_actions = time.monotonic()
        timing["exec_actions_s"] = after_exec_actions - before_exec_actions
        timing["submit_until_exception_s"] = after_exec_actions - submit_entry_monotonic
        _emit_submit_timing(
            "exec_actions_exception",
            {
                **exec_extra,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )
        raise
    after_exec_actions = time.monotonic()
    timing["exec_actions_s"] = after_exec_actions - before_exec_actions
    timing["submit_until_ack_s"] = after_exec_actions - submit_entry_monotonic
    _emit_submit_timing(
        "exec_actions_returned",
        {
            **exec_extra,
            "accepted": bool(result.accepted),
            "accepted_steps": int(result.accepted_steps),
            "reason": str(result.reason or ""),
        },
    )
    if not result.accepted:
        raise RuntimeError("Hardware rejected the B-spline plan")
    if result.accepted_steps != len(native_action):
        raise RuntimeError("Hardware did not accept the complete B-spline plan")
    if result.accepted_tail is None:
        raise RuntimeError("Hardware ACK did not return the accepted queue tail")
    return result, result.accepted_tail, actual_lead_time


def run_live(
    env,
    backend,
    policy,
    cfg,
    device,
    max_cycles,
    action_lead_time,
    min_join_lead_time,
    output_stream,
):
    frequency = float(cfg.task.bspline.control_frequency_hz)
    scheduler = EarlyInferenceScheduler(
        n_action_steps=int(policy.n_action_steps),
        n_infer_steps=int(cfg.n_infer_steps),
        prediction_alignment=PredictionAlignment.PLAN_BOUNDARY,
    )
    negotiated_lead_time = float(getattr(backend, "min_action_lead_s", 0.0) or 0.0)
    backend_submission_lead = float(
        getattr(
            backend,
            "minimum_submission_lead_s",
            negotiated_lead_time,
        )
        or 0.0
    )
    effective_bootstrap_lead = max(float(action_lead_time), backend_submission_lead)
    effective_join_lead = max(float(min_join_lead_time), backend_submission_lead)

    action_offsets_s = build_boundary_offsets(int(policy.n_action_steps), frequency)
    bootstrap_prepare_started = time.monotonic()
    bootstrap_obs_started = time.monotonic()
    prediction_obs = env.get_obs(int(cfg.n_obs_steps))
    bootstrap_obs_finished = time.monotonic()
    prediction_observation_monotonic = float(prediction_obs["timestamp"][-1])
    bootstrap_predict_started = time.monotonic()
    raw_native_action, inference_latency = _predict_live_plan(
        policy, prediction_obs, cfg.task.shape_meta, device
    )
    bootstrap_predict_finished = time.monotonic()
    bootstrap_scheduler_started = time.monotonic()
    native_action = scheduler.bootstrap(raw_native_action[None])[0]
    bootstrap_scheduler_finished = time.monotonic()
    submission_obs = prediction_obs
    bootstrap_before_submit = time.monotonic()
    result, accepted_queue_tail, actual_lead_time = _submit_live_plan(
        env,
        native_action,
        action_offsets_s,
        prediction_observation_monotonic,
        submission_obs,
        min_action_lead_time=effective_bootstrap_lead,
        interpolation="linear_se3",
        output_stream=output_stream,
        preview_context={
            "cycle": 0,
            "phase": "bootstrap",
            "prepare_timing": {
                "prepare_total_s": (bootstrap_before_submit - bootstrap_prepare_started),
                "get_obs_s": bootstrap_obs_finished - bootstrap_obs_started,
                "predict_wrapper_s": (bootstrap_predict_finished - bootstrap_predict_started),
                "inference_latency_s": inference_latency,
                "scheduler_s": (bootstrap_scheduler_finished - bootstrap_scheduler_started),
            },
        },
    )
    cycle = 1
    local_queue_deadline = accepted_queue_tail.local_deadline_monotonic
    emit_event(
        {
            "event": "live_cycle",
            "cycle": 0,
            "phase": "bootstrap",
            "interpolation": "linear_se3",
            "accepted_steps": result.accepted_steps,
            "inference_latency": inference_latency,
            "negotiated_min_action_lead_s": negotiated_lead_time,
            "backend_minimum_submission_lead_s": backend_submission_lead,
            "effective_submission_lead_s": effective_bootstrap_lead,
            "actual_submission_lead_s": actual_lead_time,
            "first_action_offset_s": float(action_offsets_s[0]),
            "bootstrap_shift_s": result.bootstrap_shift_s,
            "local_queue_deadline_monotonic": local_queue_deadline,
            "queue_token": accepted_queue_tail.queue_token,
        },
        output_stream,
    )

    while max_cycles <= 0 or cycle < max_cycles:
        cycle_prepare_started = time.monotonic()
        trigger_time = compute_inference_trigger_monotonic(
            local_queue_deadline,
            policy.n_action_steps,
            cfg.n_infer_steps,
            frequency,
            trigger_advance_s=_LIVE_INFERENCE_TRIGGER_ADVANCE_S,
        )
        wait_started = time.monotonic()
        _wait_until(trigger_time, backend=backend)
        wait_finished = time.monotonic()

        prediction_target_monotonic = continuation_prediction_target_monotonic(
            accepted_queue_tail,
            scheduler.n_latency_steps,
            frequency,
        )
        get_obs_started = time.monotonic()
        prediction_obs = env.get_obs(
            int(cfg.n_obs_steps),
            target_monotonic=prediction_target_monotonic,
        )
        get_obs_finished = time.monotonic()
        observation_timing = _observation_timing_summary(
            prediction_obs,
            target_monotonic=prediction_target_monotonic,
        )
        emit_event(
            {
                "event": "live_observation_timing",
                "cycle": cycle,
                "phase": "queue_continuation",
                "get_obs_s": get_obs_finished - get_obs_started,
                "tail_remaining_after_get_obs_s": (local_queue_deadline - get_obs_finished),
                "prediction_target_monotonic": prediction_target_monotonic,
                "observation_timing": observation_timing,
            },
            output_stream,
        )
        prediction_observation_monotonic = float(prediction_obs["timestamp"][-1])
        predict_started = time.monotonic()
        raw_native_action, inference_latency = _predict_live_plan(
            policy, prediction_obs, cfg.task.shape_meta, device
        )
        predict_finished = time.monotonic()
        scheduler_started = time.monotonic()
        native_action = scheduler.prepare_pending(raw_native_action[None])[0]
        scheduler_finished = time.monotonic()

        submission_obs = prediction_obs
        previous_local_deadline = local_queue_deadline
        before_submit = time.monotonic()
        prepare_timing = {
            "wait_until_s": wait_finished - wait_started,
            "wait_overshoot_s": wait_finished - trigger_time,
            "tail_remaining_after_wait_s": (previous_local_deadline - wait_finished),
            "get_obs_s": get_obs_finished - get_obs_started,
            "tail_remaining_after_get_obs_s": (previous_local_deadline - get_obs_finished),
            "predict_wrapper_s": predict_finished - predict_started,
            "inference_latency_s": inference_latency,
            "tail_remaining_after_predict_s": (previous_local_deadline - predict_finished),
            "scheduler_s": scheduler_finished - scheduler_started,
            "tail_remaining_before_submit_s": (previous_local_deadline - before_submit),
            "observation_timing": observation_timing,
            "first_action_lead_before_submit_s": (
                previous_local_deadline + float(action_offsets_s[0]) - before_submit
            ),
            "prepare_total_after_wait_s": before_submit - wait_finished,
            "prepare_total_s": before_submit - cycle_prepare_started,
        }
        result, accepted_queue_tail, actual_lead_time = _submit_live_plan(
            env,
            native_action,
            action_offsets_s,
            prediction_observation_monotonic,
            submission_obs,
            min_action_lead_time=effective_join_lead,
            expected_queue_tail=accepted_queue_tail,
            interpolation="linear_se3",
            output_stream=output_stream,
            preview_context={
                "cycle": cycle,
                "phase": "queue_continuation",
                "prepare_timing": prepare_timing,
            },
        )
        local_queue_deadline = accepted_queue_tail.local_deadline_monotonic
        emit_event(
            {
                "event": "live_cycle",
                "cycle": cycle,
                "phase": "queue_continuation",
                "interpolation": "linear_se3",
                "accepted_steps": result.accepted_steps,
                "inference_latency": inference_latency,
                "prediction_observation_age_s": _observation_age(prediction_obs),
                "prediction_target_monotonic": prediction_target_monotonic,
                "prediction_alignment_error_s": prediction_obs.get("diagnostics", {}).get(
                    "alignment_error_s"
                ),
                "negotiated_min_action_lead_s": negotiated_lead_time,
                "backend_minimum_submission_lead_s": backend_submission_lead,
                "effective_submission_lead_s": effective_join_lead,
                "actual_submission_lead_s": actual_lead_time,
                "inference_trigger_advance_s": _LIVE_INFERENCE_TRIGGER_ADVANCE_S,
                "previous_local_queue_deadline_monotonic": previous_local_deadline,
                "first_action_offset_s": float(action_offsets_s[0]),
                "local_queue_deadline_monotonic": local_queue_deadline,
                "queue_token": accepted_queue_tail.queue_token,
            },
            output_stream,
        )
        cycle += 1

    _wait_until(local_queue_deadline, backend=backend)
