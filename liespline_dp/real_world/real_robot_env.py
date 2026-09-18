import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Mapping, Optional, Protocol, Sequence

import cv2
import numpy as np


@contextmanager
def _defer_keyboard_interrupt(enabled: bool):
    can_defer_sigint = enabled and threading.current_thread() is threading.main_thread()
    if not can_defer_sigint:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGINT)
    pending_signal = []

    def defer_signal(signum, frame):
        pending_signal.append((signum, frame))

    signal.signal(signal.SIGINT, defer_signal)
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if pending_signal and not body_failed:
            signum, frame = pending_signal[-1]
            if callable(previous_handler):
                previous_handler(signum, frame)
            elif previous_handler == signal.SIG_DFL:
                signal.default_int_handler(signum, frame)


def _record_accepted_action_before_interrupt(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with _defer_keyboard_interrupt(self.accepted_action_callback is not None):
            return method(self, *args, **kwargs)

    return wrapped


class RealRobotObservationBackend(Protocol):
    def get_aligned_observation(
        self,
        n_steps: int,
        target_monotonic: Optional[float] = None,
    ) -> Mapping[str, np.ndarray]:
        """Return synchronized native observations, optionally at a target."""


class RealRobotCommandBackend(Protocol):
    def schedule_action_chunk(
        self,
        tcp_pose: np.ndarray,
        gripper_target_q: np.ndarray,
        action_offsets_s: np.ndarray,
        prediction_observation_monotonic: float,
        timing_reference_observation_monotonic: float,
        expected_queue_token: Optional[str],
        requested_min_lead_s: Optional[float] = None,
        interpolation: str = "linear_se3",
    ) -> "ActionScheduleResult":
        """Atomically seed or extend the synchronized action queue."""


@dataclass(frozen=True)
class ActionQueueTail:
    tcp_pose: np.ndarray
    gripper_target_q: float
    service_instance_id: str
    queue_token: str
    queue_tail_monotonic: float
    local_deadline_monotonic: float

    def __post_init__(self):
        tcp_pose = np.asarray(self.tcp_pose, dtype=np.float64)
        if tcp_pose.shape != (6,) or not np.all(np.isfinite(tcp_pose)):
            raise ValueError("ActionQueueTail.tcp_pose must be a finite [6] pose")
        gripper_target_q = float(self.gripper_target_q)
        queue_tail = float(self.queue_tail_monotonic)
        local_deadline = float(self.local_deadline_monotonic)
        instance_id = str(self.service_instance_id)
        queue_token = str(self.queue_token)
        if (
            not np.isfinite(gripper_target_q)
            or not np.isfinite(queue_tail)
            or not np.isfinite(local_deadline)
        ):
            raise ValueError("ActionQueueTail numeric values must be finite")
        if not instance_id or not queue_token:
            raise ValueError("ActionQueueTail instance and queue tokens must be non-empty")
        object.__setattr__(self, "tcp_pose", tcp_pose.copy())
        object.__setattr__(self, "gripper_target_q", gripper_target_q)
        object.__setattr__(self, "service_instance_id", instance_id)
        object.__setattr__(self, "queue_token", queue_token)
        object.__setattr__(self, "queue_tail_monotonic", queue_tail)
        object.__setattr__(self, "local_deadline_monotonic", local_deadline)


def continuation_prediction_target_monotonic(
    accepted_queue_tail: ActionQueueTail,
    n_latency_steps: int,
    frequency: float,
) -> float:
    if not isinstance(accepted_queue_tail, ActionQueueTail):
        raise TypeError("accepted_queue_tail must be an ActionQueueTail")
    n_latency_steps = int(n_latency_steps)
    frequency = float(frequency)
    if n_latency_steps < 0:
        raise ValueError("n_latency_steps must be non-negative")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive and finite")
    return accepted_queue_tail.queue_tail_monotonic - n_latency_steps / frequency


def wait_for_service_queue_drain(
    accepted_queue_tail: ActionQueueTail,
    backend,
) -> None:
    if not isinstance(accepted_queue_tail, ActionQueueTail):
        raise TypeError("accepted_queue_tail must be an ActionQueueTail")
    settled_after_tail = False
    while True:
        backend.raise_if_faulted()
        status = backend.get_status()
        backend.raise_if_faulted()
        queue_active = bool(status.get("queue_active", False))
        queue_remaining_s = float(status.get("queue_remaining_s", 0.0))
        if not np.isfinite(queue_remaining_s) or queue_remaining_s < 0.0:
            raise RuntimeError("Robot Service reported invalid queue_remaining_s while draining")
        if not queue_active:
            return
        if queue_remaining_s <= 0.0:
            if settled_after_tail:
                return
            control_rates = getattr(backend, "capabilities", {}).get("control_frequency_hz", {})
            try:
                ur_frequency_hz = float(control_rates.get("ur", 125.0))
            except (AttributeError, TypeError, ValueError):
                ur_frequency_hz = 125.0
            if not np.isfinite(ur_frequency_hz) or ur_frequency_hz <= 0.0:
                ur_frequency_hz = 125.0
            time.sleep(2.0 / ur_frequency_hz)
            settled_after_tail = True
            continue
        settled_after_tail = False
        local_remaining_s = accepted_queue_tail.local_deadline_monotonic - time.monotonic()
        sleep_s = min(queue_remaining_s, max(local_remaining_s, 0.001), 0.02)
        time.sleep(max(sleep_s, 0.001))


@dataclass(frozen=True)
class ActionScheduleResult:
    accepted: bool
    command_id: int
    accepted_steps: int
    service_instance_id: Optional[str]
    queue_token: Optional[str]
    queue_tail_monotonic: Optional[float]
    queue_remaining_s: float
    reason: str = ""
    queue_was_empty: Optional[bool] = None
    previous_queue_token: Optional[str] = None
    bootstrap_shift_s: float = 0.0
    local_deadline_monotonic: Optional[float] = None
    accepted_tail: Optional[ActionQueueTail] = None


class ActionScheduleError(RuntimeError):
    def __init__(self, result: ActionScheduleResult):
        self.result = result
        super().__init__(
            f"Action command {result.command_id} was rejected: "
            f"{result.reason or 'no reason supplied'}"
        )


class RealRobotEnv:
    """Thin native coordinator around synchronized acquisition and robot drivers."""

    required_obs_keys = (
        "wrist_rgb",
        "global_rgb",
        "eef_pos",
        "eef_quat",
        "gripper_q",
        "timestamp",
    )

    def __init__(
        self,
        observation_backend: RealRobotObservationBackend,
        command_backend: RealRobotCommandBackend,
        max_observation_age: Optional[float] = None,
        accepted_action_callback: Optional[
            Callable[[np.ndarray, np.ndarray, ActionScheduleResult], None]
        ] = None,
    ):
        self.observation_backend = observation_backend
        self.command_backend = command_backend
        self.accepted_action_callback = accepted_action_callback
        if max_observation_age is None:
            max_observation_age = getattr(observation_backend, "observation_max_age", None)
        if (
            max_observation_age is None
            or not np.isfinite(max_observation_age)
            or float(max_observation_age) <= 0.0
        ):
            raise ValueError("max_observation_age must be provided")
        self.max_observation_age = float(max_observation_age)
        self._last_obs = None
        self.last_schedule_result = None
        self.last_accepted_queue_tail = None
        self.last_schedule_shift_s = 0.0

    def _unique_backends(self, command_first: bool = False):
        ordered = (
            (self.command_backend, self.observation_backend)
            if command_first
            else (self.observation_backend, self.command_backend)
        )
        return tuple({id(backend): backend for backend in ordered}.values())

    def _raise_backend_faults(self):
        for backend in self._unique_backends():
            raise_if_faulted = getattr(backend, "raise_if_faulted", None)
            if raise_if_faulted is not None:
                raise_if_faulted()

    def start(self):
        started = []
        try:
            for backend in self._unique_backends():
                start = getattr(backend, "start", None)
                if start is not None:
                    start()
                started.append(backend)
            self._raise_backend_faults()
        except Exception:
            for backend in reversed(started):
                stop = getattr(backend, "stop", None)
                if stop is not None:
                    try:
                        stop()
                    except Exception:
                        pass
            raise

    def stop(self):
        first_error = None
        for backend in self._unique_backends(command_first=True):
            stop = getattr(backend, "stop", None)
            if stop is None:
                continue
            try:
                stop()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get_obs(
        self,
        n_steps: int,
        target_monotonic: Optional[float] = None,
    ):
        if int(n_steps) < 1:
            raise ValueError("n_steps must be positive")
        if target_monotonic is not None:
            target_monotonic = float(target_monotonic)
            if not np.isfinite(target_monotonic):
                raise ValueError("target_monotonic must be finite")
        self._raise_backend_faults()
        obs = dict(
            self.observation_backend.get_aligned_observation(
                int(n_steps),
                target_monotonic=target_monotonic,
            )
        )
        self._raise_backend_faults()
        missing = [key for key in self.required_obs_keys if key not in obs]
        if missing:
            raise KeyError(f"Observation backend is missing keys {missing}")
        timestamps = np.asarray(obs["timestamp"], dtype=np.float64)
        if timestamps.shape != (int(n_steps),):
            raise ValueError(f"Expected {n_steps} timestamps, got {timestamps.shape}")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Observation timestamps must be increasing")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("Observation timestamps must be finite")
        for key in self.required_obs_keys:
            value = np.asarray(obs[key])
            if value.shape[0] != int(n_steps):
                raise ValueError(
                    f"Observation {key!r} has {value.shape[0]} steps, expected {n_steps}"
                )
        for key in ("wrist_rgb", "global_rgb"):
            value = np.asarray(obs[key])
            if value.dtype != np.uint8:
                raise ValueError(f"Observation {key!r} must use uint8 RGB")
            if value.ndim != 4 or value.shape[-1] != 3:
                raise ValueError(f"Observation {key!r} must have shape [T,H,W,3]")
        expected_lowdim_shapes = {
            "eef_pos": (int(n_steps), 3),
            "eef_quat": (int(n_steps), 4),
            "gripper_q": (int(n_steps), 1),
        }
        for key, expected_shape in expected_lowdim_shapes.items():
            value = np.asarray(obs[key])
            if value.shape != expected_shape:
                raise ValueError(
                    f"Observation {key!r} has shape {value.shape}, expected {expected_shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Observation {key!r} contains non-finite values")
        quaternion_norms = np.linalg.norm(np.asarray(obs["eef_quat"], dtype=np.float64), axis=-1)
        if np.any(np.abs(quaternion_norms - 1.0) > 1e-2):
            raise ValueError("Observation eef_quat must contain normalized xyzw quaternions")
        diagnostics = obs.get("diagnostics", {})
        try:
            reported_age = float(diagnostics["observation_age_s"])
            received_at = float(diagnostics["observation_rpc_completed_monotonic"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Observation diagnostics must include monotonic age metadata") from exc
        age = reported_age + max(0.0, time.monotonic() - received_at)
        if not np.isfinite(age) or age < -0.05:
            raise RuntimeError(f"Invalid observation age: age={age:.3f}s")
        if age > self.max_observation_age:
            raise RuntimeError(f"Stale observation: age={age:.3f}s")
        obs["diagnostics"] = dict(diagnostics, conservative_observation_age_s=age)
        self._last_obs = obs
        return obs

    @_record_accepted_action_before_interrupt
    def exec_actions(
        self,
        tcp_pose: np.ndarray,
        gripper_target_q: np.ndarray,
        action_offsets_s: Sequence[float],
        prediction_observation_monotonic: Optional[float] = None,
        expected_queue_tail: Optional[ActionQueueTail] = None,
        interpolation: str = "linear_se3",
        requested_min_lead_s: Optional[float] = None,
    ) -> ActionScheduleResult:
        if self._last_obs is None:
            raise RuntimeError("get_obs must be called before exec_actions")
        self._raise_backend_faults()
        interpolation = str(interpolation)
        is_bootstrap = expected_queue_tail is None
        if expected_queue_tail is not None and not isinstance(expected_queue_tail, ActionQueueTail):
            raise TypeError("expected_queue_tail must be an ActionQueueTail or None")
        if requested_min_lead_s is not None:
            requested_min_lead_s = float(requested_min_lead_s)
            if not np.isfinite(requested_min_lead_s) or requested_min_lead_s <= 0.0:
                raise ValueError("requested_min_lead_s must be positive and finite")
            if not is_bootstrap:
                raise ValueError("requested_min_lead_s is only valid for an empty-queue bootstrap")

        action_offsets_s = np.asarray(action_offsets_s, dtype=np.float64)
        if action_offsets_s.ndim != 1:
            raise ValueError("action_offsets_s must be one-dimensional")
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
        gripper_target_q = np.asarray(gripper_target_q, dtype=np.float64).reshape(-1, 1)
        if len(action_offsets_s) != len(tcp_pose) or len(action_offsets_s) != len(gripper_target_q):
            raise ValueError("TCP, gripper target and action offset lengths differ")
        if len(action_offsets_s) < 1:
            raise ValueError("Action chunk cannot be empty")
        if (
            not np.all(np.isfinite(tcp_pose))
            or not np.all(np.isfinite(gripper_target_q))
            or not np.all(np.isfinite(action_offsets_s))
        ):
            raise ValueError("Action submission contains non-finite values")
        if action_offsets_s[0] <= 0.0 or np.any(np.diff(action_offsets_s) <= 0.0):
            raise ValueError("Action offsets must be positive and strictly increasing")

        latest = self._last_obs
        submission_observation_monotonic = float(latest["timestamp"][-1])
        diagnostics = latest.get("diagnostics", {})
        try:
            submission_age = float(diagnostics["observation_age_s"]) + max(
                0.0,
                time.monotonic() - float(diagnostics["observation_rpc_completed_monotonic"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Submission observation lacks monotonic age metadata") from exc
        if submission_age > self.max_observation_age:
            raise RuntimeError(
                f"Observation became stale during inference: age={submission_age:.3f}s"
            )
        if prediction_observation_monotonic is None:
            prediction_observation_monotonic = submission_observation_monotonic
        else:
            prediction_observation_monotonic = float(prediction_observation_monotonic)
            if not np.isfinite(prediction_observation_monotonic):
                raise ValueError("prediction_observation_monotonic must be finite")
            if prediction_observation_monotonic > (submission_observation_monotonic + 1e-6):
                raise ValueError(
                    "Prediction observation cannot be later than the fresh submission observation"
                )
        prediction_age = (
            submission_age + submission_observation_monotonic - prediction_observation_monotonic
        )
        if not np.isfinite(prediction_age) or prediction_age < -0.05:
            raise RuntimeError(f"Invalid prediction observation age: {prediction_age:.3f}s")
        if prediction_age > self.max_observation_age:
            raise RuntimeError(
                f"Prediction observation became stale before submission: age={prediction_age:.3f}s"
            )

        expected_queue_token = (
            None if expected_queue_tail is None else expected_queue_tail.queue_token
        )
        raw_result = self.command_backend.schedule_action_chunk(
            tcp_pose,
            gripper_target_q,
            action_offsets_s,
            prediction_observation_monotonic,
            submission_observation_monotonic,
            expected_queue_token=expected_queue_token,
            requested_min_lead_s=requested_min_lead_s,
            interpolation=interpolation,
        )
        if not isinstance(raw_result, ActionScheduleResult):
            raise TypeError("schedule_action_chunk must return ActionScheduleResult")
        result = ActionScheduleResult(
            accepted=raw_result.accepted,
            command_id=raw_result.command_id,
            accepted_steps=raw_result.accepted_steps,
            service_instance_id=raw_result.service_instance_id,
            queue_token=raw_result.queue_token,
            queue_tail_monotonic=raw_result.queue_tail_monotonic,
            queue_remaining_s=raw_result.queue_remaining_s,
            reason=raw_result.reason,
            queue_was_empty=raw_result.queue_was_empty,
            previous_queue_token=raw_result.previous_queue_token,
            bootstrap_shift_s=raw_result.bootstrap_shift_s,
            local_deadline_monotonic=raw_result.local_deadline_monotonic,
        )
        self.last_schedule_shift_s = result.bootstrap_shift_s
        self.last_schedule_result = result
        if not result.accepted:
            raise ActionScheduleError(result)

        contract_error = None
        if result.accepted_steps != len(action_offsets_s):
            contract_error = result.reason or "server did not atomically accept the complete chunk"
        elif not result.service_instance_id:
            contract_error = "accepted trajectory omitted service_instance_id"
        elif not result.queue_token:
            contract_error = "accepted trajectory omitted queue_token"
        elif result.queue_tail_monotonic is None or not np.isfinite(result.queue_tail_monotonic):
            contract_error = "accepted trajectory omitted queue_tail_monotonic"
        elif result.local_deadline_monotonic is None or not np.isfinite(
            result.local_deadline_monotonic
        ):
            contract_error = "accepted trajectory omitted local monotonic deadline"
        elif result.queue_was_empty is not None and bool(result.queue_was_empty) != is_bootstrap:
            contract_error = "server queue state disagreed with the expected queue tail"
        elif is_bootstrap and result.previous_queue_token is not None:
            contract_error = "bootstrap ACK reported a previous queue token"
        elif not is_bootstrap and result.previous_queue_token != expected_queue_token:
            contract_error = "server continued from an unexpected queue token"
        elif (
            expected_queue_tail is not None
            and result.service_instance_id != expected_queue_tail.service_instance_id
        ):
            contract_error = "service instance changed during queue continuation"

        if contract_error is not None:
            invalid = ActionScheduleResult(
                accepted=False,
                command_id=result.command_id,
                accepted_steps=result.accepted_steps,
                service_instance_id=result.service_instance_id,
                queue_token=result.queue_token,
                queue_tail_monotonic=result.queue_tail_monotonic,
                queue_remaining_s=result.queue_remaining_s,
                reason=contract_error,
                queue_was_empty=result.queue_was_empty,
                previous_queue_token=result.previous_queue_token,
                bootstrap_shift_s=result.bootstrap_shift_s,
                local_deadline_monotonic=result.local_deadline_monotonic,
            )
            self.last_schedule_result = invalid
            raise ActionScheduleError(invalid)

        accepted_tail = ActionQueueTail(
            tcp_pose=tcp_pose[-1],
            gripper_target_q=float(gripper_target_q[-1, 0]),
            service_instance_id=result.service_instance_id,
            queue_token=result.queue_token,
            queue_tail_monotonic=float(result.queue_tail_monotonic),
            local_deadline_monotonic=float(result.local_deadline_monotonic),
        )
        completed = ActionScheduleResult(
            accepted=True,
            command_id=result.command_id,
            accepted_steps=result.accepted_steps,
            service_instance_id=result.service_instance_id,
            queue_token=result.queue_token,
            queue_tail_monotonic=result.queue_tail_monotonic,
            queue_remaining_s=result.queue_remaining_s,
            reason=result.reason,
            queue_was_empty=result.queue_was_empty,
            previous_queue_token=result.previous_queue_token,
            bootstrap_shift_s=result.bootstrap_shift_s,
            local_deadline_monotonic=result.local_deadline_monotonic,
            accepted_tail=accepted_tail,
        )
        self.last_schedule_result = completed
        self.last_accepted_queue_tail = accepted_tail
        if self.accepted_action_callback is not None:
            self.accepted_action_callback(
                tcp_pose.copy(),
                action_offsets_s.copy(),
                completed,
            )
        self._raise_backend_faults()
        return completed


def wait_for_heartbeat_watchdog_disarm(backend) -> None:
    try:
        heartbeat_timeout_s = float(backend.capabilities["watchdog"]["heartbeat_timeout_s"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Robot Service omitted heartbeat watchdog timing") from exc
    if not np.isfinite(heartbeat_timeout_s) or heartbeat_timeout_s <= 0.0:
        raise RuntimeError("Robot Service reported invalid heartbeat watchdog timing")
    with _defer_keyboard_interrupt(True):
        time.sleep(heartbeat_timeout_s + 0.1)


def prepare_real_robot_policy_obs(
    env_obs: Mapping[str, np.ndarray],
    shape_meta: Mapping,
) -> dict:
    result = {}
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        expected_shape = tuple(attr["shape"])
        value = np.asarray(env_obs[key])
        if obs_type == "rgb":
            channels, output_h, output_w = expected_shape
            if channels != 3 or value.ndim != 4 or value.shape[-1] != 3:
                raise ValueError(f"Invalid RGB observation {key}: {value.shape}")
            if value.shape[1:3] != (output_h, output_w):
                value = np.stack(
                    [
                        cv2.resize(
                            image,
                            (output_w, output_h),
                            interpolation=cv2.INTER_AREA,
                        )
                        for image in value
                    ]
                )
            result[key] = np.moveaxis(value, -1, 1).astype(np.float32) / 255.0
        elif obs_type == "low_dim":
            if value.shape[1:] != expected_shape:
                raise ValueError(
                    f"Invalid lowdim observation {key}: {value.shape}, "
                    f"expected [T,{expected_shape}]"
                )
            result[key] = value.astype(np.float32)
        else:
            raise ValueError(f"Unsupported observation type {obs_type}")
    return result
