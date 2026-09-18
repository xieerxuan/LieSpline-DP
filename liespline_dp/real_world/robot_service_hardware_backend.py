"""Adapter from the UR5e robot service to the real-robot environment seam."""

from __future__ import annotations

import threading
import time
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import zmq

from liespline_dp.common.pose_action_codec import PoseActionCodec
from liespline_dp.real_world.real_robot_env import (
    ActionScheduleResult,
)
from liespline_dp.real_world.robot_service_client import (
    PROTOCOL_NAME,
    RobotServiceClient,
    RobotServiceError,
    RobotServiceInstanceChangedError,
    RobotServiceProtocolError,
    RobotServiceSubscriber,
)

POLICY_OBSERVATION_FIELDS = (
    "robot_tcp_pose",
    "gripper_actual_q",
    "wrist_rgb",
    "global_rgb",
)
_ACTION_LIMIT_KEYS = (
    "workspace_min",
    "workspace_max",
    "gripper_target_min",
    "gripper_target_max",
    "max_position_step_m",
    "max_rotation_step_rad",
    "max_gripper_step",
    "max_position_speed_m_s",
    "max_rotation_speed_rad_s",
    "max_gripper_speed_s",
    "min_action_lead_s",
    "max_action_horizon_s",
    "max_source_observation_age_s",
)


class RobotServiceBackendError(RuntimeError):
    pass


class RobotServiceCompatibilityError(RobotServiceBackendError):
    pass


class RobotServiceFaultError(RobotServiceBackendError):
    pass


class AmbiguousLeaseAcquisitionError(TimeoutError):
    """A lease reply was lost and safe reacquisition did not complete."""


class AmbiguousCommandTimeoutError(TimeoutError):
    """The service may have accepted a command whose ACK was not received."""

    def __init__(self, command_id: int, attempts: int):
        self.command_id = int(command_id)
        self.attempts = int(attempts)
        super().__init__(
            "Robot-service command acceptance is ambiguous after "
            f"{self.attempts} timed-out attempts with command_id="
            f"{self.command_id}; the backend has entered fail-safe cleanup"
        )


class AmbiguousCancelTimeoutError(TimeoutError):
    """The service may have cleared the queue whose ACK was not received."""

    def __init__(self, attempts: int):
        self.attempts = int(attempts)
        super().__init__(
            "Robot-service queue cancellation is ambiguous after "
            f"{self.attempts} timed-out attempts; the backend has entered "
            "fail-safe cleanup"
        )


class RobotServiceHardwareBackend:
    """Policy-facing client for the hardware-owned robot service.

    ``start()`` is deliberately read-only. A control lease and heartbeat are
    created only by ``arm()``. Action safety and target-to-command conversion
    remain service-owned. Timing options are in seconds, except for
    ``observation_frequency_hz``. Configuration keys match the constructor
    parameter names in ``config/real_world/robot_service_backend.yaml``.
    """

    def __init__(
        self,
        rpc_address: str = "ipc:///tmp/ur5e_robot_service/rpc.sock",
        stream_address: str = "ipc:///tmp/ur5e_robot_service/stream.sock",
        rpc_timeout: float = 2.0,
        heartbeat_interval: float = 0.1,
        status_staleness_timeout: float = 0.5,
        observation_frequency_hz: float = 20.0,
        observation_max_age: float = 0.25,
        observation_max_skew: float = 0.03,
        observation_wait_timeout: float = 0.25,
        action_lead_margin: float = 0.01,
        timeout_retry_count: int = 1,
        client_id: Optional[str] = None,
        required_service_fields: Sequence[str] = POLICY_OBSERVATION_FIELDS,
        context: Optional[zmq.Context] = None,
    ):
        positive_options = {
            "rpc_timeout": rpc_timeout,
            "heartbeat_interval": heartbeat_interval,
            "status_staleness_timeout": status_staleness_timeout,
            "observation_frequency_hz": observation_frequency_hz,
            "observation_max_age": observation_max_age,
            "observation_max_skew": observation_max_skew,
            "observation_wait_timeout": observation_wait_timeout,
        }
        for name, value in positive_options.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(action_lead_margin) or float(action_lead_margin) < 0.0:
            raise ValueError("action_lead_margin must be non-negative and finite")
        if int(timeout_retry_count) != timeout_retry_count or timeout_retry_count < 0:
            raise ValueError("timeout_retry_count must be a non-negative integer")

        required = tuple(str(field) for field in required_service_fields)
        missing_policy_fields = sorted(set(POLICY_OBSERVATION_FIELDS) - set(required))
        if missing_policy_fields:
            raise ValueError(
                f"required_service_fields must include policy fields {missing_policy_fields}"
            )
        if len(set(required)) != len(required):
            raise ValueError("required_service_fields must not contain duplicates")

        self.rpc_address = str(rpc_address)
        self.stream_address = str(stream_address)
        self.rpc_timeout = float(rpc_timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self.status_staleness_timeout = float(status_staleness_timeout)
        self.observation_frequency_hz = float(observation_frequency_hz)
        self.observation_max_age = float(observation_max_age)
        self.observation_max_skew = float(observation_max_skew)
        self.observation_wait_timeout = float(observation_wait_timeout)
        self.action_lead_margin = float(action_lead_margin)
        self.timeout_retry_count = int(timeout_retry_count)
        self.client_id = None if client_id is None else str(client_id)
        self.required_service_fields = required
        self._context = context

        self.capabilities = {}
        self.server_status = {}
        self.service_instance_id = None
        self.queue_token = None
        self.queue_tail_monotonic = None
        self.local_queue_deadline_monotonic = None
        self.queue_remaining_s = 0.0
        self.last_ipc_rtt = None
        self.last_heartbeat_error = None
        self.last_status_error = None
        self.last_status_time = None
        self.last_status_sequence = None
        self.min_action_lead_s = 0.0
        self.minimum_submission_lead_s = 0.0

        self.last_interpolation = None
        self.last_command_id = None
        self.last_command_duplicate = None
        self.last_submission_lead_s = None
        self.last_schedule_ack = None

        self._codec = PoseActionCodec()
        self._client = None
        self._started = False
        self._owns_lease = False
        self._lease_lost = False
        self._armed_expected = False
        self._command_id = 0
        self._reset_queue_epoch()
        self._fatal_error = None

        self._status_lock = threading.Lock()
        self._last_status_monotonic = None
        self._status_stop = threading.Event()
        self._status_thread = None

    @staticmethod
    def _require_mapping(value, name: str) -> Mapping:
        if not isinstance(value, Mapping):
            raise RobotServiceCompatibilityError(f"{name} must be a mapping")
        return value

    @staticmethod
    def _schema_shape(schema: Mapping, field: str) -> Tuple[int, ...]:
        try:
            return tuple(int(value) for value in schema["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                f"Service field {field!r} has an invalid shape schema"
            ) from exc

    @staticmethod
    def _schema_dtype(schema: Mapping, field: str) -> np.dtype:
        try:
            return np.dtype(schema["dtype"])
        except (KeyError, TypeError) as exc:
            raise RobotServiceCompatibilityError(
                f"Service field {field!r} has an invalid dtype schema"
            ) from exc

    def _validate_capabilities(self, capabilities: Mapping) -> None:
        capabilities = self._require_mapping(capabilities, "capabilities")
        if capabilities.get("protocol_name") != PROTOCOL_NAME:
            raise RobotServiceCompatibilityError(
                f"Robot service protocol identity does not match {PROTOCOL_NAME!r}"
            )
        if capabilities.get("tcp_pose_format") != "[x,y,z,rx,ry,rz]":
            raise RobotServiceCompatibilityError(
                "Robot service must expose TCP poses as [x,y,z,rx,ry,rz]"
            )
        rotation = str(capabilities.get("tcp_rotation", "")).lower()
        if "rotation vector" not in rotation or "radian" not in rotation:
            raise RobotServiceCompatibilityError(
                "Robot service TCP rotation must be a rotation vector in radians"
            )
        clock = self._require_mapping(capabilities.get("clock"), "capabilities.clock")
        if (
            clock.get("type") != "monotonic"
            or clock.get("scope") != "service_process"
            or capabilities.get("trajectory_time_mode") != "relative_offsets"
        ):
            raise RobotServiceCompatibilityError(
                "Robot service must expose monotonic time with relative trajectory offsets"
            )
        instance_id = str(capabilities.get("service_instance_id", ""))
        if not instance_id:
            raise RobotServiceCompatibilityError("Robot service must expose service_instance_id")

        interpolations = set(capabilities.get("trajectory_interpolation", ()))
        if "linear_se3" not in interpolations:
            raise RobotServiceCompatibilityError(
                "Robot service must support linear_se3 interpolation"
            )
        observation_alignment_modes = set(capabilities.get("observation_alignment_modes", ()))
        required_alignment_modes = {"latest", "target_snapshot"}
        missing_alignment_modes = sorted(required_alignment_modes - observation_alignment_modes)
        if missing_alignment_modes:
            raise RobotServiceCompatibilityError(
                f"Robot service is missing observation alignment modes {missing_alignment_modes}"
            )

        enabled_fields = set(capabilities.get("enabled_fields", ()))
        field_schemas = self._require_mapping(capabilities.get("fields"), "capabilities.fields")
        missing = [
            field
            for field in self.required_service_fields
            if field not in enabled_fields or field not in field_schemas
        ]
        if missing:
            raise RobotServiceCompatibilityError(
                f"Robot service is missing required fields {missing}"
            )

        expected_schemas = {
            "robot_tcp_pose": (np.dtype(np.float32), (6,)),
            "gripper_actual_q": (np.dtype(np.float64), ()),
        }
        for field, (expected_dtype, expected_shape) in expected_schemas.items():
            schema = self._require_mapping(field_schemas[field], f"capabilities.fields.{field}")
            dtype = self._schema_dtype(schema, field)
            shape = self._schema_shape(schema, field)
            if dtype != expected_dtype or shape != expected_shape:
                raise RobotServiceCompatibilityError(
                    f"Service field {field!r} must have dtype "
                    f"{expected_dtype} and shape {expected_shape}, got "
                    f"{dtype} and {shape}"
                )
        for field in ("wrist_rgb", "global_rgb"):
            schema = self._require_mapping(field_schemas[field], f"capabilities.fields.{field}")
            dtype = self._schema_dtype(schema, field)
            shape = self._schema_shape(schema, field)
            if (
                dtype != np.dtype(np.uint8)
                or len(shape) != 3
                or shape[-1] != 3
                or any(value <= 0 for value in shape)
            ):
                raise RobotServiceCompatibilityError(
                    f"Service field {field!r} must be [H,W,3] uint8"
                )

        watchdog = self._require_mapping(capabilities.get("watchdog"), "capabilities.watchdog")
        try:
            heartbeat_timeout = float(watchdog["heartbeat_timeout_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Robot service heartbeat timeout is missing or invalid"
            ) from exc
        if not np.isfinite(heartbeat_timeout) or heartbeat_timeout <= self.heartbeat_interval:
            raise RobotServiceCompatibilityError(
                "heartbeat_interval must be shorter than the service watchdog "
                f"timeout ({heartbeat_timeout!r}s)"
            )
        try:
            service_wait_timeout = float(capabilities["observation_wait_timeout_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Robot service must expose observation_wait_timeout_s"
            ) from exc
        if not np.isfinite(service_wait_timeout) or service_wait_timeout <= 0.0:
            raise RobotServiceCompatibilityError(
                "observation_wait_timeout_s must be positive and finite"
            )
        if (
            max(service_wait_timeout, self.observation_wait_timeout) + self.heartbeat_interval
            >= heartbeat_timeout
        ):
            raise RobotServiceCompatibilityError(
                "Service/request observation wait plus heartbeat_interval must "
                "stay below the heartbeat watchdog timeout"
            )

        limits = self._require_mapping(
            capabilities.get("action_limits"), "capabilities.action_limits"
        )
        missing_limits = [key for key in _ACTION_LIMIT_KEYS if key not in limits]
        if missing_limits:
            raise RobotServiceCompatibilityError(
                f"Robot service action limits are missing {missing_limits}"
            )
        try:
            workspace_min = np.asarray(limits["workspace_min"], dtype=np.float64)
            workspace_max = np.asarray(limits["workspace_max"], dtype=np.float64)
            gripper_target_min = float(limits["gripper_target_min"])
            gripper_target_max = float(limits["gripper_target_max"])
            positive_limits = np.asarray(
                [limits[key] for key in _ACTION_LIMIT_KEYS[4:]],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Robot service action limits must be numeric"
            ) from exc
        if (
            workspace_min.shape != (3,)
            or workspace_max.shape != (3,)
            or not np.all(np.isfinite(workspace_min))
            or not np.all(np.isfinite(workspace_max))
            or np.any(workspace_min >= workspace_max)
            or not np.isfinite(gripper_target_min)
            or not np.isfinite(gripper_target_max)
            or gripper_target_min >= gripper_target_max
            or not np.all(np.isfinite(positive_limits))
            or np.any(positive_limits <= 0.0)
        ):
            raise RobotServiceCompatibilityError("Robot service action limits are invalid")

        required_sources = capabilities.get("control_required_sources")
        if not isinstance(required_sources, (list, tuple)) or any(
            not str(source) for source in required_sources
        ):
            raise RobotServiceCompatibilityError(
                "control_required_sources must be a sequence of source names"
            )

        self.min_action_lead_s = float(limits["min_action_lead_s"])
        self.minimum_submission_lead_s = self.min_action_lead_s + self.action_lead_margin

    def _validate_status(self, status: Mapping) -> None:
        status = self._require_mapping(status, "status")
        required = (
            "service_instance_id",
            "status_sequence",
            "armed",
            "last_fault",
            "lease_active",
            "lease_client_id",
            "queue_active",
            "queue_tail_monotonic",
            "queue_remaining_s",
            "scheduler_error",
            "source_status",
        )
        missing = [name for name in required if name not in status]
        if missing:
            raise RobotServiceCompatibilityError(
                f"Robot service status is missing fields {missing}"
            )
        if str(status["service_instance_id"]) != str(self.service_instance_id):
            raise RobotServiceCompatibilityError(
                "Robot service instance changed during the backend session"
            )
        try:
            status_sequence = int(status["status_sequence"])
        except (TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Robot service status_sequence must be an integer"
            ) from exc
        if status_sequence < 0 or status_sequence != status["status_sequence"]:
            raise RobotServiceCompatibilityError(
                "Robot service status_sequence must be a non-negative integer"
            )
        for name in ("armed", "lease_active", "queue_active"):
            if not isinstance(status[name], (bool, np.bool_)):
                raise RobotServiceCompatibilityError(
                    f"Robot service status field {name!r} must be boolean"
                )
        queue_tail = status["queue_tail_monotonic"]
        if queue_tail is not None and not np.isfinite(float(queue_tail)):
            raise RobotServiceCompatibilityError("queue_tail_monotonic must be finite or null")
        queue_remaining = float(status["queue_remaining_s"])
        if not np.isfinite(queue_remaining) or queue_remaining < 0.0:
            raise RobotServiceCompatibilityError(
                "queue_remaining_s must be non-negative and finite"
            )
        source_status = self._require_mapping(status["source_status"], "status.source_status")
        for source, value in source_status.items():
            if not isinstance(value, Mapping):
                raise RobotServiceCompatibilityError(
                    f"Status for source {source!r} must be a mapping"
                )

    def _protocol_fail_safe_cleanup(self) -> None:
        client = self._client
        owned_lease = self._owns_lease and not self._lease_lost
        armed = self._armed_expected
        cleanup_errors = []
        if client is not None:
            try:
                client.stop_heartbeat()
            except Exception as exc:
                cleanup_errors.append(f"stop heartbeat: {exc}")
            if owned_lease and armed:
                try:
                    client.disarm("protocol response became ambiguous")
                except Exception as exc:
                    cleanup_errors.append(f"disarm: {exc}")
            if owned_lease:
                try:
                    client.release_control()
                except Exception as exc:
                    cleanup_errors.append(f"release lease: {exc}")
        self._lease_lost = True
        self._owns_lease = False
        self._armed_expected = False
        self._reset_queue_epoch()
        if cleanup_errors:
            self.last_status_error = (
                "Protocol-fault cleanup was incomplete; service watchdog remains "
                "the final stop backstop: " + "; ".join(cleanup_errors)
            )

    def _timed_call(self, function, *args, **kwargs):
        started_at = time.perf_counter()
        try:
            return function(*args, **kwargs)
        except (
            RobotServiceInstanceChangedError,
            RobotServiceProtocolError,
        ) as exc:
            fault = RobotServiceFaultError(str(exc))
            self._fatal_error = fault
            self._protocol_fail_safe_cleanup()
            raise fault from exc
        finally:
            self.last_ipc_rtt = time.perf_counter() - started_at

    def _reset_queue_epoch(self) -> None:
        self.queue_token = None
        self.queue_tail_monotonic = None
        self.local_queue_deadline_monotonic = None
        self.queue_remaining_s = 0.0

    def _update_status(self, status: Mapping) -> None:
        self._validate_status(status)
        copied = dict(status)
        status_sequence = int(copied["status_sequence"])
        monotonic_now = time.monotonic()
        status_command_id = int(copied.get("last_command_id", -1))
        with self._status_lock:
            previous_sequence = self.last_status_sequence
            if previous_sequence is not None and status_sequence <= previous_sequence:
                return
            authoritative_empty = not bool(copied["queue_active"]) and (
                self.last_command_id is None or status_command_id >= self.last_command_id
            )
            self.server_status = copied
            self.queue_remaining_s = float(copied["queue_remaining_s"])
            if authoritative_empty:
                self._reset_queue_epoch()
            self.last_status_time = monotonic_now
            self.last_status_sequence = status_sequence
            self._last_status_monotonic = monotonic_now
            self.last_status_error = None

    def _refresh_status(self) -> Mapping:
        self._require_started()
        status = self._timed_call(self._client.get_status)
        self._update_status(status)
        return status

    def _status_loop(self) -> None:
        subscriber = None
        try:
            subscriber_kwargs = {
                "stream_address": self.stream_address,
                "topics": ("status",),
            }
            if self._context is not None:
                subscriber_kwargs["context"] = self._context
            subscriber = RobotServiceSubscriber(**subscriber_kwargs)
            timeout_ms = max(
                10,
                min(100, int(round(self.status_staleness_timeout * 250.0))),
            )
            while not self._status_stop.is_set():
                try:
                    topic, header, _ = subscriber.receive(timeout_ms=timeout_ms)
                except TimeoutError:
                    continue
                except Exception as exc:
                    self.last_status_error = str(exc)
                    if self._status_stop.wait(0.01):
                        break
                    continue
                if topic != "status":
                    continue
                try:
                    self._update_status(header)
                except Exception as exc:
                    self.last_status_error = str(exc)
        except Exception as exc:
            self.last_status_error = str(exc)
        finally:
            if subscriber is not None:
                try:
                    subscriber.close()
                except Exception:
                    pass

    def start(self) -> None:
        if self._started:
            return
        self._fatal_error = None
        self._owns_lease = False
        self._lease_lost = False
        self._armed_expected = False
        self._command_id = 0
        self._reset_queue_epoch()
        self.last_heartbeat_error = None
        self.last_status_error = None
        self.last_status_sequence = None
        self._last_status_monotonic = None
        timeout_ms = max(1, int(round(self.rpc_timeout * 1000.0)))
        client_kwargs = {
            "rpc_address": self.rpc_address,
            "client_id": self.client_id,
            "timeout_ms": timeout_ms,
            "auto_release_on_close": False,
        }
        if self._context is not None:
            client_kwargs["context"] = self._context
        client = RobotServiceClient(**client_kwargs)
        self._client = client
        self.client_id = client.client_id
        try:
            capabilities = self._timed_call(client.get_capabilities)
            self._validate_capabilities(capabilities)
            self.capabilities = dict(capabilities)
            self.service_instance_id = str(capabilities["service_instance_id"])
            status = self._timed_call(client.get_status)
            self._update_status(status)
            self._status_stop.clear()
            self._started = True
            self._status_thread = threading.Thread(
                target=self._status_loop,
                name="robot-service-status",
                daemon=True,
            )
            self._status_thread.start()
        except Exception:
            self._started = False
            self._client = None
            try:
                client.close()
            except Exception:
                pass
            raise

    def _require_started(self) -> None:
        if not self._started or self._client is None:
            raise RuntimeError("RobotServiceHardwareBackend.start() has not been called")

    def get_aligned_observation(
        self,
        n_steps: int,
        target_monotonic: Optional[float] = None,
    ):
        self._require_started()
        if self._fatal_error is not None:
            raise self._fatal_error
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        request_kwargs = {
            "n_steps": n_steps,
            "fields": list(POLICY_OBSERVATION_FIELDS),
            "frequency_hz": self.observation_frequency_hz,
            "max_age_s": self.observation_max_age,
            "max_skew_s": self.observation_max_skew,
            "wait_timeout_s": self.observation_wait_timeout,
        }
        if target_monotonic is not None:
            target_monotonic = float(target_monotonic)
            if not np.isfinite(target_monotonic):
                raise ValueError("target_monotonic must be finite")
            request_kwargs.update(
                {
                    "alignment_mode": "target_snapshot",
                    "target_monotonic": target_monotonic,
                }
            )
        call_started_monotonic = time.monotonic()
        header, arrays = self._timed_call(
            self._client.get_observation,
            **request_kwargs,
        )
        observation_rpc_completed_monotonic = time.monotonic()
        client_request_timing = dict(getattr(self._client, "last_request_timing", {}) or {})
        if str(header.get("service_instance_id", "")) != str(self.service_instance_id):
            raise RobotServiceCompatibilityError(
                "Observation came from an unexpected service instance"
            )
        required_arrays = [
            *POLICY_OBSERVATION_FIELDS,
            "monotonic_timestamp",
            "source_monotonic_timestamp",
        ]
        missing = [field for field in required_arrays if field not in arrays]
        if missing:
            raise RobotServiceCompatibilityError(
                f"Observation response is missing arrays {missing}"
            )

        tcp_pose = np.asarray(arrays["robot_tcp_pose"])
        if tcp_pose.shape != (n_steps, 6) or not np.all(np.isfinite(tcp_pose)):
            raise RobotServiceCompatibilityError(
                "robot_tcp_pose observation must be finite with shape [T,6]"
            )
        timestamp = np.asarray(arrays["monotonic_timestamp"], dtype=np.float64)
        if (
            timestamp.shape != (n_steps,)
            or not np.all(np.isfinite(timestamp))
            or np.any(np.diff(timestamp) <= 0.0)
        ):
            raise RobotServiceCompatibilityError(
                "monotonic_timestamp must be finite and increasing with shape [T]"
            )
        try:
            response_frequency_hz = float(header["frequency_hz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Observation response omitted frequency_hz"
            ) from exc
        frequency_tolerance = max(
            1e-6,
            abs(self.observation_frequency_hz) * 1e-6,
        )
        if (
            not np.isfinite(response_frequency_hz)
            or abs(response_frequency_hz - self.observation_frequency_hz) > frequency_tolerance
        ):
            raise RobotServiceCompatibilityError(
                "Observation response frequency does not match the requested "
                f"{self.observation_frequency_hz:.9g} Hz"
            )
        if n_steps > 1:
            expected_period_s = 1.0 / self.observation_frequency_hz
            period_tolerance_s = max(1e-9, expected_period_s * 1e-6)
            if not np.allclose(
                np.diff(timestamp), expected_period_s, rtol=0.0, atol=period_tolerance_s
            ):
                raise RobotServiceCompatibilityError(
                    "Observation timeline spacing does not match the requested "
                    f"{self.observation_frequency_hz:.9g} Hz"
                )
        try:
            observation_monotonic = float(header["observation_monotonic"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Observation response omitted observation_monotonic"
            ) from exc
        if (
            not np.isfinite(observation_monotonic)
            or abs(observation_monotonic - float(timestamp[-1])) > 1e-6
        ):
            raise RobotServiceCompatibilityError(
                "Observation endpoint metadata does not match the timeline"
            )
        fields = header.get("fields")
        if fields != list(POLICY_OBSERVATION_FIELDS):
            raise RobotServiceCompatibilityError(
                "Observation response fields do not match the request order"
            )
        source_timestamp_raw = np.asarray(arrays["source_monotonic_timestamp"])
        source_timestamp = np.asarray(source_timestamp_raw, dtype=np.float64)
        expected_source_shape = (n_steps, len(POLICY_OBSERVATION_FIELDS))
        if (
            source_timestamp_raw.dtype != np.dtype(np.float64)
            or source_timestamp.shape != expected_source_shape
            or not np.all(np.isfinite(source_timestamp))
            or np.any(np.diff(source_timestamp, axis=0) < 0.0)
        ):
            raise RobotServiceCompatibilityError(
                "source_monotonic_timestamp must be nondecreasing float64 with shape [T,F]"
            )
        measured_skew_s = float(np.max(np.abs(source_timestamp - timestamp[:, None])))
        try:
            reported_skew_s = float(header["max_sensor_skew_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Observation response omitted max_sensor_skew_s"
            ) from exc
        skew_tolerance_s = max(1e-9, self.observation_max_skew * 1e-6)
        if (
            not np.isfinite(reported_skew_s)
            or reported_skew_s < 0.0
            or abs(reported_skew_s - measured_skew_s) > skew_tolerance_s
            or measured_skew_s > self.observation_max_skew + skew_tolerance_s
        ):
            raise RobotServiceCompatibilityError(
                "Observation source timestamps violate the requested skew contract"
            )
        if target_monotonic is not None:
            try:
                alignment_mode = str(header["alignment_mode"])
                echoed_target = float(header["target_monotonic"])
                alignment_error_s = float(header["alignment_error_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RobotServiceCompatibilityError(
                    "Target snapshot response omitted alignment metadata"
                ) from exc
            expected_alignment_error_s = target_monotonic - observation_monotonic
            if (
                alignment_mode != "target_snapshot"
                or not np.isfinite(echoed_target)
                or not np.isfinite(alignment_error_s)
                or abs(echoed_target - target_monotonic) > 1e-6
                or abs(alignment_error_s - expected_alignment_error_s) > 1e-6
            ):
                raise RobotServiceCompatibilityError(
                    "Target snapshot response has inconsistent alignment metadata"
                )
        try:
            observation_age_s = float(header["observation_age_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotServiceCompatibilityError(
                "Observation response omitted observation_age_s"
            ) from exc
        age_tolerance_s = 1e-6
        if (
            not np.isfinite(observation_age_s)
            or observation_age_s < -0.05
            or observation_age_s > self.observation_max_age + age_tolerance_s
        ):
            raise RobotServiceCompatibilityError(
                "Observation response reported an invalid monotonic age"
            )
        gripper_raw = np.asarray(arrays["gripper_actual_q"], dtype=np.float64)
        if gripper_raw.shape == (n_steps, 1):
            gripper_raw = gripper_raw[:, 0]
        if gripper_raw.shape != (n_steps,) or not np.all(np.isfinite(gripper_raw)):
            raise RobotServiceCompatibilityError(
                "gripper_actual_q observation must be finite with shape [T]"
            )
        for field in ("wrist_rgb", "global_rgb"):
            value = np.asarray(arrays[field])
            if (
                value.dtype != np.uint8
                or value.ndim != 4
                or value.shape[0] != n_steps
                or value.shape[-1] != 3
            ):
                raise RobotServiceCompatibilityError(f"{field} observation must be [T,H,W,3] uint8")

        gripper_q = gripper_raw.reshape(n_steps, 1)

        quaternion = np.asarray(
            self._codec.measured_rotvec_to_quaternion(tcp_pose[:, 3:6]),
            dtype=np.float64,
        )
        quaternion_norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        if np.any(~np.isfinite(quaternion_norm)) or np.any(quaternion_norm <= 1e-12):
            raise RobotServiceCompatibilityError(
                "robot_tcp_pose produced an invalid rotation quaternion"
            )
        quaternion = quaternion / quaternion_norm

        diagnostics = dict(header)
        if "max_sensor_skew_s" in diagnostics:
            diagnostics["max_sensor_skew"] = diagnostics["max_sensor_skew_s"]
        elif "max_sensor_skew" in diagnostics:
            diagnostics["max_sensor_skew_s"] = diagnostics["max_sensor_skew"]
        diagnostics.update(
            {
                "observation_age_s": observation_age_s,
                "observation_client_call_s": (
                    observation_rpc_completed_monotonic - call_started_monotonic
                ),
                "observation_client_request_timing": client_request_timing,
                "observation_rpc_completed_monotonic": (observation_rpc_completed_monotonic),
            }
        )
        return {
            "timestamp": timestamp,
            "wrist_rgb": np.asarray(arrays["wrist_rgb"]),
            "global_rgb": np.asarray(arrays["global_rgb"]),
            "eef_pos": np.asarray(tcp_pose[:, :3]),
            "eef_quat": quaternion,
            "gripper_q": gripper_q,
            "diagnostics": diagnostics,
        }

    def _next_command_id(self) -> int:
        command_id = self._command_id
        self._command_id += 1
        return command_id

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
    ) -> ActionScheduleResult:
        self._require_started()
        if self._fatal_error is not None:
            raise self._fatal_error
        self.raise_if_faulted()
        if not self._owns_lease or self._lease_lost or not self._armed_expected:
            raise RuntimeError("Robot service backend is not armed with an owned lease")
        interpolation = str(interpolation)
        if interpolation not in set(self.capabilities["trajectory_interpolation"]):
            raise ValueError(f"Unsupported interpolation {interpolation!r}")

        is_bootstrap = expected_queue_token is None
        if is_bootstrap:
            if self.queue_token is not None:
                raise RobotServiceFaultError(
                    "Cannot bootstrap while the local queue epoch is active"
                )
        else:
            expected_queue_token = str(expected_queue_token)
            if not expected_queue_token:
                raise ValueError("expected_queue_token must be non-empty or None")
            if self.queue_token is None:
                raise RobotServiceFaultError(
                    "Cannot continue before an empty-queue bootstrap is accepted"
                )
            if expected_queue_token != self.queue_token:
                raise RobotServiceFaultError(
                    "Expected queue token does not match the local accepted tail"
                )
            if requested_min_lead_s is not None:
                raise ValueError("requested_min_lead_s is only valid for bootstrap")

        tcp_array = np.ascontiguousarray(tcp_pose, dtype=np.float64)
        gripper_target_array = np.ascontiguousarray(gripper_target_q, dtype=np.float64)
        offset_array = np.ascontiguousarray(action_offsets_s, dtype=np.float64)
        if tcp_array.ndim != 2 or tcp_array.shape[1:] != (6,):
            raise ValueError(f"Expected tcp_pose [T,6], got {tcp_array.shape}")
        if offset_array.ndim != 1:
            raise ValueError("action_offsets_s must be one-dimensional")
        if gripper_target_array.ndim == 1:
            gripper_steps = len(gripper_target_array)
        elif gripper_target_array.ndim == 2 and gripper_target_array.shape[1:] == (1,):
            gripper_steps = len(gripper_target_array)
        else:
            raise ValueError(
                f"Expected gripper_target_q [T] or [T,1], got {gripper_target_array.shape}"
            )
        if len(tcp_array) < 1:
            raise ValueError("Action chunk cannot be empty")
        if len(tcp_array) != gripper_steps or len(tcp_array) != len(offset_array):
            raise ValueError("TCP, gripper and action offset lengths differ")
        if (
            not np.all(np.isfinite(tcp_array))
            or not np.all(np.isfinite(gripper_target_array))
            or not np.all(np.isfinite(offset_array))
            or offset_array[0] <= 0.0
            or np.any(np.diff(offset_array) <= 0.0)
            or not np.isfinite(prediction_observation_monotonic)
            or not np.isfinite(timing_reference_observation_monotonic)
        ):
            raise ValueError("Action submission contains invalid values")
        if requested_min_lead_s is not None:
            requested_min_lead_s = float(requested_min_lead_s)
            if not np.isfinite(requested_min_lead_s) or requested_min_lead_s <= 0.0:
                raise ValueError("requested_min_lead_s must be positive and finite")

        command_id = self._next_command_id()
        attempts = self.timeout_retry_count + 1
        self.last_interpolation = interpolation
        self.last_command_id = command_id
        self.last_command_duplicate = None
        self.last_submission_lead_s = float(offset_array[0])

        for attempt in range(attempts):
            rpc_started_monotonic = time.monotonic()
            try:
                ack = self._timed_call(
                    self._client.submit_trajectory,
                    tcp_pose=tcp_array,
                    gripper_target_q=gripper_target_array,
                    action_offsets_s=offset_array,
                    prediction_observation_monotonic=float(prediction_observation_monotonic),
                    timing_reference_observation_monotonic=float(
                        timing_reference_observation_monotonic
                    ),
                    expected_queue_token=expected_queue_token,
                    requested_min_lead_s=(
                        0.0 if requested_min_lead_s is None else requested_min_lead_s
                    ),
                    interpolation=interpolation,
                    command_id=command_id,
                )
                break
            except TimeoutError as exc:
                if attempt + 1 < attempts:
                    continue
                ambiguous = AmbiguousCommandTimeoutError(command_id, attempts)
                ambiguous.__cause__ = exc
                self._fatal_error = ambiguous
                self._initiate_fail_safe_cleanup()
                raise ambiguous from exc
            except RobotServiceError as exc:
                result = ActionScheduleResult(
                    accepted=False,
                    command_id=command_id,
                    accepted_steps=0,
                    service_instance_id=self.service_instance_id,
                    queue_token=self.queue_token,
                    queue_tail_monotonic=self.queue_tail_monotonic,
                    queue_remaining_s=self.queue_remaining_s,
                    reason=str(exc),
                    queue_was_empty=is_bootstrap,
                    previous_queue_token=expected_queue_token,
                )
                self.last_schedule_ack = {
                    "accepted": False,
                    "command_id": command_id,
                    "expected_queue_token": expected_queue_token,
                    "interpolation": interpolation,
                    "reason": str(exc),
                }
                return result
        else:  # pragma: no cover
            raise AssertionError("unreachable action submission state")

        if not isinstance(ack, Mapping):
            self.last_schedule_ack = {
                "invalid_ack_type": type(ack).__name__,
            }
            fault = RobotServiceFaultError("ACK must be a mapping")
            self._fatal_error = fault
            self._initiate_fail_safe_cleanup("invalid action ACK")
            raise fault
        self.last_schedule_ack = dict(ack)
        contract_errors = []

        raw_command_id = ack.get("command_id")
        if isinstance(raw_command_id, (bool, np.bool_)) or not isinstance(
            raw_command_id, (int, np.integer)
        ):
            contract_errors.append("ACK command_id must be an integer")
            acknowledged_command_id = command_id
        else:
            acknowledged_command_id = int(raw_command_id)
            if acknowledged_command_id != command_id:
                contract_errors.append("ACK command_id did not match the request")

        raw_accepted_steps = ack.get("accepted_steps")
        if isinstance(raw_accepted_steps, (bool, np.bool_)) or not isinstance(
            raw_accepted_steps, (int, np.integer)
        ):
            contract_errors.append("ACK accepted_steps must be an integer")
            accepted_steps = 0
        else:
            accepted_steps = int(raw_accepted_steps)

        raw_accepted = ack.get("accepted")
        if not isinstance(raw_accepted, (bool, np.bool_)):
            accepted = accepted_steps > 0
        else:
            accepted = bool(raw_accepted)
        expected_steps = len(offset_array) if accepted else 0
        if accepted_steps != expected_steps:
            contract_errors.append("ACK accepted_steps did not match acceptance")

        duplicate = ack.get("duplicate")
        if not isinstance(duplicate, (bool, np.bool_)):
            contract_errors.append("ACK omitted boolean duplicate")
            duplicate = False
            duplicate_valid = False
        else:
            duplicate = bool(duplicate)
            duplicate_valid = True
        acknowledged_interpolation = ack.get("interpolation")
        if acknowledged_interpolation != interpolation:
            contract_errors.append("ACK interpolation did not match the request")
        instance_id = ack.get("service_instance_id")
        if instance_id != self.service_instance_id:
            contract_errors.append("ACK came from an unexpected service instance")
        reason = ack.get("reason", "")
        if not isinstance(reason, str):
            contract_errors.append("ACK reason must be a string")
            reason = ""

        queue_was_empty = is_bootstrap
        previous_queue_token = expected_queue_token
        queue_token = self.queue_token
        queue_tail = self.queue_tail_monotonic
        queue_remaining = self.queue_remaining_s
        queue_token_live = False
        queue_token_live_valid = False
        bootstrap_shift = 0.0

        if accepted:
            queue_was_empty = ack.get("queue_was_empty")
            previous_queue_token = ack.get("previous_queue_token")
            queue_token = ack.get("queue_token")
            if not isinstance(queue_was_empty, (bool, np.bool_)):
                contract_errors.append("ACK omitted boolean queue_was_empty")
            elif bool(queue_was_empty) != is_bootstrap:
                contract_errors.append("ACK queue state disagreed with the request")
            if is_bootstrap:
                if previous_queue_token is not None:
                    contract_errors.append("bootstrap ACK reported a previous token")
            elif previous_queue_token != expected_queue_token:
                contract_errors.append("join ACK reported an unexpected previous token")
            if not isinstance(queue_token, str) or not queue_token:
                contract_errors.append("ACK omitted queue_token")
            if "queue_token_live" not in ack:
                if duplicate_valid and duplicate:
                    contract_errors.append("duplicate ACK omitted boolean queue_token_live")
                else:
                    # Services may omit token
                    # liveness. Only an original, nonduplicate ACK is trusted.
                    queue_token_live = True
                    queue_token_live_valid = True
            else:
                raw_queue_token_live = ack.get("queue_token_live")
                if not isinstance(raw_queue_token_live, (bool, np.bool_)):
                    contract_errors.append("ACK queue_token_live must be boolean")
                else:
                    queue_token_live = bool(raw_queue_token_live)
                    queue_token_live_valid = True
            numeric_values = {}
            for field in (
                "queue_tail_monotonic",
                "queue_remaining_s",
                "bootstrap_shift_s",
            ):
                raw_value = ack.get(field)
                if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                    raw_value,
                    (int, float, np.integer, np.floating),
                ):
                    contract_errors.append(f"ACK {field} must be a numeric scalar")
                    numeric_values[field] = None
                else:
                    numeric_values[field] = float(raw_value)
            queue_tail = numeric_values["queue_tail_monotonic"]
            queue_remaining = numeric_values["queue_remaining_s"]
            bootstrap_shift = numeric_values["bootstrap_shift_s"]
            if queue_tail is None or not np.isfinite(queue_tail):
                contract_errors.append("ACK omitted queue_tail_monotonic")
            if queue_remaining is None or not np.isfinite(queue_remaining) or queue_remaining < 0.0:
                contract_errors.append("ACK reported invalid queue_remaining_s")
            elif queue_token_live_valid:
                if queue_token_live and queue_remaining <= 0.0:
                    contract_errors.append(
                        "live ACK queue token must have positive queue_remaining_s"
                    )
                elif not queue_token_live and queue_remaining != 0.0:
                    contract_errors.append("dead ACK queue token must have zero queue_remaining_s")
            if bootstrap_shift is None or not np.isfinite(bootstrap_shift) or bootstrap_shift < 0.0:
                contract_errors.append("ACK reported invalid bootstrap_shift_s")

        if contract_errors:
            fault = RobotServiceFaultError("; ".join(contract_errors))
            self._fatal_error = fault
            self._initiate_fail_safe_cleanup("action ACK contract violation")
            raise fault

        local_queue_deadline = None
        if accepted and queue_token_live:
            local_queue_deadline = rpc_started_monotonic + queue_remaining
            if local_queue_deadline <= time.monotonic():
                fault = RobotServiceFaultError(
                    "Accepted ACK queue token expired before the response was processed"
                )
                self._fatal_error = fault
                self._initiate_fail_safe_cleanup("action ACK queue token expired before processing")
                raise fault

        if accepted and not queue_token_live:
            fault = RobotServiceFaultError("Accepted ACK queue token is no longer live")
            self._fatal_error = fault
            self._initiate_fail_safe_cleanup("action ACK queue token is no longer live")
            raise fault

        if accepted:
            self.queue_token = queue_token
            self.queue_tail_monotonic = queue_tail
            self.queue_remaining_s = queue_remaining
            self.local_queue_deadline_monotonic = local_queue_deadline

        self.last_interpolation = interpolation
        self.last_command_id = command_id
        self.last_command_duplicate = bool(duplicate)
        return ActionScheduleResult(
            accepted=accepted,
            command_id=command_id,
            accepted_steps=accepted_steps,
            service_instance_id=self.service_instance_id,
            queue_token=queue_token,
            queue_tail_monotonic=queue_tail,
            queue_remaining_s=queue_remaining,
            reason=reason,
            queue_was_empty=bool(queue_was_empty),
            previous_queue_token=previous_queue_token,
            bootstrap_shift_s=bootstrap_shift,
            local_deadline_monotonic=self.local_queue_deadline_monotonic,
        )

    def _acquire_control_with_recovery(self) -> None:
        try:
            self._timed_call(self._client.acquire_control, force=False)
            return
        except TimeoutError:
            heartbeat_timeout = float(self.capabilities["watchdog"]["heartbeat_timeout_s"])
            deadline = time.monotonic() + heartbeat_timeout + 0.05
            while time.monotonic() < deadline:
                time.sleep(min(0.02, deadline - time.monotonic()))
            try:
                self._timed_call(self._client.acquire_control, force=False)
                return
            except Exception as retry_error:
                raise AmbiguousLeaseAcquisitionError(
                    "Control lease acknowledgement was lost and the lease "
                    "could not be reacquired after its heartbeat timeout"
                ) from retry_error

    def arm(self) -> bool:
        self._require_started()
        if self._fatal_error is not None:
            raise self._fatal_error
        if self._armed_expected:
            return True

        try:
            if not self._owns_lease:
                self._acquire_control_with_recovery()
                self._owns_lease = True
                self._lease_lost = False
                self._command_id = 0
            self._timed_call(self._client.heartbeat)
            self.last_heartbeat_error = None
            self._client.start_heartbeat(period_s=self.heartbeat_interval)
            response = self._timed_call(self._client.arm)
            if not bool(response.get("armed", False)):
                raise RobotServiceFaultError(
                    "Robot service arm acknowledgement did not report armed=true"
                )
            self._armed_expected = True
            if str(response.get("service_instance_id", "")) != str(self.service_instance_id):
                raise RobotServiceFaultError(
                    "Arm acknowledgement came from an unexpected service instance"
                )
            self._reset_queue_epoch()
            with self._status_lock:
                updated = dict(self.server_status)
                updated.update(
                    {
                        "armed": True,
                        "lease_active": True,
                        "lease_client_id": self.client_id,
                        "last_fault": "",
                    }
                )
                self.server_status = updated
            return True
        except Exception:
            self._armed_expected = False
            try:
                self._client.stop_heartbeat()
            except Exception:
                pass
            if self._owns_lease:
                try:
                    self._timed_call(self._client.release_control)
                except Exception:
                    pass
            self._owns_lease = False
            self._lease_lost = False
            raise

    def disarm(self, reason: str = "inference client disarm") -> bool:
        if not self._started or self._client is None:
            self._reset_queue_epoch()
            return True
        if not self._owns_lease or self._lease_lost:
            self._armed_expected = False
            self._reset_queue_epoch()
            return True
        if not self._armed_expected:
            self._reset_queue_epoch()
            return True
        if not self._confirmed_owned_lease():
            if self._lease_lost:
                self._armed_expected = False
                self._reset_queue_epoch()
                return True
            raise RobotServiceFaultError(
                "Cannot confirm lease ownership; refusing an unauthenticated disarm request"
            )
        response = self._timed_call(self._client.disarm, str(reason))
        self._armed_expected = False
        self._reset_queue_epoch()
        with self._status_lock:
            updated = dict(self.server_status)
            updated.update(
                {
                    "armed": False,
                    "queue_active": False,
                    "queue_tail_monotonic": None,
                    "queue_remaining_s": 0.0,
                }
            )
            self.server_status = updated
        return not bool(response.get("armed", False))

    def cancel_actions(self) -> bool:
        self._require_started()
        if self._fatal_error is not None:
            raise self._fatal_error
        self.raise_if_faulted()
        if not self._owns_lease or self._lease_lost or not self._armed_expected:
            raise RuntimeError("Robot service backend is not armed with an owned lease")

        attempts = self.timeout_retry_count + 1
        for attempt in range(attempts):
            try:
                response = self._timed_call(self._client.cancel_actions)
                break
            except TimeoutError as exc:
                if attempt + 1 < attempts:
                    continue
                ambiguous = AmbiguousCancelTimeoutError(attempts)
                ambiguous.__cause__ = exc
                self._fatal_error = ambiguous
                self._initiate_fail_safe_cleanup()
                raise ambiguous from exc
        else:  # pragma: no cover - the loop always breaks or raises.
            raise AssertionError("unreachable queue cancellation state")

        if not bool(response.get("queue_cleared", False)) or bool(
            response.get("queue_active", True)
        ):
            fault = RobotServiceFaultError(
                "Robot service did not confirm that cancellation cleared the queue"
            )
            self._fatal_error = fault
            self._initiate_fail_safe_cleanup()
            raise fault
        self._reset_queue_epoch()
        with self._status_lock:
            updated = dict(self.server_status)
            updated.update(
                {
                    "queue_active": False,
                    "queue_tail_monotonic": None,
                    "queue_remaining_s": 0.0,
                    "has_received_command": False,
                }
            )
            self.server_status = updated
        return True

    def get_status(self) -> Mapping:
        return self._refresh_status()

    def _status_is_stale(self) -> bool:
        with self._status_lock:
            timestamp = self._last_status_monotonic
        return timestamp is None or time.monotonic() - timestamp > self.status_staleness_timeout

    def raise_if_faulted(self) -> None:
        self._require_started()
        if self._fatal_error is not None:
            raise self._fatal_error

        heartbeat_error = self._client.heartbeat_error
        self.last_heartbeat_error = None if heartbeat_error is None else str(heartbeat_error)
        needs_refresh = self._status_is_stale() or self.last_status_error is not None
        if needs_refresh:
            try:
                self._refresh_status()
            except Exception as exc:
                self.last_status_error = str(exc)
                raise RobotServiceFaultError(
                    f"Unable to refresh robot service status: {exc}"
                ) from exc

        if not self._armed_expected:
            return
        if self.last_heartbeat_error is not None:
            raise RobotServiceFaultError(
                f"Robot service heartbeat failed: {self.last_heartbeat_error}"
            )

        with self._status_lock:
            status = dict(self.server_status)
        faults = []
        if not bool(status.get("armed", False)):
            faults.append("robot service disarmed unexpectedly")
        lease_active = bool(status.get("lease_active", False))
        lease_client_id = status.get("lease_client_id")
        if not lease_active:
            self._lease_lost = True
            faults.append("control lease is no longer active")
        elif str(lease_client_id) != str(self.client_id):
            self._lease_lost = True
            faults.append(f"control lease belongs to another client ({lease_client_id!r})")
        scheduler_error = status.get("scheduler_error")
        if scheduler_error:
            faults.append(f"scheduler error: {scheduler_error}")
        last_fault = status.get("last_fault")
        if last_fault:
            faults.append(f"service fault: {last_fault}")

        source_status = status.get("source_status", {})
        required_sources = self.capabilities.get("control_required_sources", ())
        unhealthy_sources = [
            source
            for source in required_sources
            if (
                source not in source_status or not bool(source_status[source].get("healthy", False))
            )
        ]
        if unhealthy_sources:
            faults.append(f"required sources are unhealthy: {unhealthy_sources}")
        if faults:
            raise RobotServiceFaultError("; ".join(faults))

    def _confirmed_owned_lease(self) -> bool:
        if not self._owns_lease or self._lease_lost:
            return False
        try:
            status = self._refresh_status()
        except Exception as exc:
            self.last_status_error = str(exc)
            return False
        owned = bool(status.get("lease_active", False)) and str(
            status.get("lease_client_id")
        ) == str(self.client_id)
        if not owned:
            self._lease_lost = True
        return owned

    def _initiate_fail_safe_cleanup(self, disarm_reason="ambiguous command timeout") -> None:
        client = self._client
        if client is None:
            self._armed_expected = False
            self._owns_lease = False
            self._reset_queue_epoch()
            return
        try:
            client.stop_heartbeat()
        except Exception:
            pass
        if self._confirmed_owned_lease():
            try:
                self._timed_call(
                    client.disarm,
                    disarm_reason,
                )
            except Exception:
                pass
        if self._owns_lease:
            try:
                self._timed_call(client.release_control)
            except Exception:
                pass
        self._armed_expected = False
        self._reset_queue_epoch()
        self._owns_lease = False

    def stop(self) -> None:
        client = self._client
        if client is None and not self._started:
            return
        self._status_stop.set()
        if client is not None:
            try:
                client.stop_heartbeat()
            except Exception:
                pass
            if self._armed_expected and self._confirmed_owned_lease():
                try:
                    self._timed_call(client.disarm, "backend stopping")
                except Exception:
                    pass
            if self._owns_lease:
                try:
                    self._timed_call(client.release_control)
                except Exception:
                    pass
            try:
                client.close()
            except Exception:
                pass

        if self._status_thread is not None:
            self._status_thread.join(timeout=max(1.0, self.rpc_timeout + 0.1))
            self._status_thread = None
        self._client = None
        self._started = False
        self._owns_lease = False
        self._lease_lost = False
        self._armed_expected = False
        self._reset_queue_epoch()
