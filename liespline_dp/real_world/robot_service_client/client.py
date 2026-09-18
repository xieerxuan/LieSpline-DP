"""Dependency-light client for upper-layer algorithms.

The client package depends only on NumPy and PyZMQ. When vendoring it into
another repository, keep client.py and protocol.py in the same package.
"""

import threading
import time
import uuid

import numpy as np
import zmq

from .protocol import ProtocolError, pack_message, unpack_message


class RobotServiceError(RuntimeError):
    pass


class RobotServiceInstanceChangedError(RobotServiceError):
    """The endpoint now belongs to a different service process."""


class RobotServiceProtocolError(RobotServiceError):
    """The response envelope cannot be associated with the request safely."""


class RobotServiceClient:
    _SIDE_EFFECT_REQUESTS = frozenset(
        {
            "acquire_control",
            "heartbeat",
            "release_control",
            "arm",
            "disarm",
            "cancel_actions",
            "set_target",
            "submit_trajectory",
        }
    )

    def __init__(
        self,
        rpc_address="ipc:///tmp/ur5e_robot_service/rpc.sock",
        client_id=None,
        timeout_ms=3000,
        context=None,
        auto_release_on_close=True,
    ):
        self.rpc_address = str(rpc_address)
        self.client_id = str(client_id or f"algorithm-{uuid.uuid4().hex[:8]}")
        self.timeout_ms = int(timeout_ms)
        self._context = context or zmq.Context.instance()
        self.auto_release_on_close = bool(auto_release_on_close)
        self._socket = None
        self._heartbeat_thread = None
        self._heartbeat_stop = threading.Event()
        self._clear_session_state()
        self._connect()

    def _connect(self):
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IDENTITY, f"{self.client_id}-{uuid.uuid4().hex[:8]}".encode())
        self._socket.connect(self.rpc_address)

    def _clear_session_state(self):
        self._request_id = 0
        self._command_id = 0
        self.lease_id = None
        self.service_instance_id = None
        self.heartbeat_error = None
        self.last_request_timing = {}

    def reset_session(self):
        """Discard instance-scoped state and establish a fresh service session."""
        self.stop_heartbeat()
        self._clear_session_state()
        self._connect()
        return self.hello()

    def close(self):
        self.stop_heartbeat()
        if self.auto_release_on_close and self.lease_id is not None and self._socket is not None:
            try:
                self.disarm("client closing")
            except Exception:
                pass
            try:
                self.release_control()
            except Exception:
                pass
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _request(self, message_type, header=None, arrays=None):
        message_type = str(message_type)
        if self.service_instance_id is None and message_type in self._SIDE_EFFECT_REQUESTS:
            self.hello()
        self._request_id += 1
        request = {
            "message_type": message_type,
            "request_id": self._request_id,
            **(header or {}),
        }
        if self.service_instance_id is not None:
            request["expected_service_instance_id"] = self.service_instance_id
        request_started = time.perf_counter()
        frames = pack_message(request, arrays)
        after_pack = time.perf_counter()
        self._socket.send_multipart(frames, copy=False)
        after_send = time.perf_counter()
        if not self._socket.poll(self.timeout_ms, zmq.POLLIN):
            self._connect()
            raise TimeoutError(f"Timed out waiting for robot service at {self.rpc_address}")
        after_poll = time.perf_counter()
        try:
            response_frames = self._socket.recv_multipart(copy=True)
            after_recv = time.perf_counter()
            response, payload = unpack_message(response_frames)
            after_unpack = time.perf_counter()
        except (ProtocolError, AttributeError, TypeError, ValueError) as exc:
            raise RobotServiceProtocolError(f"Invalid RPC response envelope: {exc}") from exc
        self.last_request_timing = {
            "message_type": message_type,
            "request_pack_s": after_pack - request_started,
            "request_send_s": after_send - after_pack,
            "response_wait_s": after_poll - after_send,
            "response_recv_s": after_recv - after_poll,
            "response_unpack_s": after_unpack - after_recv,
            "request_total_s": after_unpack - request_started,
            "request_bytes": int(sum(getattr(frame, "nbytes", len(frame)) for frame in frames)),
            "response_bytes": int(sum(len(frame) for frame in response_frames)),
        }
        if not isinstance(response, dict):
            raise RobotServiceProtocolError("RPC response header must be a mapping")
        response_request_id = response.get("request_id")
        if (
            isinstance(response_request_id, (bool, np.bool_))
            or not isinstance(response_request_id, (int, np.integer))
            or int(response_request_id) != self._request_id
        ):
            raise RobotServiceProtocolError("RPC request_id mismatch")
        response_instance_id = response.get("service_instance_id")
        if not isinstance(response_instance_id, str) or not response_instance_id:
            raise RobotServiceProtocolError("Response is missing a valid service_instance_id")
        if self.service_instance_id is None:
            self.service_instance_id = response_instance_id
        elif response_instance_id != self.service_instance_id:
            expected = self.service_instance_id
            self.lease_id = None
            raise RobotServiceInstanceChangedError(
                f"Robot service instance changed from {expected!r} to {response_instance_id!r}"
            )
        ok = response.get("ok")
        if not isinstance(ok, (bool, np.bool_)):
            raise RobotServiceProtocolError("RPC response omitted boolean ok")
        if not bool(ok):
            if "error_type" not in response or "reason" not in response:
                raise RobotServiceProtocolError("RPC error response omitted error_type or reason")
            error_type = response["error_type"]
            reason = response["reason"]
            if (
                not isinstance(error_type, str)
                or not error_type
                or not isinstance(reason, str)
                or not reason
            ):
                raise RobotServiceProtocolError(
                    "RPC error response fields must be non-empty strings"
                )
            if error_type == "ServiceInstanceChangedError":
                self.lease_id = None
                raise RobotServiceInstanceChangedError(reason)
            raise RobotServiceError(f"{error_type}: {reason}")
        return response, payload

    def hello(self):
        return self._request("hello")[0]

    def get_capabilities(self):
        return self._request("get_capabilities")[0]

    def get_status(self):
        return self._request("get_status")[0]

    def get_latest(self, fields=None, max_age_s=None):
        header = {}
        if fields is not None:
            header["fields"] = list(fields)
        if max_age_s is not None:
            header["max_age_s"] = float(max_age_s)
        return self._request("get_latest", header)

    def get_observation(
        self,
        n_steps,
        fields=None,
        frequency_hz=None,
        max_age_s=None,
        max_skew_s=None,
        wait_timeout_s=None,
        alignment_mode=None,
        target_monotonic=None,
        max_alignment_error_s=None,
    ):
        header = {"n_steps": int(n_steps)}
        if fields is not None:
            header["fields"] = list(fields)
        if frequency_hz is not None:
            header["frequency_hz"] = float(frequency_hz)
        if max_age_s is not None:
            header["max_age_s"] = float(max_age_s)
        if max_skew_s is not None:
            header["max_skew_s"] = float(max_skew_s)
        if wait_timeout_s is not None:
            header["wait_timeout_s"] = float(wait_timeout_s)
        if alignment_mode is not None:
            header["alignment_mode"] = str(alignment_mode)
        if target_monotonic is not None:
            header["target_monotonic"] = float(target_monotonic)
        if max_alignment_error_s is not None:
            header["max_alignment_error_s"] = float(max_alignment_error_s)
        return self._request("get_observation", header)

    def acquire_control(self, force=False):
        response, _ = self._request(
            "acquire_control",
            {"client_id": self.client_id, "force": bool(force)},
        )
        self.lease_id = response["lease_id"]
        self._command_id = 0
        return response

    def _lease_header(self):
        if self.lease_id is None:
            raise RobotServiceError("Call acquire_control() first")
        return {"client_id": self.client_id, "lease_id": self.lease_id}

    def heartbeat(self):
        return self._request("heartbeat", self._lease_header())[0]

    def start_heartbeat(self, period_s=0.2):
        period_s = float(period_s)
        if not np.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("period_s must be finite and positive")
        if self.lease_id is None:
            raise RobotServiceError("Call acquire_control() first")
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_stop.clear()
        self.heartbeat_error = None
        lease_id = self.lease_id
        client_id = self.client_id

        def run():
            heartbeat_client = RobotServiceClient(
                rpc_address=self.rpc_address,
                client_id=client_id,
                timeout_ms=self.timeout_ms,
                context=self._context,
                auto_release_on_close=False,
            )
            heartbeat_client.lease_id = lease_id
            heartbeat_client.service_instance_id = self.service_instance_id
            try:
                while not self._heartbeat_stop.wait(float(period_s)):
                    heartbeat_client.heartbeat()
            except Exception as exc:
                self.heartbeat_error = exc
            finally:
                heartbeat_client.close()

        self._heartbeat_thread = threading.Thread(
            target=run, name="robot-service-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def release_control(self):
        self.stop_heartbeat()
        response = self._request("release_control", self._lease_header())[0]
        self.lease_id = None
        return response

    def arm(self):
        return self._request("arm", self._lease_header())[0]

    def disarm(self, reason="client disarm"):
        return self._request("disarm", {"reason": str(reason)})[0]

    def cancel_actions(self):
        return self._request("cancel_actions", self._lease_header())[0]

    def _next_command_id(self):
        command_id = self._command_id
        self._command_id += 1
        return command_id

    def _resolve_command_id(self, command_id):
        if command_id is None:
            return self._next_command_id()
        command_id = int(command_id)
        self._command_id = max(self._command_id, command_id + 1)
        return command_id

    def set_target(
        self,
        tcp_pose,
        gripper_target_q,
        prediction_observation_monotonic,
        timing_reference_observation_monotonic,
        duration_s,
        requested_min_lead_s=0.0,
        interpolation="linear_se3",
        command_id=None,
    ):
        return self._request(
            "set_target",
            {
                **self._lease_header(),
                "command_id": self._resolve_command_id(command_id),
                "prediction_observation_monotonic": float(prediction_observation_monotonic),
                "timing_reference_observation_monotonic": float(
                    timing_reference_observation_monotonic
                ),
                "duration_s": float(duration_s),
                "requested_min_lead_s": float(requested_min_lead_s),
                "interpolation": str(interpolation),
            },
            {
                "tcp_pose": np.asarray(tcp_pose, dtype=np.float64).reshape(6),
                "gripper_target_q": np.asarray(gripper_target_q, dtype=np.float64).reshape(1),
            },
        )[0]

    def submit_trajectory(
        self,
        tcp_pose,
        gripper_target_q,
        action_offsets_s,
        prediction_observation_monotonic,
        timing_reference_observation_monotonic,
        expected_queue_token,
        requested_min_lead_s=0.0,
        interpolation="linear_se3",
        command_id=None,
    ):
        if expected_queue_token is not None and (
            not isinstance(expected_queue_token, str) or not expected_queue_token
        ):
            raise ValueError("expected_queue_token must be a non-empty string or None")
        return self._request(
            "submit_trajectory",
            {
                **self._lease_header(),
                "command_id": self._resolve_command_id(command_id),
                "prediction_observation_monotonic": float(prediction_observation_monotonic),
                "timing_reference_observation_monotonic": float(
                    timing_reference_observation_monotonic
                ),
                "expected_queue_token": expected_queue_token,
                "requested_min_lead_s": float(requested_min_lead_s),
                "interpolation": str(interpolation),
            },
            {
                "tcp_pose": np.asarray(tcp_pose, dtype=np.float64),
                "gripper_target_q": np.asarray(gripper_target_q, dtype=np.float64),
                "action_offsets_s": np.asarray(action_offsets_s, dtype=np.float64),
            },
        )[0]


class RobotServiceSubscriber:
    def __init__(
        self,
        stream_address="ipc:///tmp/ur5e_robot_service/stream.sock",
        topics=("status", "observation"),
        context=None,
    ):
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        for topic in topics:
            self._socket.setsockopt(zmq.SUBSCRIBE, str(topic).encode("ascii"))
        self._socket.connect(str(stream_address))

    def receive(self, timeout_ms=None):
        if timeout_ms is not None and not self._socket.poll(int(timeout_ms), zmq.POLLIN):
            raise TimeoutError("Timed out waiting for robot service stream")
        frames = self._socket.recv_multipart(copy=True)
        topic = frames[0].decode("ascii")
        header, arrays = unpack_message(frames[1:])
        return topic, header, arrays

    def close(self):
        self._socket.close(linger=0)
