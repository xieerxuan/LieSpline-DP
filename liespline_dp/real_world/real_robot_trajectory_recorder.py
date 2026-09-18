from __future__ import annotations

import os
import pathlib
import tempfile
import threading
import time
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

from liespline_dp.real_world.real_robot_env import ActionScheduleResult
from liespline_dp.real_world.robot_service_client.client import (
    RobotServiceClient,
)


def serialize_checkpoint_config(cfg) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def require_zero_hold_last_action_steps(backend) -> None:
    hold_steps = backend.capabilities.get("hold_last_action_steps")
    if isinstance(hold_steps, (bool, np.bool_)) or not isinstance(hold_steps, (int, np.integer)):
        raise RuntimeError("Robot Service must advertise integer hold_last_action_steps")
    if int(hold_steps) != 0:
        raise RuntimeError("Trajectory recording requires Robot Service hold_last_action_steps=0")


class RealRobotTrajectoryRecorder:
    def __init__(
        self,
        output_path,
        *,
        checkpoint: str,
        checkpoint_config: str,
        rpc_address: str,
        rpc_timeout_s: float,
        observation_frequency_hz: float,
        observation_max_age_s: Optional[float] = None,
    ):
        self.output_path = pathlib.Path(output_path)
        self.checkpoint = str(checkpoint)
        self.checkpoint_config = str(checkpoint_config)
        self.rpc_address = str(rpc_address)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.observation_frequency_hz = float(observation_frequency_hz)
        self.observation_max_age_s = (
            None if observation_max_age_s is None else float(observation_max_age_s)
        )
        if not self.checkpoint:
            raise ValueError("checkpoint must be non-empty")
        if not self.checkpoint_config:
            raise ValueError("checkpoint_config must be non-empty")
        if not np.isfinite(self.rpc_timeout_s) or self.rpc_timeout_s <= 0.0:
            raise ValueError("rpc_timeout_s must be positive and finite")
        if not np.isfinite(self.observation_frequency_hz) or self.observation_frequency_hz <= 0.0:
            raise ValueError("observation_frequency_hz must be positive and finite")
        if self.observation_max_age_s is not None and (
            not np.isfinite(self.observation_max_age_s) or self.observation_max_age_s <= 0.0
        ):
            raise ValueError("observation_max_age_s must be positive and finite")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise FileExistsError(f"Trajectory output already exists: {self.output_path}")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.output_path.parent,
            prefix=f".{self.output_path.name}.probe.",
        ):
            pass

        self._lock = threading.Lock()
        self._command_chunks = []
        self._actual_timestamps = []
        self._actual_tcp_poses = []
        self._expected_service_instance_id = None
        self._thread = None
        self._thread_error = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._started = False
        self._closed = False

    def start(self, *, expected_service_instance_id: str) -> None:
        if self._started:
            raise RuntimeError("Trajectory recorder has already been started")
        if self._closed:
            raise RuntimeError("Trajectory recorder has already been closed")
        expected_service_instance_id = str(expected_service_instance_id)
        if not expected_service_instance_id:
            raise ValueError("expected_service_instance_id must be non-empty")
        self._expected_service_instance_id = expected_service_instance_id
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="real-robot-trajectory-sampler",
            daemon=True,
        )
        self._thread.start()
        ready_timeout_s = max(1.0, 2.0 * self.rpc_timeout_s + 1.0)
        try:
            if not self._ready.wait(timeout=ready_timeout_s):
                raise TimeoutError("Timed out starting trajectory sampler")
            self.raise_if_failed()
        except BaseException:
            self._stop.set()
            self._thread.join()
            self._thread = None
            raise
        self._started = True

    def _sample_loop(self) -> None:
        client = None
        try:
            client = RobotServiceClient(
                rpc_address=self.rpc_address,
                timeout_ms=max(1, int(round(self.rpc_timeout_s * 1000.0))),
                auto_release_on_close=False,
            )
            capabilities = client.get_capabilities()
            service_instance_id = str(capabilities.get("service_instance_id", ""))
            if service_instance_id != self._expected_service_instance_id:
                raise RuntimeError(
                    "Trajectory sampler connected to an unexpected Robot Service instance"
                )
            period_s = 1.0 / self.observation_frequency_hz
            self._sample_once(client)
            self._ready.set()
            next_sample_monotonic = time.monotonic() + period_s
            while not self._stop.is_set():
                remaining_s = next_sample_monotonic - time.monotonic()
                if remaining_s > 0.0 and self._stop.wait(remaining_s):
                    break
                self._sample_once(client)
                next_sample_monotonic += period_s
                if next_sample_monotonic < time.monotonic():
                    next_sample_monotonic = time.monotonic() + period_s
            self._sample_once(client)
        except BaseException as exc:
            self._thread_error = exc
            self._ready.set()
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _sample_once(self, client) -> None:
        header, arrays = client.get_latest(
            fields=["robot_tcp_pose"],
            max_age_s=self.observation_max_age_s,
        )
        try:
            timestamp = float(header["sample_monotonic_timestamps"]["robot_tcp_pose"])
            tcp_pose = np.asarray(arrays["robot_tcp_pose"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Robot Service returned invalid actual TCP data") from exc
        if not np.isfinite(timestamp):
            raise RuntimeError("Actual TCP timestamp must be finite")
        if tcp_pose.shape != (6,) or not np.all(np.isfinite(tcp_pose)):
            raise RuntimeError("Actual TCP pose must be a finite [6] array")
        with self._lock:
            if self._actual_timestamps:
                previous = self._actual_timestamps[-1]
                if timestamp < previous:
                    raise RuntimeError("Actual TCP timestamps moved backwards")
                if timestamp == previous:
                    return
            self._actual_timestamps.append(timestamp)
            self._actual_tcp_poses.append(tcp_pose.copy())

    def record_accepted_actions(
        self,
        tcp_pose: np.ndarray,
        action_offsets_s: np.ndarray,
        result: ActionScheduleResult,
    ) -> None:
        if not self._started:
            raise RuntimeError("Trajectory recorder has not been started")
        if not isinstance(result, ActionScheduleResult) or not result.accepted:
            raise ValueError("result must be an accepted ActionScheduleResult")
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
        action_offsets_s = np.asarray(action_offsets_s, dtype=np.float64)
        if tcp_pose.ndim != 2 or tcp_pose.shape[1] != 6:
            raise ValueError("tcp_pose must have shape [T,6]")
        if action_offsets_s.shape != (len(tcp_pose),):
            raise ValueError("action_offsets_s must match tcp_pose length")
        if len(tcp_pose) < 1:
            raise ValueError("Accepted action chunk cannot be empty")
        if not np.all(np.isfinite(tcp_pose)) or not np.all(np.isfinite(action_offsets_s)):
            raise ValueError("Accepted action chunk contains non-finite values")
        if action_offsets_s[0] <= 0.0 or np.any(np.diff(action_offsets_s) <= 0.0):
            raise ValueError("action_offsets_s must be positive and strictly increasing")
        if result.accepted_steps != len(tcp_pose):
            raise ValueError("Accepted step count does not match tcp_pose")
        command_id = int(result.command_id)
        queue_tail = float(result.queue_tail_monotonic)
        if not np.isfinite(queue_tail):
            raise ValueError("Accepted result must include a finite queue tail")
        timestamps = queue_tail - (action_offsets_s[-1] - action_offsets_s)
        if np.any(np.diff(timestamps) <= 0.0):
            raise RuntimeError("Accepted command timestamps are not increasing")
        duplicate = False
        with self._lock:
            if self._command_chunks:
                previous_command_id = self._command_chunks[-1][0]
                if command_id == previous_command_id:
                    previous_timestamps = self._command_chunks[-1][1]
                    previous_tcp_pose = self._command_chunks[-1][2]
                    if not np.array_equal(timestamps, previous_timestamps) or not np.array_equal(
                        tcp_pose, previous_tcp_pose
                    ):
                        raise RuntimeError("Repeated accepted command ID has different data")
                    duplicate = True
                elif command_id < previous_command_id:
                    raise RuntimeError("Accepted command IDs moved backwards")
                elif timestamps[0] <= self._command_chunks[-1][1][-1]:
                    raise RuntimeError("Accepted command timestamps do not advance the queue")
            if not duplicate:
                self._command_chunks.append((command_id, timestamps.copy(), tcp_pose.copy()))
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError("Trajectory sampler failed") from self._thread_error

    def close(self, *, require_complete_coverage: bool = True) -> None:
        if self._closed:
            return
        require_complete_coverage = bool(require_complete_coverage)
        final_command_timestamp = None
        with self._lock:
            if self._command_chunks:
                final_command_timestamp = float(self._command_chunks[-1][1][-1])
        if (
            require_complete_coverage
            and self._thread is not None
            and final_command_timestamp is not None
        ):
            coverage_timeout_s = max(
                1.0,
                self.rpc_timeout_s + 2.0 / self.observation_frequency_hz,
            )
            coverage_deadline = time.monotonic() + coverage_timeout_s
            while self._thread.is_alive() and self._thread_error is None:
                with self._lock:
                    covered = bool(
                        self._actual_timestamps
                        and self._actual_timestamps[-1] >= final_command_timestamp
                    )
                if covered:
                    break
                remaining_s = coverage_deadline - time.monotonic()
                if remaining_s <= 0.0:
                    break
                time.sleep(min(remaining_s, 0.01))
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

        with self._lock:
            command_timestamp = (
                np.concatenate(
                    [chunk_timestamps for _, chunk_timestamps, _ in self._command_chunks]
                ).astype(np.float64, copy=False)
                if self._command_chunks
                else np.empty((0,), dtype=np.float64)
            )
            command_tcp_target = (
                np.concatenate(
                    [chunk_tcp_pose for _, _, chunk_tcp_pose in self._command_chunks], axis=0
                ).astype(np.float64, copy=False)
                if self._command_chunks
                else np.empty((0, 6), dtype=np.float64)
            )
            actual_timestamp = np.asarray(self._actual_timestamps, dtype=np.float64)
            actual_tcp_pose = (
                np.stack(self._actual_tcp_poses, axis=0).astype(np.float64, copy=False)
                if self._actual_tcp_poses
                else np.empty((0, 6), dtype=np.float64)
            )

        self._write_archive(
            command_timestamp=command_timestamp,
            command_tcp_target=command_tcp_target,
            actual_timestamp=actual_timestamp,
            actual_tcp_pose=actual_tcp_pose,
        )
        self._closed = True

        self.raise_if_failed()
        if (
            require_complete_coverage
            and len(command_timestamp) > 0
            and (
                len(actual_timestamp) == 0
                or actual_timestamp[0] > command_timestamp[0]
                or actual_timestamp[-1] < command_timestamp[-1]
            )
        ):
            raise RuntimeError("Actual TCP trajectory does not cover the command interval")

    def _write_archive(
        self,
        *,
        command_timestamp: np.ndarray,
        command_tcp_target: np.ndarray,
        actual_timestamp: np.ndarray,
        actual_tcp_pose: np.ndarray,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = pathlib.Path(stream.name)
                np.savez_compressed(
                    stream,
                    checkpoint=np.asarray(self.checkpoint),
                    checkpoint_config=np.asarray(self.checkpoint_config),
                    command_timestamp=command_timestamp,
                    command_tcp_target=command_tcp_target,
                    actual_timestamp=actual_timestamp,
                    actual_tcp_pose=actual_tcp_pose,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, self.output_path)
            temporary_path.unlink()
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
