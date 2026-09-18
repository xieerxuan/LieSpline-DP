from __future__ import annotations

from typing import Dict

import numpy as np

from liespline_dp.common.se3_util import quat_to_mat, so3_log


def _validate_trajectory(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    control_freq_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    positions = np.asarray(positions, dtype=np.float64)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape [N,3], got {positions.shape}")
    if quaternions_xyzw.ndim != 2 or quaternions_xyzw.shape[1] != 4:
        raise ValueError(f"quaternions_xyzw must have shape [N,4], got {quaternions_xyzw.shape}")
    if len(positions) != len(quaternions_xyzw):
        raise ValueError(
            "Position and quaternion trajectory lengths differ: "
            f"{len(positions)} != {len(quaternions_xyzw)}"
        )
    if len(positions) < 4:
        raise ValueError(f"Jerk metrics require at least four measured poses, got {len(positions)}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain NaN or Inf")
    if not np.all(np.isfinite(quaternions_xyzw)):
        raise ValueError("quaternions_xyzw contain NaN or Inf")
    if np.any(np.linalg.norm(quaternions_xyzw, axis=-1) < 1e-12):
        raise ValueError("quaternions_xyzw contain a zero-norm quaternion")
    control_freq_hz = float(control_freq_hz)
    if not np.isfinite(control_freq_hz) or control_freq_hz <= 0:
        raise ValueError(f"control_freq_hz must be positive, got {control_freq_hz}")
    return positions, quaternions_xyzw, control_freq_hz


def translational_jerk_rms(
    positions: np.ndarray,
    control_freq_hz: float,
) -> float:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape [N,3], got {positions.shape}")
    if len(positions) < 4:
        raise ValueError(
            f"Translational jerk requires at least four positions, got {len(positions)}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain NaN or Inf")
    control_freq_hz = float(control_freq_hz)
    if not np.isfinite(control_freq_hz) or control_freq_hz <= 0:
        raise ValueError(f"control_freq_hz must be positive, got {control_freq_hz}")
    dt = 1.0 / control_freq_hz
    jerk = np.diff(positions, n=3, axis=0) / (dt**3)
    return float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=-1))))


def rotational_jerk_rms(
    quaternions_xyzw: np.ndarray,
    control_freq_hz: float,
) -> float:
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    if quaternions_xyzw.ndim != 2 or quaternions_xyzw.shape[1] != 4:
        raise ValueError(f"quaternions_xyzw must have shape [N,4], got {quaternions_xyzw.shape}")
    if len(quaternions_xyzw) < 4:
        raise ValueError(
            f"Rotational jerk requires at least four quaternions, got {len(quaternions_xyzw)}"
        )
    if not np.all(np.isfinite(quaternions_xyzw)):
        raise ValueError("quaternions_xyzw contain NaN or Inf")
    if np.any(np.linalg.norm(quaternions_xyzw, axis=-1) < 1e-12):
        raise ValueError("quaternions_xyzw contain a zero-norm quaternion")
    control_freq_hz = float(control_freq_hz)
    if not np.isfinite(control_freq_hz) or control_freq_hz <= 0:
        raise ValueError(f"control_freq_hz must be positive, got {control_freq_hz}")
    dt = 1.0 / control_freq_hz
    rotations = quat_to_mat(quaternions_xyzw)
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    angular_velocity = so3_log(relative) / dt
    angular_acceleration = np.diff(angular_velocity, axis=0) / dt
    rotational_jerk = np.diff(angular_acceleration, axis=0) / dt
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    rotational_jerk * rotational_jerk,
                    axis=-1,
                )
            )
        )
    )


def eef_trajectory_jerk_metrics(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    control_freq_hz: float,
) -> Dict[str, float]:
    positions, quaternions_xyzw, control_freq_hz = _validate_trajectory(
        positions,
        quaternions_xyzw,
        control_freq_hz,
    )
    return {
        "trans_jerk_m_s3": translational_jerk_rms(positions, control_freq_hz),
        "rot_jerk_rad_s3": rotational_jerk_rms(quaternions_xyzw, control_freq_hz),
    }
