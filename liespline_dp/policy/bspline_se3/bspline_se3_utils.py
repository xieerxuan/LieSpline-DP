from typing import Optional

import numpy as np
import torch

from liespline_dp.common.bspline_se3_util import (
    eval_se3_cumulative_bspline,
    get_execution_phase,
    get_zoh_execution_indices,
)
from liespline_dp.common.se3_util import se3_exp, se3_inv, se3_log, se3_mul


def local_tangent_to_control_poses(
    local_pose_cp: torch.Tensor,
    anchor_pose: torch.Tensor,
) -> torch.Tensor:
    if local_pose_cp.ndim != 3:
        raise ValueError(f"Expected local_pose_cp [B,H,6], got {local_pose_cp.shape}")
    if anchor_pose.ndim != 3 or anchor_pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected anchor_pose [B,4,4], got {anchor_pose.shape}")
    return se3_mul(anchor_pose[:, None], se3_exp(local_pose_cp))


def control_poses_to_local_tangent(
    control_poses: torch.Tensor,
    anchor_pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if control_poses.ndim != 4 or control_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected control_poses [B,H,4,4], got {control_poses.shape}")
    if anchor_pose is None:
        anchor_pose = control_poses[:, 0]
    return se3_log(se3_mul(se3_inv(anchor_pose[:, None]), control_poses))


def control_points_to_se3_waypoints(
    control_poses: torch.Tensor,
    degree: int,
    n_action_steps: int,
    exec_horizon: Optional[int] = None,
    include_start: bool = False,
) -> torch.Tensor:
    if control_poses.ndim != 4 or control_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected control_poses [B,H,4,4], got {control_poses.shape}")
    if control_poses.shape[1] <= degree:
        raise ValueError(
            f"B-spline needs more control points than degree, got "
            f"{control_poses.shape[1]} control points and degree {degree}."
        )

    phase_start = float(degree)
    if exec_horizon is None:
        phase_end = float(control_poses.shape[1])
    else:
        phase_end = min(
            phase_start + float(max(int(exec_horizon), 1)),
            float(control_poses.shape[1]),
        )

    if not include_start and exec_horizon is not None:
        if int(n_action_steps) % int(exec_horizon) != 0:
            raise ValueError(
                "n_action_steps must be divisible by exec_horizon for exact "
                "control-point phase alignment"
            )
        phase = get_execution_phase(
            degree=degree,
            n_action_steps=n_action_steps,
            control_point_stride=int(n_action_steps) // int(exec_horizon),
            exec_horizon=exec_horizon,
        )
    elif include_start:
        phase = np.linspace(
            phase_start,
            phase_end,
            num=max(n_action_steps, 1),
            dtype=np.float64,
        )
    else:
        phase = np.linspace(
            phase_start,
            phase_end,
            num=max(n_action_steps, 1) + 1,
            dtype=np.float64,
        )[1:]

    pose_np = control_poses.detach().cpu().numpy()
    waypoint_poses = []
    for this_pose_cp in pose_np:
        waypoint_poses.append(
            eval_se3_cumulative_bspline(
                control_poses=this_pose_cp,
                phase=phase,
                degree=degree,
            )
        )

    waypoint_poses = torch.from_numpy(np.stack(waypoint_poses, axis=0)).to(
        device=control_poses.device,
        dtype=control_poses.dtype,
    )
    return waypoint_poses


def control_points_to_zoh_waypoints(
    control_points: torch.Tensor,
    degree: int,
    n_action_steps: int,
    exec_horizon: int,
) -> torch.Tensor:
    """Evaluate zero-order-held gripper control points for execution steps."""
    if control_points.ndim != 3:
        raise ValueError(f"Expected ZOH control points [B,H,D], got {control_points.shape}")
    if int(n_action_steps) % int(exec_horizon) != 0:
        raise ValueError(
            "n_action_steps must be divisible by exec_horizon for exact "
            "control-point index alignment"
        )
    indices = get_zoh_execution_indices(
        degree=degree,
        n_action_steps=n_action_steps,
        control_point_stride=int(n_action_steps) // int(exec_horizon),
        exec_horizon=exec_horizon,
    )
    if indices[-1] > control_points.shape[1] - 1:
        raise ValueError(
            "Execution index exceeds gripper control-point horizon: "
            f"{indices[-1]} > {control_points.shape[1] - 1}"
        )
    index_tensor = torch.as_tensor(
        indices,
        device=control_points.device,
        dtype=torch.long,
    )
    return control_points[:, index_tensor]


def get_next_control_point_prefix(
    control_points: torch.Tensor,
    prefix_cp_steps: int,
    exec_horizon: Optional[int],
) -> torch.Tensor:
    if prefix_cp_steps < 1:
        raise ValueError(f"prefix_cp_steps must be positive, got {prefix_cp_steps}")
    if control_points.shape[1] < prefix_cp_steps:
        raise ValueError(
            f"Need at least {prefix_cp_steps} control points to build prefix, "
            f"got {control_points.shape[1]}"
        )
    if exec_horizon is None:
        start_idx = control_points.shape[1] - prefix_cp_steps
    else:
        max_start_idx = control_points.shape[1] - prefix_cp_steps
        start_idx = min(max(int(exec_horizon), 1), max_start_idx)
    return control_points[:, start_idx : start_idx + prefix_cp_steps]


def get_next_bspline_se3_prefix(
    control_poses: torch.Tensor,
    prefix_cp_steps: int,
    exec_horizon: Optional[int],
) -> torch.Tensor:
    return get_next_control_point_prefix(
        control_points=control_poses,
        prefix_cp_steps=prefix_cp_steps,
        exec_horizon=exec_horizon,
    )
