from typing import Tuple, Union

import numpy as np
import torch

from liespline_dp.common.real_robot_action_format import (
    normalize_real_robot_action_format,
    real_robot_has_gripper,
)
from liespline_dp.common.se3_util import (
    mat_to_quat,
    mat_to_rot6d,
    quat_to_mat,
    rot6d_to_mat,
    so3_exp,
    so3_log,
)

ArrayLike = Union[np.ndarray, torch.Tensor]


class PoseActionCodec:
    """Shared native action representation for real-robot policies.

    External target poses use position plus an ``xyzw`` quaternion.  The
    diffusion model uses position plus the continuous 6D rotation
    representation and a continuous gripper target:

    ``[xyz(3), rotation_6d(6), gripper_target_q(1)]`` for pose-gripper tasks
    and ``[xyz(3), rotation_6d(6)]`` for pose-only tasks.
    """

    tcp_pose_dim = 6

    def __init__(self, action_format=None, include_gripper=None):
        self.action_format = normalize_real_robot_action_format(
            action_format=action_format,
            include_gripper=include_gripper,
        )
        self.include_gripper = real_robot_has_gripper(self.action_format)

    @property
    def model_action_dim(self):
        return 10 if self.include_gripper else 9

    @staticmethod
    def _concat(values, axis=-1):
        if any(torch.is_tensor(value) for value in values):
            values = [
                value if torch.is_tensor(value) else torch.as_tensor(value) for value in values
            ]
            return torch.cat(values, dim=axis)
        return np.concatenate(values, axis=axis)

    @staticmethod
    def _ensure_gripper_dim(gripper_q: ArrayLike) -> ArrayLike:
        if gripper_q.ndim == 0:
            return gripper_q.reshape(1)
        if gripper_q.shape[-1:] != (1,):
            return gripper_q[..., None]
        return gripper_q

    def encode_target(
        self,
        position: ArrayLike,
        quaternion_xyzw: ArrayLike,
        gripper_q: ArrayLike,
    ) -> ArrayLike:
        if position.shape[-1] != 3:
            raise ValueError(f"Expected position [...,3], got {position.shape}")
        if quaternion_xyzw.shape[-1] != 4:
            raise ValueError(f"Expected quaternion_xyzw [...,4], got {quaternion_xyzw.shape}")
        rotation_6d = mat_to_rot6d(quat_to_mat(quaternion_xyzw))
        if not self.include_gripper:
            return self._concat([position, rotation_6d], axis=-1)
        if gripper_q is None:
            raise ValueError("pose_gripper action format requires gripper_q")
        gripper_q = self._ensure_gripper_dim(gripper_q)
        return self._concat([position, rotation_6d, gripper_q], axis=-1)

    def decode_model_action(
        self,
        action: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        if action.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"Expected model action [...,{self.model_action_dim}], got {action.shape}"
            )
        position = action[..., :3]
        rotation_6d = action[..., 3:9]
        gripper_q = action[..., 9:10] if self.include_gripper else None
        rotvec = so3_log(rot6d_to_mat(rotation_6d))
        tcp_pose = self._concat([position, rotvec], axis=-1)
        return tcp_pose, gripper_q

    def model_action_to_target(
        self,
        action: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        if action.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"Expected model action [...,{self.model_action_dim}], got {action.shape}"
            )
        position = action[..., :3]
        quaternion = mat_to_quat(rot6d_to_mat(action[..., 3:9]))
        gripper_q = action[..., 9:10] if self.include_gripper else None
        return position, quaternion, gripper_q

    @staticmethod
    def measured_rotvec_to_quaternion(rotvec: ArrayLike) -> ArrayLike:
        if rotvec.shape[-1] != 3:
            raise ValueError(f"Expected rotvec [...,3], got {rotvec.shape}")
        return mat_to_quat(so3_exp(rotvec))
