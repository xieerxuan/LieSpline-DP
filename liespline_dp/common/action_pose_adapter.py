from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from liespline_dp.common.pose_action_codec import PoseActionCodec
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


def _is_numpy(x) -> bool:
    return isinstance(x, np.ndarray)


def _concat(xs, axis=-1):
    if any(torch.is_tensor(x) for x in xs):
        return torch.cat(xs, dim=axis)
    return np.concatenate(xs, axis=axis)


def _zeros(shape, like):
    if torch.is_tensor(like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=np.asarray(like).dtype)


def _eye4(batch_shape, like):
    if torch.is_tensor(like):
        eye = torch.eye(4, dtype=like.dtype, device=like.device)
        return eye.expand(batch_shape + (4, 4)).clone()
    eye = np.eye(4, dtype=np.asarray(like).dtype)
    return np.broadcast_to(eye, batch_shape + (4, 4)).copy()


def _constant_rotmat(rotvec, like):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if rotvec.shape != (3,):
        raise ValueError(f"Expected rotation offset rotvec with shape (3,), got {rotvec.shape}")
    if torch.is_tensor(like):
        rotvec_t = torch.as_tensor(rotvec, dtype=like.dtype, device=like.device)
        return so3_exp(rotvec_t)
    return so3_exp(rotvec).astype(np.asarray(like).dtype)


def _mean(x, axis=-1, keepdims=True):
    if torch.is_tensor(x):
        return x.mean(dim=axis, keepdim=keepdims)
    return np.mean(x, axis=axis, keepdims=keepdims)


def _take_obs_key(obs, key: str):
    if isinstance(obs, Dict):
        return obs[key]
    raise TypeError("obs must be a dict unless obs_key_shapes are configured")


class ActionPoseAdapter(ABC):
    @abstractmethod
    def action_to_pose_grip(self, action, obs=None):
        """Convert raw environment actions to absolute SE(3) pose and gripper command."""

    @abstractmethod
    def pose_grip_to_action(self, transform, grip, obs=None, prev_action=None):
        """Convert absolute SE(3) pose and gripper command to raw environment actions."""

    @abstractmethod
    def obs_to_current_pose_grip(self, obs):
        """Extract current measured SE(3) pose and gripper state from observations."""


class RobomimicAbsActionPoseAdapter(ActionPoseAdapter):
    """Adapter for robomimic absolute OSC pose actions.

    robomimic raw absolute datasets store one arm as
    ``[x, y, z, axis_angle(3), gripper]``. The existing lowdim dataset may
    convert the rotation block to PyTorch3D ``rotation_6d``, yielding
    ``[x, y, z, rot6d(6), gripper]``. This adapter supports both formats.
    """

    _ROT_DIMS = {
        "axis_angle": 3,
        "rotation_6d": 6,
        "quaternion": 4,
        "matrix": 9,
    }

    def __init__(
        self,
        action_rotation_rep: str = "axis_angle",
        gripper_dim: int = 1,
        robot_index: int = 0,
        obs_keys: Sequence[str] = (
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        obs_key_shapes: Optional[Dict[str, Sequence[int]]] = None,
        obs_gripper_dim: int = 2,
        obs_eef_to_action_rotvec: Optional[Sequence[float]] = None,
        obs_eef_to_action_compose_side: str = "left",
        n_robots: Optional[int] = None,
    ):
        if action_rotation_rep not in self._ROT_DIMS:
            raise ValueError(f"Unsupported action_rotation_rep: {action_rotation_rep}")
        self.action_rotation_rep = action_rotation_rep
        self.gripper_dim = int(gripper_dim)
        self.robot_index = int(robot_index)
        self.obs_keys = tuple(obs_keys)
        self.obs_key_shapes = obs_key_shapes
        self.obs_gripper_dim = int(obs_gripper_dim)
        self.obs_eef_to_action_compose_side = str(obs_eef_to_action_compose_side)
        if self.obs_eef_to_action_compose_side not in {"left", "right"}:
            raise ValueError("obs_eef_to_action_compose_side must be 'left' or 'right'")
        self.n_robots = self._infer_n_robots(n_robots)
        if obs_eef_to_action_rotvec is None:
            self.obs_eef_to_action_rotvec = None
        else:
            self.obs_eef_to_action_rotvec = tuple(float(x) for x in obs_eef_to_action_rotvec)

    def _infer_n_robots(self, n_robots: Optional[int]) -> int:
        if n_robots is not None:
            return int(n_robots)
        max_robot_idx = -1
        for key in self.obs_keys:
            if not key.startswith("robot"):
                continue
            suffix = key[len("robot") :]
            idx_text = suffix.split("_", 1)[0]
            if idx_text.isdigit():
                max_robot_idx = max(max_robot_idx, int(idx_text))
        return max_robot_idx + 1 if max_robot_idx >= 0 else 1

    @staticmethod
    def _robot_key(robot_index: int, suffix: str) -> str:
        return f"robot{robot_index}_{suffix}"

    @property
    def per_robot_action_dim(self) -> int:
        return 3 + self._ROT_DIMS[self.action_rotation_rep] + self.gripper_dim

    def _select_robot_action(self, action):
        action_dim = action.shape[-1]
        per_robot_dim = self.per_robot_action_dim
        if action_dim == per_robot_dim:
            if self.robot_index != 0:
                raise ValueError("robot_index is nonzero but action contains only one robot")
            return action, None
        if action_dim % per_robot_dim != 0:
            raise ValueError(
                f"Action dim {action_dim} is incompatible with per-robot dim {per_robot_dim}"
            )
        n_robot = action_dim // per_robot_dim
        if self.robot_index >= n_robot:
            raise ValueError(f"robot_index {self.robot_index} out of range for {n_robot} robots")
        start = self.robot_index * per_robot_dim
        end = start + per_robot_dim
        return action[..., start:end], (start, end, action_dim)

    def _rot_to_mat(self, rot):
        if self.action_rotation_rep == "axis_angle":
            return so3_exp(rot)
        if self.action_rotation_rep == "rotation_6d":
            return rot6d_to_mat(rot)
        if self.action_rotation_rep == "quaternion":
            return quat_to_mat(rot)
        if self.action_rotation_rep == "matrix":
            return rot.reshape(rot.shape[:-1] + (3, 3))
        raise AssertionError("unreachable")

    def _mat_to_rot(self, mat):
        if self.action_rotation_rep == "axis_angle":
            return so3_log(mat)
        if self.action_rotation_rep == "rotation_6d":
            return mat_to_rot6d(mat)
        if self.action_rotation_rep == "quaternion":
            from liespline_dp.common.se3_util import mat_to_quat

            return mat_to_quat(mat)
        if self.action_rotation_rep == "matrix":
            return mat.reshape(mat.shape[:-2] + (9,))
        raise AssertionError("unreachable")

    def action_to_pose_grip(self, action, obs=None):
        robot_action, _ = self._select_robot_action(action)
        pos = robot_action[..., :3]
        rot_dim = self._ROT_DIMS[self.action_rotation_rep]
        rot = robot_action[..., 3 : 3 + rot_dim]
        grip = robot_action[..., 3 + rot_dim : 3 + rot_dim + self.gripper_dim]

        transform = _eye4(pos.shape[:-1], pos)
        transform[..., :3, :3] = self._rot_to_mat(rot)
        transform[..., :3, 3] = pos
        return transform, grip

    def pose_grip_to_action(self, transform, grip, obs=None, prev_action=None):
        pos = transform[..., :3, 3]
        rot = self._mat_to_rot(transform[..., :3, :3])
        robot_action = _concat([pos, rot, grip[..., : self.gripper_dim]], axis=-1)

        if prev_action is None:
            if self.n_robots > 1 and obs is not None:
                batch_shape = robot_action.shape[:-1]
                robot_actions = []
                for robot_idx in range(self.n_robots):
                    if robot_idx == self.robot_index:
                        robot_actions.append(robot_action)
                    else:
                        robot_actions.append(
                            self._robot_action_from_obs(obs, robot_idx, batch_shape)
                        )
                return _concat(robot_actions, axis=-1)
            return robot_action

        _, selection = self._select_robot_action(prev_action)
        if selection is None:
            return robot_action
        start, end, _ = selection
        if _is_numpy(prev_action):
            out = np.array(prev_action, copy=True)
        else:
            out = prev_action.clone()
        out[..., start:end] = robot_action
        return out

    def pose_grip_to_action_all(self, transform, grip):
        if transform.ndim < 4 or transform.shape[-2:] != (4, 4):
            raise ValueError(f"Expected transform [..., R, 4, 4], got {transform.shape}")
        if grip.ndim < 2:
            raise ValueError(f"Expected grip [..., R, Dg], got {grip.shape}")
        if transform.shape[-3] != self.n_robots:
            raise ValueError(
                f"Expected {self.n_robots} robot transforms, got {transform.shape[-3]}"
            )
        if grip.shape[-2] != self.n_robots:
            raise ValueError(f"Expected {self.n_robots} robot grips, got {grip.shape[-2]}")

        robot_actions = []
        for robot_idx in range(self.n_robots):
            pos = transform[..., robot_idx, :3, 3]
            rot = self._mat_to_rot(transform[..., robot_idx, :3, :3])
            robot_grip = grip[..., robot_idx, : self.gripper_dim]
            robot_actions.append(_concat([pos, rot, robot_grip], axis=-1))
        return _concat(robot_actions, axis=-1)

    def _known_obs_key_dim(self, key: str) -> Optional[int]:
        if key.endswith("_eef_pos"):
            return 3
        if key.endswith("_eef_quat"):
            return 4
        if key.endswith("_gripper_qpos"):
            return self.obs_gripper_dim
        return None

    def _obs_slices(self, obs=None):
        if self.obs_key_shapes is None:
            if obs is None:
                return None
            total_dim = int(obs.shape[-1])
            dims = []
            unknown_indices = []
            known_total = 0
            for idx, key in enumerate(self.obs_keys):
                dim = self._known_obs_key_dim(key)
                if dim is None:
                    unknown_indices.append(idx)
                    dims.append(None)
                else:
                    dims.append(dim)
                    known_total += dim
            if len(unknown_indices) != 1:
                return None
            unknown_dim = total_dim - known_total
            if unknown_dim <= 0:
                return None
            dims[unknown_indices[0]] = unknown_dim
            slices = {}
            offset = 0
            for key, dim in zip(self.obs_keys, dims):
                slices[key] = slice(offset, offset + dim)
                offset += dim
            return slices
        slices = {}
        offset = 0
        for key in self.obs_keys:
            dim = int(np.prod(self.obs_key_shapes[key]))
            slices[key] = slice(offset, offset + dim)
            offset += dim
        return slices

    def _obs_from_concat(self, obs, key: str):
        slices = self._obs_slices(obs)
        if slices is not None:
            return obs[..., slices[key]]
        if key == "robot0_eef_pos":
            start = -(self.obs_gripper_dim + 4 + 3)
            return obs[..., start : start + 3]
        if key == "robot0_eef_quat":
            start = -(self.obs_gripper_dim + 4)
            return obs[..., start : start + 4]
        if key == "robot0_gripper_qpos":
            return obs[..., -self.obs_gripper_dim :]
        raise KeyError(key)

    def _obs_value(self, obs, key: str):
        if isinstance(obs, Dict):
            return _take_obs_key(obs, key)
        return self._obs_from_concat(obs, key)

    def _obs_robot_pose_grip(self, obs, robot_index: int):
        pos = self._obs_value(obs, self._robot_key(robot_index, "eef_pos"))
        quat = self._obs_value(obs, self._robot_key(robot_index, "eef_quat"))
        gripper_qpos = self._obs_value(obs, self._robot_key(robot_index, "gripper_qpos"))
        transform = _eye4(pos.shape[:-1], pos)
        obs_rot = quat_to_mat(quat)
        if self.obs_eef_to_action_rotvec is not None:
            offset = _constant_rotmat(self.obs_eef_to_action_rotvec, obs_rot)
            if self.obs_eef_to_action_compose_side == "left":
                obs_rot = offset @ obs_rot
            else:
                obs_rot = obs_rot @ offset
        transform[..., :3, :3] = obs_rot
        transform[..., :3, 3] = pos
        grip = _mean(gripper_qpos, axis=-1, keepdims=True)
        if self.gripper_dim != 1:
            if torch.is_tensor(grip):
                grip = grip.repeat_interleave(self.gripper_dim, dim=-1)
            else:
                grip = np.repeat(grip, self.gripper_dim, axis=-1)
        return transform, grip

    def obs_to_current_pose_grip(self, obs):
        return self._obs_robot_pose_grip(obs, self.robot_index)

    def obs_to_current_pose_grip_all(self, obs):
        poses = []
        grips = []
        for robot_idx in range(self.n_robots):
            pose, grip = self._obs_robot_pose_grip(obs, robot_idx)
            poses.append(pose)
            grips.append(grip)
        if torch.is_tensor(poses[0]):
            return torch.stack(poses, dim=-3), torch.stack(grips, dim=-2)
        return np.stack(poses, axis=-3), np.stack(grips, axis=-2)

    def _expand_to_batch_shape(self, value, batch_shape):
        if tuple(value.shape[:-1]) == tuple(batch_shape):
            return value
        expand_shape = batch_shape + value.shape[-1:]
        reshape_shape = (
            value.shape[:-1] + (1,) * (len(batch_shape) - value.ndim + 1) + value.shape[-1:]
        )
        if torch.is_tensor(value):
            return value.reshape(reshape_shape).expand(expand_shape)
        return np.broadcast_to(
            np.reshape(value, reshape_shape),
            expand_shape,
        )

    def _robot_action_from_obs(self, obs, robot_index: int, batch_shape):
        transform, grip = self._obs_robot_pose_grip(obs, robot_index)
        pos = self._expand_to_batch_shape(transform[..., :3, 3], batch_shape)
        rot = self._expand_to_batch_shape(self._mat_to_rot(transform[..., :3, :3]), batch_shape)
        grip = self._expand_to_batch_shape(grip[..., : self.gripper_dim], batch_shape)
        return _concat([pos, rot, grip], axis=-1)


class RealRobotActionPoseAdapter(ActionPoseAdapter):
    """Adapter for absolute TCP poses and optional physical gripper targets."""

    def __init__(
        self,
        gripper_bounds: Sequence[float] = (0.0, 5.0),
        obs_keys: Sequence[str] = (
            "eef_pos",
            "eef_quat",
            "gripper_q",
        ),
        action_format=None,
        include_gripper=None,
    ):
        self.action_format = normalize_real_robot_action_format(
            action_format=action_format,
            include_gripper=include_gripper,
        )
        self.include_gripper = real_robot_has_gripper(self.action_format)
        if len(gripper_bounds) != 2:
            raise ValueError("gripper_bounds must contain [min, max]")
        lower, upper = (float(value) for value in gripper_bounds)
        if not lower < upper:
            raise ValueError("gripper_bounds must satisfy min < max")
        expected_obs_keys = 3 if self.include_gripper else 2
        if len(obs_keys) != expected_obs_keys:
            raise ValueError(
                "obs_keys must contain eef position and quaternion keys"
                + (" plus a gripper key" if self.include_gripper else "")
            )
        self.gripper_bounds = (lower, upper)
        self.obs_keys = tuple(str(key) for key in obs_keys)
        self.codec = PoseActionCodec(action_format=self.action_format)
        self.gripper_dim = 1 if self.include_gripper else 0
        self.n_robots = 1

    @staticmethod
    def _clip(value, lower, upper):
        if torch.is_tensor(value):
            return value.clamp(min=lower, max=upper)
        return np.clip(value, lower, upper)

    def normalize_gripper(self, gripper_q):
        if not self.include_gripper:
            return _zeros(gripper_q.shape[:-1] + (0,), gripper_q)
        lower, upper = self.gripper_bounds
        gripper_q = self.codec._ensure_gripper_dim(gripper_q)
        normalized = 2.0 * (gripper_q - lower) / (upper - lower) - 1.0
        return self._clip(normalized, -1.0, 1.0)

    def denormalize_gripper(self, normalized_gripper):
        if not self.include_gripper:
            return None
        lower, upper = self.gripper_bounds
        normalized_gripper = self.codec._ensure_gripper_dim(normalized_gripper)
        normalized_gripper = self._clip(normalized_gripper, -1.0, 1.0)
        return lower + 0.5 * (normalized_gripper + 1.0) * (upper - lower)

    def action_to_pose_grip(self, action, obs=None):
        if action.shape[-1] != self.codec.model_action_dim:
            raise ValueError(
                "Real-robot action must have shape "
                f"[...,{self.codec.model_action_dim}], got {action.shape}"
            )
        position, quaternion, physical_grip = self.codec.model_action_to_target(action)
        transform = _eye4(position.shape[:-1], position)
        transform[..., :3, :3] = quat_to_mat(quaternion)
        transform[..., :3, 3] = position
        if not self.include_gripper:
            return transform, _zeros(position.shape[:-1] + (0,), position)
        return transform, self.normalize_gripper(physical_grip)

    def pose_grip_to_action(self, transform, grip, obs=None, prev_action=None):
        if transform.shape[-2:] != (4, 4):
            raise ValueError(f"Expected transform [...,4,4], got {transform.shape}")
        position = transform[..., :3, 3]
        quaternion = mat_to_quat(transform[..., :3, :3])
        physical_grip = self.denormalize_gripper(grip)
        return self.codec.encode_target(
            position,
            quaternion,
            physical_grip,
        )

    def pose_grip_to_tcp_command(self, transform, grip):
        """Convert predicted poses directly to the hardware rotvec schema."""
        if transform.shape[-2:] != (4, 4):
            raise ValueError(f"Expected transform [...,4,4], got {transform.shape}")
        position = transform[..., :3, 3]
        rotvec = so3_log(transform[..., :3, :3])
        physical_grip = self.denormalize_gripper(grip)
        return _concat([position, rotvec], axis=-1), physical_grip

    def obs_to_current_pose_grip(self, obs):
        if not isinstance(obs, Mapping):
            raise TypeError("Real-robot observations must be a mapping")
        position_key, quaternion_key = self.obs_keys[:2]
        missing = [key for key in self.obs_keys if key not in obs]
        if missing:
            raise KeyError(f"Missing real-robot observation keys: {missing}")
        position = obs[position_key]
        quaternion = obs[quaternion_key]
        if position.shape[-1] != 3:
            raise ValueError(f"Expected {position_key} [...,3], got {position.shape}")
        if quaternion.shape[-1] != 4:
            raise ValueError(f"Expected {quaternion_key} [...,4], got {quaternion.shape}")
        transform = _eye4(position.shape[:-1], position)
        transform[..., :3, :3] = quat_to_mat(quaternion)
        transform[..., :3, 3] = position
        if not self.include_gripper:
            return transform, _zeros(position.shape[:-1] + (0,), position)
        gripper_key = self.obs_keys[2]
        gripper_q = obs[gripper_key]
        return transform, self.normalize_gripper(gripper_q)
