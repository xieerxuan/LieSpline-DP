import copy
from typing import Dict

import numpy as np
import torch

from liespline_dp.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.common.replay_buffer import ReplayBuffer
from liespline_dp.common.sampler import downsample_mask, get_val_mask
from liespline_dp.dataset.base_dataset import BaseImageDataset
from liespline_dp.dataset.bspline_se3_dataset_validation import (
    validate_bspline_se3_anchor_contract,
    validate_bspline_se3_dataset_parameters,
)
from liespline_dp.dataset.spline_sampling import build_sample_indices
from liespline_dp.model.common.normalizer import LinearNormalizer


class BsplineSE3ImageDataset(BaseImageDataset):
    """Shared image-window storage for SE(3) B-spline datasets."""

    def __init__(
        self,
        shape_meta: dict,
        zarr_path,
        degree,
        prefix_cp_steps,
        control_point_stride,
        exec_horizon,
        n_action_steps,
        n_latency_steps,
        anchor_contract=None,
        validate_anchor_contract=True,
        n_obs_steps=2,
        trajectory_horizon=None,
        trajectory_key="trajectory",
        condition_data_key="condition_data",
        condition_mask_key="condition_mask",
        pose_waypoint_key="pose_waypoint",
        gripper_waypoint_key="gripper_waypoint",
        gripper_control_point_key=None,
        split_key=None,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        max_train_source_episodes=None,
    ):
        super().__init__()
        validate_bspline_se3_dataset_parameters(
            zarr_path,
            degree=degree,
            prefix_cp_steps=prefix_cp_steps,
            control_point_stride=control_point_stride,
            exec_horizon=exec_horizon,
            n_action_steps=n_action_steps,
            n_latency_steps=n_latency_steps,
        )
        if validate_anchor_contract:
            if anchor_contract is None:
                raise RuntimeError(
                    "The active simulation task config is missing anchor_contract. "
                    "Update the config and regenerate the B-spline SE(3) dataset."
                )
            validate_bspline_se3_anchor_contract(zarr_path, anchor_contract)
        self.shape_meta = shape_meta
        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = [
            key for key, attr in obs_shape_meta.items() if attr.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            key for key, attr in obs_shape_meta.items() if attr.get("type", "low_dim") == "low_dim"
        ]
        obs_keys = self.rgb_keys + self.lowdim_keys
        keys = obs_keys + [
            trajectory_key,
            condition_data_key,
            condition_mask_key,
            pose_waypoint_key,
        ]
        if gripper_waypoint_key is not None:
            keys.append(gripper_waypoint_key)
        if gripper_control_point_key is not None:
            keys.append(gripper_control_point_key)
        if split_key is not None:
            keys.append(split_key)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        for key in obs_keys:
            assert self.replay_buffer[key].shape[1] == n_obs_steps, key
        if trajectory_horizon is not None:
            assert self.replay_buffer[trajectory_key].shape[1] == trajectory_horizon
            assert self.replay_buffer[condition_data_key].shape[1] == trajectory_horizon
            assert self.replay_buffer[condition_mask_key].shape[1] == trajectory_horizon
            if gripper_control_point_key is not None:
                assert self.replay_buffer[gripper_control_point_key].shape[1] == trajectory_horizon
        assert (
            self.replay_buffer[trajectory_key].shape == self.replay_buffer[condition_data_key].shape
        )
        assert (
            self.replay_buffer[trajectory_key].shape == self.replay_buffer[condition_mask_key].shape
        )

        self.trajectory_key = trajectory_key
        self.condition_data_key = condition_data_key
        self.condition_mask_key = condition_mask_key
        self.pose_waypoint_key = pose_waypoint_key
        self.gripper_waypoint_key = gripper_waypoint_key
        self.gripper_control_point_key = gripper_control_point_key
        self.split_key = split_key
        self.n_obs_steps = n_obs_steps
        self.trajectory_horizon = trajectory_horizon

        if split_key is None:
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            self.train_mask = downsample_mask(
                mask=~val_mask,
                max_n=max_train_episodes,
                seed=seed,
            )
            self.val_mask = ~self.train_mask
            self.indices = build_sample_indices(
                episode_ends=self.replay_buffer.episode_ends[:],
                episode_mask=self.train_mask,
            )
            self.val_indices = build_sample_indices(
                episode_ends=self.replay_buffer.episode_ends[:],
                episode_mask=self.val_mask,
            )
        else:
            if max_train_episodes is not None and max_train_source_episodes is None:
                max_train_source_episodes = max_train_episodes
            self.source_ids = np.asarray(self.replay_buffer[split_key][:]).reshape(-1)
            unique_source_ids = np.unique(self.source_ids)
            val_source_mask = get_val_mask(
                n_episodes=len(unique_source_ids),
                val_ratio=val_ratio,
                seed=seed,
            )
            train_source_mask = downsample_mask(
                mask=~val_source_mask,
                max_n=max_train_source_episodes,
                seed=seed,
            )
            train_source_ids = unique_source_ids[train_source_mask]
            val_source_ids = unique_source_ids[val_source_mask]
            self.train_mask = np.isin(self.source_ids, train_source_ids)
            self.val_mask = np.isin(self.source_ids, val_source_ids)
            self.indices = np.flatnonzero(self.train_mask).astype(np.int64)
            self.val_indices = np.flatnonzero(self.val_mask).astype(np.int64)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.indices = self.val_indices
        return val_set

    def set_pose_waypoint_key(self, key: str):
        if key not in self.replay_buffer.root["data"]:
            raise KeyError(f"pose waypoint key not found in replay buffer: {key}")
        self.pose_waypoint_key = str(key)

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "trajectory": self.replay_buffer[self.trajectory_key],
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["condition_data"] = normalizer["trajectory"]

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("pos") or key.endswith("qpos"):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                normalizer[key] = get_range_normalizer_from_stat(stat)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.trajectory_key])

    def __len__(self) -> int:
        return len(self.indices)

    def _read_rgb(self, key: str, sample_idx: int) -> np.ndarray:
        data = self.replay_buffer[key][sample_idx]
        expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
        if data.shape[1:] == expected_shape:
            image = data
        elif data.shape[-1] == expected_shape[0]:
            image = np.moveaxis(data, -1, 1)
        else:
            raise ValueError(
                f"Unexpected image shape for {key}: {data.shape}, "
                f"expected T,{expected_shape} or T,H,W,C"
            )
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        obs = {}
        for key in self.rgb_keys:
            obs[key] = self._read_rgb(key, sample_idx)
        for key in self.lowdim_keys:
            obs[key] = self.replay_buffer[key][sample_idx].astype(np.float32)

        data = {
            "sample_idx": np.asarray(sample_idx, dtype=np.int64),
            "obs": obs,
            "trajectory": self.replay_buffer[self.trajectory_key][sample_idx].astype(np.float32),
            "condition_data": self.replay_buffer[self.condition_data_key][sample_idx].astype(
                np.float32
            ),
            "condition_mask": self.replay_buffer[self.condition_mask_key][sample_idx].astype(bool),
            "pose_waypoint": self.replay_buffer[self.pose_waypoint_key][sample_idx].astype(
                np.float32
            ),
        }
        if self.gripper_waypoint_key is not None:
            data["gripper_waypoint"] = self.replay_buffer[self.gripper_waypoint_key][
                sample_idx
            ].astype(np.float32)
        if self.gripper_control_point_key is not None:
            data["gripper_control_point"] = self.replay_buffer[self.gripper_control_point_key][
                sample_idx
            ].astype(np.float32)
        return dict_apply(data, torch.from_numpy)
