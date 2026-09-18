import copy
from typing import Dict

import numpy as np
import torch

from liespline_dp.common.normalize_util import (
    array_to_stats,
    robomimic_lowdim_obs_normalizer_from_stat,
)
from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.common.replay_buffer import ReplayBuffer
from liespline_dp.common.sampler import downsample_mask, get_val_mask
from liespline_dp.dataset.base_dataset import BaseLowdimDataset
from liespline_dp.dataset.bspline_se3_dataset_validation import (
    validate_bspline_se3_anchor_contract,
    validate_bspline_se3_dataset_parameters,
)
from liespline_dp.dataset.spline_sampling import build_sample_indices
from liespline_dp.model.common.normalizer import LinearNormalizer


class RobomimicBsplineSE3LowdimDataset(BaseLowdimDataset):
    """Pose-CP SE(3) B-spline dataset with binary gripper waypoint labels."""

    def __init__(
        self,
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
        obs_key="obs",
        trajectory_key="trajectory",
        condition_data_key="condition_data",
        condition_mask_key="condition_mask",
        pose_waypoint_key="pose_waypoint",
        gripper_waypoint_key="gripper_waypoint",
        split_key=None,
        obs_normalizer_mode="limits",
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
        keys = [
            obs_key,
            trajectory_key,
            condition_data_key,
            condition_mask_key,
            pose_waypoint_key,
            gripper_waypoint_key,
        ]
        if split_key is not None:
            keys.append(split_key)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        assert self.replay_buffer[obs_key].shape[1] == n_obs_steps
        if trajectory_horizon is not None:
            assert self.replay_buffer[trajectory_key].shape[1] == trajectory_horizon
            assert self.replay_buffer[condition_data_key].shape[1] == trajectory_horizon
            assert self.replay_buffer[condition_mask_key].shape[1] == trajectory_horizon
        assert (
            self.replay_buffer[trajectory_key].shape == self.replay_buffer[condition_data_key].shape
        )
        assert (
            self.replay_buffer[trajectory_key].shape == self.replay_buffer[condition_mask_key].shape
        )

        self.obs_key = obs_key
        self.trajectory_key = trajectory_key
        self.condition_data_key = condition_data_key
        self.condition_mask_key = condition_mask_key
        self.pose_waypoint_key = pose_waypoint_key
        self.gripper_waypoint_key = gripper_waypoint_key
        self.split_key = split_key
        if obs_normalizer_mode not in {"global_max", "limits"}:
            raise ValueError(
                f"obs_normalizer_mode must be 'global_max' or 'limits', got {obs_normalizer_mode!r}"
            )
        self.obs_normalizer_mode = obs_normalizer_mode
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
            data={"trajectory": self.replay_buffer[self.trajectory_key]},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        if self.obs_normalizer_mode == "global_max":
            obs = self.replay_buffer[self.obs_key][:]
            obs_stat = array_to_stats(obs.reshape(-1, obs.shape[-1]))
            normalizer["obs"] = robomimic_lowdim_obs_normalizer_from_stat(obs_stat)
        else:
            obs_normalizer = LinearNormalizer()
            obs_normalizer.fit(
                data={"obs": self.replay_buffer[self.obs_key]},
                last_n_dims=1,
                mode=mode,
                **kwargs,
            )
            normalizer["obs"] = obs_normalizer["obs"]
        normalizer["condition_data"] = normalizer["trajectory"]
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.trajectory_key])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        data = {
            "sample_idx": np.asarray(sample_idx, dtype=np.int64),
            "obs": self.replay_buffer[self.obs_key][sample_idx].astype(np.float32),
            "trajectory": self.replay_buffer[self.trajectory_key][sample_idx].astype(np.float32),
            "condition_data": self.replay_buffer[self.condition_data_key][sample_idx].astype(
                np.float32
            ),
            "condition_mask": self.replay_buffer[self.condition_mask_key][sample_idx].astype(bool),
            "pose_waypoint": self.replay_buffer[self.pose_waypoint_key][sample_idx].astype(
                np.float32
            ),
            "gripper_waypoint": self.replay_buffer[self.gripper_waypoint_key][sample_idx].astype(
                np.float32
            ),
        }
        return dict_apply(data, torch.from_numpy)
