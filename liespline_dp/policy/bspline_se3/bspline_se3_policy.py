import itertools
import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from liespline_dp.common.action_pose_adapter import ActionPoseAdapter
from liespline_dp.model.common.normalizer import LinearNormalizer
from liespline_dp.policy.base_policy import BasePolicy
from liespline_dp.policy.bspline_se3.bspline_se3_utils import (
    control_points_to_se3_waypoints,
    control_points_to_zoh_waypoints,
    control_poses_to_local_tangent,
    get_next_bspline_se3_prefix,
    get_next_control_point_prefix,
    local_tangent_to_control_poses,
)

CONTROL_POINT_GRIPPER_REPRESENTATIONS = ("zoh_control_point",)


def resolve_diffusion_prefix_cp_steps(
    prefix_cp_steps: int, diffusion_prefix_cp_steps=None, future_only=None
) -> int:
    prefix_cp_steps = int(prefix_cp_steps)
    if diffusion_prefix_cp_steps is None:
        if future_only is None:
            diffusion_prefix_cp_steps = prefix_cp_steps
        else:
            diffusion_prefix_cp_steps = 0 if bool(future_only) else prefix_cp_steps
    diffusion_prefix_cp_steps = int(diffusion_prefix_cp_steps)
    if not 0 <= diffusion_prefix_cp_steps <= prefix_cp_steps:
        raise ValueError(
            "diffusion_prefix_cp_steps must be in "
            f"[0, {prefix_cp_steps}], got {diffusion_prefix_cp_steps}"
        )
    if future_only is True and diffusion_prefix_cp_steps != 0:
        raise ValueError("future_only=True requires diffusion_prefix_cp_steps=0")
    return diffusion_prefix_cp_steps


class BsplineSE3Policy(BasePolicy):
    """Network-agnostic SE(3) B-spline diffusion policy.

    Modality subclasses adapt observations; the diffusion network is a model
    component. SE(3) spline decoding, prefix caching, and joint pose/gripper
    diffusion are shared by lowdim and image policies.
    """

    def __init__(
        self,
        model: Any,
        noise_scheduler: Any,
        horizon,
        obs_dim,
        action_dim,
        trajectory_dim,
        prefix_cp_steps,
        n_action_steps,
        n_obs_steps,
        gripper_dim=1,
        action_pose_adapter=None,
        initial_gripper=None,
        degree=3,
        exec_horizon=None,
        binarize_gripper=True,
        gripper_representation="waypoint",
        gripper_spline_degree=1,
        num_inference_steps=None,
        num_robots=None,
        gripper_loss_weight=1.0 / 9.0,
        prefix_corruption_probability=0.0,
        prefix_corruption_scale_min=0.0,
        prefix_corruption_scale_max=1.0,
        prefix_corruption_translation_std=0.0,
        prefix_corruption_rotation_std_deg=0.0,
        diffusion_prefix_cp_steps=None,
        future_only=None,
        target_attn_mode=None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.trajectory_dim = int(trajectory_dim)
        self.prefix_cp_steps = int(prefix_cp_steps)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.gripper_dim = int(gripper_dim)
        if self.gripper_dim < 0:
            raise ValueError("gripper_dim must be non-negative")
        self.has_gripper = self.gripper_dim > 0
        self.degree = int(degree)
        self.exec_horizon = exec_horizon
        self.gripper_representation = str(gripper_representation)
        if self.gripper_representation not in ("waypoint", *CONTROL_POINT_GRIPPER_REPRESENTATIONS):
            raise ValueError(
                "gripper_representation must be 'waypoint', "
                "or 'zoh_control_point', got "
                f"{self.gripper_representation!r}"
            )
        self.gripper_spline_degree = int(gripper_spline_degree)
        if self.gripper_representation == "zoh_control_point" and self.gripper_spline_degree < 0:
            raise ValueError(
                "zoh_control_point gripper representation requires non-negative degree offset"
            )
        self.diffusion_prefix_cp_steps = resolve_diffusion_prefix_cp_steps(
            prefix_cp_steps=self.prefix_cp_steps,
            diffusion_prefix_cp_steps=diffusion_prefix_cp_steps,
            future_only=future_only,
        )
        self.excluded_prefix_cp_steps = self.prefix_cp_steps - self.diffusion_prefix_cp_steps
        self.future_only = self.diffusion_prefix_cp_steps == 0
        self.future_horizon = self.horizon - self.prefix_cp_steps
        if self.future_horizon < 1:
            raise ValueError(
                "future control-point horizon must be positive, got "
                f"horizon={self.horizon}, prefix_cp_steps={self.prefix_cp_steps}"
            )
        self.pose_diffusion_horizon = self.diffusion_prefix_cp_steps + self.future_horizon
        self.pose_pred_horizon = self.future_horizon
        if self.has_gripper:
            self.gripper_diffusion_horizon = (
                self.pose_diffusion_horizon
                if self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
                else self.n_action_steps
            )
        else:
            self.gripper_diffusion_horizon = 0
        self.diffusion_horizon = max(
            self.pose_diffusion_horizon,
            self.gripper_diffusion_horizon,
        )
        if num_robots is None:
            if self.trajectory_dim % 6 != 0:
                raise ValueError(
                    f"pose trajectory_dim must be divisible by 6, got {trajectory_dim}"
                )
            num_robots = self.trajectory_dim // 6
        self.num_robots = int(num_robots)
        expected_trajectory_dim = self.num_robots * 6
        if self.trajectory_dim != expected_trajectory_dim:
            raise ValueError(
                f"pose trajectory_dim must be num_robots * 6 "
                f"({expected_trajectory_dim}), got {self.trajectory_dim}"
            )
        self.gripper_waypoint_dim = self.num_robots * self.gripper_dim
        self.gripper_loss_weight = float(gripper_loss_weight)
        if not math.isfinite(self.gripper_loss_weight) or self.gripper_loss_weight < 0.0:
            raise ValueError("gripper_loss_weight must be finite and non-negative")
        self.diffusion_dim = self.trajectory_dim + self.gripper_waypoint_dim
        self.diffusion_output_dim = self.diffusion_dim
        if self.gripper_diffusion_horizon > self.diffusion_horizon:
            raise ValueError(
                "gripper diffusion horizon exceeds the joint horizon: "
                f"{self.gripper_diffusion_horizon} > {self.diffusion_horizon}"
            )
        model_horizon = getattr(self.model, "horizon", None)
        if model_horizon is not None and int(model_horizon) != self.diffusion_horizon:
            raise ValueError(
                f"Diffusion model horizon={model_horizon} does not match "
                f"policy diffusion_horizon={self.diffusion_horizon}"
            )
        self.initial_gripper = initial_gripper
        self.binarize_gripper = bool(binarize_gripper)
        self.prefix_corruption_probability = float(prefix_corruption_probability)
        self.prefix_corruption_scale_min = float(prefix_corruption_scale_min)
        self.prefix_corruption_scale_max = float(prefix_corruption_scale_max)
        self.prefix_corruption_translation_std = float(prefix_corruption_translation_std)
        self.prefix_corruption_rotation_std = math.radians(
            float(prefix_corruption_rotation_std_deg)
        )
        if not 0.0 <= self.prefix_corruption_probability <= 1.0:
            raise ValueError("prefix_corruption_probability must be in [0, 1]")
        if not 0.0 <= self.prefix_corruption_scale_min <= self.prefix_corruption_scale_max:
            raise ValueError("prefix corruption scales must satisfy 0 <= scale_min <= scale_max")
        if self.prefix_corruption_translation_std < 0.0:
            raise ValueError("prefix_corruption_translation_std must be non-negative")
        if self.prefix_corruption_rotation_std < 0.0:
            raise ValueError("prefix_corruption_rotation_std_deg must be non-negative")
        self.boundary_encoder = None
        if self.excluded_prefix_cp_steps > 0:
            boundary_token_dim = self.trajectory_dim
            if (
                self.has_gripper
                and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
            ):
                boundary_token_dim += self.gripper_waypoint_dim
            boundary_dim = self.excluded_prefix_cp_steps * boundary_token_dim
            self.boundary_encoder = torch.nn.Sequential(
                torch.nn.Linear(boundary_dim, self.obs_dim),
                torch.nn.Mish(),
                torch.nn.Linear(self.obs_dim, self.obs_dim),
            )
        self.kwargs = kwargs
        self._last_loss_components = {}

        if not isinstance(action_pose_adapter, ActionPoseAdapter):
            raise TypeError("Provide an ActionPoseAdapter in the task configuration")
        self.action_pose_adapter = action_pose_adapter

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = int(num_inference_steps)

        self._cached_prefix_pose_cp = None
        self._cached_prefix_gripper_cp = None
        self._last_planned_waypoint_pose = None
        self._last_planned_waypoint_grip = None
        if target_attn_mode is not None:
            self.model.set_target_attention(target_attn_mode, self.diffusion_prefix_cp_steps)

    def _corrupt_training_prefix(self, condition_data: torch.Tensor):
        """Weaken prefix motion cues while preserving the boundary control point.

        Each selected sample's older prefix control points are contracted toward
        the final prefix point. Small tangent-space SE(3) noise can additionally
        perturb those older points. The final prefix point remains exact so the
        current-pose anchor is not corrupted.
        """
        batch_size = int(condition_data.shape[0])
        applied = torch.zeros(
            batch_size,
            device=condition_data.device,
            dtype=torch.bool,
        )
        if (
            not self.training
            or self.prefix_corruption_probability <= 0.0
            or self.prefix_cp_steps <= 1
        ):
            return condition_data, applied

        applied = torch.rand(batch_size, device=condition_data.device) < (
            self.prefix_corruption_probability
        )
        if not bool(applied.any()):
            return condition_data, applied

        corrupted = condition_data.clone()
        prefix = corrupted[applied, : self.prefix_cp_steps].reshape(
            -1, self.prefix_cp_steps, self.num_robots, 6
        )
        boundary = prefix[:, -1:].clone()
        num_selected = prefix.shape[0]
        scale = torch.empty(
            num_selected,
            1,
            1,
            1,
            device=prefix.device,
            dtype=prefix.dtype,
        ).uniform_(
            self.prefix_corruption_scale_min,
            self.prefix_corruption_scale_max,
        )
        prefix[:, :-1] = boundary + scale * (prefix[:, :-1] - boundary)

        noise_std = prefix.new_tensor(
            [
                self.prefix_corruption_translation_std,
                self.prefix_corruption_translation_std,
                self.prefix_corruption_translation_std,
                self.prefix_corruption_rotation_std,
                self.prefix_corruption_rotation_std,
                self.prefix_corruption_rotation_std,
            ]
        )
        if bool((noise_std > 0.0).any()):
            prefix[:, :-1] += torch.randn_like(prefix[:, :-1]) * noise_std

        # Assignment through the reshaped advanced-index result does not write
        # back into ``corrupted``, so explicitly store the selected prefix.
        corrupted[applied, : self.prefix_cp_steps] = prefix.reshape(
            num_selected, self.prefix_cp_steps, self.trajectory_dim
        )
        return corrupted, applied

    def _make_model_condition(
        self, normalized_obs: torch.Tensor, boundary_embedding: torch.Tensor = None
    ):
        raise NotImplementedError

    def _make_boundary_embedding(
        self, prefix_cp: torch.Tensor, prefix_gripper_cp: torch.Tensor = None
    ):
        if self.excluded_prefix_cp_steps == 0:
            return None
        if prefix_cp.shape[-2:] != (self.prefix_cp_steps, self.trajectory_dim):
            raise ValueError(
                "Expected prefix control points shape "
                f"[...,{self.prefix_cp_steps},{self.trajectory_dim}], got "
                f"{tuple(prefix_cp.shape)}"
            )
        excluded_prefix = prefix_cp[:, : self.excluded_prefix_cp_steps]
        normalized_pose = self.normalizer["condition_data"].normalize(excluded_prefix)
        boundary_values = normalized_pose
        if (
            self.has_gripper
            and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
        ):
            expected = (
                prefix_cp.shape[0],
                self.prefix_cp_steps,
                self.gripper_waypoint_dim,
            )
            if prefix_gripper_cp is None or tuple(prefix_gripper_cp.shape) != expected:
                raise ValueError(
                    f"Expected gripper prefix shape {expected}, got "
                    f"{None if prefix_gripper_cp is None else tuple(prefix_gripper_cp.shape)}"
                )
            boundary_values = torch.cat(
                [
                    normalized_pose,
                    prefix_gripper_cp[:, : self.excluded_prefix_cp_steps],
                ],
                dim=-1,
            )
        return self.boundary_encoder(boundary_values.reshape(boundary_values.shape[0], -1))

    def _normalize_obs(self, obs):
        raise NotImplementedError

    def _obs_batch_size(self, obs) -> int:
        raise NotImplementedError

    def _latest_obs(self, obs):
        raise NotImplementedError

    def _predict_model_output(self, sample, timestep, model_cond):
        return self.model(sample, timestep, model_cond)

    def _pose_optimizer_params(self):
        if self.boundary_encoder is None:
            return self.model.parameters()
        return itertools.chain(
            self.model.parameters(),
            self.boundary_encoder.parameters(),
        )

    def get_optimizer(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]):
        if hasattr(self.model, "get_optim_groups"):
            groups = list(self.model.get_optim_groups(weight_decay=weight_decay))
        else:
            groups = [{"params": list(self.model.parameters()), "weight_decay": weight_decay}]
        if self.boundary_encoder is not None:
            groups.append(
                {
                    "params": list(self.boundary_encoder.parameters()),
                    "weight_decay": weight_decay,
                }
            )
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=tuple(betas),
        )

    def _pack_diffusion_trajectory(
        self, pose_trajectory: torch.Tensor, gripper_trajectory: torch.Tensor = None
    ):
        """Pack pose and representation-native gripper diffusion targets."""
        if pose_trajectory.shape[-2:] != (self.pose_diffusion_horizon, self.trajectory_dim):
            raise ValueError(
                "Expected pose diffusion shape "
                f"[...,{self.pose_diffusion_horizon},{self.trajectory_dim}], got "
                f"{tuple(pose_trajectory.shape)}"
            )
        packed = pose_trajectory.new_zeros(
            *pose_trajectory.shape[:-2],
            self.diffusion_horizon,
            self.diffusion_dim,
        )
        packed[..., : self.pose_diffusion_horizon, : self.trajectory_dim] = pose_trajectory
        if gripper_trajectory is not None:
            gripper_trajectory = gripper_trajectory.to(
                device=pose_trajectory.device,
                dtype=pose_trajectory.dtype,
            )
            expected = (
                *pose_trajectory.shape[:-2],
                self.gripper_diffusion_horizon,
                self.gripper_waypoint_dim,
            )
            if tuple(gripper_trajectory.shape) != expected:
                raise ValueError(
                    f"Expected gripper diffusion shape {expected}, got "
                    f"{tuple(gripper_trajectory.shape)}"
                )
            packed[..., : self.gripper_diffusion_horizon, self.trajectory_dim :] = (
                gripper_trajectory
            )
        return packed

    def _pack_diffusion_condition(
        self,
        pose_condition_data: torch.Tensor,
        pose_condition_mask: torch.Tensor,
        gripper_condition_data: torch.Tensor = None,
        gripper_condition_mask: torch.Tensor = None,
    ):
        condition_data = self._pack_diffusion_trajectory(
            pose_condition_data,
            gripper_condition_data,
        )
        pose_mask = torch.ones(
            *pose_condition_mask.shape[:-2],
            self.diffusion_horizon,
            self.trajectory_dim,
            device=pose_condition_mask.device,
            dtype=torch.bool,
        )
        pose_mask[..., : self.pose_diffusion_horizon, :] = pose_condition_mask.to(dtype=torch.bool)
        gripper_mask = torch.ones(
            *pose_condition_mask.shape[:-2],
            self.diffusion_horizon,
            self.gripper_waypoint_dim,
            device=pose_condition_mask.device,
            dtype=torch.bool,
        )
        if gripper_condition_mask is None:
            gripper_mask[..., : self.gripper_diffusion_horizon, :] = False
        else:
            expected = (
                *pose_condition_mask.shape[:-2],
                self.gripper_diffusion_horizon,
                self.gripper_waypoint_dim,
            )
            if tuple(gripper_condition_mask.shape) != expected:
                raise ValueError(
                    f"Expected gripper condition mask {expected}, got "
                    f"{tuple(gripper_condition_mask.shape)}"
                )
            gripper_mask[..., : self.gripper_diffusion_horizon, :] = gripper_condition_mask.to(
                dtype=torch.bool
            )
        condition_mask = torch.cat([pose_mask, gripper_mask], dim=-1)
        return condition_data, condition_mask

    def conditional_sample(
        self, condition_data, condition_mask, global_cond=None, generator=None, **kwargs
    ):
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self._predict_model_output(trajectory, t, global_cond)
            if model_output.shape[-1] != self.diffusion_output_dim:
                raise ValueError(
                    f"SE(3) diffusion model must output {self.diffusion_output_dim} "
                    f"channels ({self.trajectory_dim} pose + "
                    f"{self.gripper_waypoint_dim} gripper), got "
                    f"{model_output.shape[-1]}"
                )
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        pose_trajectory = trajectory[..., : self.pose_diffusion_horizon, : self.trajectory_dim]
        gripper_trajectory = trajectory[
            ..., : self.gripper_diffusion_horizon, self.trajectory_dim :
        ]
        return pose_trajectory, gripper_trajectory

    def _initial_gripper_like(self, current_grip: torch.Tensor):
        if self.initial_gripper is None:
            return current_grip
        initial_gripper = torch.as_tensor(
            self.initial_gripper,
            dtype=current_grip.dtype,
            device=current_grip.device,
        )
        if initial_gripper.ndim == 0:
            initial_gripper = initial_gripper.reshape(1)
        if initial_gripper.shape[-1] != self.gripper_dim:
            raise ValueError(
                f"initial_gripper must have dim {self.gripper_dim}, got {initial_gripper.shape[-1]}"
            )
        return initial_gripper.reshape(
            *((1,) * (current_grip.ndim - 1)),
            self.gripper_dim,
        ).expand_as(current_grip)

    def _current_pose_grip(self, obs: torch.Tensor, use_initial_gripper: bool = True):
        raise NotImplementedError

    def _flatten_gripper(self, gripper: torch.Tensor):
        return gripper.reshape(gripper.shape[0], self.gripper_waypoint_dim)

    def _split_pose_trajectory(self, trajectory: torch.Tensor):
        if self.num_robots == 1:
            return trajectory
        B, H, D = trajectory.shape
        if D != self.num_robots * 6:
            raise ValueError(f"Expected pose trajectory dim {self.num_robots * 6}, got {D}")
        return trajectory.reshape(B, H, self.num_robots, 6).permute(0, 2, 1, 3).contiguous()

    def _flatten_pose_trajectory(self, local_pose_cp: torch.Tensor):
        if self.num_robots == 1:
            return local_pose_cp
        local_pose_cp = local_pose_cp.permute(0, 2, 1, 3).contiguous()
        return local_pose_cp.reshape(
            local_pose_cp.shape[0], local_pose_cp.shape[1], self.trajectory_dim
        )

    def _local_tangent_to_control_poses(
        self, local_pose_cp: torch.Tensor, anchor_pose: torch.Tensor
    ):
        if self.num_robots == 1:
            return local_tangent_to_control_poses(local_pose_cp, anchor_pose)
        B, R, H = local_pose_cp.shape[:3]
        control_poses = local_tangent_to_control_poses(
            local_pose_cp.reshape(B * R, H, 6),
            anchor_pose.reshape(B * R, 4, 4),
        )
        return control_poses.reshape(B, R, H, 4, 4)

    def _control_poses_to_local_tangent(
        self, control_poses: torch.Tensor, anchor_pose: torch.Tensor
    ):
        if self.num_robots == 1:
            return control_poses_to_local_tangent(control_poses, anchor_pose)
        B, R, H = control_poses.shape[:3]
        local_pose_cp = control_poses_to_local_tangent(
            control_poses=control_poses.reshape(B * R, H, 4, 4),
            anchor_pose=anchor_pose.reshape(B * R, 4, 4),
        )
        return local_pose_cp.reshape(B, R, H, 6)

    def _control_points_to_pose_waypoints(self, control_poses: torch.Tensor):
        if self.num_robots == 1:
            return control_points_to_se3_waypoints(
                control_poses=control_poses,
                degree=self.degree,
                exec_horizon=self.exec_horizon,
                n_action_steps=self.n_action_steps,
            )
        B, R, H = control_poses.shape[:3]
        waypoint_poses = control_points_to_se3_waypoints(
            control_poses=control_poses.reshape(B * R, H, 4, 4),
            degree=self.degree,
            exec_horizon=self.exec_horizon,
            n_action_steps=self.n_action_steps,
        )
        A = waypoint_poses.shape[1]
        return waypoint_poses.reshape(B, R, A, 4, 4).permute(0, 2, 1, 3, 4)

    def _pose_waypoints_to_local(self, waypoint_poses: torch.Tensor, anchor_pose: torch.Tensor):
        if self.num_robots == 1:
            return control_poses_to_local_tangent(waypoint_poses, anchor_pose)
        B, A, R = waypoint_poses.shape[:3]
        local = control_poses_to_local_tangent(
            control_poses=waypoint_poses.permute(0, 2, 1, 3, 4).reshape(B * R, A, 4, 4),
            anchor_pose=anchor_pose.reshape(B * R, 4, 4),
        )
        return local.reshape(B, R, A, 6).permute(0, 2, 1, 3).reshape(B, A, self.trajectory_dim)

    def _get_next_prefix(self, control_poses: torch.Tensor):
        if self.num_robots == 1:
            return get_next_bspline_se3_prefix(
                control_poses=control_poses,
                prefix_cp_steps=self.prefix_cp_steps,
                exec_horizon=self.exec_horizon,
            )
        B, R, H = control_poses.shape[:3]
        next_pose = get_next_bspline_se3_prefix(
            control_poses=control_poses.reshape(B * R, H, 4, 4),
            prefix_cp_steps=self.prefix_cp_steps,
            exec_horizon=self.exec_horizon,
        )
        return next_pose.reshape(B, R, self.prefix_cp_steps, 4, 4)

    def _get_prefix_from_obs_or_cache(self, obs: torch.Tensor):
        device = self.device
        dtype = self.dtype
        if self._cached_prefix_pose_cp is None:
            current_pose, _ = self._current_pose_grip(obs)
            current_pose = current_pose.to(device=device, dtype=dtype)
            if self.num_robots == 1:
                self._cached_prefix_pose_cp = current_pose[:, None].repeat(
                    1, self.prefix_cp_steps, 1, 1
                )
            else:
                self._cached_prefix_pose_cp = current_pose[:, :, None].repeat(
                    1, 1, self.prefix_cp_steps, 1, 1
                )
        return self._cached_prefix_pose_cp

    def _get_gripper_prefix_from_obs_or_cache(self, obs, prefix_gripper_cp=None):
        if (
            not self.has_gripper
            or self.gripper_representation not in CONTROL_POINT_GRIPPER_REPRESENTATIONS
        ):
            return None
        device = self.device
        dtype = self.dtype
        if prefix_gripper_cp is not None:
            prefix_gripper_cp = prefix_gripper_cp.to(device=device, dtype=dtype)
            expected = (
                self._obs_batch_size(obs),
                self.prefix_cp_steps,
                self.gripper_waypoint_dim,
            )
            if tuple(prefix_gripper_cp.shape) != expected:
                raise ValueError(
                    f"Expected prefix_gripper_cp shape {expected}, got "
                    f"{tuple(prefix_gripper_cp.shape)}"
                )
            return prefix_gripper_cp
        if self._cached_prefix_gripper_cp is None:
            _, current_grip = self._current_pose_grip(obs)
            current_grip = self._flatten_gripper(current_grip).to(device=device, dtype=dtype)
            self._cached_prefix_gripper_cp = current_grip[:, None].repeat(
                1, self.prefix_cp_steps, 1
            )
        return self._cached_prefix_gripper_cp

    def _make_condition_from_prefix(
        self,
        obs: torch.Tensor,
        prefix_cp=None,
        prefix_pose_cp=None,
        anchor_pose=None,
    ):
        device = self.device
        dtype = self.dtype
        if anchor_pose is None:
            anchor_pose, _ = self._current_pose_grip(obs)
        anchor_pose = anchor_pose.to(device=device, dtype=dtype)

        if prefix_cp is not None:
            prefix_cp = prefix_cp.to(device=device, dtype=dtype)
            local_pose_cp = self._split_pose_trajectory(prefix_cp)
            prefix_pose_cp = self._local_tangent_to_control_poses(local_pose_cp, anchor_pose)
            return prefix_cp, prefix_pose_cp, anchor_pose

        if prefix_pose_cp is None:
            prefix_pose_cp = self._get_prefix_from_obs_or_cache(obs)
        prefix_pose_cp = prefix_pose_cp.to(device=device, dtype=dtype)
        local_pose_cp = self._control_poses_to_local_tangent(prefix_pose_cp, anchor_pose)
        prefix_cp = self._flatten_pose_trajectory(local_pose_cp)
        return prefix_cp, prefix_pose_cp, anchor_pose

    def _control_points_to_gripper_waypoints(self, gripper_control_points: torch.Tensor):
        if self.exec_horizon is None:
            raise ValueError("gripper control-point representation requires exec_horizon")
        return control_points_to_zoh_waypoints(
            control_points=gripper_control_points,
            degree=self.gripper_spline_degree,
            n_action_steps=self.n_action_steps,
            exec_horizon=self.exec_horizon,
        )

    def _predict_gripper_waypoint(self, gripper_prediction: torch.Tensor):
        """Decode denoised per-execution-waypoint gripper values."""
        if gripper_prediction.shape[1:] != (self.n_action_steps, self.gripper_waypoint_dim):
            raise ValueError(
                "Expected gripper prediction "
                f"[B,{self.n_action_steps},{self.gripper_waypoint_dim}], got "
                f"{tuple(gripper_prediction.shape)}"
            )
        if self.binarize_gripper:
            grip = torch.where(
                gripper_prediction >= 0.0,
                torch.ones_like(gripper_prediction),
                -torch.ones_like(gripper_prediction),
            )
        else:
            grip = gripper_prediction.clamp(-1.0, 1.0)
        if self.num_robots == 1:
            return grip
        return grip.reshape(grip.shape[0], grip.shape[1], self.num_robots, self.gripper_dim)

    def _make_diffusion_condition(
        self, prefix_cp: torch.Tensor, batch_size: int, prefix_gripper_cp: torch.Tensor = None
    ):
        pose_shape = (
            batch_size,
            self.pose_diffusion_horizon,
            self.trajectory_dim,
        )
        pose_cond_data = torch.zeros(
            size=pose_shape,
            device=self.device,
            dtype=self.dtype,
        )
        pose_cond_mask = torch.zeros_like(pose_cond_data, dtype=torch.bool)
        gripper_cond_data = None
        gripper_cond_mask = None
        if (
            self.has_gripper
            and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
        ):
            gripper_shape = (
                batch_size,
                self.gripper_diffusion_horizon,
                self.gripper_waypoint_dim,
            )
            gripper_cond_data = torch.zeros(
                size=gripper_shape,
                device=self.device,
                dtype=self.dtype,
            )
            gripper_cond_mask = torch.zeros_like(gripper_cond_data, dtype=torch.bool)
        if self.diffusion_prefix_cp_steps > 0:
            diffusion_prefix = prefix_cp[:, self.excluded_prefix_cp_steps :]
            pose_cond_data[:, : self.diffusion_prefix_cp_steps] = self.normalizer[
                "condition_data"
            ].normalize(diffusion_prefix)
            pose_cond_mask[:, : self.diffusion_prefix_cp_steps] = True
            if (
                self.has_gripper
                and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
            ):
                if prefix_gripper_cp is None:
                    raise ValueError("gripper prefix is required in control-point mode")
                gripper_cond_data[:, : self.diffusion_prefix_cp_steps] = prefix_gripper_cp[
                    :, self.excluded_prefix_cp_steps :
                ]
                gripper_cond_mask[:, : self.diffusion_prefix_cp_steps] = True
        return self._pack_diffusion_condition(
            pose_cond_data,
            pose_cond_mask,
            gripper_cond_data,
            gripper_cond_mask,
        )

    def _assemble_pose_trajectory(self, prefix_cp: torch.Tensor, pose_prediction: torch.Tensor):
        pose_prediction = self.normalizer["trajectory"].unnormalize(pose_prediction)
        future_prediction = pose_prediction[:, self.diffusion_prefix_cp_steps :]
        return torch.cat([prefix_cp, future_prediction], dim=1)

    def _assemble_gripper_trajectory(
        self, prefix_gripper_cp: torch.Tensor, gripper_prediction: torch.Tensor
    ):
        if (
            not self.has_gripper
            or self.gripper_representation not in CONTROL_POINT_GRIPPER_REPRESENTATIONS
        ):
            return None
        future_prediction = gripper_prediction[:, self.diffusion_prefix_cp_steps :]
        return torch.cat([prefix_gripper_cp, future_prediction], dim=1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        obs = obs_dict["obs"]
        prefix_cp = obs_dict.get("prefix_cp", None)
        prefix_pose_cp = obs_dict.get("prefix_pose_cp", None)
        prefix_gripper_cp = obs_dict.get("prefix_gripper_cp", None)
        anchor_pose = obs_dict.get("anchor_pose", None)
        prev_action = obs_dict.get("prev_action", None)

        nobs_obs = self._normalize_obs(obs)
        B = self._obs_batch_size(nobs_obs)
        device = self.device
        dtype = self.dtype
        cached_batch_mismatch = (
            self._cached_prefix_pose_cp is not None and self._cached_prefix_pose_cp.shape[0] != B
        ) or (
            self._cached_prefix_gripper_cp is not None
            and self._cached_prefix_gripper_cp.shape[0] != B
        )
        if cached_batch_mismatch:
            self.reset()

        prefix_cp, _, anchor_pose = self._make_condition_from_prefix(
            obs=obs,
            prefix_cp=prefix_cp,
            prefix_pose_cp=prefix_pose_cp,
            anchor_pose=anchor_pose,
        )
        prefix_gripper_cp = self._get_gripper_prefix_from_obs_or_cache(
            obs,
            prefix_gripper_cp=prefix_gripper_cp,
        )
        boundary_embedding = self._make_boundary_embedding(
            prefix_cp,
            prefix_gripper_cp,
        )
        global_cond = self._make_model_condition(
            nobs_obs,
            boundary_embedding=boundary_embedding,
        )
        cond_data, cond_mask = self._make_diffusion_condition(
            prefix_cp,
            B,
            prefix_gripper_cp,
        )
        pose_prediction, gripper_prediction = self.conditional_sample(
            cond_data, cond_mask, global_cond=global_cond, **self.kwargs
        )

        pose_trajectory = self._assemble_pose_trajectory(
            prefix_cp,
            pose_prediction,
        )
        local_pose_cp = self._split_pose_trajectory(pose_trajectory)
        control_poses = self._local_tangent_to_control_poses(local_pose_cp, anchor_pose)
        waypoint_poses = self._control_points_to_pose_waypoints(control_poses)
        pose_waypoint_local = self._pose_waypoints_to_local(waypoint_poses, anchor_pose)
        gripper_control_point = self._assemble_gripper_trajectory(
            prefix_gripper_cp,
            gripper_prediction,
        )
        gripper_waypoint_prediction = None
        waypoint_grip = None
        if self.has_gripper:
            gripper_waypoint_prediction = (
                self._control_points_to_gripper_waypoints(gripper_control_point)
                if gripper_control_point is not None
                else gripper_prediction
            )
            waypoint_grip = self._predict_gripper_waypoint(gripper_waypoint_prediction)

        if prev_action is not None:
            prev_action = prev_action.to(device=device, dtype=dtype)
            if prev_action.ndim == 2:
                prev_action = prev_action[:, None].expand(-1, waypoint_poses.shape[1], -1).clone()
        if self.num_robots == 1:
            action = self.action_pose_adapter.pose_grip_to_action(
                waypoint_poses,
                waypoint_grip,
                obs=self._latest_obs(obs),
                prev_action=prev_action,
            )
        else:
            action = self.action_pose_adapter.pose_grip_to_action_all(
                waypoint_poses,
                waypoint_grip,
            )

        self._cached_prefix_pose_cp = self._get_next_prefix(control_poses).detach()
        if gripper_control_point is not None:
            self._cached_prefix_gripper_cp = get_next_control_point_prefix(
                control_points=gripper_control_point,
                prefix_cp_steps=self.prefix_cp_steps,
                exec_horizon=self.exec_horizon,
            ).detach()
        self._last_planned_waypoint_pose = waypoint_poses[:, -1].detach()
        self._last_planned_waypoint_grip = (
            waypoint_grip[:, -1].detach() if waypoint_grip is not None else None
        )

        result = {
            "action": action,
            "action_pred": pose_trajectory,
            "pose_trajectory_pred": pose_trajectory,
            "control_pose_pred": control_poses,
            "waypoint_pose": waypoint_poses,
            "pose_waypoint_local": pose_waypoint_local,
            "diffusion_output_pred": self._pack_diffusion_trajectory(
                pose_trajectory[:, self.excluded_prefix_cp_steps :],
                gripper_prediction if self.has_gripper else None,
            ),
        }
        if self.has_gripper:
            result.update(
                {
                    "waypoint_grip": waypoint_grip,
                    "gripper_waypoint_pred": gripper_waypoint_prediction,
                    "gripper_pred": gripper_prediction,
                }
            )
        if gripper_control_point is not None:
            result["gripper_control_point_pred"] = gripper_control_point
        return result

    def predict_pose_waypoint(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        obs = obs_dict["obs"]
        prefix_cp = obs_dict.get("prefix_cp", None)
        prefix_pose_cp = obs_dict.get("prefix_pose_cp", None)
        prefix_gripper_cp = obs_dict.get("prefix_gripper_cp", None)
        anchor_pose = obs_dict.get("anchor_pose", None)

        nobs_obs = self._normalize_obs(obs)
        B = self._obs_batch_size(nobs_obs)

        prefix_cp, _, anchor_pose = self._make_condition_from_prefix(
            obs=obs,
            prefix_cp=prefix_cp,
            prefix_pose_cp=prefix_pose_cp,
            anchor_pose=anchor_pose,
        )
        prefix_gripper_cp = self._get_gripper_prefix_from_obs_or_cache(
            obs,
            prefix_gripper_cp=prefix_gripper_cp,
        )
        boundary_embedding = self._make_boundary_embedding(
            prefix_cp,
            prefix_gripper_cp,
        )
        global_cond = self._make_model_condition(
            nobs_obs,
            boundary_embedding=boundary_embedding,
        )
        cond_data, cond_mask = self._make_diffusion_condition(
            prefix_cp,
            B,
            prefix_gripper_cp,
        )
        pose_prediction, gripper_prediction = self.conditional_sample(
            cond_data, cond_mask, global_cond=global_cond, **self.kwargs
        )
        pose_trajectory = self._assemble_pose_trajectory(
            prefix_cp,
            pose_prediction,
        )
        local_pose_cp = self._split_pose_trajectory(pose_trajectory)
        control_poses = self._local_tangent_to_control_poses(local_pose_cp, anchor_pose)
        waypoint_poses = self._control_points_to_pose_waypoints(control_poses)
        pose_waypoint_local = self._pose_waypoints_to_local(waypoint_poses, anchor_pose)
        gripper_control_point = self._assemble_gripper_trajectory(
            prefix_gripper_cp,
            gripper_prediction,
        )
        gripper_waypoint_prediction = None
        waypoint_grip = None
        if self.has_gripper:
            gripper_waypoint_prediction = (
                self._control_points_to_gripper_waypoints(gripper_control_point)
                if gripper_control_point is not None
                else gripper_prediction
            )
            waypoint_grip = self._predict_gripper_waypoint(gripper_waypoint_prediction)
        result = {
            "pose_trajectory_pred": pose_trajectory,
            "control_pose_pred": control_poses,
            "waypoint_pose": waypoint_poses,
            "pose_waypoint_local": pose_waypoint_local,
        }
        if self.has_gripper:
            result.update(
                {
                    "waypoint_grip": waypoint_grip,
                    "gripper_waypoint_pred": gripper_waypoint_prediction,
                    "gripper_pred": gripper_prediction,
                }
            )
        if gripper_control_point is not None:
            result["gripper_control_point_pred"] = gripper_control_point
        return result

    def reset(self):
        self._cached_prefix_pose_cp = None
        self._cached_prefix_gripper_cp = None
        self._last_planned_waypoint_pose = None
        self._last_planned_waypoint_grip = None

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _diffusion_losses(
        self,
        trajectory,
        condition_data,
        condition_mask,
        global_cond,
        gripper_target,
        gripper_condition_data=None,
        gripper_condition_mask=None,
    ):
        """Compute joint pose-control and gripper diffusion noise loss."""
        diffusion_trajectory = self._pack_diffusion_trajectory(trajectory, gripper_target)
        diffusion_condition_data, diffusion_condition_mask = self._pack_diffusion_condition(
            condition_data,
            condition_mask,
            gripper_condition_data,
            gripper_condition_mask,
        )
        noise = torch.randn_like(diffusion_trajectory)
        bsz = diffusion_trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(diffusion_trajectory, noise, timesteps)
        noisy_trajectory[diffusion_condition_mask] = diffusion_condition_data[
            diffusion_condition_mask
        ]

        pred = self._predict_model_output(noisy_trajectory, timesteps, global_cond)
        if pred.shape[-1] != self.diffusion_output_dim:
            raise ValueError(
                f"Expected diffusion output dim {self.diffusion_output_dim}, got {pred.shape[-1]}"
            )
        pose_pred = pred[:, : self.pose_diffusion_horizon, : self.trajectory_dim]
        gripper_pred = pred[:, : self.gripper_diffusion_horizon, self.trajectory_dim :]
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = diffusion_trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        pose_target = target[:, : self.pose_diffusion_horizon, : self.trajectory_dim]
        pose_loss_values = F.mse_loss(
            pose_pred,
            pose_target,
            reduction="none",
        )
        pose_valid = (~condition_mask).to(pose_loss_values.dtype)
        pose_loss = (pose_loss_values * pose_valid).sum() / pose_valid.sum().clamp_min(1.0)

        if self.has_gripper:
            gripper_noise_target = target[
                :, : self.gripper_diffusion_horizon, self.trajectory_dim :
            ]
            gripper_loss_values = F.mse_loss(
                gripper_pred,
                gripper_noise_target,
                reduction="none",
            )
            if gripper_condition_mask is None:
                gripper_valid = torch.ones_like(gripper_loss_values)
            else:
                gripper_valid = (~gripper_condition_mask).to(gripper_loss_values.dtype)
            gripper_loss = (
                gripper_loss_values * gripper_valid
            ).sum() / gripper_valid.sum().clamp_min(1.0)
        else:
            gripper_loss = pose_loss.new_zeros(())
        return pose_loss, gripper_loss, gripper_pred

    @torch.no_grad()
    def compute_gripper_validation_metrics(self, batch, prediction=None):
        """Score dense execution waypoints from observations and inherited prefixes.

        An existing inference result can be supplied to score the same rollout
        as pose metrics without running diffusion twice. Otherwise sampling uses
        the configured inference schedule and leaves the rollout caches untouched.
        """
        if not self.has_gripper:
            return {}
        if prediction is None:
            obs = self._normalize_obs(batch["obs"])
            pose_prefix = batch["condition_data"][:, : self.prefix_cp_steps]
            gripper_prefix = None
            if self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS:
                gripper_prefix = batch["gripper_control_point"][:, : self.prefix_cp_steps]
            condition_data, condition_mask = self._make_diffusion_condition(
                pose_prefix,
                self._obs_batch_size(obs),
                gripper_prefix,
            )
            boundary_embedding = self._make_boundary_embedding(pose_prefix, gripper_prefix)
            global_cond = self._make_model_condition(obs, boundary_embedding=boundary_embedding)
            _, gripper_pred = self.conditional_sample(
                condition_data, condition_mask, global_cond=global_cond, **self.kwargs
            )
            if self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS:
                full_gripper_pred = self._assemble_gripper_trajectory(gripper_prefix, gripper_pred)
                gripper_pred = self._control_points_to_gripper_waypoints(full_gripper_pred)
        else:
            gripper_pred = prediction["gripper_waypoint_pred"]

        target_value = batch["gripper_waypoint"].to(
            device=gripper_pred.device, dtype=gripper_pred.dtype
        )
        if gripper_pred.shape != target_value.shape:
            raise ValueError(
                "Gripper prediction and target shapes must match, got "
                f"{tuple(gripper_pred.shape)} and {tuple(target_value.shape)}"
            )
        if not self.binarize_gripper:
            prediction_value = gripper_pred.clamp(-1.0, 1.0)
            error = prediction_value - target_value
            return {
                "absolute_error_sum": error.abs().sum(),
                "squared_error_sum": error.square().sum(),
                "count": target_value.new_tensor(target_value.numel()),
            }

        target = target_value >= 0.0
        predicted_open = gripper_pred >= 0.0
        positive = target
        negative = ~target
        return {
            "correct": (predicted_open == target).sum(),
            "count": torch.tensor(target.numel(), device=target.device),
            "positive_correct": (predicted_open & positive).sum(),
            "positive_count": positive.sum(),
            "negative_correct": ((~predicted_open) & negative).sum(),
            "negative_count": negative.sum(),
        }

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        obs = self._normalize_obs(batch["obs"])
        full_trajectory = self.normalizer["trajectory"].normalize(batch["trajectory"])
        raw_condition_data = batch["condition_data"]
        corruption_mask = torch.zeros(
            raw_condition_data.shape[0],
            device=raw_condition_data.device,
            dtype=torch.bool,
        )
        raw_condition_data, corruption_mask = self._corrupt_training_prefix(raw_condition_data)
        raw_prefix = raw_condition_data[:, : self.prefix_cp_steps]
        full_gripper_cp = None
        gripper_prefix = None
        if (
            self.has_gripper
            and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
        ):
            if "gripper_control_point" not in batch:
                raise KeyError("gripper control-point mode requires batch['gripper_control_point']")
            full_gripper_cp = batch["gripper_control_point"]
            gripper_prefix = full_gripper_cp[:, : self.prefix_cp_steps]
        boundary_embedding = self._make_boundary_embedding(
            raw_prefix,
            gripper_prefix,
        )
        global_cond = self._make_model_condition(
            obs,
            boundary_embedding=boundary_embedding,
        )

        trajectory = full_trajectory[:, self.excluded_prefix_cp_steps :]
        condition_data = torch.zeros_like(trajectory)
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        gripper_condition_data = None
        gripper_condition_mask = None
        if (
            self.has_gripper
            and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
        ):
            gripper_target = full_gripper_cp[:, self.excluded_prefix_cp_steps :]
            gripper_condition_data = torch.zeros_like(gripper_target)
            gripper_condition_mask = torch.zeros_like(gripper_target, dtype=torch.bool)
        elif self.has_gripper:
            gripper_target = batch["gripper_waypoint"]
        else:
            gripper_target = None
        if self.diffusion_prefix_cp_steps > 0:
            diffusion_prefix = raw_prefix[:, self.excluded_prefix_cp_steps :]
            condition_data[:, : self.diffusion_prefix_cp_steps] = self.normalizer[
                "condition_data"
            ].normalize(diffusion_prefix)
            condition_mask[:, : self.diffusion_prefix_cp_steps] = True
            if (
                self.has_gripper
                and self.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
            ):
                gripper_condition_data[:, : self.diffusion_prefix_cp_steps] = gripper_prefix[
                    :, self.excluded_prefix_cp_steps :
                ]
                gripper_condition_mask[:, : self.diffusion_prefix_cp_steps] = True

        pose_loss, gripper_loss, _ = self._diffusion_losses(
            trajectory=trajectory,
            condition_data=condition_data,
            condition_mask=condition_mask,
            global_cond=global_cond,
            gripper_target=gripper_target,
            gripper_condition_data=gripper_condition_data,
            gripper_condition_mask=gripper_condition_mask,
        )
        self._last_loss_components = {
            "pose_loss": pose_loss.detach(),
            "gripper_loss": gripper_loss.detach(),
            "prefix_corruption_rate": corruption_mask.float().mean().detach(),
        }
        return pose_loss + self.gripper_loss_weight * gripper_loss
