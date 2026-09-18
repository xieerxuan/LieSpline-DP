import itertools
from typing import Any, Dict, Mapping, Tuple

import torch
from liespline_dp.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.model.vision.image_obs_encoder import (
    get_image_obs_encoder,
)
from liespline_dp.policy.bspline_se3.bspline_se3_policy import (
    CONTROL_POINT_GRIPPER_REPRESENTATIONS,
    BsplineSE3Policy,
    resolve_diffusion_prefix_cp_steps,
)


class BsplineSE3ImagePolicy(BsplineSE3Policy):
    """Image observation adapter for the shared SE(3) B-spline algorithm.

    Accepts a diffusion model component; defaults to TransformerForDiffusion.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: Any,
        horizon,
        trajectory_dim,
        prefix_cp_steps,
        n_action_steps,
        n_obs_steps,
        action_pose_adapter,
        diffusion_model_kwargs: Mapping = None,
        model=None,
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.01,
        time_as_cond=True,
        obs_as_cond=True,
        gripper_dim=1,
        num_robots=None,
        initial_gripper=None,
        degree=3,
        exec_horizon=None,
        binarize_gripper=True,
        gripper_representation="waypoint",
        gripper_spline_degree=1,
        num_inference_steps=None,
        obs_encoder=None,
        crop_shape=(76, 76),
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        diffusion_prefix_cp_steps=None,
        future_only=None,
        **kwargs,
    ):
        if not obs_as_cond:
            raise ValueError("SE(3) bspline image policy requires obs_as_cond=True")
        model_kwargs = dict(
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
        )
        model_kwargs.update(diffusion_model_kwargs or {})
        action_dim = int(shape_meta["action"]["shape"][0])
        if obs_encoder is None:
            obs_encoder = get_image_obs_encoder(
                shape_meta=shape_meta,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
        obs_feature_dim = int(obs_encoder.output_shape()[0])
        if num_robots is None:
            num_robots = int(trajectory_dim) // 6
        has_gripper = int(gripper_dim) > 0
        gripper_waypoint_dim = int(num_robots) * int(gripper_dim)
        diffusion_dim = int(trajectory_dim) + gripper_waypoint_dim
        diffusion_prefix_cp_steps = resolve_diffusion_prefix_cp_steps(
            prefix_cp_steps=prefix_cp_steps,
            diffusion_prefix_cp_steps=diffusion_prefix_cp_steps,
            future_only=future_only,
        )
        excluded_prefix_cp_steps = int(prefix_cp_steps) - diffusion_prefix_cp_steps
        future_horizon = int(horizon) - int(prefix_cp_steps)
        if future_horizon < 1:
            raise ValueError(
                "future control-point horizon must be positive, got "
                f"horizon={horizon}, prefix_cp_steps={prefix_cp_steps}"
            )
        pose_diffusion_horizon = diffusion_prefix_cp_steps + future_horizon
        if has_gripper:
            gripper_diffusion_horizon = (
                pose_diffusion_horizon
                if str(gripper_representation) in CONTROL_POINT_GRIPPER_REPRESENTATIONS
                else int(n_action_steps)
            )
        else:
            gripper_diffusion_horizon = 0
        diffusion_horizon = max(
            pose_diffusion_horizon,
            gripper_diffusion_horizon,
        )
        model_n_obs_steps = int(n_obs_steps) + int(excluded_prefix_cp_steps > 0)
        if model is None:
            model = self._build_diffusion_model(
                obs_feature_dim=obs_feature_dim,
                horizon=diffusion_horizon,
                trajectory_dim=diffusion_dim,
                output_dim=diffusion_dim,
                n_obs_steps=model_n_obs_steps,
                model_kwargs=model_kwargs,
            )

        super().__init__(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            obs_dim=obs_feature_dim,
            action_dim=action_dim,
            trajectory_dim=trajectory_dim,
            prefix_cp_steps=prefix_cp_steps,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            gripper_dim=gripper_dim,
            action_pose_adapter=action_pose_adapter,
            initial_gripper=initial_gripper,
            degree=degree,
            exec_horizon=exec_horizon,
            binarize_gripper=binarize_gripper,
            gripper_representation=gripper_representation,
            gripper_spline_degree=gripper_spline_degree,
            num_inference_steps=num_inference_steps,
            num_robots=num_robots,
            diffusion_prefix_cp_steps=diffusion_prefix_cp_steps,
            future_only=future_only,
            **kwargs,
        )
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.obs_feature_dim = obs_feature_dim

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    @staticmethod
    def _build_diffusion_model(
        obs_feature_dim: int,
        horizon: int,
        trajectory_dim: int,
        output_dim: int,
        n_obs_steps: int,
        model_kwargs: dict,
    ):
        return TransformerForDiffusion(
            input_dim=trajectory_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            **model_kwargs,
        )

    def _wrap_obs_input(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "obs" in obs_dict:
            return obs_dict
        policy_keys = {
            "prefix_cp",
            "prefix_pose_cp",
            "prefix_gripper_cp",
            "anchor_pose",
            "prev_action",
            "past_action",
        }
        wrapped = {"obs": {key: value for key, value in obs_dict.items() if key not in policy_keys}}
        wrapped.update({key: obs_dict[key] for key in policy_keys if key in obs_dict})
        return wrapped

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_action(self._wrap_obs_input(obs_dict))

    def predict_pose_waypoint(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_pose_waypoint(self._wrap_obs_input(obs_dict))

    def _normalize_obs(self, obs):
        nobs = self.normalizer.normalize(obs)
        return dict_apply(nobs, lambda x: x.to(device=self.device, dtype=self.dtype))

    def _obs_batch_size(self, obs) -> int:
        return int(next(iter(obs.values())).shape[0])

    def _latest_obs(self, obs):
        return dict_apply(obs, lambda x: x[:, -1])

    def _adapter_obs(self, obs):
        latest = self._latest_obs(obs)
        return {key: latest[key] for key in self.action_pose_adapter.obs_keys if key in latest}

    def _current_pose_grip(self, obs, use_initial_gripper: bool = True):
        adapter_obs = self._adapter_obs(obs)
        if self.num_robots == 1:
            current_pose, current_grip = self.action_pose_adapter.obs_to_current_pose_grip(
                adapter_obs
            )
        else:
            current_pose, current_grip = self.action_pose_adapter.obs_to_current_pose_grip_all(
                adapter_obs
            )
        if use_initial_gripper:
            current_grip = self._initial_gripper_like(current_grip)
        return current_pose, current_grip

    def _make_model_condition(self, normalized_obs, boundary_embedding: torch.Tensor = None):
        batch_size = self._obs_batch_size(normalized_obs)
        obs = dict_apply(
            normalized_obs,
            lambda x: x[:, : self.n_obs_steps].reshape(-1, *x.shape[2:]),
        )
        features = self.obs_encoder(obs).reshape(batch_size, self.n_obs_steps, -1)
        if boundary_embedding is not None:
            features = torch.cat(
                [boundary_embedding[:, None], features],
                dim=1,
            )
        return self._format_model_condition(features)

    def _format_model_condition(self, obs_features: torch.Tensor) -> torch.Tensor:
        return obs_features

    def _pose_optimizer_groups(self, weight_decay: float):
        if hasattr(self.model, "get_optim_groups"):
            groups = list(self.model.get_optim_groups(weight_decay=weight_decay))
        else:
            groups = [
                {
                    "params": list(self.model.parameters()),
                    "weight_decay": weight_decay,
                }
            ]
        if getattr(self, "boundary_encoder", None) is not None:
            groups.append(
                {
                    "params": list(self.boundary_encoder.parameters()),
                    "weight_decay": weight_decay,
                }
            )
        groups.append(
            {
                "params": list(self.obs_encoder.parameters()),
                "weight_decay": weight_decay,
            }
        )
        return groups

    def get_pose_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ):
        return torch.optim.AdamW(
            self._pose_optimizer_groups(weight_decay),
            lr=learning_rate,
            betas=tuple(betas),
        )

    def get_optimizer(
        self,
        learning_rate: float,
        betas: Tuple[float, float],
        weight_decay: float = None,
        transformer_weight_decay: float = None,
        obs_encoder_weight_decay: float = None,
    ):
        if transformer_weight_decay is None:
            if weight_decay is None:
                raise ValueError("weight_decay or transformer_weight_decay is required")
            transformer_weight_decay = weight_decay
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = transformer_weight_decay
        groups = self._pose_optimizer_groups(transformer_weight_decay)
        groups[-1]["weight_decay"] = obs_encoder_weight_decay
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=tuple(betas),
        )

    def _pose_optimizer_params(self):
        modules = [self.obs_encoder, self.model]
        if getattr(self, "boundary_encoder", None) is not None:
            modules.append(self.boundary_encoder)
        return itertools.chain.from_iterable(module.parameters() for module in modules)
