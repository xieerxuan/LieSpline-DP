from typing import Tuple

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from liespline_dp.model.vision.image_obs_encoder import (
    get_image_obs_encoder,
)
from liespline_dp.policy.baseline.diffusion_policy import DiffusionPolicy


class DiffusionImagePolicy(DiffusionPolicy):
    """Action diffusion conditioned on encoded image observations."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # image
        crop_shape=(76, 76),
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        # arch
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        pred_action_steps_only=False,
        model=None,
        obs_encoder=None,
        # parameters passed to step
        **kwargs,
    ):

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        if obs_encoder is None:
            obs_encoder = get_image_obs_encoder(
                shape_meta=shape_meta,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        if model is None:
            model = TransformerForDiffusion(
                input_dim=input_dim,
                output_dim=output_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                cond_dim=cond_dim,
                n_layer=n_layer,
                n_head=n_head,
                n_emb=n_emb,
                p_drop_emb=p_drop_emb,
                p_drop_attn=p_drop_attn,
                causal_attn=causal_attn,
                time_as_cond=time_as_cond,
                obs_as_cond=obs_as_cond,
                n_cond_layers=n_cond_layers,
            )

        super().__init__(
            model=model, noise_scheduler=noise_scheduler, horizon=horizon,
            obs_dim=obs_feature_dim, action_dim=action_dim,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps, obs_as_cond=obs_as_cond,
            pred_action_steps_only=pred_action_steps_only, **kwargs,
        )
        self.obs_encoder = obs_encoder
        self.obs_feature_dim = obs_feature_dim

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append(
            {"params": self.obs_encoder.parameters(), "weight_decay": obs_encoder_weight_decay}
        )
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def predict_action(self, obs_dict):
        assert "past_action" not in obs_dict
        return super().predict_action({"obs": obs_dict})

    def _encode_obs(self, obs, steps=None):
        obs = self.normalizer.normalize(obs)
        value = next(iter(obs.values()))
        batch_size = value.shape[0]
        steps = value.shape[1] if steps is None else steps
        flat_obs = dict_apply(obs, lambda x: x[:, :steps].reshape(-1, *x.shape[2:]))
        return self.obs_encoder(flat_obs).reshape(batch_size, steps, -1)

    def _inpainting_trajectory(self, action, obs):
        # Preserve the original image inpainting encoder gradient behavior.
        return torch.cat([action, obs], dim=-1).detach()
