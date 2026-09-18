from liespline_dp.policy.baseline.diffusion_policy import DiffusionPolicy


class DiffusionLowdimPolicy(DiffusionPolicy):
    """Action diffusion conditioned on vector observations."""

    def _encode_obs(self, obs, steps=None):
        obs = self.normalizer["obs"].normalize(obs)
        return obs if steps is None else obs[:, :steps]

    def _add_observation_predictions(self, result, sample, start):
        if not self.obs_as_cond:
            obs = self.normalizer["obs"].unnormalize(sample[..., self.action_dim:])
            result["obs_pred"] = obs
            result["action_obs_pred"] = obs[:, start:start + self.n_action_steps]
