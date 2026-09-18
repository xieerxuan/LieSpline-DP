import torch

from liespline_dp.policy.bspline_se3.bspline_se3_policy import BsplineSE3Policy


class BsplineSE3LowdimPolicy(BsplineSE3Policy):
    """SE(3) B-spline diffusion with vector observations."""

    def _normalize_obs(self, obs):
        return self.normalizer["obs"].normalize(obs)

    def _obs_batch_size(self, obs) -> int:
        return int(obs.shape[0])

    def _latest_obs(self, obs):
        return obs[:, -1]

    def _current_pose_grip(self, obs: torch.Tensor, use_initial_gripper: bool = True):
        if self.num_robots == 1:
            current_pose, current_grip = self.action_pose_adapter.obs_to_current_pose_grip(
                obs[:, -1]
            )
        else:
            current_pose, current_grip = self.action_pose_adapter.obs_to_current_pose_grip_all(
                obs[:, -1]
            )
        if use_initial_gripper:
            current_grip = self._initial_gripper_like(current_grip)
        return current_pose, current_grip

    def _make_model_condition(
        self, normalized_obs: torch.Tensor, boundary_embedding: torch.Tensor = None
    ):
        condition = normalized_obs[:, : self.n_obs_steps]
        if boundary_embedding is not None:
            condition = torch.cat(
                [boundary_embedding[:, None], condition],
                dim=1,
            )
        return condition
