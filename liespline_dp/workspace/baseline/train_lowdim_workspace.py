from liespline_dp.workspace.baseline.base_baseline_workspace import BaseBaselineWorkspace


class TrainBaselineLowdimWorkspace(BaseBaselineWorkspace):
    """Baseline training for simulation with lowdim observations."""

    scene = "lowdim"
    policy_type = "liespline_dp.policy.baseline.diffusion_lowdim_policy.DiffusionLowdimPolicy"

    def _add_sampling_log(self, policy, task_cfg, batch, step_log):
        import torch

        result = policy.predict_action({"obs": batch["obs"]})
        gt_action = batch["action"]
        if task_cfg.pred_action_steps_only:
            pred_action = result["action"]
            start = task_cfg.n_obs_steps - 1
            gt_action = gt_action[:, start:start + task_cfg.n_action_steps]
        else:
            pred_action = result["action_pred"]
        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
            pred_action, gt_action
        ).item()
