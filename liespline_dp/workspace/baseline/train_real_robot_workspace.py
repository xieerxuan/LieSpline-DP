from liespline_dp.workspace.baseline.base_baseline_workspace import BaseBaselineWorkspace


class TrainBaselineRealRobotWorkspace(BaseBaselineWorkspace):
    """Baseline training for native robot image demonstrations without simulator rollout."""

    scene = "real_robot"
    policy_type = "liespline_dp.policy.baseline.diffusion_image_policy.DiffusionImagePolicy"
