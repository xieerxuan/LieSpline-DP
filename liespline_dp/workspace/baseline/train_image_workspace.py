from liespline_dp.workspace.baseline.base_baseline_workspace import BaseBaselineWorkspace


class TrainBaselineImageWorkspace(BaseBaselineWorkspace):
    """Baseline training for simulation with image observations."""

    scene = "image"
    policy_type = "liespline_dp.policy.baseline.diffusion_image_policy.DiffusionImagePolicy"
