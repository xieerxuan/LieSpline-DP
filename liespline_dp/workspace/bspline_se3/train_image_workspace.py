from liespline_dp.workspace.bspline_se3.base_liespline_workspace import BaseLieSplineWorkspace


class TrainLieSplineImageWorkspace(BaseLieSplineWorkspace):
    """LieSpline training for simulation with image observations."""

    scene = "image"
    policy_type = "liespline_dp.policy.bspline_se3.bspline_se3_image_policy.BsplineSE3ImagePolicy"
