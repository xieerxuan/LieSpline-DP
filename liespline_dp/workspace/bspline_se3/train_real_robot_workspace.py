from liespline_dp.workspace.bspline_se3.base_liespline_workspace import BaseLieSplineWorkspace


class TrainLieSplineRealRobotWorkspace(BaseLieSplineWorkspace):
    """LieSpline training for native robot image demonstrations without simulator rollout."""

    scene = "real_robot"
    policy_type = "liespline_dp.policy.bspline_se3.bspline_se3_image_policy.BsplineSE3ImagePolicy"
