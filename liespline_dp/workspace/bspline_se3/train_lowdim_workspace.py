from liespline_dp.workspace.bspline_se3.base_liespline_workspace import BaseLieSplineWorkspace


class TrainLieSplineLowdimWorkspace(BaseLieSplineWorkspace):
    """LieSpline training for simulation with lowdim observations."""

    scene = "lowdim"
    policy_type = "liespline_dp.policy.bspline_se3.bspline_se3_lowdim_policy.BsplineSE3LowdimPolicy"
