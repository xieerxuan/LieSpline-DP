import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.optimize as so
import scipy.sparse as ss

from liespline_dp.common.bspline_util import (
    build_bspline_design_matrix,
    get_padded_observation_sequence,
    get_uniform_phase,
)
from liespline_dp.common.se3_util import se3_exp, se3_inv, se3_log, se3_mul, so3_log


@dataclass
class SE3SplineFit:
    control_poses: np.ndarray
    phase: np.ndarray
    tangent_control_points: np.ndarray
    tangent_points: np.ndarray


def cumulative_basis_from_bspline_design(design: np.ndarray) -> np.ndarray:
    """Convert Euclidean B-spline basis B_i(s) to cumulative basis Lambda_j(s).

    For Euclidean control points C_i:
        sum_i B_i C_i = C_0 + sum_{j=1}^{L-1} Lambda_j (C_j - C_{j-1})
    where Lambda_j = sum_{i=j}^{L-1} B_i.
    """
    design = np.asarray(design, dtype=np.float64)
    assert design.ndim >= 2
    if design.shape[-1] <= 1:
        return np.zeros(design.shape[:-1] + (0,), dtype=np.float64)
    reverse_cumsum = np.cumsum(design[..., ::-1], axis=-1)[..., ::-1]
    return reverse_cumsum[..., 1:]


def build_cumulative_bspline_basis(
    phase: np.ndarray, num_control_points: int, degree: int
) -> Tuple[np.ndarray, np.ndarray]:
    design = build_bspline_design_matrix(
        phase=phase,
        num_control_points=num_control_points,
        degree=degree,
    )
    cumulative = cumulative_basis_from_bspline_design(design)
    return design, cumulative


def eval_se3_cumulative_bspline(
    control_poses: np.ndarray,
    phase: np.ndarray,
    degree: int = 3,
    return_basis: bool = False,
):
    """Evaluate a cumulative B-spline on SE(3).

    The curve follows the project convention:
        T(s) = Q_0 prod_j Exp(Lambda_j(s) Log(Q_{j-1}^{-1} Q_j)).
    """
    control_poses = np.asarray(control_poses, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64).reshape(-1)
    assert control_poses.ndim == 3
    assert control_poses.shape[-2:] == (4, 4)
    assert control_poses.shape[0] > degree

    _, cumulative = build_cumulative_bspline_basis(
        phase=phase,
        num_control_points=control_poses.shape[0],
        degree=degree,
    )
    out = _eval_se3_cumulative_bspline_from_basis(
        control_poses=control_poses,
        cumulative=cumulative,
    )

    if return_basis:
        return out, cumulative
    return out


def _eval_se3_cumulative_bspline_from_basis(
    control_poses: np.ndarray,
    cumulative: np.ndarray,
) -> np.ndarray:
    """Evaluate a cumulative SE(3) spline using a precomputed basis.

    Each phase only touches a small, contiguous group of control-pose
    increments. Grouping the same local-support position across all phases
    preserves the product order while replacing thousands of tiny Python /
    NumPy-to-Torch calls with a handful of batched SE(3) operations.
    """
    control_poses = np.asarray(control_poses, dtype=np.float64)
    cumulative = np.asarray(cumulative, dtype=np.float64)
    if control_poses.ndim != 3 or control_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected control_poses [L,4,4], got {control_poses.shape}")
    if cumulative.ndim != 2 or cumulative.shape[1] != len(control_poses) - 1:
        raise ValueError(
            "Expected cumulative basis [T,L-1], got "
            f"{cumulative.shape} for {len(control_poses)} control poses"
        )

    increments = se3_log(se3_mul(se3_inv(control_poses[:-1]), control_poses[1:]))
    if len(cumulative) == 0:
        return np.empty((0, 4, 4), dtype=np.float64)

    tolerance = 1e-14
    leading_one_mask = np.abs(cumulative - 1.0) < tolerance
    leading_ones = np.logical_and.accumulate(leading_one_mask, axis=1).sum(axis=1)
    out = np.array(control_poses[leading_ones], copy=True)

    nonzero_count = np.sum(np.abs(cumulative) > tolerance, axis=1)
    max_active_terms = int(np.max(nonzero_count - leading_ones, initial=0))
    row_indices = np.arange(len(cumulative), dtype=np.int64)
    num_increments = cumulative.shape[1]
    for local_idx in range(max_active_terms):
        increment_indices = leading_ones + local_idx
        valid = increment_indices < num_increments
        if not np.any(valid):
            break
        valid_rows = row_indices[valid]
        valid_increment_indices = increment_indices[valid]
        lambdas = cumulative[valid_rows, valid_increment_indices]
        nonzero = np.abs(lambdas) > tolerance
        if not np.any(nonzero):
            continue
        active_rows = valid_rows[nonzero]
        active_increment_indices = valid_increment_indices[nonzero]
        delta = se3_exp(lambdas[nonzero, None] * increments[active_increment_indices])
        out[active_rows] = se3_mul(out[active_rows], delta)
    return out


def fit_tangent_lsq_se3_spline(
    poses: np.ndarray,
    degree: int = 3,
    control_point_stride: int = 3,
    point_weights: Optional[np.ndarray] = None,
    start_pose: Optional[np.ndarray] = None,
) -> SE3SplineFit:
    """Fit SE(3) control poses by least squares in the start-pose tangent chart.

    The first ``degree`` control poses are fixed to ``start_pose``.  Repeating
    the start pose p times is the endpoint-padding convention for the uniform
    degree-p B-spline used in this project, and makes the curve interpolate the
    start pose at phase ``p``.  If no explicit start pose is supplied, retain
    the historical behavior of using the first target pose.
    """
    poses = np.asarray(poses, dtype=np.float64)
    assert poses.ndim == 3
    assert poses.shape[-2:] == (4, 4)
    assert poses.shape[0] >= 1

    if start_pose is None:
        start_pose = poses[0]
    start_pose = np.asarray(start_pose, dtype=np.float64)
    if start_pose.shape != (4, 4):
        raise ValueError(f"Expected start_pose [4,4], got {start_pose.shape}")

    anchor = start_pose
    tangent_points = se3_log(se3_mul(se3_inv(anchor), poses))
    num_points = len(tangent_points)
    # Phase ``degree`` is the measured observation boundary.  The first action
    # target is one environment step after that boundary, at degree + 1 / R.
    num_segments = max(math.ceil(num_points / control_point_stride), 1)
    num_control_points = max(degree + num_segments, 2 * (degree + 1))
    phase = degree + (np.arange(num_points, dtype=np.float64) + 1.0) / control_point_stride
    design = build_bspline_design_matrix(phase, num_control_points, degree)
    tangent_control_points = np.empty((num_control_points, 6), dtype=np.float64)
    tangent_control_points[:degree] = 0.0
    tangent_control_points[-degree:] = tangent_points[-1]
    free_start = degree
    free_end = num_control_points - degree
    if free_end > free_start:
        residual = (
            tangent_points
            - design[:, :free_start] @ tangent_control_points[:free_start]
            - design[:, free_end:] @ tangent_control_points[free_end:]
        )
        free_design = design[:, free_start:free_end]
        if point_weights is not None:
            weight_scale = np.sqrt(np.asarray(point_weights, dtype=np.float64))[:, None]
            free_design = free_design * weight_scale
            residual = residual * weight_scale
        tangent_control_points[free_start:free_end], *_ = np.linalg.lstsq(
            free_design, residual, rcond=None
        )
    control_poses = se3_mul(anchor, se3_exp(tangent_control_points.astype(np.float64)))
    return SE3SplineFit(
        control_poses=control_poses.astype(np.float64),
        phase=phase.astype(np.float64),
        tangent_control_points=tangent_control_points.astype(np.float64),
        tangent_points=tangent_points.astype(np.float64),
    )


def fit_cumulative_lie_lsq_se3_spline(
    poses: np.ndarray,
    degree: int = 3,
    control_point_stride: int = 3,
    point_weights: Optional[np.ndarray] = None,
    start_pose: Optional[np.ndarray] = None,
    translation_scale_m: float = 1.0,
    rotation_scale_rad: float = 1.0,
    max_nfev: int = 100,
) -> SE3SplineFit:
    """Fit the same cumulative Lie-group B-spline used at execution time.

    The tangent-chart least-squares fit is retained only as an initializer.
    Optimization residuals are right-invariant pose errors
    ``Log(T_target^-1 T_spline)``.  ``point_weights`` multiply the squared
    dimensionless six-dimensional residual per timestep, matching the weighting
    semantics of the former linear fit (implemented via their square roots
    here). Translation and rotation are normalized by their configured physical
    scales before being combined. The defaults encode 1 m == 1 rad.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected poses [T,4,4], got {poses.shape}")
    if len(poses) < 1:
        raise ValueError("poses must be non-empty")
    if degree < 1:
        raise ValueError(f"degree must be positive, got {degree}")
    if control_point_stride < 1:
        raise ValueError("control_point_stride must be positive")
    if start_pose is None:
        start_pose = poses[0]
    start_pose = np.asarray(start_pose, dtype=np.float64)
    if start_pose.shape != (4, 4):
        raise ValueError(f"Expected start_pose [4,4], got {start_pose.shape}")
    translation_scale_m = float(translation_scale_m)
    rotation_scale_rad = float(rotation_scale_rad)
    if not np.isfinite(translation_scale_m) or translation_scale_m <= 0.0:
        raise ValueError("translation_scale_m must be finite and positive")
    if not np.isfinite(rotation_scale_rad) or rotation_scale_rad <= 0.0:
        raise ValueError("rotation_scale_rad must be finite and positive")

    if point_weights is None:
        weights = np.ones(len(poses), dtype=np.float64)
    else:
        weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(poses):
            raise ValueError("point_weights and poses must have the same length")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("point_weights must be finite and non-negative")
    weight_scale = np.sqrt(weights)[:, None]
    component_scale = np.asarray(
        [translation_scale_m] * 3 + [rotation_scale_rad] * 3,
        dtype=np.float64,
    )[None, :]

    initial = fit_tangent_lsq_se3_spline(
        poses=poses,
        degree=degree,
        control_point_stride=control_point_stride,
        point_weights=weights,
        start_pose=start_pose,
    )
    tangent_cp = initial.tangent_control_points.copy()
    num_control_points = len(tangent_cp)
    free_start = min(degree, num_control_points)
    free_end = max(free_start, num_control_points - degree)
    if free_end <= free_start:
        return initial

    anchor = start_pose
    free_shape = tangent_cp[free_start:free_end].shape

    def unpack(free_values):
        candidate = tangent_cp.copy()
        candidate[free_start:free_end] = free_values.reshape(free_shape)
        return se3_mul(anchor, se3_exp(candidate))

    design, cumulative = build_cumulative_bspline_basis(
        phase=initial.phase,
        num_control_points=num_control_points,
        degree=degree,
    )
    inverse_poses = se3_inv(poses)

    def residual(free_values):
        prediction = _eval_se3_cumulative_bspline_from_basis(
            control_poses=unpack(free_values),
            cumulative=cumulative,
        )
        pose_error = se3_log(se3_mul(inverse_poses, prediction))
        return (pose_error / component_scale * weight_scale).reshape(-1)

    # A B-spline sample only depends on its locally supported control poses.
    # Supplying this structure makes finite-difference nonlinear fitting
    # practical for full demonstrations rather than scaling quadratically in
    # the number of control points.
    support = np.abs(design[:, free_start:free_end]) > 1e-14
    jac_sparsity = ss.lil_matrix((len(poses) * 6, np.prod(free_shape)), dtype=np.int8)
    for t_idx, supported_cp in enumerate(support):
        cp_indices = np.flatnonzero(supported_cp)
        for cp_idx in cp_indices:
            jac_sparsity[t_idx * 6 : (t_idx + 1) * 6, cp_idx * 6 : (cp_idx + 1) * 6] = 1

    result = so.least_squares(
        residual,
        tangent_cp[free_start:free_end].reshape(-1),
        jac_sparsity=jac_sparsity.tocsr(),
        max_nfev=int(max_nfev),
        method="trf",
    )
    optimized_control_poses = unpack(result.x)
    optimized_tangent_cp = se3_log(se3_mul(se3_inv(anchor), optimized_control_poses))
    return SE3SplineFit(
        control_poses=optimized_control_poses.astype(np.float64),
        phase=initial.phase.astype(np.float64),
        tangent_control_points=optimized_tangent_cp.astype(np.float64),
        tangent_points=initial.tangent_points.astype(np.float64),
    )


def uniform_phase_for_pose_count(
    num_poses: int,
    num_control_points: int,
    degree: int,
) -> np.ndarray:
    return get_uniform_phase(
        num_points=num_poses,
        num_control_points=num_control_points,
        degree=degree,
    ).astype(np.float64)


def get_execution_phase(
    degree: int,
    n_action_steps: int,
    control_point_stride: int,
    exec_horizon: Optional[int] = None,
) -> np.ndarray:
    """Return the shared pose/gripper phase for dense execution waypoints."""
    degree = int(degree)
    n_action_steps = int(n_action_steps)
    control_point_stride = int(control_point_stride)
    if degree < 1:
        raise ValueError("degree must be positive")
    if n_action_steps < 1:
        raise ValueError("n_action_steps must be positive")
    if control_point_stride < 1:
        raise ValueError("control_point_stride must be positive")
    if exec_horizon is not None:
        expected = int(exec_horizon) * control_point_stride
        if n_action_steps != expected:
            raise ValueError(
                "n_action_steps must equal exec_horizon * "
                f"control_point_stride, got {n_action_steps} != {expected}"
            )
    return degree + (np.arange(n_action_steps, dtype=np.float64) + 1.0) / control_point_stride


def get_zoh_execution_indices(
    degree: int,
    n_action_steps: int,
    control_point_stride: int,
    exec_horizon: Optional[int] = None,
) -> np.ndarray:
    """Return held gripper control-point indices for dense execution steps."""
    degree = int(degree)
    n_action_steps = int(n_action_steps)
    control_point_stride = int(control_point_stride)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if n_action_steps < 1:
        raise ValueError("n_action_steps must be positive")
    if control_point_stride < 1:
        raise ValueError("control_point_stride must be positive")
    if exec_horizon is not None:
        expected = int(exec_horizon) * control_point_stride
        if n_action_steps != expected:
            raise ValueError(
                "n_action_steps must equal exec_horizon * "
                f"control_point_stride, got {n_action_steps} != {expected}"
            )
    return degree + (np.arange(n_action_steps, dtype=np.int64) // control_point_stride)


def evaluate_zoh_control_points(
    control_points: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Evaluate zero-order held control-point values at integer indices."""
    control_points = np.asarray(control_points, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if control_points.ndim != 2:
        raise ValueError(f"Expected ZOH control points [L,D], got {control_points.shape}")
    if len(control_points) < 1:
        raise ValueError("At least one ZOH control point is required")
    if np.any(indices < 0) or np.any(indices >= len(control_points)):
        raise ValueError(
            "ZOH index lies outside the control-point domain: "
            f"[{indices.min()}, {indices.max()}] vs [0, {len(control_points) - 1}]"
        )
    return control_points[indices]


def _binary_intent_from_target(
    target_gripper: np.ndarray,
    intent_hysteresis: float,
) -> np.ndarray:
    target_gripper = np.clip(target_gripper, -1.0, 1.0)
    intent = np.empty_like(target_gripper)
    state = np.where(target_gripper[0] >= 0.0, 1.0, -1.0)
    for t_idx, value in enumerate(target_gripper):
        state = np.where(value >= intent_hysteresis, 1.0, state)
        state = np.where(value <= -intent_hysteresis, -1.0, state)
        intent[t_idx] = state
    return intent


def build_causal_zoh_gripper_control_points(
    target_gripper: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    control_point_stride: int,
    intent_hysteresis: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build target-derived causal zero-order-held gripper control points."""
    target_gripper = np.asarray(target_gripper, dtype=np.float64)
    if target_gripper.ndim == 1:
        target_gripper = target_gripper[:, None]
    if target_gripper.ndim != 2 or len(target_gripper) < 1:
        raise ValueError("target_gripper must have shape [T,D] with T >= 1")
    if not np.all(np.isfinite(target_gripper)):
        raise ValueError("Gripper values must be finite")
    degree = int(degree)
    control_point_stride = int(control_point_stride)
    num_control_points = int(num_control_points)
    intent_hysteresis = float(intent_hysteresis)
    if degree < 0 or control_point_stride < 1:
        raise ValueError("degree must be non-negative and stride positive")
    if num_control_points <= degree:
        raise ValueError("num_control_points must exceed degree")
    if not 0.0 <= intent_hysteresis < 1.0:
        raise ValueError("intent_hysteresis must lie in [0,1)")

    intent = _binary_intent_from_target(target_gripper, intent_hysteresis)
    control_points = np.repeat(intent[0:1], num_control_points, axis=0)
    for cp_idx in range(degree, num_control_points):
        segment_start = (cp_idx - degree) * control_point_stride
        intent_idx = min(segment_start, len(intent) - 1)
        control_points[cp_idx] = intent[intent_idx]
    return control_points, intent


def se3_pose_errors(reference: np.ndarray, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    assert reference.shape == prediction.shape
    assert reference.shape[-2:] == (4, 4)

    trans_l2 = np.linalg.norm(
        reference[..., :3, 3] - prediction[..., :3, 3],
        axis=-1,
    )
    rel_rot = np.swapaxes(reference[..., :3, :3], -1, -2) @ prediction[..., :3, :3]
    rot_angle = np.linalg.norm(so3_log(rel_rot), axis=-1)
    return trans_l2, rot_angle


def summarize_se3_pose_errors(reference: np.ndarray, prediction: np.ndarray) -> dict:
    trans_l2, rot_angle = se3_pose_errors(reference, prediction)
    return {
        "all_step_trans_l2_mean": float(np.mean(trans_l2)),
        "all_step_trans_l2_max": float(np.max(trans_l2)),
        "all_step_rot_angle_deg_mean": float(np.rad2deg(np.mean(rot_angle))),
        "all_step_rot_angle_deg_max": float(np.rad2deg(np.max(rot_angle))),
        "endpoint_trans_l2": float(trans_l2[-1]),
        "endpoint_rot_deg": float(np.rad2deg(rot_angle[-1])),
    }


def pad_terminal_hold(sequence: np.ndarray, control_point_stride: int) -> np.ndarray:
    """Repeat the final item so the sequence length is a stride multiple."""
    sequence = np.asarray(sequence)
    control_point_stride = int(control_point_stride)
    if sequence.ndim < 1 or len(sequence) < 1:
        raise ValueError("sequence must be non-empty")
    if control_point_stride < 1:
        raise ValueError("control_point_stride must be positive")
    padding_steps = (-len(sequence)) % control_point_stride
    if padding_steps == 0:
        return sequence
    return np.concatenate(
        [sequence, np.repeat(sequence[-1:], padding_steps, axis=0)],
        axis=0,
    )


def build_bspline_se3_training_episode(
    poses: np.ndarray,
    gripper: np.ndarray,
    observations,
    anchor_poses: Optional[np.ndarray] = None,
    observation_key: str = "obs",
    degree: int = 3,
    prefix_cp_steps: int = None,
    pred_horizon: int = 8,
    control_point_stride: int = 2,
    sample_stride: int = 1,
    n_obs_steps: int = 2,
    n_action_steps: int = 1,
    n_infer_steps: Optional[int] = None,
    exec_horizon: Optional[int] = None,
    point_weights: Optional[np.ndarray] = None,
    fit_translation_scale_m: float = 1.0,
    fit_rotation_scale_rad: float = 1.0,
    terminal_point_weight: float = 1.0,
    pad_terminal: bool = False,
    binarize_gripper: bool = True,
    gripper_representation: str = "waypoint",
    gripper_spline_degree: int = 1,
    gripper_intent_hysteresis: float = 0.0,
    include_gripper: bool = True,
):
    """Build pose-control samples and representation-aware gripper targets."""
    poses = np.asarray(poses, dtype=np.float64)
    include_gripper = bool(include_gripper)
    if include_gripper:
        gripper = np.asarray(gripper, dtype=np.float64)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
    else:
        gripper = None
    if anchor_poses is not None:
        anchor_poses = np.asarray(anchor_poses, dtype=np.float64)
    assert poses.ndim == 3
    assert poses.shape[-2:] == (4, 4)
    if include_gripper:
        assert gripper.ndim == 2
        assert len(poses) == len(gripper)
    assert len(poses) >= 1
    if anchor_poses is not None:
        assert anchor_poses.ndim == 3
        assert anchor_poses.shape[-2:] == (4, 4)
        assert len(anchor_poses) == len(poses)
    gripper_representation = str(gripper_representation)
    if gripper_representation not in ("waypoint", "zoh_control_point"):
        raise ValueError(
            "gripper_representation must be 'waypoint', "
            "or 'zoh_control_point', got "
            f"{gripper_representation!r}"
        )
    gripper_spline_degree = int(gripper_spline_degree)
    if gripper_representation == "zoh_control_point":
        if gripper_spline_degree < 0:
            raise ValueError(
                "zoh_control_point gripper representation requires non-negative degree offset"
            )

    if prefix_cp_steps is None:
        prefix_cp_steps = degree
    assert degree >= 1
    assert prefix_cp_steps >= 1
    assert pred_horizon >= 1
    assert control_point_stride >= 1
    assert sample_stride >= 1
    assert n_obs_steps >= 1
    n_action_steps = int(n_action_steps)
    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be positive, got {n_action_steps}.")
    if n_infer_steps is None:
        n_infer_steps = n_action_steps
    n_infer_steps = int(n_infer_steps)
    if not 1 <= n_infer_steps <= n_action_steps:
        raise ValueError(
            "n_infer_steps must satisfy 1 <= n_infer_steps <= "
            f"n_action_steps, got {n_infer_steps} and {n_action_steps}."
        )
    n_latency_steps = n_action_steps - n_infer_steps

    if isinstance(observations, dict):
        observation_map = {key: np.asarray(value) for key, value in observations.items()}
    else:
        observation_map = {observation_key: np.asarray(observations)}

    for key, value in observation_map.items():
        assert value.ndim >= 1, key
        assert len(value) == len(poses), key

    if exec_horizon is not None:
        expected_action_steps = int(exec_horizon) * int(control_point_stride)
        if int(n_action_steps) != expected_action_steps:
            raise ValueError(
                "Exact timestep/phase alignment requires n_action_steps == "
                "exec_horizon * control_point_stride, got "
                f"{n_action_steps} != {exec_horizon} * {control_point_stride}."
            )

    padding_steps = (-len(poses)) % control_point_stride if pad_terminal else 0
    if padding_steps:
        poses = pad_terminal_hold(poses, control_point_stride)
        if include_gripper:
            gripper = pad_terminal_hold(gripper, control_point_stride)
        if anchor_poses is not None:
            anchor_poses = pad_terminal_hold(anchor_poses, control_point_stride)
        observation_map = {
            key: pad_terminal_hold(value, control_point_stride)
            for key, value in observation_map.items()
        }
        if point_weights is not None:
            point_weights = np.concatenate(
                [
                    np.asarray(point_weights, dtype=np.float64),
                    np.full(padding_steps, float(terminal_point_weight)),
                ]
            )
    num_steps = len(poses)

    # Phase 2: fit the whole demo into cumulative Lie B-spline control poses.
    pose_fit = fit_cumulative_lie_lsq_se3_spline(
        poses=poses,
        degree=degree,
        control_point_stride=control_point_stride,
        point_weights=point_weights,
        start_pose=(anchor_poses[0] if anchor_poses is not None else poses[0]),
        translation_scale_m=fit_translation_scale_m,
        rotation_scale_rad=fit_rotation_scale_rad,
    )
    num_control_points = len(pose_fit.control_poses)
    gripper_control_points = None
    gripper_intent = None
    if include_gripper and gripper_representation == "zoh_control_point":
        gripper_control_points, gripper_intent = build_causal_zoh_gripper_control_points(
            target_gripper=gripper,
            num_control_points=num_control_points,
            degree=gripper_spline_degree,
            control_point_stride=control_point_stride,
            intent_hysteresis=gripper_intent_hysteresis,
        )
    window_size = prefix_cp_steps + pred_horizon
    if num_control_points < window_size:
        if (not pad_terminal) or num_control_points < prefix_cp_steps:
            return None

    max_window_start_with_observation = (num_steps - 1) // control_point_stride
    max_window_start = min(
        num_control_points - window_size,
        max_window_start_with_observation,
    )
    if pad_terminal:
        # A terminal window keeps a complete prefix and at least one fitted
        # future control pose. Missing control poses at the end repeat the final
        # fitted control pose.
        max_window_start = min(
            num_control_points - prefix_cp_steps - 1,
            max_window_start_with_observation,
        )
    if max_window_start < 0:
        return None
    # Phase 3: slide control-point windows to create training samples.
    window_starts = np.arange(0, max_window_start + 1, sample_stride, dtype=np.int64)
    if pad_terminal and len(window_starts) > 0 and window_starts[-1] != max_window_start:
        window_starts = np.concatenate(
            [window_starts, np.asarray([max_window_start], dtype=np.int64)],
            axis=0,
        )
    anchor_phase = window_starts + degree
    # A control-point window starting at i predicts actions from the plan
    # boundary after i * R actions. Early inference observes the active plan
    # n_latency_steps before that boundary. At episode start there is no older
    # observation, so the inference boundary is padded with timestep zero.
    fit_anchor_t_idx = window_starts * int(control_point_stride)
    if np.any(fit_anchor_t_idx >= num_steps):
        raise RuntimeError(
            "B-spline SE(3) window generation produced an observation anchor "
            "outside the fitted sub-episode."
        )
    anchor_t_idx = np.maximum(
        fit_anchor_t_idx - n_latency_steps,
        0,
    ).astype(np.int64)

    trajectory = []
    condition_data = []
    condition_mask = []
    pose_waypoint = []
    gripper_control_point = []
    gripper_waypoint = []
    waypoint_phase = []
    gripper_waypoint_phase = []
    waypoint_t_idx = []
    anchor_pose = []
    anchor_grip = []
    observation_sequences = {key: [] for key in observation_map.keys()}

    execution_phase = get_execution_phase(
        degree=degree,
        n_action_steps=n_action_steps,
        control_point_stride=control_point_stride,
        exec_horizon=exec_horizon,
    )
    gripper_execution_indices = None
    if include_gripper and gripper_representation == "zoh_control_point":
        gripper_execution_indices = get_zoh_execution_indices(
            degree=gripper_spline_degree,
            n_action_steps=n_action_steps,
            control_point_stride=control_point_stride,
            exec_horizon=exec_horizon,
        )
        if gripper_execution_indices[-1] > window_size - 1:
            raise ValueError(
                "Gripper execution index exceeds the control-point window: "
                f"index={gripper_execution_indices[-1]}, "
                f"window_size={window_size}"
            )

    for i, boundary_t_idx, t_idx in zip(window_starts, fit_anchor_t_idx, anchor_t_idx):
        pose_window = pose_fit.control_poses[i : i + window_size]
        if len(pose_window) < window_size:
            pad = np.repeat(
                pose_fit.control_poses[-1:],
                repeats=window_size - len(pose_window),
                axis=0,
            )
            pose_window = np.concatenate([pose_window, pad], axis=0)
        if anchor_poses is not None:
            base_pose = anchor_poses[int(t_idx)]
        elif n_latency_steps > 0:
            base_pose = poses[int(t_idx)]
        else:
            base_pose = pose_window[0]
        # Express control poses in the observation anchor frame.
        local_pose_cp = se3_log(se3_mul(se3_inv(base_pose), pose_window))
        sample_trajectory = local_pose_cp.astype(np.float32)

        sample_condition_data = np.zeros_like(sample_trajectory, dtype=np.float32)
        sample_condition_mask = np.zeros(sample_trajectory.shape, dtype=bool)
        # Keep prefix control points fixed as diffusion conditions.
        sample_condition_data[:prefix_cp_steps] = sample_trajectory[:prefix_cp_steps]
        sample_condition_mask[:prefix_cp_steps] = True

        raw_waypoint_t_idx = int(boundary_t_idx) + np.arange(n_action_steps, dtype=np.int64)
        sample_waypoint_t_idx = np.clip(raw_waypoint_t_idx, 0, num_steps - 1)
        local_waypoint_phase = execution_phase
        sample_waypoint_phase = local_waypoint_phase + float(i)
        if include_gripper:
            local_gripper_waypoint_phase = (
                gripper_execution_indices.astype(np.float64)
                if gripper_representation == "zoh_control_point"
                else execution_phase
            )
            sample_gripper_waypoint_phase = local_gripper_waypoint_phase + float(i)
        waypoint_pose_abs = eval_se3_cumulative_bspline(
            control_poses=pose_window,
            phase=local_waypoint_phase,
            degree=degree,
        )
        sample_pose_waypoint = se3_log(se3_mul(se3_inv(base_pose), waypoint_pose_abs))
        if include_gripper and gripper_representation == "zoh_control_point":
            sample_gripper_cp = gripper_control_points[i : i + window_size]
            if len(sample_gripper_cp) < window_size:
                sample_gripper_cp = np.concatenate(
                    [
                        sample_gripper_cp,
                        np.repeat(
                            gripper_control_points[-1:],
                            window_size - len(sample_gripper_cp),
                            axis=0,
                        ),
                    ],
                    axis=0,
                )
            sample_gripper_waypoint = evaluate_zoh_control_points(
                sample_gripper_cp,
                gripper_execution_indices,
            )
            sample_gripper_waypoint = np.clip(sample_gripper_waypoint, -1.0, 1.0)
            gripper_control_point.append(sample_gripper_cp.astype(np.float32))
        elif include_gripper:
            sample_gripper_waypoint = gripper[sample_waypoint_t_idx]
            if binarize_gripper:
                sample_gripper_waypoint = np.where(sample_gripper_waypoint >= 0.0, 1.0, -1.0)
            else:
                sample_gripper_waypoint = np.clip(sample_gripper_waypoint, -1.0, 1.0)
        if include_gripper:
            sample_gripper_waypoint = sample_gripper_waypoint.astype(np.float32)

        trajectory.append(sample_trajectory)
        condition_data.append(sample_condition_data)
        condition_mask.append(sample_condition_mask)
        pose_waypoint.append(sample_pose_waypoint)
        if include_gripper:
            gripper_waypoint.append(sample_gripper_waypoint)
        waypoint_phase.append(sample_waypoint_phase)
        if include_gripper:
            gripper_waypoint_phase.append(sample_gripper_waypoint_phase)
        waypoint_t_idx.append(sample_waypoint_t_idx)
        anchor_pose.append(base_pose)
        if include_gripper:
            anchor_grip.append(gripper[int(t_idx)])

        for key, value in observation_map.items():
            observation_sequences[key].append(
                get_padded_observation_sequence(
                    observations=value,
                    end_idx=int(t_idx),
                    n_obs_steps=n_obs_steps,
                )
            )

    data = {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "condition_data": np.asarray(condition_data, dtype=np.float32),
        "condition_mask": np.asarray(condition_mask, dtype=bool),
        "pose_waypoint": np.asarray(pose_waypoint, dtype=np.float32),
        "waypoint_phase": np.asarray(waypoint_phase, dtype=np.float32),
        "waypoint_t_idx": np.asarray(waypoint_t_idx, dtype=np.int64),
        "window_starts": window_starts.astype(np.int64),
        "anchor_t_idx": anchor_t_idx.astype(np.int64),
        "fit_anchor_t_idx": fit_anchor_t_idx.astype(np.int64),
        "anchor_phase": anchor_phase.astype(np.float32),
        "anchor_pose": np.asarray(anchor_pose, dtype=np.float32),
        "phase_per_timestep": pose_fit.phase.astype(np.float32),
        "control_poses": pose_fit.control_poses.astype(np.float32),
    }
    if include_gripper:
        data["gripper_waypoint"] = np.asarray(gripper_waypoint, dtype=np.float32)
        data["gripper_waypoint_phase"] = np.asarray(gripper_waypoint_phase, dtype=np.float32)
        data["anchor_grip"] = np.asarray(anchor_grip, dtype=np.float32)
        data["raw_gripper"] = gripper.astype(np.float32)
    if include_gripper and gripper_representation == "zoh_control_point":
        data["gripper_control_point"] = np.asarray(gripper_control_point, dtype=np.float32)
        data["gripper_intent"] = gripper_intent.astype(np.float32)
        data["gripper_control_points"] = gripper_control_points.astype(np.float32)
    for key, value in observation_sequences.items():
        data[key] = np.asarray(value)
    return data
