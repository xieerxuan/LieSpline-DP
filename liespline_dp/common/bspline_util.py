import numpy as np
import scipy.interpolate as si


def build_open_uniform_knot_vector(num_control_points: int, degree: int) -> np.ndarray:
    """
    input:
        num_control_points: number of control points (N)
        degree: degree of the B-spline (p)
    output:
        u = [0, 1, 2, ..., N + p]
    """
    assert degree >= 0
    assert num_control_points > degree
    return np.arange(num_control_points + degree + 1, dtype=np.float64)


def build_bspline_design_matrix(
    phase: np.ndarray, num_control_points: int, degree: int
) -> np.ndarray:
    """
    input:
        phase: array of shape (T,) with values in [p, L] (s in [p, L])
        num_control_points: number of control points (L)
        degree: degree of the B-spline (p)
    output:
        design: array of shape (T, L) with design[t, l] = B_l(phase[t])
    """
    phase = np.asarray(phase, dtype=np.float64)
    if not np.all(np.isfinite(phase)):
        raise ValueError("B-spline phase must contain only finite values")
    phase_min = float(degree)
    phase_max = float(num_control_points)
    endpoint_tolerance = 8.0 * np.finfo(np.float32).eps * max(1.0, abs(phase_min), abs(phase_max))
    if np.any(phase < phase_min - endpoint_tolerance) or np.any(
        phase > phase_max + endpoint_tolerance
    ):
        raise ValueError(
            "B-spline phase must lie in "
            f"[{phase_min}, {phase_max}] within endpoint tolerance "
            f"{endpoint_tolerance}, got range "
            f"[{float(np.min(phase))}, {float(np.max(phase))}]"
        )
    phase = np.clip(phase, phase_min, phase_max)
    knots = build_open_uniform_knot_vector(num_control_points=num_control_points, degree=degree)

    design = np.empty((len(phase), num_control_points), dtype=np.float64)
    for i in range(num_control_points):
        coeff = np.zeros((num_control_points,), dtype=np.float64)
        coeff[i] = 1.0
        basis = si.BSpline(knots, coeff, degree, extrapolate=False)
        design[:, i] = basis(phase)
    if not np.all(np.isfinite(design)):
        raise RuntimeError("B-spline basis evaluation produced non-finite values")
    return design


def get_uniform_phase(num_points: int, num_control_points: int, degree: int) -> np.ndarray:
    """
    input:
        num_points: number of points to be fitted (T)
        num_control_points: number of control points (L)
        degree: degree of the B-spline (p)
    output:
        phase: s in [p, L], s = p + (L - p) * (i / (T - 1)), i = 0, 1, ..., T-1
    """
    assert num_points >= 1
    if num_points == 1:
        return np.array([degree], dtype=np.float64)

    points_idx = np.arange(num_points, dtype=np.float64)
    return degree + (num_control_points - degree) * (points_idx / (num_points - 1))


def get_padded_observation_sequence(
    observations: np.ndarray, end_idx: int, n_obs_steps: int
) -> np.ndarray:
    observations = np.asarray(observations)
    assert observations.ndim >= 1
    assert len(observations) >= 1
    assert 0 <= end_idx < len(observations)
    assert n_obs_steps >= 1

    start_idx = max(end_idx - n_obs_steps + 1, 0)
    sequence = observations[start_idx : end_idx + 1]
    if len(sequence) < n_obs_steps:
        pad = np.repeat(sequence[:1], repeats=n_obs_steps - len(sequence), axis=0)
        sequence = np.concatenate([pad, sequence], axis=0)
    return sequence
