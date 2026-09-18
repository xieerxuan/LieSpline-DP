import numpy as np


def ema_sequence(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    x = np.asarray(x, dtype=np.float64)
    if alpha <= 0.0 or alpha >= 1.0 or len(x) <= 1:
        return x.copy()
    out = x.copy()
    for idx in range(1, len(out)):
        out[idx] = alpha * out[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def robust_threshold(
    score: np.ndarray,
    mad_multiplier: float,
    min_threshold: float,
) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(score) == 0:
        return float(min_threshold)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    sigma = 1.4826 * mad
    return max(float(min_threshold), median + float(mad_multiplier) * sigma)


def select_event_indices_from_score(
    score: np.ndarray,
    threshold: float,
    cooldown_steps: int = 0,
    warmup_steps: int = 1,
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(score) == 0:
        return np.zeros((0,), dtype=np.int64)
    mask = score > float(threshold)
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps > 0:
        mask[:warmup_steps] = False
    if not np.any(mask):
        return np.zeros((0,), dtype=np.int64)

    candidates = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        stop = idx
        candidates.append(start + int(np.argmax(score[start:stop])))

    cooldown_steps = max(int(cooldown_steps), 0)
    if cooldown_steps <= 0:
        return np.asarray(candidates, dtype=np.int64)

    selected = []
    for idx in candidates:
        if selected and idx - selected[-1] <= cooldown_steps:
            if score[idx] > score[selected[-1]]:
                selected[-1] = idx
            continue
        selected.append(idx)
    return np.asarray(selected, dtype=np.int64)


def cfrc_ext_delta_score(
    wrench_sequence: np.ndarray,
    torque_delta_weight: float = 0.2,
    ema_alpha: float = 0.35,
) -> np.ndarray:
    """Score changes in MuJoCo external body wrenches.

    MuJoCo stores ``cfrc_ext`` as ``[torque_xyz, force_xyz]``. Scores sum the
    force-change norm over selected bodies and add a configurable fraction of
    the corresponding torque-change norm.
    """
    wrench_sequence = np.asarray(wrench_sequence, dtype=np.float64)
    if wrench_sequence.ndim != 3 or wrench_sequence.shape[-1] != 6:
        raise ValueError(
            f"Expected wrench_sequence with shape [T, B, 6], got {wrench_sequence.shape}"
        )
    if len(wrench_sequence) == 0:
        return np.zeros((0,), dtype=np.float64)

    smooth = ema_sequence(wrench_sequence, alpha=ema_alpha)
    delta = np.zeros_like(smooth)
    delta[1:] = smooth[1:] - smooth[:-1]
    torque_delta = np.linalg.norm(delta[..., :3], axis=-1).sum(axis=-1)
    force_delta = np.linalg.norm(delta[..., 3:6], axis=-1).sum(axis=-1)
    return force_delta + float(torque_delta_weight) * torque_delta
