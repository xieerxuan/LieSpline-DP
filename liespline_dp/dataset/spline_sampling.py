"""Index converted spline windows by their source demonstration."""

import numpy as np


def build_sample_indices(episode_ends: np.ndarray, episode_mask: np.ndarray) -> np.ndarray:
    assert len(episode_ends) == len(episode_mask)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    indices = [
        np.arange(start_idx, end_idx, dtype=np.int64)
        for start_idx, end_idx, enabled in zip(episode_starts, episode_ends, episode_mask)
        if enabled
    ]
    if not indices:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(indices, axis=0)
