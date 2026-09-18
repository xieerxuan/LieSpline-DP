from typing import Optional, Sequence

import numpy as np

from liespline_dp.common.normalize_util import (
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from liespline_dp.common.real_robot_action_format import (
    normalize_real_robot_action_format,
    real_robot_has_gripper,
    real_robot_model_action_dim,
)
from liespline_dp.model.common.normalizer import (
    SingleFieldLinearNormalizer,
)


def _slice_stat(stat, index):
    return {key: np.asarray(value[index], dtype=np.float32) for key, value in stat.items()}


def get_real_robot_action_normalizer(
    stat,
    gripper_bounds: Optional[Sequence[float]] = None,
    action_format=None,
) -> SingleFieldLinearNormalizer:
    """Normalize xyz and continuous gripper q while preserving rotation-6D."""
    action_format = normalize_real_robot_action_format(action_format)
    include_gripper = real_robot_has_gripper(action_format)
    expected_shape = (real_robot_model_action_dim(action_format),)
    if np.asarray(stat["min"]).shape != expected_shape:
        raise ValueError(f"Real-robot baseline action statistics must have shape {expected_shape}")

    position_stat = _slice_stat(stat, slice(0, 3))
    rotation_stat = _slice_stat(stat, slice(3, 9))
    gripper_stat = None
    if include_gripper:
        gripper_stat = _slice_stat(stat, slice(9, 10))
    if include_gripper and gripper_bounds is not None:
        if len(gripper_bounds) != 2:
            raise ValueError("gripper_bounds must contain [min, max]")
        lower, upper = (float(x) for x in gripper_bounds)
        if not lower < upper:
            raise ValueError("gripper_bounds must satisfy min < max")
        gripper_stat["min"] = np.asarray([lower], dtype=np.float32)
        gripper_stat["max"] = np.asarray([upper], dtype=np.float32)

    position = get_range_normalizer_from_stat(position_stat)
    rotation = get_identity_normalizer_from_stat(rotation_stat)
    fields = [position, rotation]
    stats = [position_stat, rotation_stat]
    if include_gripper:
        fields.append(get_range_normalizer_from_stat(gripper_stat))
        stats.append(gripper_stat)

    scale = np.concatenate(
        [field.params_dict["scale"].detach().cpu().numpy() for field in fields]
    ).astype(np.float32)
    offset = np.concatenate(
        [field.params_dict["offset"].detach().cpu().numpy() for field in fields]
    ).astype(np.float32)
    input_stats = {
        key: np.concatenate([field_stat[key] for field_stat in stats]).astype(np.float32)
        for key in ("min", "max", "mean", "std")
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats,
    )
