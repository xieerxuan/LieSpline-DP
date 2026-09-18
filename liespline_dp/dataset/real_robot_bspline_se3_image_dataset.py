import numpy as np
import zarr

from liespline_dp.codecs.imagecodecs_numcodecs import register_codecs
from liespline_dp.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from liespline_dp.common.real_robot_action_format import (
    normalize_real_robot_action_format,
    real_robot_has_gripper,
)
from liespline_dp.dataset.bspline_se3_image_dataset import (
    BsplineSE3ImageDataset,
)
from liespline_dp.model.common.normalizer import LinearNormalizer

CONTINUOUS_GRIPPER_MODE = "continuous_normalized"
ZOH_GRIPPER_REPRESENTATION = "zoh_control_point"
GRIPPER_LABEL_SOURCE = "gripper_target_q"
GRIPPER_LABEL_CONSTRUCTION = "binary_intent_causal_zoh"

register_codecs()


def _meta_value(meta, key):
    if key not in meta:
        raise RuntimeError(f"Real-robot B-spline dataset is missing meta/{key}.")
    value = np.asarray(meta[key][:])
    if value.size == 1:
        value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return value


def validate_real_robot_bspline_metadata(
    zarr_path,
    *,
    control_frequency_hz,
    gripper_bounds,
    gripper_representation,
    gripper_spline_degree,
    gripper_label_source,
    gripper_label_construction,
    gripper_control_point_horizon,
    start_offset_augmentation_enabled=False,
    num_start_offsets=1,
    action_format=None,
):
    root = zarr.open(str(zarr_path), mode="r")
    meta = root["meta"]
    action_format = normalize_real_robot_action_format(action_format)
    include_gripper = real_robot_has_gripper(action_format)
    gripper_representation = str(gripper_representation)
    augmentation_enabled = bool(start_offset_augmentation_enabled)
    configured_num_offsets = int(num_start_offsets)
    if configured_num_offsets < 1:
        raise ValueError("num_start_offsets must be positive")
    expected_start_offsets = np.arange(
        configured_num_offsets if augmentation_enabled else 1,
        dtype=np.int64,
    )
    if not include_gripper:
        expected_scalars = {
            "action_format": action_format,
            "tcp_target_quat_order": "xyzw",
            "start_offset_augmentation_enabled": int(augmentation_enabled),
            "num_start_offsets": len(expected_start_offsets),
        }
    elif gripper_representation == ZOH_GRIPPER_REPRESENTATION:
        expected_scalars = {
            "action_format": action_format,
            "tcp_target_quat_order": "xyzw",
            "gripper_mode": CONTINUOUS_GRIPPER_MODE,
            "gripper_representation": gripper_representation,
            "gripper_spline_degree": int(gripper_spline_degree),
            "gripper_label_source": str(gripper_label_source),
            "gripper_label_construction": str(gripper_label_construction),
            "gripper_control_point_horizon": int(gripper_control_point_horizon),
            "start_offset_augmentation_enabled": int(augmentation_enabled),
            "num_start_offsets": len(expected_start_offsets),
        }
    else:
        raise ValueError(f"gripper_representation must be '{ZOH_GRIPPER_REPRESENTATION}'")
    mismatches = []
    for key, expected in expected_scalars.items():
        actual = _meta_value(meta, key)
        if str(actual) != str(expected):
            mismatches.append(f"{key}: dataset={actual}, task={expected}")

    if not include_gripper or gripper_representation == ZOH_GRIPPER_REPRESENTATION:
        actual_start_offsets = np.asarray(
            _meta_value(meta, "start_offsets"), dtype=np.int64
        ).reshape(-1)
        if not np.array_equal(actual_start_offsets, expected_start_offsets):
            mismatches.append(
                "start_offsets: "
                f"dataset={actual_start_offsets.tolist()}, "
                f"task={expected_start_offsets.tolist()}"
            )
        if "source_start_offset" not in root["data"]:
            mismatches.append("data/source_start_offset is missing")
        else:
            source_start_offsets = np.asarray(
                root["data/source_start_offset"][:], dtype=np.int64
            ).reshape(-1)
            realized_offsets = np.unique(source_start_offsets)
            if not np.array_equal(realized_offsets, expected_start_offsets):
                mismatches.append(
                    "data/source_start_offset values: "
                    f"dataset={realized_offsets.tolist()}, "
                    f"task={expected_start_offsets.tolist()}"
                )
            if "source_episode_index" not in root["data"]:
                mismatches.append("data/source_episode_index is missing")
            else:
                source_episode_indices = np.asarray(
                    root["data/source_episode_index"][:], dtype=np.int64
                ).reshape(-1)
                episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64).reshape(-1)
                if len(source_episode_indices) != len(source_start_offsets):
                    mismatches.append("source provenance arrays have different lengths")
                else:
                    episode_starts = np.insert(episode_ends[:-1], 0, 0)
                    for episode_idx, (start, end) in enumerate(zip(episode_starts, episode_ends)):
                        provenance = np.unique(
                            np.stack(
                                [
                                    source_episode_indices[start:end],
                                    source_start_offsets[start:end],
                                ],
                                axis=-1,
                            ),
                            axis=0,
                        )
                        if provenance.shape != (1, 2):
                            mismatches.append(
                                "source provenance must be constant within "
                                f"converted episode {episode_idx}"
                            )
                            break

    actual_frequency = float(_meta_value(meta, "control_frequency_hz"))
    if not np.isclose(actual_frequency, float(control_frequency_hz), rtol=0.0, atol=1e-9):
        mismatches.append(
            f"control_frequency_hz: dataset={actual_frequency}, task={control_frequency_hz}"
        )

    if include_gripper:
        actual_bounds = np.asarray(_meta_value(meta, "gripper_bounds"), dtype=np.float64).reshape(
            -1
        )
        expected_bounds = np.asarray(gripper_bounds, dtype=np.float64).reshape(-1)
        if (
            actual_bounds.shape != (2,)
            or expected_bounds.shape != (2,)
            or not np.allclose(actual_bounds, expected_bounds, rtol=0.0, atol=1e-9)
        ):
            mismatches.append(
                f"gripper_bounds: dataset={actual_bounds.tolist()}, task={expected_bounds.tolist()}"
            )

    if mismatches:
        raise RuntimeError(
            "Real-robot B-spline dataset metadata does not match the task: " + "; ".join(mismatches)
        )


class RealRobotBsplineSE3ImageDataset(BsplineSE3ImageDataset):
    """Image B-spline dataset built from the native real-robot cache."""

    def __init__(
        self,
        shape_meta,
        zarr_path,
        degree,
        prefix_cp_steps,
        control_point_stride,
        exec_horizon,
        n_action_steps,
        n_latency_steps,
        control_frequency_hz,
        trajectory_horizon=None,
        gripper_bounds=(0.0, 5.0),
        gripper_representation="zoh_control_point",
        gripper_spline_degree=None,
        gripper_label_source=None,
        gripper_label_construction=None,
        gripper_control_point_key=None,
        action_format=None,
        start_offset_augmentation_enabled=False,
        num_start_offsets=1,
        **kwargs,
    ):
        self.control_frequency_hz = float(control_frequency_hz)
        self.action_format = normalize_real_robot_action_format(action_format)
        self.include_gripper = real_robot_has_gripper(self.action_format)
        self.gripper_bounds = tuple(float(x) for x in gripper_bounds)
        if trajectory_horizon is None:
            raise ValueError("trajectory_horizon is required")
        validate_real_robot_bspline_metadata(
            zarr_path,
            control_frequency_hz=self.control_frequency_hz,
            gripper_bounds=self.gripper_bounds,
            gripper_representation=gripper_representation,
            gripper_spline_degree=gripper_spline_degree,
            gripper_label_source=gripper_label_source,
            gripper_label_construction=gripper_label_construction,
            gripper_control_point_horizon=trajectory_horizon,
            action_format=self.action_format,
            start_offset_augmentation_enabled=(start_offset_augmentation_enabled),
            num_start_offsets=num_start_offsets,
        )
        kwargs.pop("split_key", None)
        super().__init__(
            shape_meta=shape_meta,
            zarr_path=zarr_path,
            degree=degree,
            prefix_cp_steps=prefix_cp_steps,
            control_point_stride=control_point_stride,
            exec_horizon=exec_horizon,
            n_action_steps=n_action_steps,
            n_latency_steps=n_latency_steps,
            trajectory_horizon=trajectory_horizon,
            gripper_control_point_key=gripper_control_point_key,
            split_key="source_episode_index",
            validate_anchor_contract=False,
            **kwargs,
        )
        self.normalizer_indices = self.indices.copy()
        train_sources = set(np.asarray(self.source_ids[self.train_mask]).reshape(-1).tolist())
        val_sources = set(np.asarray(self.source_ids[self.val_mask]).reshape(-1).tolist())
        overlap = train_sources.intersection(val_sources)
        if overlap:
            raise RuntimeError(
                f"Real-robot train/validation source episodes overlap: {sorted(overlap)}"
            )

    def get_normalizer(self, mode="limits", **kwargs):
        if len(self.normalizer_indices) == 0:
            raise RuntimeError("Cannot fit a normalizer without real-robot training samples")
        normalizer = LinearNormalizer()
        trajectory = self.replay_buffer[self.trajectory_key][self.normalizer_indices]
        normalizer.fit(
            data={"trajectory": trajectory},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["condition_data"] = normalizer["trajectory"]

        for key in self.lowdim_keys:
            values = self.replay_buffer[key][self.normalizer_indices]
            stat = array_to_stats(values)
            if key == "eef_quat":
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                if key == "gripper_q":
                    lower, upper = self.gripper_bounds
                    stat["min"] = np.full_like(stat["min"], lower, dtype=np.float32)
                    stat["max"] = np.full_like(stat["max"], upper, dtype=np.float32)
                normalizer[key] = get_range_normalizer_from_stat(stat)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer
