import os
import pathlib
import shutil

import click
import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from liespline_dp.codecs.imagecodecs_numcodecs import register_codecs
from liespline_dp.common.bspline_se3_util import build_bspline_se3_training_episode
from liespline_dp.common.real_robot_action_format import (
    normalize_real_robot_action_format,
    real_robot_has_gripper,
)
from liespline_dp.common.replay_buffer import ReplayBuffer
from liespline_dp.common.task_config_util import load_task_config
from liespline_dp.dataset.real_robot_bspline_se3_image_dataset import (
    CONTINUOUS_GRIPPER_MODE,
    GRIPPER_LABEL_CONSTRUCTION,
    GRIPPER_LABEL_SOURCE,
    ZOH_GRIPPER_REPRESENTATION,
)
from liespline_dp.dataset.real_robot_image_dataset import (
    REAL_ROBOT_GRIPPER_LABEL_SOURCE,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
register_codecs()


def _meta_scalar(meta, key):
    if key not in meta:
        raise RuntimeError(f"Source real-robot cache is missing meta/{key}")
    value = np.asarray(meta[key][:]).reshape(-1)
    if len(value) != 1:
        raise RuntimeError(f"Source cache meta/{key} must contain one value")
    value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _meta_scalar_default(meta, key, default):
    if key not in meta:
        return default
    return _meta_scalar(meta, key)


def _episode_slices(episode_ends, max_episodes=None):
    ends = np.asarray(episode_ends, dtype=np.int64)
    if max_episodes is not None:
        ends = ends[: int(max_episodes)]
    starts = np.insert(ends[:-1], 0, 0)
    return list(zip(starts.tolist(), ends.tolist()))


def _get_start_offsets(task_cfg):
    cfg = task_cfg.conversion.get("start_offset_augmentation", {})
    if not bool(cfg.get("enabled", False)):
        return np.asarray([0], dtype=np.int64)

    control_point_stride = int(task_cfg.bspline.control_point_stride)
    num_offsets = int(cfg.get("num_offsets", control_point_stride))
    if num_offsets < 1:
        raise ValueError("start_offset_augmentation.num_offsets must be positive.")
    if num_offsets > control_point_stride:
        raise ValueError(
            "start_offset_augmentation.num_offsets must be <= bspline.control_point_stride."
        )
    return np.arange(num_offsets, dtype=np.int64)


def _fit_point_weight_cfg(task_cfg):
    return task_cfg.conversion.get("fit_point_weight", {})


def _fit_point_weight_enabled(task_cfg):
    cfg = _fit_point_weight_cfg(task_cfg)
    return bool(cfg.get("enabled", False))


def _positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value, name):
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _event_indices_from_runs(mask, min_run_steps):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    min_run_steps = _positive_int(min_run_steps, "min_run_steps")
    true_idx = np.flatnonzero(mask)
    if len(true_idx) == 0:
        return np.empty(0, dtype=np.int64)
    split = np.flatnonzero(np.diff(true_idx) != 1) + 1
    event_indices = []
    for run in np.split(true_idx, split):
        if len(run) < min_run_steps:
            continue
        # A true entry at diff index i marks the transition i -> i + 1.
        event_indices.extend(range(int(run[0]), int(run[-1]) + 2))
    return np.asarray(event_indices, dtype=np.int64)


def _build_actual_gripper_jump_event_indices(actual_gripper_q, cfg):
    gripper_cfg = cfg.get("gripper", {})
    if not bool(gripper_cfg.get("enabled", True)):
        return np.empty(0, dtype=np.int64)
    if actual_gripper_q is None:
        return np.empty(0, dtype=np.int64)
    actual = np.asarray(actual_gripper_q, dtype=np.float64).reshape(-1)
    if len(actual) < 2:
        return np.empty(0, dtype=np.int64)
    min_delta_q = float(
        gripper_cfg.get(
            "min_delta_q",
            cfg.get("gripper_min_delta_q", 0.03),
        )
    )
    if min_delta_q < 0.0 or not np.isfinite(min_delta_q):
        raise ValueError("fit_point_weight.gripper.min_delta_q must be finite and non-negative")
    min_run_steps = int(
        gripper_cfg.get(
            "min_run_steps",
            cfg.get("gripper_min_run_steps", 1),
        )
    )
    delta = np.diff(actual)
    return _event_indices_from_runs(
        np.isfinite(delta) & (np.abs(delta) >= min_delta_q),
        min_run_steps,
    )


def _resolve_tactile_fz_sequence(source, start, end, cfg):
    tactile_cfg = cfg.get("tactile", {})
    key = str(
        tactile_cfg.get(
            "key",
            cfg.get("tactile_key", "tactile_active_total_raw"),
        )
    )
    candidate_keys = [key]
    for fallback in (
        "tactile_active_total_raw",
        "tactile_resultants_raw",
        "tactile_resultant_raw",
        "tactile_taxels_mean_raw",
    ):
        if fallback not in candidate_keys:
            candidate_keys.append(fallback)
    for candidate in candidate_keys:
        if candidate not in source.keys():
            continue
        values = np.asarray(source[candidate][start:end], dtype=np.float64)
        if values.ndim == 1:
            return values.reshape(-1)
        axis = int(
            tactile_cfg.get(
                "fz_axis",
                cfg.get("tactile_fz_axis", values.shape[-1] - 1),
            )
        )
        if axis < 0:
            axis += values.shape[-1]
        if axis < 0 or axis >= values.shape[-1]:
            raise ValueError(
                f"fit_point_weight.tactile.fz_axis={axis} is out of bounds "
                f"for {candidate} shape {values.shape}"
            )
        fz = values[..., axis]
        if fz.ndim == 1:
            return fz
        return np.nansum(fz, axis=tuple(range(1, fz.ndim)))
    return None


def _build_tactile_fz_jump_event_indices(tactile_fz, cfg):
    tactile_cfg = cfg.get("tactile", {})
    if not bool(tactile_cfg.get("enabled", True)):
        return np.empty(0, dtype=np.int64)
    if tactile_fz is None:
        return np.empty(0, dtype=np.int64)
    fz = np.asarray(tactile_fz, dtype=np.float64).reshape(-1)
    if len(fz) < 2:
        return np.empty(0, dtype=np.int64)
    delta = np.diff(fz)
    finite_delta = delta[np.isfinite(delta)]
    if len(finite_delta) == 0:
        return np.empty(0, dtype=np.int64)
    positive_delta = finite_delta[finite_delta > 0.0]
    min_delta_raw = float(
        tactile_cfg.get(
            "min_delta_raw",
            cfg.get("tactile_fz_min_delta_raw", 30.0),
        )
    )
    robust_multiplier = float(
        tactile_cfg.get(
            "robust_multiplier",
            cfg.get("tactile_fz_robust_multiplier", 6.0),
        )
    )
    if min_delta_raw < 0.0 or not np.isfinite(min_delta_raw):
        raise ValueError("fit_point_weight.tactile.min_delta_raw must be finite and non-negative")
    if robust_multiplier < 0.0 or not np.isfinite(robust_multiplier):
        raise ValueError(
            "fit_point_weight.tactile.robust_multiplier must be finite and non-negative"
        )
    if len(positive_delta) > 0:
        center = float(np.median(positive_delta))
        mad = float(np.median(np.abs(positive_delta - center)))
        robust_threshold = center + robust_multiplier * 1.4826 * mad
    else:
        robust_threshold = 0.0
    threshold = max(min_delta_raw, robust_threshold)
    min_run_steps = int(
        tactile_cfg.get(
            "min_run_steps",
            cfg.get("tactile_fz_min_run_steps", 1),
        )
    )
    return _event_indices_from_runs(
        np.isfinite(delta) & (delta >= threshold),
        min_run_steps,
    )


def _apply_event_weights(
    weights,
    event_indices,
    *,
    proximity_weight,
    event_weight,
    proximity_neighborhood,
    event_neighborhood,
):
    event_indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
    event_indices = event_indices[(event_indices >= 0) & (event_indices < len(weights))]
    if len(event_indices) == 0:
        return 0
    for idx in np.unique(event_indices):
        lo = max(0, int(idx) - proximity_neighborhood)
        hi = min(len(weights), int(idx) + proximity_neighborhood + 1)
        weights[lo:hi] = np.maximum(weights[lo:hi], proximity_weight)
    for idx in np.unique(event_indices):
        lo = max(0, int(idx) - event_neighborhood)
        hi = min(len(weights), int(idx) + event_neighborhood + 1)
        weights[lo:hi] = np.maximum(weights[lo:hi], event_weight)
    return len(np.unique(event_indices))


def _build_real_robot_fit_point_weights(
    task_cfg,
    source,
    start,
    end,
    actual_gripper_q,
):
    cfg = _fit_point_weight_cfg(task_cfg)
    if not _fit_point_weight_enabled(task_cfg):
        return None, {
            "gripper_events": 0,
            "tactile_events": 0,
            "weighted_steps": 0,
        }
    length = int(end) - int(start)
    base_weight = float(cfg.get("base_weight", 1.0))
    proximity_weight = float(cfg.get("proximity_weight", 2.0))
    event_weight = float(
        cfg.get(
            "event_weight",
            cfg.get("contact_weight", 8.0),
        )
    )
    proximity_neighborhood = _nonnegative_int(
        cfg.get("proximity_neighborhood", cfg.get("neighborhood", 2)),
        "fit_point_weight.proximity_neighborhood",
    )
    event_neighborhood = _nonnegative_int(
        cfg.get("event_neighborhood", cfg.get("contact_neighborhood", 1)),
        "fit_point_weight.event_neighborhood",
    )
    if base_weight <= 0.0 or not np.isfinite(base_weight):
        raise ValueError("fit_point_weight.base_weight must be finite and positive")
    if proximity_weight < base_weight or not np.isfinite(proximity_weight):
        raise ValueError("fit_point_weight.proximity_weight must be finite and >= base_weight")
    if event_weight < proximity_weight or not np.isfinite(event_weight):
        raise ValueError("fit_point_weight.event_weight must be finite and >= proximity_weight")

    weights = np.full(length, base_weight, dtype=np.float64)
    gripper_events = _build_actual_gripper_jump_event_indices(
        actual_gripper_q,
        cfg,
    )
    tactile_events = _build_tactile_fz_jump_event_indices(
        _resolve_tactile_fz_sequence(source, start, end, cfg),
        cfg,
    )
    num_gripper_events = _apply_event_weights(
        weights,
        gripper_events,
        proximity_weight=proximity_weight,
        event_weight=event_weight,
        proximity_neighborhood=proximity_neighborhood,
        event_neighborhood=event_neighborhood,
    )
    num_tactile_events = _apply_event_weights(
        weights,
        tactile_events,
        proximity_weight=proximity_weight,
        event_weight=event_weight,
        proximity_neighborhood=proximity_neighborhood,
        event_neighborhood=event_neighborhood,
    )
    return weights.astype(np.float32), {
        "gripper_events": int(num_gripper_events),
        "tactile_events": int(num_tactile_events),
        "weighted_steps": int(np.count_nonzero(weights > base_weight)),
    }


def _conversion_metadata(task_cfg, source, stats):
    n_action_steps = int(task_cfg.bspline.n_action_steps)
    n_infer_steps = int(task_cfg.n_infer_steps)
    n_latency_steps = int(task_cfg.n_latency_steps)
    action_format = normalize_real_robot_action_format(
        task_cfg.get("action_format", "pose_gripper")
    )
    include_gripper = real_robot_has_gripper(action_format)
    metadata = {
        "source_fingerprint": [str(_meta_scalar(source.meta, "source_fingerprint"))],
        "action_format": [action_format],
        "tcp_target_quat_order": ["xyzw"],
        "control_frequency_hz": [float(task_cfg.bspline.control_frequency_hz)],
        "bspline_degree": [int(task_cfg.bspline.degree)],
        "prefix_cp_steps": [int(task_cfg.bspline.prefix_cp_steps)],
        "pred_horizon": [int(task_cfg.bspline.pred_horizon)],
        "trajectory_horizon": [
            int(task_cfg.bspline.prefix_cp_steps + task_cfg.bspline.pred_horizon)
        ],
        "control_point_stride": [int(task_cfg.bspline.control_point_stride)],
        "exec_horizon": [int(task_cfg.bspline.exec_horizon)],
        "sample_stride": [int(task_cfg.bspline.sample_stride)],
        "pad_terminal": [bool(task_cfg.bspline.pad_terminal)],
        "n_obs_steps": [int(task_cfg.bspline.n_obs_steps)],
        "n_action_steps": [n_action_steps],
        "n_infer_steps": [n_infer_steps],
        "n_latency_steps": [n_latency_steps],
        "fit_method": [str(task_cfg.conversion.get("fit_method", "cumulative_lie_lsq"))],
        "fit_point_weight_enabled": [int(_fit_point_weight_enabled(task_cfg))],
        "fit_point_weight_gripper_events": [int(stats.get("fit_point_weight_gripper_events", 0))],
        "fit_point_weight_tactile_events": [int(stats.get("fit_point_weight_tactile_events", 0))],
        "fit_point_weight_weighted_steps": [int(stats.get("fit_point_weight_weighted_steps", 0))],
        "fit_translation_scale_m": [float(task_cfg.se3.fit_translation_scale_m)],
        "fit_rotation_scale_rad": [float(task_cfg.se3.fit_rotation_scale_rad)],
        "gripper_dim": [int(task_cfg.se3.get("gripper_dim", 0))],
        "num_robots": [int(task_cfg.se3.num_robots)],
        "observation_keys": np.asarray(list(task_cfg.conversion.obs_keys), dtype="<U32"),
        "start_offset_augmentation_enabled": [int(stats["start_offset_augmentation_enabled"])],
        "start_offsets": np.asarray(stats["start_offsets"], dtype=np.int64),
        "num_start_offsets": [int(stats["num_start_offsets"])],
        "num_source_episodes": [int(stats["num_source_episodes"])],
        "num_source_subepisodes": [int(stats["num_source_subepisodes"])],
        "num_converted_episodes": [int(stats["num_converted_episodes"])],
        "num_skipped_episodes": [int(stats["num_skipped_episodes"])],
        "num_skipped_subepisodes": [int(stats["num_skipped_subepisodes"])],
        "num_samples": [int(stats["num_samples"])],
    }
    if include_gripper:
        metadata.update(
            {
                "gripper_mode": [CONTINUOUS_GRIPPER_MODE],
                "gripper_representation": [ZOH_GRIPPER_REPRESENTATION],
                "gripper_spline_degree": [int(task_cfg.se3.gripper_spline_degree)],
                "gripper_label_source": [GRIPPER_LABEL_SOURCE],
                "gripper_label_construction": [GRIPPER_LABEL_CONSTRUCTION],
                "gripper_control_point_horizon": [
                    int(task_cfg.bspline.prefix_cp_steps + task_cfg.bspline.pred_horizon)
                ],
                "gripper_intent_hysteresis_q": [float(task_cfg.se3.gripper_intent_hysteresis_q)],
                "gripper_bounds": np.asarray(task_cfg.se3.gripper_bounds, dtype=np.float64),
            }
        )
    return metadata


def convert_real_robot_bspline_se3_dataset(
    task_cfg,
    input_path=None,
    output_path=None,
    max_episodes=None,
):
    task_cfg = OmegaConf.create(task_cfg)
    input_path = pathlib.Path(
        input_path if input_path is not None else task_cfg.conversion.input_path
    ).expanduser()
    output_path = pathlib.Path(
        output_path if output_path is not None else task_cfg.conversion.output_path
    ).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Source baseline cache not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output cache already exists: {output_path}")

    action_format = normalize_real_robot_action_format(
        task_cfg.get("action_format", "pose_gripper")
    )
    include_gripper = real_robot_has_gripper(action_format)
    n_action_steps = int(task_cfg.bspline.n_action_steps)
    expected_action_steps = int(
        task_cfg.bspline.exec_horizon * task_cfg.bspline.control_point_stride
    )
    if n_action_steps != expected_action_steps:
        raise ValueError("n_action_steps must equal exec_horizon * control_point_stride")
    n_infer_steps = int(task_cfg.n_infer_steps)
    n_latency_steps = int(task_cfg.n_latency_steps)
    if not 1 <= n_infer_steps <= n_action_steps:
        raise ValueError(
            "n_infer_steps must satisfy 1 <= n_infer_steps <= "
            f"n_action_steps, got n_infer_steps={n_infer_steps}, "
            f"n_action_steps={n_action_steps}"
        )
    derived_latency_steps = n_action_steps - n_infer_steps
    if n_latency_steps != derived_latency_steps:
        raise ValueError(
            "n_latency_steps must equal n_action_steps - n_infer_steps, "
            f"got n_latency_steps={n_latency_steps}, "
            f"n_action_steps={n_action_steps}, "
            f"n_infer_steps={n_infer_steps}, "
            f"derived_latency_steps={derived_latency_steps}"
        )
    if include_gripper and str(task_cfg.se3.gripper_representation) != ZOH_GRIPPER_REPRESENTATION:
        raise ValueError(
            f"Real-robot conversion requires gripper_representation={ZOH_GRIPPER_REPRESENTATION}"
        )
    if include_gripper and int(task_cfg.se3.gripper_spline_degree) < 0:
        raise ValueError("ZOH gripper control points require a non-negative degree offset")
    intent_hysteresis_normalized = 0.0
    if include_gripper:
        lower_q, upper_q = (float(value) for value in task_cfg.se3.gripper_bounds)
        gripper_span_q = upper_q - lower_q
        if gripper_span_q <= 0.0:
            raise ValueError("gripper_bounds must satisfy min < max")
        intent_hysteresis_normalized = (
            2.0 * float(task_cfg.se3.gripper_intent_hysteresis_q) / gripper_span_q
        )

    source = ReplayBuffer.create_from_path(str(input_path), mode="r")
    if str(_meta_scalar(source.meta, "tcp_target_quat_order")) != "xyzw":
        raise RuntimeError("Source baseline cache must declare xyzw quaternions")
    source_action_format = str(_meta_scalar_default(source.meta, "action_format", "pose_gripper"))
    if source_action_format != action_format:
        raise RuntimeError(
            "Source baseline cache action_format must be "
            f"{action_format}, got {source_action_format}"
        )
    if include_gripper:
        source_gripper_label = str(_meta_scalar(source.meta, "gripper_label_source"))
        if source_gripper_label != REAL_ROBOT_GRIPPER_LABEL_SOURCE:
            raise RuntimeError(
                "Source baseline cache gripper label must be "
                f"{REAL_ROBOT_GRIPPER_LABEL_SOURCE}, got {source_gripper_label}"
            )

    adapter = hydra.utils.instantiate(task_cfg.action_pose_adapter)
    obs_keys = list(task_cfg.conversion.obs_keys)
    required_keys = set(
        obs_keys
        + [
            "action",
            "timestamp",
            "source_episode_index",
        ]
    )
    if include_gripper:
        required_keys.update(["gripper_q", "gripper_target_q"])
    missing = sorted(required_keys.difference(source.keys()))
    if missing:
        raise KeyError(f"Source baseline cache is missing data keys: {missing}")

    temp_path = output_path.with_name(f"{output_path.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = ReplayBuffer.create_from_path(str(temp_path), mode="a")

    slices = _episode_slices(source.episode_ends[:], max_episodes)
    start_offsets = _get_start_offsets(task_cfg)
    stats = {
        "start_offset_augmentation_enabled": bool(
            task_cfg.conversion.get("start_offset_augmentation", {}).get("enabled", False)
        ),
        "start_offsets": start_offsets.tolist(),
        "num_start_offsets": len(start_offsets),
        "num_source_episodes": len(slices),
        "num_source_subepisodes": len(slices) * len(start_offsets),
        "num_converted_episodes": 0,
        "num_skipped_episodes": 0,
        "num_skipped_subepisodes": 0,
        "num_samples": 0,
        "fit_point_weight_gripper_events": 0,
        "fit_point_weight_tactile_events": 0,
        "fit_point_weight_weighted_steps": 0,
    }
    try:
        for _, (start, end) in enumerate(
            tqdm(slices, desc="Converting real-robot bspline SE(3) dataset")
        ):
            source_ids = np.unique(source["source_episode_index"][start:end])
            if len(source_ids) != 1:
                raise RuntimeError(
                    "Each baseline cache episode must contain exactly one source_episode_index"
                )
            source_episode_index = int(source_ids[0])
            action = source["action"][start:end].astype(np.float32)
            actual_gripper_q = None
            target_gripper_q = None
            if include_gripper:
                actual_gripper_q = source["gripper_q"][start:end].astype(np.float32)
                target_gripper_q = source["gripper_target_q"][start:end].astype(np.float32)
            observations = {key: source[key][start:end] for key in obs_keys}
            observations["timestamp"] = source["timestamp"][start:end].astype(np.float64)
            target_poses, action_gripper = adapter.action_to_pose_grip(action)
            normalized_target_gripper = None
            if include_gripper:
                normalized_target_gripper = adapter.normalize_gripper(target_gripper_q)
                if not np.allclose(action_gripper, normalized_target_gripper, rtol=0.0, atol=1e-6):
                    raise RuntimeError("Baseline action gripper does not match gripper_target_q")
            point_weights, weight_stats = _build_real_robot_fit_point_weights(
                task_cfg,
                source,
                start,
                end,
                actual_gripper_q,
            )
            stats["fit_point_weight_gripper_events"] += weight_stats["gripper_events"]
            stats["fit_point_weight_tactile_events"] += weight_stats["tactile_events"]
            stats["fit_point_weight_weighted_steps"] += weight_stats["weighted_steps"]
            measured_poses, _ = adapter.obs_to_current_pose_grip(observations)
            source_episode_converted = False
            for start_offset in start_offsets:
                start_offset = int(start_offset)
                if start_offset >= len(target_poses):
                    stats["num_skipped_subepisodes"] += 1
                    continue

                sub_observations = {
                    key: value[start_offset:] for key, value in observations.items()
                }
                sub_point_weights = (
                    point_weights[start_offset:] if point_weights is not None else None
                )

                episode = build_bspline_se3_training_episode(
                    poses=np.asarray(target_poses[start_offset:], dtype=np.float64),
                    gripper=(
                        np.asarray(
                            normalized_target_gripper[start_offset:],
                            dtype=np.float64,
                        )
                        if include_gripper
                        else None
                    ),
                    observations=sub_observations,
                    anchor_poses=np.asarray(measured_poses[start_offset:], dtype=np.float64),
                    degree=int(task_cfg.bspline.degree),
                    prefix_cp_steps=int(task_cfg.bspline.prefix_cp_steps),
                    pred_horizon=int(task_cfg.bspline.pred_horizon),
                    control_point_stride=int(task_cfg.bspline.control_point_stride),
                    sample_stride=int(task_cfg.bspline.sample_stride),
                    n_obs_steps=int(task_cfg.bspline.n_obs_steps),
                    n_action_steps=n_action_steps,
                    n_infer_steps=int(task_cfg.n_infer_steps),
                    exec_horizon=int(task_cfg.bspline.exec_horizon),
                    fit_translation_scale_m=float(task_cfg.se3.fit_translation_scale_m),
                    fit_rotation_scale_rad=float(task_cfg.se3.fit_rotation_scale_rad),
                    point_weights=sub_point_weights,
                    terminal_point_weight=float(
                        _fit_point_weight_cfg(task_cfg).get("base_weight", 1.0)
                    ),
                    pad_terminal=bool(task_cfg.bspline.pad_terminal),
                    binarize_gripper=False,
                    gripper_representation=ZOH_GRIPPER_REPRESENTATION,
                    gripper_spline_degree=int(task_cfg.se3.gripper_spline_degree),
                    gripper_intent_hysteresis=intent_hysteresis_normalized,
                    include_gripper=include_gripper,
                )
                if episode is None:
                    stats["num_skipped_subepisodes"] += 1
                    continue

                num_samples = len(episode["trajectory"])
                sample_keys = [
                    "trajectory",
                    "condition_data",
                    "condition_mask",
                    "pose_waypoint",
                    "waypoint_phase",
                    "waypoint_t_idx",
                    "window_starts",
                    "anchor_t_idx",
                    "fit_anchor_t_idx",
                    "anchor_phase",
                    "anchor_pose",
                ]
                if include_gripper:
                    sample_keys.extend(
                        [
                            "gripper_control_point",
                            "gripper_waypoint",
                            "gripper_waypoint_phase",
                            "anchor_grip",
                        ]
                    )
                payload = {key: episode[key] for key in sample_keys}
                for key in obs_keys:
                    payload[key] = episode[key]
                payload["source_episode_index"] = np.full(
                    num_samples,
                    source_episode_index,
                    dtype=np.int64,
                )
                payload["source_start_offset"] = np.full(
                    num_samples,
                    start_offset,
                    dtype=np.int64,
                )
                payload["source_observation_timestamp"] = episode["timestamp"].astype(np.float64)
                chunks = {}
                compressors = {}
                for key, value in payload.items():
                    if key in ("wrist_rgb", "global_rgb"):
                        chunks[key] = (1,) + value.shape[1:]
                    compressors[key] = "disk"
                destination.add_episode(
                    payload,
                    chunks=chunks,
                    compressors=compressors,
                )
                source_episode_converted = True
                stats["num_converted_episodes"] += 1
                stats["num_samples"] += num_samples
            if not source_episode_converted:
                stats["num_skipped_episodes"] += 1

        if stats["num_converted_episodes"] == 0:
            raise RuntimeError("No real-robot episodes produced B-spline samples")
        destination.update_meta(_conversion_metadata(task_cfg, source, stats))
        temp_path.rename(output_path)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise
    return stats


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "--task-config",
    default="liespline_dp/config/task/real_robot/bspline_se3_image.yaml",
    show_default=True,
)
@click.option("-i", "--input", default=None)
@click.option("-o", "--output", default=None)
@click.option("--max-episodes", default=None, type=int)
@click.option("--overwrite", is_flag=True, default=False)
@click.pass_context
def main(ctx, task_config, input, output, max_episodes, overwrite):
    task_cfg = load_task_config(task_config)
    if ctx.args:
        task_cfg = OmegaConf.merge(
            task_cfg,
            OmegaConf.from_dotlist(list(ctx.args)),
        )
        OmegaConf.resolve(task_cfg)

    output_path = pathlib.Path(
        output if output is not None else task_cfg.conversion.output_path
    ).expanduser()
    if output_path.exists():
        if not overwrite:
            raise click.ClickException(
                f"Output path already exists: {output_path}. Use --overwrite to replace it."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    stats = convert_real_robot_bspline_se3_dataset(
        task_cfg=task_cfg,
        input_path=input,
        output_path=output,
        max_episodes=max_episodes,
    )
    click.echo("Real-robot B-spline SE(3) conversion finished.")
    for key, value in stats.items():
        click.echo(f"{key}: {value}")


if __name__ == "__main__":
    main()
