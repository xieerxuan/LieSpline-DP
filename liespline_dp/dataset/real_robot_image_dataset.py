import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits

from liespline_dp.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from liespline_dp.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from liespline_dp.common.pose_action_codec import PoseActionCodec
from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.common.real_robot_action_format import (
    normalize_real_robot_action_format,
    real_robot_has_gripper,
    real_robot_model_action_dim,
)
from liespline_dp.common.real_robot_normalizer import (
    get_real_robot_action_normalizer,
)
from liespline_dp.common.replay_buffer import ReplayBuffer
from liespline_dp.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from liespline_dp.common.se3_util import quat_to_mat, so3_log
from liespline_dp.dataset.base_dataset import BaseImageDataset
from liespline_dp.dataset.real_robot_episode_reader import (
    discover_episode_dirs,
    load_real_robot_episode,
    resolve_target_pose_source,
)
from liespline_dp.model.common.normalizer import LinearNormalizer

register_codecs()

REAL_ROBOT_GRIPPER_LABEL_SOURCE = "gripper_target_q"
RGB_KEYS = ("wrist_rgb", "global_rgb")
LOWDIM_KEYS = ("eef_pos", "eef_quat", "gripper_q")
TACTILE_DIAGNOSTIC_KEYS = ("tactile_active_total_raw",)


def _shape_meta_to_dict(shape_meta) -> dict:
    result = {"obs": {}, "action": {"shape": list(shape_meta["action"]["shape"])}}
    for key, attr in shape_meta["obs"].items():
        result["obs"][key] = {
            "shape": list(attr["shape"]),
            "type": attr.get("type", "low_dim"),
        }
    return result


def _cache_fingerprint(
    episode_dirs,
    shape_meta,
    active_only: bool,
    min_episode_steps: int,
    target_quat_order: Optional[str],
    action_format,
) -> str:
    action_format = normalize_real_robot_action_format(action_format)
    sources = []
    for episode_dir in episode_dirs:
        steps_path = episode_dir / "steps.npz"
        stat = steps_path.stat()
        with np.load(steps_path, allow_pickle=False) as steps:
            target_source = resolve_target_pose_source(
                steps,
                episode_dir,
                target_quat_order=target_quat_order,
            )
        sources.append(
            {
                "name": episode_dir.name,
                "steps_size": stat.st_size,
                "steps_mtime_ns": stat.st_mtime_ns,
                "raw_target_source_format": target_source.raw_format,
                "tcp_target_quat_order": target_source.target_quat_order,
            }
        )
    payload = {
        "action_format": action_format,
        "shape_meta": _shape_meta_to_dict(shape_meta),
        "active_only": bool(active_only),
        "min_episode_steps": int(min_episode_steps),
        "target_quat_order": target_quat_order,
        "tactile_diagnostic_keys": TACTILE_DIAGNOSTIC_KEYS,
        "sources": sources,
    }
    if real_robot_has_gripper(action_format):
        payload["gripper_label_source"] = REAL_ROBOT_GRIPPER_LABEL_SOURCE
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_rgb(path: Path, output_hw: Tuple[int, int]) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode RGB image: {path}")
    output_h, output_w = output_hw
    if image_bgr.shape[:2] != (output_h, output_w):
        interpolation = (
            cv2.INTER_AREA
            if output_h <= image_bgr.shape[0] and output_w <= image_bgr.shape[1]
            else cv2.INTER_LINEAR
        )
        image_bgr = cv2.resize(
            image_bgr,
            (output_w, output_h),
            interpolation=interpolation,
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _rgb_output_shapes(shape_meta) -> Dict[str, Tuple[int, int]]:
    result = {}
    for key in RGB_KEYS:
        if key not in shape_meta["obs"]:
            raise KeyError(f"shape_meta.obs is missing required RGB key {key}")
        shape = tuple(shape_meta["obs"][key]["shape"])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"{key} must have CHW shape [3,H,W], got {shape}")
        result[key] = (int(shape[1]), int(shape[2]))
    return result


def build_real_robot_baseline_cache(
    dataset_root,
    cache_path,
    shape_meta,
    *,
    action_format=None,
    active_only: bool = True,
    min_episode_steps: int = 2,
    target_quat_order: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """Build a native real-robot cache directly from episode directories."""
    action_format = normalize_real_robot_action_format(action_format)
    dataset_root = Path(dataset_root).expanduser()
    cache_path = Path(cache_path).expanduser()
    episode_dirs = discover_episode_dirs(dataset_root)
    fingerprint = _cache_fingerprint(
        episode_dirs,
        shape_meta,
        active_only,
        min_episode_steps,
        target_quat_order,
        action_format,
    )
    if cache_path.exists():
        if not overwrite:
            existing = ReplayBuffer.create_from_path(str(cache_path), mode="r")
            cached = str(existing.meta["source_fingerprint"][0])
            if cached == fingerprint:
                result = {
                    "cache_path": str(cache_path),
                    "fingerprint": fingerprint,
                    "num_episodes": existing.n_episodes,
                    "num_steps": int(existing.n_steps),
                    "reused": True,
                }
                for key in (
                    "target_actual_orientation_error_median_deg",
                    "target_actual_orientation_error_p99_deg",
                ):
                    if key in existing.meta:
                        result[key] = float(existing.meta[key][0])
                return result
            raise RuntimeError(
                f"Stale real-robot cache at {cache_path}. Rebuild it with overwrite=True."
            )
        shutil.rmtree(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    buffer = ReplayBuffer.create_from_path(str(temp_path), mode="a")
    rgb_shapes = _rgb_output_shapes(shape_meta)
    image_compressor = Jpeg2k(level=50)
    episode_names = []
    warning_counts = []
    orientation_errors_deg = []
    codec = PoseActionCodec(action_format=action_format)

    cv2.setNumThreads(1)
    try:
        with threadpool_limits(1):
            for episode_index, episode_dir in enumerate(episode_dirs):
                episode = load_real_robot_episode(
                    episode_dir,
                    active_only=active_only,
                    min_episode_steps=min_episode_steps,
                    codec=codec,
                    target_quat_order=target_quat_order,
                )
                wrist_rgb = np.stack(
                    [_read_rgb(path, rgb_shapes["wrist_rgb"]) for path in episode.wrist_rgb_paths]
                )
                global_rgb = np.stack(
                    [_read_rgb(path, rgb_shapes["global_rgb"]) for path in episode.global_rgb_paths]
                )
                _, target_quaternion, _ = codec.model_action_to_target(episode.action)
                relative_rotation = np.swapaxes(
                    quat_to_mat(episode.eef_quat), -1, -2
                ) @ quat_to_mat(target_quaternion)
                orientation_errors_deg.append(
                    np.rad2deg(np.linalg.norm(so3_log(relative_rotation), axis=-1))
                )
                payload = {
                    "wrist_rgb": wrist_rgb,
                    "global_rgb": global_rgb,
                    "eef_pos": episode.eef_pos,
                    "eef_quat": episode.eef_quat,
                    "action": episode.action,
                    "timestamp": episode.timestamp,
                    "source_frame_index": episode.source_frame_index,
                    "source_episode_index": np.full(
                        episode.length,
                        episode_index,
                        dtype=np.int64,
                    ),
                }
                if codec.include_gripper:
                    payload["gripper_q"] = episode.gripper_q
                    payload["gripper_target_q"] = episode.gripper_target_q
                if episode.tactile_active_total_raw is not None:
                    payload["tactile_active_total_raw"] = episode.tactile_active_total_raw
                chunks = {
                    "wrist_rgb": (1,) + wrist_rgb.shape[1:],
                    "global_rgb": (1,) + global_rgb.shape[1:],
                }
                compressors = {key: "disk" for key in payload}
                compressors.update({key: image_compressor for key in RGB_KEYS})
                buffer.add_episode(
                    payload,
                    chunks=chunks,
                    compressors=compressors,
                )
                episode_names.append(episode.episode_name)
                warning_counts.append(episode.warning_count)

        max_name_len = max(len(name) for name in episode_names)
        all_orientation_errors = np.concatenate(orientation_errors_deg)
        orientation_error_median_deg = float(np.median(all_orientation_errors))
        orientation_error_p99_deg = float(np.percentile(all_orientation_errors, 99))
        metadata = {
            "action_format": np.asarray([action_format], dtype="<U16"),
            "source_fingerprint": np.asarray([fingerprint], dtype="<U64"),
            "episode_names": np.asarray(episode_names, dtype=f"<U{max_name_len}"),
            "warning_counts": np.asarray(warning_counts, dtype=np.int64),
            "tcp_target_quat_order": np.asarray(["xyzw"], dtype="<U4"),
            "tactile_diagnostic_keys": np.asarray(
                [",".join(TACTILE_DIAGNOSTIC_KEYS)], dtype="<U96"
            ),
            "target_actual_orientation_error_median_deg": np.asarray(
                [orientation_error_median_deg], dtype=np.float64
            ),
            "target_actual_orientation_error_p99_deg": np.asarray(
                [orientation_error_p99_deg], dtype=np.float64
            ),
        }
        if codec.include_gripper:
            metadata["gripper_label_source"] = np.asarray(
                [REAL_ROBOT_GRIPPER_LABEL_SOURCE], dtype="<U32"
            )
        buffer.update_meta(metadata)
        num_steps = int(buffer.n_steps)
        temp_path.rename(cache_path)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise

    return {
        "cache_path": str(cache_path),
        "fingerprint": fingerprint,
        "num_episodes": len(episode_names),
        "num_steps": num_steps,
        "reused": False,
        "target_actual_orientation_error_median_deg": (orientation_error_median_deg),
        "target_actual_orientation_error_p99_deg": (orientation_error_p99_deg),
    }


class RealRobotImageDataset(BaseImageDataset):
    """Image observations and target actions from real-robot demonstrations."""

    def __init__(
        self,
        shape_meta: Mapping,
        dataset_root,
        cache_path,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        n_latency_steps: int = 0,
        active_only: bool = True,
        min_episode_steps: int = 2,
        rebuild_cache: bool = False,
        target_quat_order: Optional[str] = None,
        gripper_bounds: Optional[Sequence[float]] = None,
        action_format=None,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        self.action_format = normalize_real_robot_action_format(action_format)
        self.shape_meta = shape_meta
        self.dataset_root = str(Path(dataset_root).expanduser())
        self.cache_path = str(Path(cache_path).expanduser())
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = int(n_latency_steps)
        if self.n_latency_steps < 0:
            raise ValueError("n_latency_steps must be non-negative")
        self.gripper_bounds = gripper_bounds
        expected_action_shape = (real_robot_model_action_dim(self.action_format),)
        if tuple(shape_meta["action"]["shape"]) != expected_action_shape:
            raise ValueError(
                f"Real-robot baseline action shape must be {list(expected_action_shape)}"
            )

        lock_path = self.cache_path + ".lock"
        with FileLock(lock_path):
            build_real_robot_baseline_cache(
                dataset_root=self.dataset_root,
                cache_path=self.cache_path,
                shape_meta=shape_meta,
                action_format=self.action_format,
                active_only=active_only,
                min_episode_steps=min_episode_steps,
                target_quat_order=target_quat_order,
                overwrite=rebuild_cache,
            )
        self.replay_buffer = ReplayBuffer.create_from_path(self.cache_path, mode="r")

        self.rgb_keys = [
            key for key, attr in shape_meta["obs"].items() if attr.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            key
            for key, attr in shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "low_dim"
        ]
        allowed_obs = set(RGB_KEYS + LOWDIM_KEYS)
        required_obs = set(RGB_KEYS + ("eef_pos", "eef_quat"))
        actual_obs = set(self.rgb_keys + self.lowdim_keys)
        if not required_obs.issubset(actual_obs) or not actual_obs.issubset(allowed_obs):
            raise ValueError(
                "RealRobotImageDataset expects RGB plus eef pose obs keys "
                "and optional gripper_q, got "
                f"{sorted(actual_obs)}"
            )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            ~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.sampler = self._make_sampler(train_mask)

    def _make_sampler(self, episode_mask):
        keys = self.rgb_keys + self.lowdim_keys + ["action"]
        key_first_k = {}
        if self.n_obs_steps is not None:
            key_first_k = {key: int(self.n_obs_steps) for key in self.rgb_keys + self.lowdim_keys}
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=keys,
            key_first_k=key_first_k,
            episode_mask=episode_mask,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = self._make_sampler(self.val_mask)
        return val_set

    def _training_step_mask(self) -> np.ndarray:
        mask = np.zeros(int(self.replay_buffer.n_steps), dtype=bool)
        starts = np.insert(self.replay_buffer.episode_ends[:-1], 0, 0)
        for episode_idx, (start, end) in enumerate(zip(starts, self.replay_buffer.episode_ends[:])):
            if self.train_mask[episode_idx]:
                mask[int(start) : int(end)] = True
        return mask

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        train_steps = self._training_step_mask()
        if not np.any(train_steps):
            raise RuntimeError("Cannot fit normalizer without training episodes")
        normalizer = LinearNormalizer()

        action = self.replay_buffer["action"][:][train_steps]
        normalizer["action"] = get_real_robot_action_normalizer(
            array_to_stats(action),
            gripper_bounds=self.gripper_bounds,
            action_format=self.action_format,
        )
        for key in self.lowdim_keys:
            values = self.replay_buffer[key][:][train_steps]
            stat = array_to_stats(values)
            if key == "eef_quat":
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                if key == "gripper_q" and self.gripper_bounds is not None:
                    lower, upper = (float(x) for x in self.gripper_bounds)
                    if not lower < upper:
                        raise ValueError("gripper_bounds must satisfy min < max")
                    stat["min"] = np.asarray([lower], dtype=np.float32)
                    stat["max"] = np.asarray([upper], dtype=np.float32)
                normalizer[key] = get_range_normalizer_from_stat(stat)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int):
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        obs = {}
        obs_slice = slice(self.n_obs_steps)
        for key in self.rgb_keys:
            image = data.pop(key)[obs_slice]
            obs[key] = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0
        for key in self.lowdim_keys:
            obs[key] = data.pop(key)[obs_slice].astype(np.float32)
        action_start = self.n_latency_steps
        action_stop = action_start + self.horizon
        result = {
            "obs": obs,
            "action": data["action"][action_start:action_stop].astype(np.float32),
        }
        return dict_apply(result, torch.from_numpy)
