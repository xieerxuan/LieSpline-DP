import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from liespline_dp.common.pose_action_codec import PoseActionCodec

COMMON_REQUIRED_STEP_KEYS = (
    "active",
    "timestamp",
    "record_frame_index",
    "tcp_actual",
    "wrist_rgb_path",
    "global_rgb_path",
)
GRIPPER_REQUIRED_STEP_KEYS = ("gripper_actual_q", "gripper_target_q")
COMBINED_TARGET_KEY = "tcp_target"
SPLIT_TARGET_KEYS = ("tcp_target_position", "tcp_target_quat")
SUPPORTED_TARGET_QUAT_ORDERS = ("xyzw", "wxyz")
TARGET_SOURCE_XYZ_ROTVEC = "tcp_target_xyz_rotvec_rad"
TARGET_SOURCE_POSITION_QUATERNION = "tcp_target_position_plus_quaternion"
COMBINED_TARGET_METADATA = {
    "tcp_pose_format": "[x,y,z,rx,ry,rz]",
    "tcp_rotation_representation": "rotation vector in radians",
    "robot_state_frame": "UR base frame",
}


@dataclass(frozen=True)
class TargetPoseSource:
    raw_format: str
    required_step_keys: Tuple[str, ...]
    target_quat_order: Optional[str]


@dataclass(frozen=True)
class RealRobotEpisode:
    episode_dir: Path
    episode_name: str
    indices: np.ndarray
    timestamp: np.ndarray
    source_frame_index: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_q: Optional[np.ndarray]
    gripper_target_q: Optional[np.ndarray]
    tactile_active_total_raw: Optional[np.ndarray]
    action: np.ndarray
    wrist_rgb_paths: List[Path]
    global_rgb_paths: List[Path]
    warning_count: int
    raw_target_source_format: str
    target_quat_order: Optional[str]

    @property
    def length(self) -> int:
        return len(self.timestamp)


def discover_episode_dirs(dataset_root) -> List[Path]:
    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    episodes = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and (path / "steps.npz").is_file()
    )
    if not episodes:
        raise FileNotFoundError(
            f"No episode directories containing steps.npz found under {dataset_root}"
        )
    return episodes


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    true_idx = np.flatnonzero(mask)
    if len(true_idx) == 0:
        return true_idx
    split = np.flatnonzero(np.diff(true_idx) != 1) + 1
    runs = np.split(true_idx, split)
    return max(runs, key=len)


def _validate_step_table(
    steps: Mapping[str, np.ndarray],
    episode_dir: Path,
    target_source: TargetPoseSource,
    *,
    include_gripper: bool = True,
) -> int:
    required_keys = COMMON_REQUIRED_STEP_KEYS + target_source.required_step_keys
    if include_gripper:
        required_keys += GRIPPER_REQUIRED_STEP_KEYS
    missing = [key for key in required_keys if key not in steps]
    if missing:
        raise KeyError(f"{episode_dir}: missing steps.npz fields {missing}")
    length = len(steps["timestamp"])
    if length < 1:
        raise ValueError(f"{episode_dir}: empty episode")
    if include_gripper:
        for key in GRIPPER_REQUIRED_STEP_KEYS:
            shape = tuple(steps[key].shape)
            if shape not in ((length,), (length, 1)):
                raise ValueError(
                    f"{episode_dir}: {key} must have shape {(length,)} or "
                    f"{(length, 1)}, got {shape}"
                )
    for key in required_keys:
        if len(steps[key]) != length:
            raise ValueError(
                f"{episode_dir}: field {key} has {len(steps[key])} rows, expected {length}"
            )

    expected_shapes = {
        COMBINED_TARGET_KEY: (length, 6),
        "tcp_target_position": (length, 3),
        "tcp_target_quat": (length, 4),
    }
    for key in target_source.required_step_keys:
        shape = tuple(steps[key].shape)
        if shape != expected_shapes[key]:
            raise ValueError(
                f"{episode_dir}: {key} must have shape {expected_shapes[key]}, got {shape}"
            )
    return length


def _resolve_paths(
    episode_dir: Path,
    relative_paths: Iterable[str],
    field_name: str,
) -> List[Path]:
    paths = [episode_dir / str(value) for value in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = missing[:5]
        raise FileNotFoundError(f"{episode_dir}: missing {field_name} files {preview}")
    return paths


def _load_tactile_active_total_raw(
    episode_dir: Path,
    steps: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> Optional[np.ndarray]:
    if "tactile_active_total_raw" in steps:
        tactile_active_total_raw = np.asarray(
            steps["tactile_active_total_raw"][indices], dtype=np.float32
        )
        if tactile_active_total_raw.shape != (len(indices), 3):
            raise ValueError(
                f"{episode_dir}: tactile_active_total_raw must have shape "
                f"{(len(indices), 3)}, got {tactile_active_total_raw.shape}"
            )
        return tactile_active_total_raw
    if "tactile_taxels_path" not in steps:
        return None

    paths = _resolve_paths(
        episode_dir,
        steps["tactile_taxels_path"][indices],
        "tactile_taxels_path",
    )
    totals = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as tactile:
                if "active_total_raw" in tactile:
                    total = np.asarray(tactile["active_total_raw"], dtype=np.float32)
                elif "resultants_raw_by_module" in tactile:
                    total = np.nansum(
                        np.asarray(
                            tactile["resultants_raw_by_module"],
                            dtype=np.float32,
                        ),
                        axis=0,
                    )
                else:
                    continue
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{path}: invalid tactile archive: {exc}") from exc
        total = np.asarray(total, dtype=np.float32).reshape(-1)
        if total.shape != (3,) or not np.all(np.isfinite(total)):
            raise ValueError(
                f"{path}: tactile active total must be finite shape (3,), got {total.shape}"
            )
        totals.append(total)
    if len(totals) != len(indices):
        raise ValueError(
            f"{episode_dir}: could not derive tactile_active_total_raw for every selected step"
        )
    return np.stack(totals).astype(np.float32)


def read_episode_metadata(episode_dir) -> dict:
    episode_dir = Path(episode_dir).expanduser()
    meta_path = episode_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{meta_path}: invalid episode metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{meta_path}: metadata root must be a JSON object")
    return metadata


def resolve_target_quat_order(
    episode_dir,
    *,
    target_quat_order: Optional[str] = None,
) -> str:
    """Resolve the on-disk target quaternion order without guessing.

    Episodes declare ``tcp_target_quat_order`` in ``meta.json``.
    A dataset-level target_quat_order supplies the default for split targets.
    """
    episode_dir = Path(episode_dir).expanduser()
    metadata = read_episode_metadata(episode_dir)
    declared = metadata.get("tcp_target_quat_order")
    if declared is None:
        if target_quat_order is None:
            raise ValueError(
                f"{episode_dir}: meta.json does not declare "
                "tcp_target_quat_order; set target_quat_order to "
                "'wxyz' or 'xyzw' explicitly"
            )
        order = str(target_quat_order).lower()
    else:
        order = str(declared).lower()
    if order not in SUPPORTED_TARGET_QUAT_ORDERS:
        raise ValueError(
            f"{episode_dir}: unsupported tcp_target_quat_order={order!r}; "
            f"expected one of {SUPPORTED_TARGET_QUAT_ORDERS}"
        )
    return order


def resolve_target_pose_source(
    steps: Mapping[str, np.ndarray],
    episode_dir,
    *,
    target_quat_order: Optional[str] = None,
) -> TargetPoseSource:
    episode_dir = Path(episode_dir).expanduser()
    has_combined = COMBINED_TARGET_KEY in steps
    present_split = [key for key in SPLIT_TARGET_KEYS if key in steps]

    if has_combined and present_split:
        raise ValueError(
            f"{episode_dir}: ambiguous target pose fields; {COMBINED_TARGET_KEY} "
            f"cannot coexist with {present_split}"
        )
    if has_combined:
        metadata = read_episode_metadata(episode_dir)
        for key, expected in COMBINED_TARGET_METADATA.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"{episode_dir}: meta.json {key} must be {expected!r}, got {actual!r}"
                )
        return TargetPoseSource(
            raw_format=TARGET_SOURCE_XYZ_ROTVEC,
            required_step_keys=(COMBINED_TARGET_KEY,),
            target_quat_order=None,
        )

    if present_split and len(present_split) != len(SPLIT_TARGET_KEYS):
        missing = [key for key in SPLIT_TARGET_KEYS if key not in steps]
        raise KeyError(f"{episode_dir}: incomplete split target pose fields; missing {missing}")
    if len(present_split) == len(SPLIT_TARGET_KEYS):
        return TargetPoseSource(
            raw_format=TARGET_SOURCE_POSITION_QUATERNION,
            required_step_keys=SPLIT_TARGET_KEYS,
            target_quat_order=resolve_target_quat_order(
                episode_dir,
                target_quat_order=target_quat_order,
            ),
        )

    raise KeyError(
        f"{episode_dir}: missing target pose fields; expected "
        f"{COMBINED_TARGET_KEY!r} or {SPLIT_TARGET_KEYS}"
    )


def load_real_robot_episode(
    episode_dir,
    *,
    active_only: bool = True,
    min_episode_steps: int = 2,
    codec: Optional[PoseActionCodec] = None,
    target_quat_order: Optional[str] = None,
) -> RealRobotEpisode:
    episode_dir = Path(episode_dir).expanduser()
    codec = PoseActionCodec() if codec is None else codec
    read_keys = set(COMMON_REQUIRED_STEP_KEYS + SPLIT_TARGET_KEYS)
    read_keys.update(
        (COMBINED_TARGET_KEY, "warning", "tactile_active_total_raw", "tactile_taxels_path")
    )
    if codec.include_gripper:
        read_keys.update(GRIPPER_REQUIRED_STEP_KEYS)
    with np.load(episode_dir / "steps.npz", allow_pickle=False) as steps_file:
        # Ignore recorder diagnostics and fields unused by the selected action
        # format, including legacy arrays that cannot be loaded without pickle.
        steps: Dict[str, np.ndarray] = {
            key: steps_file[key] for key in steps_file.files if key in read_keys
        }
    target_source = resolve_target_pose_source(
        steps,
        episode_dir,
        target_quat_order=target_quat_order,
    )
    length = _validate_step_table(
        steps, episode_dir, target_source, include_gripper=codec.include_gripper
    )

    if active_only:
        indices = _longest_true_run(np.asarray(steps["active"], dtype=bool))
    else:
        indices = np.arange(length, dtype=np.int64)
    if len(indices) < int(min_episode_steps):
        raise ValueError(
            f"{episode_dir}: only {len(indices)} usable steps, minimum is {min_episode_steps}"
        )

    timestamp = np.asarray(steps["timestamp"][indices], dtype=np.float64)
    if not np.all(np.isfinite(timestamp)) or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"{episode_dir}: timestamps must be finite and increasing")

    tcp_actual = np.asarray(steps["tcp_actual"][indices], dtype=np.float32)
    if target_source.raw_format == TARGET_SOURCE_XYZ_ROTVEC:
        target_pose = np.asarray(steps[COMBINED_TARGET_KEY][indices], dtype=np.float32)
        target_position = target_pose[:, :3]
        target_quaternion = codec.measured_rotvec_to_quaternion(target_pose[:, 3:6])
    else:
        target_position = np.asarray(steps["tcp_target_position"][indices], dtype=np.float32)
        target_quaternion = np.asarray(steps["tcp_target_quat"][indices], dtype=np.float32)
        if target_source.target_quat_order == "wxyz":
            target_quaternion = target_quaternion[:, [1, 2, 3, 0]]
    actual_gripper = None
    target_gripper = None
    if codec.include_gripper:
        gripper_values = []
        for key in GRIPPER_REQUIRED_STEP_KEYS:
            value = np.asarray(steps[key][indices], dtype=np.float32).reshape(len(indices), 1)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{episode_dir}: {key} must contain only finite values")
            gripper_values.append(value)
        actual_gripper, target_gripper = gripper_values
    tactile_active_total_raw = _load_tactile_active_total_raw(
        episode_dir,
        steps,
        indices,
    )
    numeric_values = (
        tcp_actual,
        target_position,
        target_quaternion,
    )
    if not all(np.all(np.isfinite(value)) for value in numeric_values):
        raise ValueError(f"{episode_dir}: non-finite robot values")
    target_quaternion_norm = np.linalg.norm(target_quaternion, axis=-1)
    if np.any(target_quaternion_norm < 1e-6):
        raise ValueError(f"{episode_dir}: zero-norm target quaternion")
    if np.any(np.abs(target_quaternion_norm - 1.0) > 1e-2):
        raise ValueError(
            f"{episode_dir}: target quaternion norm differs from one by more than 1e-2"
        )
    target_quaternion = (target_quaternion / target_quaternion_norm[:, None]).astype(np.float32)

    eef_quat = codec.measured_rotvec_to_quaternion(tcp_actual[:, 3:6])
    action = codec.encode_target(
        target_position,
        target_quaternion,
        target_gripper,
    )
    warnings = steps.get("warning", np.full(length, "", dtype="<U1"))[indices]
    warning_count = int(np.count_nonzero(np.asarray(warnings).astype(str) != ""))

    return RealRobotEpisode(
        episode_dir=episode_dir,
        episode_name=episode_dir.name,
        indices=indices.astype(np.int64),
        timestamp=timestamp,
        source_frame_index=np.asarray(steps["record_frame_index"][indices], dtype=np.int64),
        eef_pos=tcp_actual[:, :3].astype(np.float32),
        eef_quat=np.asarray(eef_quat, dtype=np.float32),
        gripper_q=actual_gripper,
        gripper_target_q=target_gripper,
        tactile_active_total_raw=tactile_active_total_raw,
        action=np.asarray(action, dtype=np.float32),
        wrist_rgb_paths=_resolve_paths(
            episode_dir,
            steps["wrist_rgb_path"][indices],
            "wrist_rgb_path",
        ),
        global_rgb_paths=_resolve_paths(
            episode_dir,
            steps["global_rgb_path"][indices],
            "global_rgb_path",
        ),
        warning_count=warning_count,
        raw_target_source_format=target_source.raw_format,
        target_quat_order=target_source.target_quat_order,
    )
