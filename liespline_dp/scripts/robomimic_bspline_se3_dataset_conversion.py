import atexit
import concurrent.futures
import multiprocessing
import pathlib
import shutil
from collections import deque

import click
import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from liespline_dp.common.bspline_se3_util import (
    build_bspline_se3_training_episode,
    se3_pose_errors,
)
from liespline_dp.common.contact_force_util import (
    cfrc_ext_delta_score as _shared_cfrc_ext_delta_score,
)
from liespline_dp.common.contact_force_util import (
    robust_threshold as _shared_robust_threshold,
)
from liespline_dp.common.contact_force_util import (
    select_event_indices_from_score as _shared_select_event_indices_from_score,
)
from liespline_dp.common.replay_buffer import ReplayBuffer
from liespline_dp.common.robomimic_util import (
    normalize_robosuite_controller_metadata,
)
from liespline_dp.common.task_config_util import (
    instantiate_task_action_pose_adapter,
    load_task_config,
)
from liespline_dp.dataset.bspline_se3_dataset_validation import (
    build_bspline_se3_anchor_contract_metadata,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


_CONVERSION_WORKER_STATE = {}


def _get_n_infer_steps(task_cfg):
    return int(
        task_cfg.get(
            "n_infer_steps",
            task_cfg.bspline.n_action_steps,
        )
    )


def _validate_anchor_contract_config(task_cfg):
    contract = task_cfg.anchor_contract
    adapter_cfg = task_cfg.action_pose_adapter
    mismatches = []
    if str(contract.family) != "robomimic":
        mismatches.append(f"family={contract.family!s}")
    if str(contract.adapter_target) != str(adapter_cfg._target_):
        mismatches.append("adapter_target")
    if str(adapter_cfg.action_rotation_rep) != str(task_cfg.conversion.action_rep):
        mismatches.append("action_rotation_rep")
    if int(adapter_cfg.gripper_dim) != int(task_cfg.se3.gripper_dim):
        mismatches.append("gripper_dim")
    if list(adapter_cfg.obs_keys) != list(task_cfg.conversion.obs_keys):
        mismatches.append("obs_keys")
    if str(contract.compose_side) != str(adapter_cfg.obs_eef_to_action_compose_side):
        mismatches.append("compose_side")
    contract_rotvec = np.asarray(contract.rotvec, dtype=np.float64).reshape(-1)
    adapter_rotvec = np.asarray(
        adapter_cfg.obs_eef_to_action_rotvec,
        dtype=np.float64,
    ).reshape(-1)
    if (
        contract_rotvec.shape != (3,)
        or adapter_rotvec.shape != (3,)
        or not np.allclose(
            contract_rotvec,
            adapter_rotvec,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        mismatches.append("rotvec")
    if int(contract.num_robots) != int(adapter_cfg.n_robots) or int(contract.num_robots) != int(
        task_cfg.se3.num_robots
    ):
        mismatches.append("num_robots")
    if mismatches:
        raise ValueError(
            "Robomimic anchor_contract does not match action_pose_adapter: " + ", ".join(mismatches)
        )


def _read_demo_obs(raw_obs, obs_keys):
    return np.concatenate(
        [raw_obs[key][:].astype(np.float32) for key in obs_keys],
        axis=-1,
    )


def _infer_num_robots(task_cfg, obs_keys):
    if "num_robots" in task_cfg.se3:
        return int(task_cfg.se3.num_robots)
    max_robot_idx = -1
    for key in obs_keys:
        if not key.startswith("robot"):
            continue
        idx_text = key[len("robot") :].split("_", 1)[0]
        if idx_text.isdigit():
            max_robot_idx = max(max_robot_idx, int(idx_text))
    return max_robot_idx + 1 if max_robot_idx >= 0 else 1


def _combine_robot_episodes(robot_episodes):
    if len(robot_episodes) == 1:
        return robot_episodes[0]

    ref = robot_episodes[0]
    metadata_keys = {
        "trajectory",
        "condition_data",
        "condition_mask",
        "pose_waypoint",
        "gripper_waypoint",
        "waypoint_phase",
        "waypoint_t_idx",
        "window_starts",
        "anchor_t_idx",
        "anchor_phase",
        "anchor_pose",
        "anchor_grip",
        "phase_per_timestep",
        "control_poses",
        "raw_gripper",
        "fit_anchor_t_idx",
    }
    observation_keys = [key for key in ref.keys() if key not in metadata_keys]
    for episode in robot_episodes[1:]:
        for key in ("window_starts", "anchor_t_idx", "anchor_phase"):
            np.testing.assert_array_equal(ref[key], episode[key])
        for key in observation_keys:
            np.testing.assert_array_equal(ref[key], episode[key])

    combined = {
        "trajectory": np.concatenate(
            [episode["trajectory"] for episode in robot_episodes],
            axis=-1,
        ),
        "condition_data": np.concatenate(
            [episode["condition_data"] for episode in robot_episodes],
            axis=-1,
        ),
        "condition_mask": np.concatenate(
            [episode["condition_mask"] for episode in robot_episodes],
            axis=-1,
        ),
        "pose_waypoint": np.concatenate(
            [episode["pose_waypoint"] for episode in robot_episodes],
            axis=-1,
        ),
        "gripper_waypoint": np.concatenate(
            [episode["gripper_waypoint"] for episode in robot_episodes],
            axis=-1,
        ),
        "waypoint_phase": ref["waypoint_phase"],
        "waypoint_t_idx": ref["waypoint_t_idx"],
        "window_starts": ref["window_starts"],
        "anchor_t_idx": ref["anchor_t_idx"],
        "anchor_phase": ref["anchor_phase"],
        "anchor_pose": np.stack(
            [episode["anchor_pose"] for episode in robot_episodes],
            axis=1,
        ),
        "anchor_grip": np.stack(
            [episode["anchor_grip"] for episode in robot_episodes],
            axis=1,
        ),
    }
    for key in observation_keys:
        combined[key] = ref[key]
    return combined


def _get_fit_point_weight_cfg(task_cfg):
    return task_cfg.conversion.get("fit_point_weight", None)


def _is_fit_point_weight_enabled(task_cfg):
    cfg = _get_fit_point_weight_cfg(task_cfg)
    return cfg is not None and bool(cfg.get("enabled", True))


def _get_fit_point_weight_detector(task_cfg):
    cfg = _get_fit_point_weight_cfg(task_cfg)
    if cfg is None or not _is_fit_point_weight_enabled(task_cfg):
        return "none"
    return str(cfg.get("detector", "pose_deviation"))


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


def _build_event_point_weights(task_cfg, length, event_indices):
    cfg = _get_fit_point_weight_cfg(task_cfg)
    if not _is_fit_point_weight_enabled(task_cfg):
        return None

    base_weight = float(cfg.get("base_weight", 1.0))
    proximity_weight = float(cfg.get("proximity_weight", 2.0))
    contact_weight = float(cfg.get("contact_weight", 4.0))
    proximity_neighborhood = int(cfg.get("proximity_neighborhood", cfg.get("neighborhood", 1)))
    contact_neighborhood = int(cfg.get("contact_neighborhood", 0))

    if base_weight <= 0.0:
        raise ValueError("fit_point_weight.base_weight must be positive.")
    if proximity_weight < base_weight:
        raise ValueError("fit_point_weight.proximity_weight must be >= base_weight.")
    if contact_weight < proximity_weight:
        raise ValueError("fit_point_weight.contact_weight must be >= proximity_weight.")
    if proximity_neighborhood < 0 or contact_neighborhood < 0:
        raise ValueError("fit_point_weight neighborhoods must be non-negative.")

    weights = np.full(int(length), fill_value=base_weight, dtype=np.float64)
    event_indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
    event_indices = event_indices[(event_indices >= 0) & (event_indices < len(weights))]
    if len(event_indices) == 0:
        return weights.astype(np.float32)

    for idx in np.unique(event_indices):
        lo = max(0, int(idx) - proximity_neighborhood)
        hi = min(len(weights), int(idx) + proximity_neighborhood + 1)
        weights[lo:hi] = np.maximum(weights[lo:hi], proximity_weight)

    for idx in np.unique(event_indices):
        lo = max(0, int(idx) - contact_neighborhood)
        hi = min(len(weights), int(idx) + contact_neighborhood + 1)
        weights[lo:hi] = contact_weight

    return weights.astype(np.float32)


def _find_gripper_switch_indices(gripper):
    gripper = np.asarray(gripper)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    if len(gripper) < 2:
        return np.zeros((0,), dtype=np.int64)
    return (np.flatnonzero(np.any(gripper[1:] != gripper[:-1], axis=-1)) + 1).astype(np.int64)


def _apply_gripper_switch_point_weights(task_cfg, weights, gripper):
    cfg = _get_fit_point_weight_cfg(task_cfg)
    if not _is_fit_point_weight_enabled(task_cfg) or gripper is None:
        return weights

    gripper_switch_weight = float(cfg.get("gripper_switch_weight", 16.0))
    gripper_switch_neighborhood = int(cfg.get("gripper_switch_neighborhood", 2))
    if gripper_switch_weight <= 0.0:
        raise ValueError("fit_point_weight.gripper_switch_weight must be positive.")
    if gripper_switch_neighborhood < 0:
        raise ValueError("fit_point_weight.gripper_switch_neighborhood must be non-negative.")

    if weights is None:
        weights = np.full(
            len(gripper),
            fill_value=float(cfg.get("base_weight", 1.0)),
            dtype=np.float32,
        )
    else:
        weights = np.asarray(weights, dtype=np.float32).copy()
    if len(weights) != len(gripper):
        raise ValueError(
            f"point weights and gripper length differ: {len(weights)} vs {len(gripper)}"
        )

    switch_indices = _find_gripper_switch_indices(gripper)
    for idx in switch_indices:
        lo = max(0, int(idx) - gripper_switch_neighborhood)
        hi = min(len(weights), int(idx) + gripper_switch_neighborhood + 1)
        weights[lo:hi] = gripper_switch_weight
    return weights.astype(np.float32)


def _build_pose_deviation_point_weights(task_cfg, planned_pose, observed_pose):
    cfg = _get_fit_point_weight_cfg(task_cfg)
    if not _is_fit_point_weight_enabled(task_cfg):
        return None

    detector = cfg.get("detector", "pose_deviation")
    if detector != "pose_deviation":
        raise ValueError(f"Unsupported robomimic fit point weight detector: {detector}")

    trans_threshold = float(cfg.get("trans_threshold", 0.05))
    rot_threshold_deg = float(cfg.get("rot_threshold_deg", 25.0))
    if trans_threshold <= 0.0:
        raise ValueError("fit_point_weight.trans_threshold must be positive.")
    if rot_threshold_deg <= 0.0:
        raise ValueError("fit_point_weight.rot_threshold_deg must be positive.")

    trans_l2, rot_angle = se3_pose_errors(observed_pose, planned_pose)
    contact_mask = (trans_l2 > trans_threshold) | (rot_angle > np.deg2rad(rot_threshold_deg))
    return _build_event_point_weights(
        task_cfg=task_cfg,
        length=len(planned_pose),
        event_indices=np.flatnonzero(contact_mask),
    )


def _robust_threshold(score, mad_multiplier, min_threshold):
    return _shared_robust_threshold(score, mad_multiplier, min_threshold)


def _select_event_indices_from_score(score, threshold, cooldown_steps=0, warmup_steps=1):
    return _shared_select_event_indices_from_score(
        score, threshold, cooldown_steps=cooldown_steps, warmup_steps=warmup_steps
    )


def _cfrc_ext_delta_score(wrench_sequence, torque_delta_weight=0.2, ema_alpha=0.35):
    return _shared_cfrc_ext_delta_score(
        wrench_sequence,
        torque_delta_weight=torque_delta_weight,
        ema_alpha=ema_alpha,
    )


def _build_cfrc_ext_delta_point_weights(task_cfg, wrench_sequence):
    cfg = _get_fit_point_weight_cfg(task_cfg)
    if not _is_fit_point_weight_enabled(task_cfg):
        return None

    score = _cfrc_ext_delta_score(
        wrench_sequence=wrench_sequence,
        torque_delta_weight=float(cfg.get("torque_delta_weight", 0.2)),
        ema_alpha=float(cfg.get("ema_alpha", 0.35)),
    )
    explicit_threshold = cfg.get("force_delta_threshold", None)
    if explicit_threshold is None:
        threshold = _robust_threshold(
            score=score,
            mad_multiplier=float(cfg.get("mad_multiplier", 6.0)),
            min_threshold=float(cfg.get("min_force_delta_threshold", 1e-6)),
        )
    else:
        threshold = float(explicit_threshold)
    event_indices = _select_event_indices_from_score(
        score=score,
        threshold=threshold,
        cooldown_steps=int(cfg.get("cooldown_steps", 2)),
        warmup_steps=int(cfg.get("warmup_steps", 1)),
    )
    return _build_event_point_weights(
        task_cfg=task_cfg,
        length=len(score),
        event_indices=event_indices,
    )


def _create_robomimic_replay_env(dataset_path, obs_keys):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env_meta = normalize_robosuite_controller_metadata(
        env_meta,
        absolute_action=False,
    )
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )


def _close_env(env):
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def _get_mujoco_sim(env):
    current = env
    visited = set()
    for _ in range(8):
        if id(current) in visited:
            break
        visited.add(id(current))
        if hasattr(current, "sim"):
            return current.sim
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    raise AttributeError("Could not find MuJoCo sim on robomimic environment.")


def _body_names(model):
    if hasattr(model, "body_names"):
        return [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in model.body_names
        ]
    nbody = int(getattr(model, "nbody"))
    if hasattr(model, "body_id2name"):
        return [str(model.body_id2name(i)) for i in range(nbody)]
    raise AttributeError("MuJoCo model does not expose body names.")


def _select_cfrc_ext_body_ids(sim, task_cfg, robot_index, num_robots, body_cfg=None):
    cfg = (_get_fit_point_weight_cfg(task_cfg) or {}) if body_cfg is None else body_cfg
    explicit_names = [str(x) for x in cfg.get("body_names", [])]
    names = _body_names(sim.model)
    if explicit_names:
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in explicit_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Configured cfrc_ext body_names not found: {missing}")
        return [name_to_idx[name] for name in explicit_names]

    patterns = [
        str(x).lower() for x in cfg.get("body_name_patterns", ["eef", "gripper", "finger", "hand"])
    ]
    exclude_patterns = [
        str(x).lower()
        for x in cfg.get("exclude_body_name_patterns", ["camera", "mount", "base", "world"])
    ]

    def select(robot_scoped):
        ids = []
        for idx, name in enumerate(names):
            lname = name.lower()
            if any(pattern in lname for pattern in exclude_patterns):
                continue
            if not any(pattern in lname for pattern in patterns):
                continue
            if robot_scoped and num_robots > 1:
                robot_tags = (f"robot{robot_index}", f"gripper{robot_index}")
                if not any(tag in lname for tag in robot_tags):
                    continue
            ids.append(idx)
        return ids

    body_ids = select(robot_scoped=True)
    if not body_ids:
        body_ids = select(robot_scoped=False)
    if not body_ids:
        preview = ", ".join(names[:40])
        raise ValueError(
            "No cfrc_ext bodies matched fit_point_weight body_name_patterns "
            f"{patterns}. First body names: {preview}"
        )
    return body_ids


def _read_cfrc_ext_wrench_sequences(replay_env, states, task_cfg, num_robots, body_cfg=None):
    states = np.asarray(states)
    if states.ndim < 2:
        raise ValueError(f"Expected states with shape [T, D], got {states.shape}")
    if len(states) == 0:
        return [np.zeros((0, 0, 6), dtype=np.float64) for _ in range(num_robots)]

    replay_env.reset_to({"states": states[0]})
    sim = _get_mujoco_sim(replay_env)
    if not hasattr(sim.data, "cfrc_ext"):
        raise AttributeError("MuJoCo sim.data does not expose cfrc_ext.")
    body_ids_by_robot = [
        _select_cfrc_ext_body_ids(
            sim=sim,
            task_cfg=task_cfg,
            robot_index=robot_idx,
            num_robots=num_robots,
            body_cfg=body_cfg,
        )
        for robot_idx in range(num_robots)
    ]
    wrench_sequences = [
        np.zeros((len(states), len(body_ids), 6), dtype=np.float64)
        for body_ids in body_ids_by_robot
    ]

    for t_idx, state in enumerate(states):
        if t_idx != 0:
            replay_env.reset_to({"states": state})
            sim = _get_mujoco_sim(replay_env)
        cfrc_ext = np.asarray(sim.data.cfrc_ext, dtype=np.float64)
        for robot_idx, body_ids in enumerate(body_ids_by_robot):
            wrench_sequences[robot_idx][t_idx] = cfrc_ext[body_ids]
    return wrench_sequences


def _build_fit_point_weights_for_demo(
    task_cfg,
    planned_poses,
    observed_poses,
    grippers=None,
    demo=None,
    replay_env=None,
    num_robots=None,
):
    detector = _get_fit_point_weight_detector(task_cfg)
    num_robots = len(planned_poses) if num_robots is None else int(num_robots)
    if grippers is None:
        grippers = [None for _ in range(num_robots)]
    if len(grippers) != num_robots:
        raise ValueError(f"Expected {num_robots} gripper sequences, got {len(grippers)}")
    if detector == "none":
        return [None for _ in range(num_robots)]
    if detector == "pose_deviation":
        weights_by_robot = [
            _build_pose_deviation_point_weights(
                task_cfg=task_cfg,
                planned_pose=planned_poses[robot_idx],
                observed_pose=observed_poses[robot_idx],
            )
            for robot_idx in range(num_robots)
        ]
        return [
            _apply_gripper_switch_point_weights(
                task_cfg=task_cfg,
                weights=weights_by_robot[robot_idx],
                gripper=grippers[robot_idx],
            )
            for robot_idx in range(num_robots)
        ]
    if detector == "cfrc_ext_delta":
        if replay_env is None:
            raise ValueError("fit_point_weight.detector=cfrc_ext_delta requires replay_env.")
        if demo is None or "states" not in demo:
            raise ValueError("fit_point_weight.detector=cfrc_ext_delta requires demo states.")
        wrench_sequences = _read_cfrc_ext_wrench_sequences(
            replay_env=replay_env,
            states=demo["states"][:],
            task_cfg=task_cfg,
            num_robots=num_robots,
        )
        weights_by_robot = [
            _build_cfrc_ext_delta_point_weights(
                task_cfg=task_cfg,
                wrench_sequence=wrench_sequences[robot_idx],
            )
            for robot_idx in range(num_robots)
        ]
        return [
            _apply_gripper_switch_point_weights(
                task_cfg=task_cfg,
                weights=weights_by_robot[robot_idx],
                gripper=grippers[robot_idx],
            )
            for robot_idx in range(num_robots)
        ]
    raise ValueError(f"Unsupported robomimic fit point weight detector: {detector}")


def _make_action_pose_adapters(task_cfg, obs_keys, num_robots):
    return [
        instantiate_task_action_pose_adapter(
            task_cfg,
            robot_index=robot_idx,
            obs_keys=obs_keys,
            n_robots=num_robots,
        )
        for robot_idx in range(num_robots)
    ]


def _build_demo_payloads(
    task_cfg,
    demo,
    source_demo_idx,
    adapters,
    replay_env,
    robot_point_weights_override=None,
):
    obs_keys = list(task_cfg.conversion.obs_keys)
    image_obs_keys = list(task_cfg.conversion.get("image_obs_keys", []))
    action_key = task_cfg.conversion.get("action_key", "actions")
    num_robots = len(adapters)
    raw_action = demo[action_key][:].astype(np.float32)
    raw_obs = demo["obs"]

    # Phase 0: read O_t and build SE(3) action targets T_t.
    raw_obs_dict = {key: raw_obs[key][:].astype(np.float32) for key in obs_keys}
    obs = _read_demo_obs(raw_obs=raw_obs, obs_keys=obs_keys)
    image_obs_dict = {key: raw_obs[key][:] for key in image_obs_keys}
    robot_poses = []
    robot_grippers = []
    robot_current_poses = []

    # one adapter for every robot
    for adapter in adapters:
        poses, gripper = adapter.action_to_pose_grip(raw_action)
        current_pose, _ = adapter.obs_to_current_pose_grip(raw_obs_dict)
        robot_poses.append(poses)
        robot_grippers.append(gripper)
        robot_current_poses.append(current_pose)
    robot_point_weights = robot_point_weights_override
    if robot_point_weights is None:
        robot_point_weights = _build_fit_point_weights_for_demo(
            task_cfg=task_cfg,
            planned_poses=robot_poses,
            observed_poses=robot_current_poses,
            grippers=robot_grippers,
            demo=demo,
            replay_env=replay_env,
            num_robots=num_robots,
        )
    payloads = []
    n_skipped_episodes = 0

    # Phase 1: make offset-augmented sub-demos before spline fitting.
    start_offsets = _get_start_offsets(task_cfg)
    for start_offset in start_offsets:
        start_offset = int(start_offset)
        if start_offset >= len(robot_poses[0]):
            n_skipped_episodes += 1
            continue

        sub_observations = {
            "obs": obs[start_offset:].astype(np.float32),
        }
        for key, value in image_obs_dict.items():
            sub_observations[key] = value[start_offset:]

        robot_episodes = []
        for robot_idx in range(num_robots):
            sub_poses = robot_poses[robot_idx][start_offset:]
            sub_gripper = robot_grippers[robot_idx][start_offset:]
            sub_current_pose = robot_current_poses[robot_idx][start_offset:]
            point_weights = robot_point_weights[robot_idx]
            sub_point_weights = point_weights[start_offset:] if point_weights is not None else None
            # Fit this sub-demo into one full SE(3) B-spline trajectory.
            robot_episode = build_bspline_se3_training_episode(
                poses=sub_poses,
                gripper=sub_gripper,
                observations=sub_observations,
                anchor_poses=sub_current_pose,
                degree=task_cfg.bspline.degree,
                prefix_cp_steps=task_cfg.bspline.prefix_cp_steps,
                pred_horizon=task_cfg.bspline.pred_horizon,
                control_point_stride=task_cfg.bspline.control_point_stride,
                sample_stride=task_cfg.bspline.sample_stride,
                n_action_steps=task_cfg.bspline.n_action_steps,
                n_infer_steps=_get_n_infer_steps(task_cfg),
                exec_horizon=task_cfg.bspline.exec_horizon,
                n_obs_steps=task_cfg.bspline.n_obs_steps,
                point_weights=sub_point_weights,
                fit_translation_scale_m=float(task_cfg.se3.get("fit_translation_scale_m", 1.0)),
                fit_rotation_scale_rad=float(task_cfg.se3.get("fit_rotation_scale_rad", 1.0)),
                terminal_point_weight=float(
                    _get_fit_point_weight_cfg(task_cfg).get("base_weight", 1.0)
                    if _get_fit_point_weight_cfg(task_cfg) is not None
                    else 1.0
                ),
                pad_terminal=bool(task_cfg.bspline.get("pad_terminal", False)),
            )
            if robot_episode is None:
                robot_episodes = None
                break
            robot_episodes.append(robot_episode)
        if robot_episodes is None:
            n_skipped_episodes += 1
            continue
        se3_episode = _combine_robot_episodes(robot_episodes)

        num_episode_samples = len(se3_episode["trajectory"])

        # Phase 2: store pre-windowed training samples for the dataset.
        payload = {
            "obs": se3_episode["obs"],
            "trajectory": se3_episode["trajectory"],
            "condition_data": se3_episode["condition_data"],
            "condition_mask": se3_episode["condition_mask"],
            "pose_waypoint": se3_episode["pose_waypoint"],
            "gripper_waypoint": se3_episode["gripper_waypoint"],
            "waypoint_phase": se3_episode["waypoint_phase"],
            "waypoint_t_idx": se3_episode["waypoint_t_idx"],
            "window_starts": se3_episode["window_starts"],
            "anchor_t_idx": se3_episode["anchor_t_idx"],
            "anchor_phase": se3_episode["anchor_phase"],
            "anchor_pose": se3_episode["anchor_pose"],
            "anchor_grip": se3_episode["anchor_grip"],
            "source_demo_idx": np.full(
                num_episode_samples,
                fill_value=source_demo_idx,
                dtype=np.int64,
            ),
            "source_start_offset": np.full(
                num_episode_samples,
                fill_value=start_offset,
                dtype=np.int64,
            ),
        }
        for key in image_obs_keys:
            payload[key] = se3_episode[key]
        payloads.append(payload)

    return {
        "payloads": payloads,
        "num_source_subepisodes": len(start_offsets),
        "num_skipped_episodes": n_skipped_episodes,
    }


def _close_conversion_worker():
    state = _CONVERSION_WORKER_STATE
    _close_env(state.get("replay_env"))
    src_file = state.get("src_file")
    if src_file is not None:
        src_file.close()
    state.clear()


def _initialize_conversion_worker(task_cfg_container, input_path):
    task_cfg = OmegaConf.create(task_cfg_container)
    import torch

    torch.set_num_threads(int(task_cfg.conversion.get("threads_per_worker", 4)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    obs_keys = list(task_cfg.conversion.obs_keys)
    num_robots = _infer_num_robots(task_cfg, obs_keys)
    src_file = h5py.File(input_path, "r")
    _CONVERSION_WORKER_STATE.update(
        {
            "task_cfg": task_cfg,
            "src_file": src_file,
            "demos": src_file[task_cfg.conversion.get("demo_group", "data")],
            "adapters": _make_action_pose_adapters(task_cfg, obs_keys, num_robots),
            "replay_env": None,
        }
    )
    atexit.register(_close_conversion_worker)


def _conversion_worker(job):
    state = _CONVERSION_WORKER_STATE
    if isinstance(job, tuple):
        demo_key, robot_point_weights = job
    else:
        demo_key = job
        robot_point_weights = None
    source_demo_idx = int(demo_key.split("_")[-1])
    return _build_demo_payloads(
        task_cfg=state["task_cfg"],
        demo=state["demos"][demo_key],
        source_demo_idx=source_demo_idx,
        adapters=state["adapters"],
        replay_env=state["replay_env"],
        robot_point_weights_override=robot_point_weights,
    )


def _build_parallel_demo_jobs(task_cfg, input_path, demo_keys):
    needs_fit_wrench = _get_fit_point_weight_detector(task_cfg) == "cfrc_ext_delta"
    if not needs_fit_wrench:
        return list(demo_keys)

    obs_keys = list(task_cfg.conversion.obs_keys)
    action_key = task_cfg.conversion.get("action_key", "actions")
    demo_group = task_cfg.conversion.get("demo_group", "data")
    num_robots = _infer_num_robots(task_cfg, obs_keys)
    adapters = _make_action_pose_adapters(task_cfg, obs_keys, num_robots)
    replay_env = _create_robomimic_replay_env(
        dataset_path=input_path,
        obs_keys=obs_keys,
    )
    jobs = []
    try:
        with h5py.File(input_path, "r") as src_file:
            demos = src_file[demo_group]
            for demo_key in demo_keys:
                demo = demos[demo_key]
                raw_action = demo[action_key][:].astype(np.float32)
                raw_obs = demo["obs"]
                raw_obs_dict = {key: raw_obs[key][:].astype(np.float32) for key in obs_keys}
                robot_poses = []
                robot_grippers = []
                robot_current_poses = []
                for adapter in adapters:
                    poses, gripper = adapter.action_to_pose_grip(raw_action)
                    current_pose, _ = adapter.obs_to_current_pose_grip(raw_obs_dict)
                    robot_poses.append(poses)
                    robot_grippers.append(gripper)
                    robot_current_poses.append(current_pose)
                weights = _build_fit_point_weights_for_demo(
                    task_cfg=task_cfg,
                    planned_poses=robot_poses,
                    observed_poses=robot_current_poses,
                    grippers=robot_grippers,
                    demo=demo,
                    replay_env=replay_env,
                    num_robots=num_robots,
                )
                jobs.append((demo_key, weights))
    finally:
        _close_env(replay_env)
    return jobs


def _iter_parallel_demo_results(task_cfg, input_path, demo_keys, num_workers):
    # Spawn avoids inheriting live HDF5, MuJoCo, and Torch runtime state.
    context = multiprocessing.get_context("spawn")
    task_cfg_container = OmegaConf.to_container(task_cfg, resolve=True)
    demo_jobs = _build_parallel_demo_jobs(task_cfg, input_path, demo_keys)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_initialize_conversion_worker,
        initargs=(task_cfg_container, str(input_path)),
    ) as executor:
        demo_iter = iter(demo_jobs)
        pending = deque()
        for _ in range(num_workers):
            job = next(demo_iter, None)
            if job is None:
                break
            pending.append(executor.submit(_conversion_worker, job))

        while pending:
            future = pending.popleft()
            yield future.result()
            job = next(demo_iter, None)
            if job is not None:
                pending.append(executor.submit(_conversion_worker, job))


def _iter_serial_demo_results(task_cfg, input_path, demo_keys):
    obs_keys = list(task_cfg.conversion.obs_keys)
    num_robots = _infer_num_robots(task_cfg, obs_keys)
    adapters = _make_action_pose_adapters(task_cfg, obs_keys, num_robots)
    needs_replay_env = _get_fit_point_weight_detector(task_cfg) == "cfrc_ext_delta"
    replay_env = None
    if needs_replay_env:
        replay_env = _create_robomimic_replay_env(
            dataset_path=input_path,
            obs_keys=obs_keys,
        )

    demo_group = task_cfg.conversion.get("demo_group", "data")
    try:
        with h5py.File(input_path, "r") as src_file:
            demos = src_file[demo_group]
            for demo_key in demo_keys:
                yield _build_demo_payloads(
                    task_cfg=task_cfg,
                    demo=demos[demo_key],
                    source_demo_idx=int(demo_key.split("_")[-1]),
                    adapters=adapters,
                    replay_env=replay_env,
                )
    finally:
        _close_env(replay_env)


def _read_demo_keys(input_path, demo_group, max_episodes):
    with h5py.File(input_path, "r") as src_file:
        demo_keys = sorted(
            src_file[demo_group].keys(),
            key=lambda key: int(key.split("_")[-1]),
        )
    if max_episodes is not None:
        demo_keys = demo_keys[: int(max_episodes)]
    return demo_keys


def _write_conversion_result(dst_buffer, result):
    num_samples = 0
    for payload in result["payloads"]:
        dst_buffer.add_episode(payload, compressors="disk")
        num_samples += len(payload["trajectory"])
    return len(result["payloads"]), num_samples


def _finalize_conversion(
    dst_buffer,
    task_cfg,
    num_robots,
    n_source_episodes,
    n_source_subepisodes,
    n_converted_episodes,
    n_skipped_episodes,
    n_samples,
    start_offsets,
    num_workers,
):
    trajectory_horizon = int(task_cfg.bspline.prefix_cp_steps + task_cfg.bspline.pred_horizon)
    n_action_steps = int(task_cfg.bspline.n_action_steps)
    n_infer_steps = _get_n_infer_steps(task_cfg)
    metadata = {
        "bspline_degree": [task_cfg.bspline.degree],
        "prefix_cp_steps": [task_cfg.bspline.prefix_cp_steps],
        "pred_horizon": [task_cfg.bspline.pred_horizon],
        "trajectory_horizon": [trajectory_horizon],
        "exec_horizon": [task_cfg.bspline.exec_horizon],
        "control_point_stride": [task_cfg.bspline.control_point_stride],
        "sample_stride": [task_cfg.bspline.sample_stride],
        "pad_terminal": [bool(task_cfg.bspline.get("pad_terminal", False))],
        "n_obs_steps": [task_cfg.bspline.n_obs_steps],
        "n_action_steps": [n_action_steps],
        "n_infer_steps": [n_infer_steps],
        "n_latency_steps": [n_action_steps - n_infer_steps],
        "action_rep": [task_cfg.conversion.get("action_rep", "axis_angle")],
        "fit_method": [task_cfg.conversion.get("fit_method", "cumulative_lie_lsq")],
        "fit_translation_scale_m": [float(task_cfg.se3.get("fit_translation_scale_m", 1.0))],
        "fit_rotation_scale_rad": [float(task_cfg.se3.get("fit_rotation_scale_rad", 1.0))],
        "gripper_dim": [task_cfg.se3.gripper_dim],
        "num_robots": [num_robots],
        "num_source_episodes": [n_source_episodes],
        "num_source_subepisodes": [n_source_subepisodes],
        "num_converted_episodes": [n_converted_episodes],
        "num_skipped_episodes": [n_skipped_episodes],
        "num_samples": [n_samples],
        "num_start_offsets": [len(start_offsets)],
        "conversion_num_workers": [num_workers],
        "fit_point_weight_enabled": [int(_is_fit_point_weight_enabled(task_cfg))],
        "fit_point_weight_detector": [
            str(_get_fit_point_weight_cfg(task_cfg).get("detector", "pose_deviation"))
            if _get_fit_point_weight_cfg(task_cfg) is not None
            else "none"
        ],
    }
    metadata.update(build_bspline_se3_anchor_contract_metadata(task_cfg.anchor_contract))
    dst_buffer.update_meta(metadata)
    return {
        "num_source_episodes": n_source_episodes,
        "num_source_subepisodes": n_source_subepisodes,
        "num_converted_episodes": n_converted_episodes,
        "num_skipped_episodes": n_skipped_episodes,
        "num_samples": n_samples,
        "num_start_offsets": len(start_offsets),
        "num_workers": num_workers,
    }


def convert_robomimic_bspline_se3_dataset(
    task_cfg,
    input_path=None,
    output_path=None,
    max_episodes=None,
    num_workers=None,
):
    task_cfg = OmegaConf.create(task_cfg)
    _validate_anchor_contract_config(task_cfg)
    input_path = pathlib.Path(
        input_path if input_path is not None else task_cfg.conversion.input_path
    ).expanduser()
    output_path = pathlib.Path(
        output_path if output_path is not None else task_cfg.conversion.output_path
    ).expanduser()

    assert input_path.exists()
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dst_buffer = ReplayBuffer.create_from_path(str(output_path), mode="a")
    demo_group = task_cfg.conversion.get("demo_group", "data")
    obs_keys = list(task_cfg.conversion.obs_keys)
    num_robots = _infer_num_robots(task_cfg, obs_keys)
    num_workers = int(
        task_cfg.conversion.get("num_workers", 1) if num_workers is None else num_workers
    )
    if num_workers < 1:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    start_offsets = _get_start_offsets(task_cfg)
    demo_keys = _read_demo_keys(
        input_path=input_path,
        demo_group=demo_group,
        max_episodes=max_episodes,
    )
    if num_workers > 1:
        results = _iter_parallel_demo_results(
            task_cfg=task_cfg,
            input_path=input_path,
            demo_keys=demo_keys,
            num_workers=num_workers,
        )
    else:
        results = _iter_serial_demo_results(
            task_cfg=task_cfg,
            input_path=input_path,
            demo_keys=demo_keys,
        )

    n_converted_episodes = 0
    n_skipped_episodes = 0
    n_samples = 0
    n_source_subepisodes = 0
    for result in tqdm(
        results, total=len(demo_keys), desc="Converting robomimic bspline SE(3) dataset"
    ):
        converted, samples = _write_conversion_result(dst_buffer, result)
        n_converted_episodes += converted
        n_samples += samples
        n_source_subepisodes += result["num_source_subepisodes"]
        n_skipped_episodes += result["num_skipped_episodes"]

    return _finalize_conversion(
        dst_buffer=dst_buffer,
        task_cfg=task_cfg,
        num_robots=num_robots,
        n_source_episodes=len(demo_keys),
        n_source_subepisodes=n_source_subepisodes,
        n_converted_episodes=n_converted_episodes,
        n_skipped_episodes=n_skipped_episodes,
        n_samples=n_samples,
        start_offsets=start_offsets,
        num_workers=num_workers,
    )


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "--task_config",
    default="liespline_dp/config/task/lift/bspline_se3_lowdim_abs.yaml",
    show_default=True,
)
@click.option(
    "-i", "--input", default=None, help="Override input robomimic absolute lowdim hdf5 path."
)
@click.option("-o", "--output", default=None, help="Override output bspline SE(3) zarr path.")
@click.option("--max_episodes", default=None, type=int)
@click.option("--num_workers", default=None, type=int, help="Parallel demo conversion workers.")
@click.option("--overwrite", is_flag=True, default=False)
@click.pass_context
def main(ctx, task_config, input, output, max_episodes, num_workers, overwrite):
    task_cfg = load_task_config(task_config, overrides=ctx.args)
    _validate_anchor_contract_config(task_cfg)

    input_path = pathlib.Path(
        input if input is not None else task_cfg.conversion.input_path
    ).expanduser()
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

    stats = convert_robomimic_bspline_se3_dataset(
        task_cfg=task_cfg,
        input_path=input_path,
        output_path=output_path,
        max_episodes=max_episodes,
        num_workers=num_workers,
    )

    print("robomimic bspline SE(3) dataset conversion finished.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
