import collections
import csv
import functools
import json
import math
import os
import pathlib
import uuid

import dill
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import tqdm
import wandb

from liespline_dp.common.inference_scheduler import (
    EarlyInferenceScheduler,
    resolve_n_infer_steps,
    run_early_inference_control_loop,
)
from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.common.robomimic_util import (
    normalize_robosuite_controller_metadata,
)
from liespline_dp.common.trajectory_metrics import (
    eef_trajectory_jerk_metrics,
)
from liespline_dp.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from liespline_dp.env_runner.base_lowdim_runner import BaseLowdimRunner
from liespline_dp.gym_util.async_vector_env import AsyncVectorEnv
from liespline_dp.gym_util.multistep_wrapper import MultiStepWrapper
from liespline_dp.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from liespline_dp.model.common.rotation_transformer import RotationTransformer
from liespline_dp.policy.base_policy import BasePolicy


def create_env(env_meta, obs_keys, enable_render=False):
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=False,
    )
    return env


def _get_eef_pose_history(env):
    if not isinstance(env, MultiStepWrapper):
        raise TypeError(f"Expected MultiStepWrapper, got {type(env).__name__}")
    video_env = env.env
    lowdim_env = video_env.env
    if not isinstance(lowdim_env, RobomimicLowdimWrapper):
        raise TypeError(
            f"Expected RobomimicLowdimWrapper inside video wrapper, got {type(lowdim_env).__name__}"
        )
    return lowdim_env.get_eef_pose_history()


def _uses_legacy_square_object_order(env_meta):
    if env_meta.get("env_name") != "NutAssemblySquare":
        return False
    env_version = env_meta.get("env_version")
    if env_version is None:
        return True
    try:
        major, minor = str(env_version).split(".")[:2]
        return (int(major), int(minor)) < (1, 5)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid Robosuite env_version: {env_version!r}")


def _validate_observation_contract(
    dataset_path,
    env_meta,
    obs_keys,
    legacy_object_obs_keys,
    atol=1e-5,
):
    with h5py.File(dataset_path, "r") as dataset:
        demo_keys = sorted(
            dataset["data"].keys(),
            key=lambda key: int(key.split("_")[-1]),
        )
        if not demo_keys:
            raise RuntimeError(f"No demonstrations found in {dataset_path}")
        demo = dataset[f"data/{demo_keys[0]}"]
        length = int(demo["states"].shape[0])
        sample_indices = sorted({0, length // 2, length - 1})
        states = [demo["states"][idx] for idx in sample_indices]
        expected = [
            np.concatenate([demo[f"obs/{key}"][idx] for key in obs_keys], axis=-1)
            for idx in sample_indices
        ]

    probe_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
    max_error = 0.0
    try:
        probe_env.env.hard_reset = False
        for state, expected_obs in zip(states, expected):
            probe_env.reset_to({"states": state})
            raw_obs = probe_env.get_observation()
            if legacy_object_obs_keys is not None and "object" in obs_keys:
                raw_obs = dict(raw_obs)
                robosuite_obs = probe_env.env._get_observations(force_update=True)
                raw_obs["object"] = np.concatenate(
                    [robosuite_obs[key] for key in legacy_object_obs_keys], axis=0
                )
            actual_obs = np.concatenate([raw_obs[key] for key in obs_keys], axis=-1)
            if expected_obs.shape != actual_obs.shape:
                max_error = math.inf
                break
            max_error = max(
                max_error,
                float(np.max(np.abs(expected_obs - actual_obs))),
            )
    finally:
        try:
            probe_env.close()
        except Exception:
            pass
    if max_error > atol:
        raise RuntimeError(
            "Lowdim observation contract mismatch for "
            f"{dataset_path}: max_abs_error={max_error:.6g}, atol={atol}. "
            "Regenerate the lowdim observations with the documented data "
            "preparation command before training."
        )


def _compute_render_slot_mask(init_render_enabled, n_envs):
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}")
    slot_mask = [False] * n_envs
    for init_idx, enable_render in enumerate(init_render_enabled):
        if enable_render:
            slot_mask[init_idx % n_envs] = True
    return slot_mask


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    supports_step_rollout_context = True

    def __init__(
        self,
        output_dir,
        dataset_path,
        obs_keys,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        n_infer_steps=None,
        prediction_alignment="observation",
        render_hw=(256, 256),
        render_camera_name="agentview",
        fps=10,
        crf=22,
        control_freq=None,
        past_action=False,
        abs_action=False,
        action_rotation_rep="rotation_6d",
        tqdm_interval_sec=5.0,
        n_envs=None,
        n_latency_steps=None,
        measure_eef_jerk=False,
        save_eef_trajectories=False,
        task_name=None,
        method_name=None,
        train_seed=None,
    ):
        """
        Inference starts after n_infer_steps of the active plan. Its remaining
        actions run during inference, then a boundary-aligned plan is joined.
        """

        super().__init__(output_dir)
        n_infer_steps = resolve_n_infer_steps(
            n_action_steps=n_action_steps,
            n_infer_steps=n_infer_steps,
            n_latency_steps=n_latency_steps,
        )
        action_scheduler = EarlyInferenceScheduler(
            n_action_steps=n_action_steps,
            n_infer_steps=n_infer_steps,
            prediction_alignment=prediction_alignment,
        )
        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = os.path.expanduser(dataset_path)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta = normalize_robosuite_controller_metadata(env_meta, absolute_action=abs_action)
        legacy_object_obs_keys = None
        if _uses_legacy_square_object_order(env_meta):
            legacy_object_obs_keys = [
                "SquareNut_pos",
                "SquareNut_quat",
                "SquareNut_to_robot0_eef_pos",
                "SquareNut_to_robot0_eef_quat",
            ]
        if env_meta.get("env_version") is not None:
            _validate_observation_contract(
                dataset_path=dataset_path,
                env_meta=env_meta,
                obs_keys=obs_keys,
                legacy_object_obs_keys=legacy_object_obs_keys,
            )
        dataset_control_freq = env_meta["env_kwargs"].get("control_freq", 20)
        if control_freq is None:
            control_freq = dataset_control_freq
        control_freq = float(control_freq)
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        env_meta["env_kwargs"]["control_freq"] = control_freq
        steps_per_render = max(int(round(control_freq / fps)), 1)

        rotation_transformer = None
        if abs_action and action_rotation_rep != "axis_angle":
            rotation_transformer = RotationTransformer("axis_angle", action_rotation_rep)

        def env_fn(enable_render=False):
            robomimic_env = create_env(
                env_meta=env_meta,
                obs_keys=obs_keys,
                enable_render=enable_render,
            )
            # hard reset doesn't influence lowdim env
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name,
                        legacy_object_obs_keys=legacy_object_obs_keys,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        env_init_render_enabled = list()
        padding_init_fn_dill = None

        # train
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", uuid.uuid4().hex + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train/")
                env_init_fn_dills.append(dill.dumps(init_fn))
                env_init_render_enabled.append(enable_render)

                if padding_init_fn_dill is None:

                    def padding_init_fn(env, init_state=init_state):
                        assert isinstance(env.env, VideoRecordingWrapper)
                        env.env.video_recoder.stop()
                        env.env.file_path = None
                        assert isinstance(env.env.env, RobomimicLowdimWrapper)
                        env.env.env.init_state = init_state

                    padding_init_fn_dill = dill.dumps(padding_init_fn)

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath("media", uuid.uuid4().hex + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))
            env_init_render_enabled.append(enable_render)

            if padding_init_fn_dill is None:

                def padding_init_fn(env, seed=seed):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = None
                    env.seed(seed)

                padding_init_fn_dill = dill.dumps(padding_init_fn)

        if not env_init_fn_dills:
            raise ValueError("At least one train or test rollout is required")

        render_slot_mask = _compute_render_slot_mask(env_init_render_enabled, n_envs)
        env_fns = [
            functools.partial(env_fn, enable_render=enable_render)
            for enable_render in render_slot_mask
        ]
        dummy_env_fn = functools.partial(env_fn, enable_render=False)
        # MuJoCo workers must not inherit the initialized parent process.
        context = "spawn"
        env = AsyncVectorEnv(
            env_fns,
            dummy_env_fn=dummy_env_fn,
            context=context,
            filter_known_worker_import_notices=True,
        )

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.padding_init_fn_dill = padding_init_fn_dill
        self.render_slot_mask = render_slot_mask
        self.control_freq = control_freq
        self.dataset_control_freq = dataset_control_freq
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_infer_steps = n_infer_steps
        self.n_latency_steps = action_scheduler.n_latency_steps
        self.action_scheduler = action_scheduler
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.action_rotation_rep = action_rotation_rep
        self.tqdm_interval_sec = tqdm_interval_sec
        self.measure_eef_jerk = bool(measure_eef_jerk or save_eef_trajectories)
        self.save_eef_trajectories = bool(save_eef_trajectories)
        self.task_name = None if task_name is None else str(task_name)
        self.method_name = None if method_name is None else str(method_name)
        self.train_seed = None if train_seed is None else int(train_seed)

    def _eef_episode_results(self, histories, rewards):
        rows = []
        trajectories = []
        for idx, (prefix, seed) in enumerate(zip(self.env_prefixs, self.env_seeds)):
            if prefix != "test/":
                continue
            history = histories[idx]
            positions = np.asarray(history["positions"], dtype=np.float64)
            quaternions = np.asarray(history["quaternions_xyzw"], dtype=np.float64)
            success = bool(np.max(rewards[idx]) > 0)
            metrics = eef_trajectory_jerk_metrics(
                positions,
                quaternions,
                self.control_freq,
            )
            rows.append(
                {
                    "task": self.task_name or self.env_meta["env_name"],
                    "method": self.method_name or "unknown",
                    "train_seed": "" if self.train_seed is None else self.train_seed,
                    "eval_seed": int(seed),
                    "success": int(success),
                    "pose_count": len(positions),
                    "control_freq_hz": self.control_freq,
                    **metrics,
                }
            )
            trajectories.append((positions, quaternions))
        return rows, trajectories

    def _write_eef_trajectory_results(self, histories, rewards):
        output_dir = pathlib.Path(self.output_dir)
        trajectory_dir = output_dir / "trajectories"
        if self.save_eef_trajectories:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "task",
            "method",
            "train_seed",
            "eval_seed",
            "success",
            "pose_count",
            "control_freq_hz",
            "trajectory_path",
            "trans_jerk_m_s3",
            "rot_jerk_rad_s3",
        )
        rows, trajectories = self._eef_episode_results(histories, rewards)
        for row, (positions, quaternions) in zip(rows, trajectories):
            trajectory_path = ""
            if self.save_eef_trajectories:
                path = trajectory_dir / f"test_seed_{row['eval_seed']}.npz"
                np.savez_compressed(
                    path,
                    positions=positions,
                    quaternions_xyzw=quaternions,
                    control_freq_hz=np.asarray(self.control_freq, dtype=np.float64),
                    eval_seed=np.asarray(row["eval_seed"], dtype=np.int64),
                    success=np.asarray(row["success"], dtype=np.bool_),
                )
                trajectory_path = str(path.relative_to(output_dir))
            row["trajectory_path"] = trajectory_path

        result_path = output_dir / "episodes.csv"
        with result_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def _write_periodic_eef_results(
        self,
        evaluation_context,
        histories,
        rewards,
    ):
        required_context = {
            "checkpoint_index",
            "baseline_epoch",
            "optimizer_step",
            "train_seed",
            "evaluation_seed",
        }
        missing = required_context - set(evaluation_context)
        if missing:
            raise ValueError("Evaluation context is missing " + ", ".join(sorted(missing)))
        context = {key: int(evaluation_context[key]) for key in required_context}
        if self.train_seed is not None and context["train_seed"] != self.train_seed:
            raise ValueError("Evaluation context training seed does not match runner seed.")

        output_dir = pathlib.Path(self.output_dir)
        aggregate_path = output_dir / "rollout_evaluations.jsonl"
        if aggregate_path.exists():
            with aggregate_path.open() as stream:
                existing = [json.loads(line) for line in stream if line.strip()]
            for item in existing:
                if (
                    int(item["checkpoint_index"]) == context["checkpoint_index"]
                    or int(item["optimizer_step"]) == context["optimizer_step"]
                ):
                    raise RuntimeError(
                        "Periodic rollout checkpoint index or optimizer step already exists."
                    )

        point_name = (
            f"checkpoint-{context['checkpoint_index']:04d}-step-{context['optimizer_step']}"
        )
        point_dir = output_dir / "rollouts" / point_name
        point_dir.mkdir(parents=True, exist_ok=False)

        rows, trajectories = self._eef_episode_results(histories, rewards)
        if not rows:
            raise RuntimeError("Periodic EEF evaluation requires at least one test episode.")
        trajectory_path = ""
        if self.save_eef_trajectories:
            trajectory_file = point_dir / "trajectories.npz"
            episode_ends = np.cumsum(
                [len(positions) for positions, _ in trajectories], dtype=np.int64
            )
            np.savez_compressed(
                trajectory_file,
                positions=np.concatenate([positions for positions, _ in trajectories], axis=0),
                quaternions_xyzw=np.concatenate(
                    [quaternions for _, quaternions in trajectories], axis=0
                ),
                episode_ends=episode_ends,
                eval_seeds=np.asarray([row["eval_seed"] for row in rows], dtype=np.int64),
                success=np.asarray([row["success"] for row in rows], dtype=np.bool_),
                control_freq_hz=np.asarray(self.control_freq, dtype=np.float64),
            )
            trajectory_path = str(trajectory_file.relative_to(output_dir))

        fieldnames = (
            "checkpoint_index",
            "baseline_epoch",
            "optimizer_step",
            "task",
            "method",
            "train_seed",
            "eval_seed",
            "success",
            "pose_count",
            "control_freq_hz",
            "trajectory_path",
            "trajectory_episode_index",
            "trans_jerk_m_s3",
            "rot_jerk_rad_s3",
        )
        for episode_index, row in enumerate(rows):
            row.update(
                {
                    "checkpoint_index": context["checkpoint_index"],
                    "baseline_epoch": context["baseline_epoch"],
                    "optimizer_step": context["optimizer_step"],
                    "trajectory_path": trajectory_path,
                    "trajectory_episode_index": (episode_index if trajectory_path else ""),
                }
            )
        with (point_dir / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        success_count = int(sum(row["success"] for row in rows))
        metrics = {
            **context,
            "task": rows[0]["task"],
            "method": rows[0]["method"],
            "eval_start_seed": min(row["eval_seed"] for row in rows),
            "eval_end_seed": max(row["eval_seed"] for row in rows),
            "episode_count": len(rows),
            "success_count": success_count,
            "test_mean_score": success_count / len(rows),
            "jerk_episode_count": len(rows),
            "mean_trans_jerk_m_s3": float(np.mean([row["trans_jerk_m_s3"] for row in rows])),
            "mean_rot_jerk_rad_s3": float(np.mean([row["rot_jerk_rad_s3"] for row in rows])),
            "control_freq_hz": self.control_freq,
            "rollout_path": str(point_dir.relative_to(output_dir)),
        }
        (point_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        with aggregate_path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        return rows, metrics

    def run(self, policy: BasePolicy, evaluation_context=None):
        if evaluation_context is not None and not self.measure_eef_jerk:
            raise ValueError("Periodic rollout evaluation requires measure_eef_jerk=true.")
        device = policy.device
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_eef_histories = [None] * n_inits if self.measure_eef_jerk else None

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.padding_init_fn_dill] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            policy.reset()

            env_name = self.env_meta["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Lowdim {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            def predict_plan(current_obs, action_history):
                np_obs_dict = {"obs": current_obs.astype(np.float32)}
                if self.past_action and action_history is not None and self.n_obs_steps > 1:
                    np_obs_dict["past_action"] = action_history[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)
                if action_history is not None:
                    np_obs_dict["prev_action"] = action_history[:, -1].astype(np.float32)

                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                return np_action_dict["action"]

            def step_action_segment(action):
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                next_obs, reward, segment_done, info = env.step(env_action)
                pbar.update(action.shape[1])
                return next_obs, np.all(segment_done)

            obs, _ = run_early_inference_control_loop(
                scheduler=self.action_scheduler,
                initial_observation=obs,
                predict_plan=predict_plan,
                step_action=step_action_segment,
                action_history_steps=max(self.n_obs_steps - 1, 1),
            )
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[this_local_slice]
            if all_eef_histories is not None:
                histories = env.call(
                    "run_dill_function",
                    dill.dumps(_get_eef_pose_history),
                )
                all_eef_histories[this_global_slice] = histories[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # Aggregate every evaluated initial condition across all vectorized batches.
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            log_data[name] = float(np.mean(value))
            if prefix == "test/":
                log_data["test/success_count"] = int(np.sum(value))
                log_data["test/episode_count"] = len(value)

        if all_eef_histories is not None:
            if evaluation_context is None:
                rows = self._write_eef_trajectory_results(
                    all_eef_histories,
                    all_rewards,
                )
                if not rows:
                    raise RuntimeError("EEF jerk measurement requires at least one test episode.")
                metrics = {
                    "mean_trans_jerk_m_s3": float(
                        np.mean([row["trans_jerk_m_s3"] for row in rows])
                    ),
                    "mean_rot_jerk_rad_s3": float(
                        np.mean([row["rot_jerk_rad_s3"] for row in rows])
                    ),
                }
            else:
                rows, metrics = self._write_periodic_eef_results(
                    evaluation_context,
                    all_eef_histories,
                    all_rewards,
                )
                log_data.update(
                    {
                        "evaluation/checkpoint_index": metrics["checkpoint_index"],
                        "evaluation/baseline_epoch": metrics["baseline_epoch"],
                        "evaluation/optimizer_step": metrics["optimizer_step"],
                        "evaluation/evaluation_seed": metrics["evaluation_seed"],
                    }
                )
            if not rows:
                raise RuntimeError("EEF jerk measurement requires at least one test episode.")
            log_data["test/trajectory_episode_count"] = len(rows)
            log_data["test/mean_trans_jerk_m_s3"] = metrics["mean_trans_jerk_m_s3"]
            log_data["test/mean_rot_jerk_rad_s3"] = metrics["mean_rot_jerk_rad_s3"]

        return log_data

    def undo_transform_action(self, action):
        if self.rotation_transformer is None:
            return action

        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
