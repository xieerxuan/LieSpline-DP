import collections
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
from liespline_dp.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from liespline_dp.env_runner.base_image_runner import BaseImageRunner
from liespline_dp.gym_util.async_vector_env import AsyncVectorEnv
from liespline_dp.gym_util.multistep_wrapper import MultiStepWrapper
from liespline_dp.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from liespline_dp.model.common.rotation_transformer import RotationTransformer
from liespline_dp.policy.base_policy import BasePolicy


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
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
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        action_rotation_rep="rotation_6d",
        tqdm_interval_sec=5.0,
        n_envs=None,
        n_latency_steps=None,
    ):
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

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta = normalize_robosuite_controller_metadata(env_meta, absolute_action=abs_action)
        env_meta["env_kwargs"]["use_object_obs"] = False
        control_freq = float(env_meta["env_kwargs"].get("control_freq", 20))
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        steps_per_render = max(int(round(control_freq / fps)), 1)

        rotation_transformer = None
        if abs_action and action_rotation_rep != "axis_angle":
            rotation_transformer = RotationTransformer("axis_angle", action_rotation_rep)

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
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

        # Initialize spaces without opening an OpenGL context in the parent.
        # EGL workers use the spawn context so they do not inherit parent-side
        # MuJoCo / EGL global state.
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, shape_meta=shape_meta, enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
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

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

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
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train/")
                env_init_fn_dills.append(dill.dumps(init_fn))

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
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(
            env_fns,
            dummy_env_fn=dummy_env_fn,
            context="spawn",
            filter_known_worker_import_notices=True,
        )

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.control_freq = control_freq
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

    def run(self, policy: BasePolicy):
        device = policy.device
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            policy.reset()

            env_name = self.env_meta["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            def predict_plan(current_obs, action_history):
                np_obs_dict = dict(current_obs)
                if self.past_action and action_history is not None and self.n_obs_steps > 1:
                    np_obs_dict["past_action"] = action_history[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)

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
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # Aggregate every evaluated initial condition across all vectorized batches.
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
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
            value = np.mean(value)
            log_data[name] = value

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
