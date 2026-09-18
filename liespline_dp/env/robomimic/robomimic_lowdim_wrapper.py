from typing import List, Optional

import gym
import numpy as np
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicLowdimWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        obs_keys: List[str] = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        init_state: Optional[np.ndarray] = None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        legacy_object_obs_keys: Optional[List[str]] = None,
    ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.legacy_object_obs_keys = legacy_object_obs_keys
        self.seed_state_map = dict()
        self._seed = None
        self._eef_positions = []
        self._eef_quaternions_xyzw = []

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        if self.legacy_object_obs_keys is not None and "object" in self.obs_keys:
            robosuite_obs = self.env.env._get_observations(force_update=True)
            raw_obs = dict(raw_obs)
            raw_obs["object"] = np.concatenate(
                [robosuite_obs[key] for key in self.legacy_object_obs_keys], axis=0
            )
        return np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def _reset_eef_pose_history(self, raw_obs):
        self._eef_positions = []
        self._eef_quaternions_xyzw = []
        self._append_eef_pose(raw_obs)

    def _append_eef_pose(self, raw_obs):
        self._eef_positions.append(np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64).copy())
        self._eef_quaternions_xyzw.append(
            np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64).copy()
        )

    def get_eef_pose_history(self):
        return {
            "positions": np.asarray(self._eef_positions, dtype=np.float64).copy(),
            "quaternions_xyzw": np.asarray(
                self._eef_quaternions_xyzw,
                dtype=np.float64,
            ).copy(),
        }

    def reset(self):
        if self.init_state is not None:
            # Always reset to the same recorded state. Modern Robomimic can
            # additionally restore the demonstration model and episode metadata.
            reset_state = self.init_state
            if not isinstance(reset_state, dict):
                reset_state = {"states": reset_state}
            self.env.reset_to(reset_state)
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs and start the measured primitive-step trajectory
        raw_obs = self.env.get_observation()
        self._reset_eef_pose_history(raw_obs)
        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self._append_eef_pose(raw_obs)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(mode=mode, height=h, width=w, camera_name=self.render_camera_name)

    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
