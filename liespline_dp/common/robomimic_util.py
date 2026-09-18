import copy

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from scipy.spatial.transform import Rotation


def _get_robot_arm_names(robot_type):
    from robosuite.models.robots.robot_model import create_robot

    robot_model = create_robot(robot_type, idn=0)
    arms = list(robot_model.arms)
    if not arms:
        raise ValueError(f"Robot {robot_type!r} does not declare any arms")
    return arms


def normalize_robosuite_controller_metadata(env_meta, absolute_action):
    """Convert legacy Robosuite controller metadata to the 1.5 schema."""
    from robosuite.controllers.composite.composite_controller_factory import (
        refactor_composite_controller_config,
    )

    normalized = copy.deepcopy(env_meta)
    env_kwargs = normalized.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ValueError("Robosuite metadata must contain an env_kwargs mapping")

    robots = env_kwargs.get("robots")
    if isinstance(robots, str):
        robot_types = [robots]
    elif isinstance(robots, (list, tuple)) and robots:
        robot_types = list(robots)
    else:
        raise ValueError("Robosuite metadata must declare at least one robot")

    controller_configs = env_kwargs.get("controller_configs")
    if isinstance(controller_configs, (list, tuple)):
        if len(controller_configs) != len(robot_types):
            raise ValueError(
                "controller_configs must have one entry per robot: "
                f"got {len(controller_configs)} for {len(robot_types)} robots"
            )
        configs = list(controller_configs)
        return_list = True
    elif isinstance(controller_configs, dict):
        configs = [controller_configs] * len(robot_types)
        return_list = False
    else:
        raise ValueError("Robosuite metadata must contain controller_configs")

    normalized_configs = []
    input_type = "absolute" if absolute_action else "delta"
    for robot_type, controller_config in zip(robot_types, configs):
        if not isinstance(controller_config, dict):
            raise ValueError(f"Controller config for {robot_type!r} must be a mapping")
        arms = _get_robot_arm_names(robot_type)
        composite_config = refactor_composite_controller_config(
            copy.deepcopy(controller_config),
            robot_type=robot_type,
            arms=arms,
        )
        body_parts = composite_config.get("body_parts")
        if not isinstance(body_parts, dict):
            raise ValueError(f"Controller config for {robot_type!r} has no body_parts mapping")
        for arm in arms:
            arm_config = body_parts.get(arm)
            if not isinstance(arm_config, dict):
                raise ValueError(f"Controller config for {robot_type!r} has no {arm!r} arm")
            controller_type = arm_config.get("type")
            if controller_type != "OSC_POSE":
                raise ValueError(
                    "Robomimic simulation requires OSC_POSE arm controllers; "
                    f"{robot_type!r} {arm!r} uses {controller_type!r}"
                )
            arm_config.pop("control_delta", None)
            arm_config["input_type"] = input_type
            if absolute_action:
                arm_config["input_ref_frame"] = "world"
        normalized_configs.append(composite_config)

    env_kwargs["controller_configs"] = normalized_configs if return_list else normalized_configs[0]
    return normalized


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name="bc"):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        delta_env_meta = normalize_robosuite_controller_metadata(env_meta, absolute_action=False)
        abs_env_meta = normalize_robosuite_controller_metadata(env_meta, absolute_action=True)

        env = EnvUtils.create_env_from_metadata(
            env_meta=delta_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        for robot in abs_env.env.robots:
            if len(robot.arms) != 1:
                raise NotImplementedError(
                    "Absolute action conversion supports single-arm robots only"
                )
            controller = robot.part_controllers[robot.arms[0]]
            if controller.input_type != "absolute":
                raise RuntimeError("Absolute Robosuite controller was not configured")

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.file["data"])

    def convert_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # This converter supports one 6D OSC pose plus one gripper command
        # per single-arm robot.
        expected_action_dim = 7 * len(self.env.env.robots)
        if actions.shape[-1] != expected_action_dim:
            raise ValueError(
                f"Expected action dimension {expected_action_dim}, got {actions.shape[-1]}"
            )
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[..., [-1]]
        for i in range(len(states)):
            _ = env.reset_to({"states": states[i]})

            for idx, robot in enumerate(env.env.robots):
                if len(robot.arms) != 1:
                    raise NotImplementedError(
                        "Absolute action conversion supports single-arm robots only"
                    )
                if stacked_actions.shape[-1] != 7:
                    raise ValueError(
                        "Expected 7 action values per single-arm robot (6D OSC pose and 1D gripper)"
                    )

                robot.control(stacked_actions[i, idx], policy_step=True)

                controller = robot.part_controllers[robot.arms[0]]
                action_goal_pos[i, idx] = controller.goal_pos
                action_goal_ori[i, idx] = Rotation.from_matrix(controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate(
            [action_goal_pos, action_goal_ori, action_gripper], axis=-1
        )
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
        robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, metric_skip_steps=eval_skip_steps
        )
        abs_error_info = self.evaluate_rollout_error(
            abs_env,
            states,
            abs_actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )

        info = {"delta_max_error": delta_error_info, "abs_max_error": abs_error_info}
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(
        env, states, actions, robot0_eef_pos, robot0_eef_quat, metric_skip_steps=1
    ):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({"states": states[0]})
        for i in range(len(states)):
            obs = env.reset_to({"states": states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()["states"])
            rollout_next_eef_pos.append(obs["robot0_eef_pos"])
            rollout_next_eef_quat.append(obs["robot0_eef_quat"])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = (
            Rotation.from_quat(robot0_eef_quat[1:])
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        )
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            "state": max_next_state_diff,
            "pos": max_next_eef_pos_dist,
            "rot": max_next_eef_rot_dist,
        }
        return info
