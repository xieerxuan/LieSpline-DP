POSE_GRIPPER = "pose_gripper"
POSE_ONLY = "pose_only"
REAL_ROBOT_ACTION_FORMATS = (POSE_GRIPPER, POSE_ONLY)


def normalize_real_robot_action_format(action_format=None, include_gripper=None):
    if action_format is None:
        if include_gripper is None:
            action_format = POSE_GRIPPER
        else:
            action_format = POSE_GRIPPER if bool(include_gripper) else POSE_ONLY
    action_format = str(action_format)
    if action_format not in REAL_ROBOT_ACTION_FORMATS:
        raise ValueError(
            f"action_format must be 'pose_gripper' or 'pose_only', got {action_format!r}"
        )
    if include_gripper is not None:
        expected = action_format == POSE_GRIPPER
        if bool(include_gripper) != expected:
            raise ValueError(
                "include_gripper disagrees with action_format: "
                f"{include_gripper} vs {action_format}"
            )
    return action_format


def real_robot_has_gripper(action_format=None, include_gripper=None):
    return (
        normalize_real_robot_action_format(
            action_format=action_format,
            include_gripper=include_gripper,
        )
        == POSE_GRIPPER
    )


def real_robot_model_action_dim(action_format=None, include_gripper=None):
    return 10 if real_robot_has_gripper(action_format, include_gripper) else 9
