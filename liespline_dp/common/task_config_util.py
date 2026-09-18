from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def instantiate_task_action_pose_adapter(task_cfg, **overrides):
    """Instantiate the action-pose adapter declared by a task config."""
    try:
        adapter_cfg = task_cfg["action_pose_adapter"]
    except (KeyError, TypeError) as exc:
        raise KeyError("Task config has no action_pose_adapter") from exc
    return instantiate(adapter_cfg, **overrides)


def _prefix_task_override(override, task_prefix):
    for operator in ("++", "+", "~"):
        if override.startswith(operator):
            return f"{operator}{task_prefix}.{override[len(operator) :]}"
    return f"{task_prefix}.{override}"


def load_task_config(task_config, overrides=None):
    """Load a repository task config with Hydra composition.

    Standalone YAML files outside ``liespline_dp/config`` remain supported and
    are loaded directly. Task-relative overrides are applied before interpolation
    resolution so linked adapter and cache-contract fields stay synchronized.
    """
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    override_list = [str(value) for value in (overrides or [])]
    config_path = Path(task_config).expanduser().resolve()
    config_dir = Path(__file__).parents[1].joinpath("config").resolve()
    try:
        relative_path = config_path.relative_to(config_dir)
    except ValueError:
        config = OmegaConf.load(config_path)
        if override_list:
            config = OmegaConf.merge(
                config,
                OmegaConf.from_dotlist(override_list),
            )
        OmegaConf.resolve(config)
        return config

    config_name = relative_path.with_suffix("").as_posix()
    task_prefix = ".".join(relative_path.parent.parts)
    prefixed_overrides = [
        _prefix_task_override(override, task_prefix) for override in override_list
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(
            config_name=config_name,
            overrides=prefixed_overrides,
        )

    node = composed
    for part in relative_path.parent.parts:
        node = node[part]
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))
