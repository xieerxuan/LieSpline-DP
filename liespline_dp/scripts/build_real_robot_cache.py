import click
from omegaconf import OmegaConf

from liespline_dp.common.task_config_util import load_task_config
from liespline_dp.dataset.real_robot_image_dataset import (
    build_real_robot_baseline_cache,
)


@click.command()
@click.option(
    "--task-config",
    default="liespline_dp/config/task/real_robot/baseline_image.yaml",
    show_default=True,
)
@click.option("--dataset-root", default=None)
@click.option("--cache-path", default=None)
@click.option("--overwrite", is_flag=True, default=False)
def main(task_config, dataset_root, cache_path, overwrite):
    task_cfg = load_task_config(task_config)
    OmegaConf.resolve(task_cfg)
    dataset_cfg = task_cfg.dataset
    stats = build_real_robot_baseline_cache(
        dataset_root=(dataset_cfg.dataset_root if dataset_root is None else dataset_root),
        cache_path=(dataset_cfg.cache_path if cache_path is None else cache_path),
        shape_meta=task_cfg.shape_meta,
        action_format=dataset_cfg.get(
            "action_format",
            task_cfg.get("action_format", "pose_gripper"),
        ),
        active_only=bool(dataset_cfg.get("active_only", True)),
        min_episode_steps=int(dataset_cfg.get("min_episode_steps", 2)),
        target_quat_order=dataset_cfg.get("target_quat_order", None),
        overwrite=overwrite,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
