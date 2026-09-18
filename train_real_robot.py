"""Train DP or LieSpline-DP on real-robot demonstrations.

See docs/real_robot_training.md for data preparation and training commands.
"""

import hydra

from train import CONFIG_DIR, run_training


@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name="bspline_se3/train_real_robot_image_workspace",
)
def main(cfg):
    if cfg.task.get("env_runner") is not None:
        raise ValueError("Select a real_robot task for this training entrypoint")
    run_training(cfg)


if __name__ == "__main__":
    main()
