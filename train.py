"""Train DP or LieSpline-DP using the selected task and workspace configuration."""

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
CONFIG_DIR = str(Path(__file__).parent / "liespline_dp" / "config")


def run_training(cfg):
    # Resolve time/path interpolations once, before logging and checkpointing.
    OmegaConf.resolve(cfg)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.run()


@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name="bspline_se3/train_lowdim_workspace",
)
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
