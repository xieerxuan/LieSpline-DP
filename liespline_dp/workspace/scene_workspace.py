"""Contracts shared by the six concrete training entry points."""

import dill
import hydra

from liespline_dp.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from liespline_dp.env_runner.base_image_runner import BaseImageRunner
from liespline_dp.env_runner.base_lowdim_runner import BaseLowdimRunner
from liespline_dp.workspace.base_workspace import BaseWorkspace


class SceneWorkspace(BaseWorkspace):
    # Shared bases must be specialized by a concrete scene entry point.
    scene = None
    policy_type = None

    def __init__(self, cfg, output_dir=None):
        self.validate_config(cfg)
        super().__init__(cfg, output_dir=output_dir)

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        super().load_payload(
            payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs
        )
        ema = getattr(self, "ema", None)
        if (
            ema is not None
            and "ema" not in payload["state_dicts"]
            and "ema" not in (exclude_keys or ())
            and "ema_model" in payload["state_dicts"]
            and "ema_model" not in (exclude_keys or ())
            and "global_step" in payload["pickles"]
        ):
            # Checkpoints with state_dicts["ema_model"] and pickles["global_step"],
            # but no state_dicts["ema"], reconstruct the EMA update counter
            # from the stored batch count (one EMA update per training batch).
            # Explicit EMA updater state, when present, is loaded above.
            ema.optimization_step = int(dill.loads(payload["pickles"]["global_step"]))
            ema.decay = (
                ema.get_decay(ema.optimization_step - 1)
                if ema.optimization_step > 0 else 0.0
            )

    @classmethod
    def validate_config(cls, cfg):
        """Reject incompatible experiments before constructing models or data."""
        if cls.scene not in ("lowdim", "image", "real_robot"):
            raise TypeError(f"{cls.__name__} requires a concrete training scene")
        policy_cls = hydra.utils.get_class(cfg.policy._target_)
        if not issubclass(policy_cls, hydra.utils.get_class(cls.policy_type)):
            raise TypeError(f"{cls.__name__} requires {cls.policy_type}")
        dataset_cls = hydra.utils.get_class(cfg.task.dataset._target_)
        expected = BaseLowdimDataset if cls.scene == "lowdim" else BaseImageDataset
        if not issubclass(dataset_cls, expected):
            raise TypeError(f"{cls.__name__} requires a {expected.__name__}")
        # Import native datasets only for concrete scene validation.
        from liespline_dp.dataset.real_robot_image_dataset import RealRobotImageDataset
        from liespline_dp.dataset.real_robot_bspline_se3_image_dataset import (
            RealRobotBsplineSE3ImageDataset,
        )

        native = issubclass(dataset_cls, (RealRobotImageDataset, RealRobotBsplineSE3ImageDataset))
        runner = cfg.task.get("env_runner")
        if cls.scene == "real_robot":
            native_type = (RealRobotImageDataset if cls.algorithm == "baseline"
                           else RealRobotBsplineSE3ImageDataset)
            if not issubclass(dataset_cls, native_type):
                raise TypeError(f"{cls.__name__} requires a native robot dataset")
            rollout_keys = ("rollout_every", "rollout_first_step", "rollout_every_steps",
                            "rollout_num_points", "rollout_epoch_stride")
            if runner is not None or any(cfg.training.get(key) is not None for key in rollout_keys):
                raise ValueError("Real-robot workspaces do not support simulator rollout")
        else:
            if native:
                raise TypeError("Use the real_robot workspace for native robot datasets")
            if runner is None:
                raise ValueError(f"{cls.__name__} requires a simulation env_runner configuration")
            # Runner type is checked when instantiated, keeping optional simulator
            # dependencies out of checkpoint loading and offline training.

    def _check_dataset(self, dataset):
        expected = BaseLowdimDataset if self.scene == "lowdim" else BaseImageDataset
        if not isinstance(dataset, expected):
            raise TypeError(f"Unexpected dataset for {type(self).__name__}: {type(dataset).__name__}")

    def _check_env_runner(self, runner):
        if self.scene == "real_robot":
            raise ValueError("Real-robot workspaces do not support simulator rollout")
        expected = BaseLowdimRunner if self.scene == "lowdim" else BaseImageRunner
        if not isinstance(runner, expected):
            raise TypeError(f"Unexpected runner for {type(self).__name__}: {type(runner).__name__}")
