import copy
import os
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from liespline_dp.common.checkpoint_util import TopKCheckpointManager
from liespline_dp.common.json_logger import JsonLogger
from liespline_dp.common.pytorch_util import dict_apply, optimizer_to
from liespline_dp.model.common.lr_scheduler import get_scheduler
from liespline_dp.policy.base_policy import BasePolicy
from liespline_dp.workspace.scene_workspace import SceneWorkspace
from liespline_dp.workspace.step_rollout_util import (
    StepRolloutSchedule,
    isolated_rollout_rng,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class BaseLieSplineWorkspace(SceneWorkspace):
    """Shared LieSpline-DP trainer for low-dimensional and image observations.

    Pose-control and gripper channels are optimized jointly with diffusion
    MSE loss, predicting noise or clean samples as configured by the scheduler.
    """

    algorithm = "liespline-dp"

    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.ema = (
            hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            if self.ema_model is not None else None
        )
        if "_target_" in cfg.optimizer:
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer,
                params=self.model.parameters(),
            )
        else:
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0

    def save_ema_checkpoint(self, path=None, tag=None, use_thread=True):
        """Save an inference checkpoint containing only the EMA model state."""
        if self.ema_model is None:
            raise RuntimeError("Cannot save an EMA checkpoint when training.use_ema is false.")
        if path is None:
            if tag is None:
                tag = f"epoch={self.epoch:04d}-ema"
            path = os.path.join(self.output_dir, "checkpoints", f"{tag}.ckpt")
        return self.save_checkpoint(
            path=path, exclude_keys=("model", "optimizer", "ema"), use_thread=use_thread
        )

    @staticmethod
    def _accumulate_gripper_metrics(counts, metrics):
        for key, value in metrics.items():
            counts[key] = counts.get(key, 0.0) + float(value.detach().item())

    @staticmethod
    def _finish_gripper_metrics(step_log, counts):
        count = counts.get("count", 0.0)
        if count <= 0:
            return
        if "absolute_error_sum" in counts:
            step_log["val_gripper_mae"] = counts["absolute_error_sum"] / count
            step_log["val_gripper_rmse"] = float(np.sqrt(counts["squared_error_sum"] / count))
            return

        specs = {
            "val_gripper_accuracy": ("correct", "count"),
            "val_gripper_open_accuracy": ("positive_correct", "positive_count"),
            "val_gripper_closed_accuracy": ("negative_correct", "negative_count"),
        }
        for metric_name, (numerator, denominator) in specs.items():
            metric_count = counts.get(denominator, 0.0)
            if metric_count > 0:
                step_log[metric_name] = counts.get(numerator, 0.0) / metric_count
        open_accuracy = step_log.get("val_gripper_open_accuracy")
        closed_accuracy = step_log.get("val_gripper_closed_accuracy")
        if open_accuracy is not None and closed_accuracy is not None:
            step_log["val_gripper_balanced_accuracy"] = 0.5 * (open_accuracy + closed_accuracy)

    def _get_evaluation_policy(self):
        policy = self.ema_model if self.ema_model is not None else self.model
        policy.eval()
        return policy

    def _run_validation_epoch(
        self,
        val_dataloader,
        *,
        device,
        max_val_steps=None,
    ):
        policy = self._get_evaluation_policy()
        has_gripper = bool(getattr(policy, "has_gripper", True))
        val_losses = []
        val_pose_losses = []
        val_gripper_losses = []
        counts = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = dict_apply(
                    batch,
                    lambda x: x.to(device, non_blocking=True),
                )
                val_losses.append(policy.compute_loss(batch))
                components = policy._last_loss_components
                val_pose_losses.append(components["pose_loss"])
                if has_gripper:
                    val_gripper_losses.append(components["gripper_loss"])
                    self._accumulate_gripper_metrics(
                        counts,
                        policy.compute_gripper_validation_metrics(batch),
                    )
                if max_val_steps is not None and batch_idx >= max_val_steps - 1:
                    break

        metrics = {}
        if val_losses:
            metrics["val_loss"] = torch.stack(val_losses).mean().item()
            metrics["val_pose_loss"] = torch.stack(val_pose_losses).mean().item()
            if has_gripper and val_gripper_losses:
                metrics["val_gripper_loss"] = torch.stack(val_gripper_losses).mean().item()
        if has_gripper:
            self._finish_gripper_metrics(metrics, counts)
        return metrics

    def _add_sampling_log(self, policy, task_cfg, batch, step_log):
        policy.reset()
        prefix_cp_steps = int(task_cfg.bspline.prefix_cp_steps)
        obs_dict = {
            "obs": batch["obs"],
            "prefix_cp": batch["condition_data"][:, :prefix_cp_steps],
        }
        result = policy.predict_pose_waypoint(obs_dict)
        pred_action = result["pose_trajectory_pred"]
        gt_action = batch["trajectory"]
        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
            pred_action, gt_action
        ).item()
        policy.reset()

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        task_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=False))
        OmegaConf.resolve(task_cfg)
        cfg_for_logging = copy.deepcopy(cfg)
        cfg_for_logging.task = task_cfg

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset = hydra.utils.instantiate(task_cfg.dataset)
        self._check_dataset(dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataloader = DataLoader(
            dataset.get_validation_dataset(),
            **cfg.val_dataloader,
        )

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        fixed_training_steps = cfg.training.get("num_training_steps", None)
        if fixed_training_steps is not None:
            fixed_training_steps = int(fixed_training_steps)
            if fixed_training_steps <= 0:
                raise ValueError("training.num_training_steps must be positive.")
            if int(cfg.training.gradient_accumulate_every) != 1:
                raise ValueError("Fixed-step training requires gradient_accumulate_every=1.")

        step_rollout_schedule = StepRolloutSchedule.from_training_cfg(cfg.training)
        if (
            step_rollout_schedule is not None
            and fixed_training_steps is not None
            and step_rollout_schedule.last_step > fixed_training_steps
        ):
            raise ValueError("Step-based rollout schedule extends past the fixed training budget.")

        lr_num_training_steps = cfg.training.get("lr_num_training_steps", None)
        if lr_num_training_steps is None:
            lr_num_epochs = cfg.training.get("lr_num_epochs", None)
            if lr_num_epochs is None:
                lr_num_epochs = cfg.training.num_epochs
            lr_num_training_steps = (
                len(train_dataloader) * int(lr_num_epochs)
            ) // cfg.training.gradient_accumulate_every
        else:
            lr_num_training_steps = int(lr_num_training_steps)
        if lr_num_training_steps <= 0:
            raise ValueError("training.lr_num_training_steps must be positive.")
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=lr_num_training_steps,
            last_epoch=self.global_step - 1,
        )
        # Reuse the updater whose schedule and counter are restored with the model.
        ema = self.ema

        env_runner = None
        if (
            cfg.training.rollout_every is not None or step_rollout_schedule is not None
        ) and task_cfg.get("env_runner") is not None:
            env_runner = hydra.utils.instantiate(
                task_cfg.env_runner,
                output_dir=self.output_dir,
            )
            self._check_env_runner(env_runner)
        if step_rollout_schedule is not None:
            if env_runner is None:
                raise ValueError("Step-based rollout scheduling requires task.env_runner.")
            if not getattr(env_runner, "supports_step_rollout_context", False):
                raise ValueError(
                    f"{type(env_runner).__name__} does not support step-based "
                    "rollout evaluation context."
                )

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg_for_logging, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        ema_checkpoint_every = cfg.checkpoint.get("ema_checkpoint_every", None)
        if ema_checkpoint_every is not None:
            ema_checkpoint_every = int(ema_checkpoint_every)
            if ema_checkpoint_every <= 0:
                raise ValueError("checkpoint.ema_checkpoint_every must be positive.")
            if not cfg.training.use_ema:
                raise ValueError("Periodic EMA checkpoints require training.use_ema=true.")

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            if cfg.training.rollout_every is not None:
                cfg.training.rollout_every = 1
            if cfg.training.checkpoint_every is not None:
                cfg.training.checkpoint_every = 1
            if cfg.training.val_every is not None:
                cfg.training.val_every = 1
            if cfg.training.sample_every is not None:
                cfg.training.sample_every = 1

        steps_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            steps_per_epoch = min(steps_per_epoch, int(cfg.training.max_train_steps))
        if steps_per_epoch <= 0:
            raise RuntimeError("Training dataset produced no batches.")
        if fixed_training_steps is None:
            loop_epochs = max(0, int(cfg.training.num_epochs) - self.epoch)
        else:
            remaining_steps = max(0, fixed_training_steps - self.global_step)
            loop_epochs = int(np.ceil(remaining_steps / steps_per_epoch))

        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        log_every = max(1, int(cfg.training.get("log_every", 1)))
        with JsonLogger(log_path) as json_logger:
            for _ in range(loop_epochs):
                train_losses = []
                train_pose_losses = []
                train_gripper_losses = []
                train_prefix_corruption_rates = []
                has_gripper = bool(getattr(self.model, "has_gripper", True))
                step_log = {}
                self.model.train()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        (raw_loss / cfg.training.gradient_accumulate_every).backward()
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        if ema is not None:
                            ema.step(self.model)

                        loss_value = float(raw_loss.item())
                        train_losses.append(loss_value)
                        components = self.model._last_loss_components
                        train_pose_losses.append(float(components["pose_loss"].item()))
                        if has_gripper:
                            train_gripper_losses.append(float(components["gripper_loss"].item()))
                        train_prefix_corruption_rates.append(
                            float(components["prefix_corruption_rate"].item())
                        )
                        step_log = {
                            "train_loss": loss_value,
                            "train_pose_loss": train_pose_losses[-1],
                            "train_prefix_corruption_rate": train_prefix_corruption_rates[-1],
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if has_gripper:
                            step_log["train_gripper_loss"] = train_gripper_losses[-1]

                        completed_step = self.global_step + 1
                        rollout_context = (
                            None
                            if step_rollout_schedule is None
                            else step_rollout_schedule.context_for_step(
                                completed_step,
                                int(cfg.training.seed),
                            )
                        )
                        if rollout_context is not None:
                            policy = self.ema_model if self.ema_model is not None else self.model
                            was_training = policy.training
                            policy.eval()
                            with isolated_rollout_rng(rollout_context["evaluation_seed"]):
                                runner_log = env_runner.run(
                                    policy,
                                    evaluation_context=rollout_context,
                                )
                            if was_training:
                                policy.train()
                            step_log.update(runner_log)

                        tepoch.set_postfix(loss=loss_value, refresh=False)
                        reached_epoch_limit = (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        )
                        reached_training_limit = (
                            fixed_training_steps is not None
                            and self.global_step + 1 >= fixed_training_steps
                        )
                        is_terminal_batch = (
                            batch_idx == len(train_dataloader) - 1
                            or reached_epoch_limit
                            or reached_training_limit
                        )
                        if not is_terminal_batch:
                            if rollout_context is not None or self.global_step % log_every == 0:
                                wandb_run.log(
                                    step_log,
                                    step=self.global_step,
                                )
                                json_logger.log(step_log)
                            self.global_step += 1
                        else:
                            break

                step_log.update(
                    {
                        "train_loss": float(np.mean(train_losses)),
                        "train_pose_loss": float(np.mean(train_pose_losses)),
                        "train_prefix_corruption_rate": float(
                            np.mean(train_prefix_corruption_rates)
                        ),
                    }
                )
                if has_gripper:
                    step_log["train_gripper_loss"] = float(np.mean(train_gripper_losses))

                policy = self._get_evaluation_policy()
                if (
                    env_runner is not None
                    and cfg.training.rollout_every is not None
                    and self.epoch % cfg.training.rollout_every == 0
                ):
                    step_log.update(env_runner.run(policy))

                if cfg.training.val_every is not None and self.epoch % cfg.training.val_every == 0:
                    step_log.update(
                        self._run_validation_epoch(
                            val_dataloader,
                            device=device,
                            max_val_steps=cfg.training.max_val_steps,
                        )
                    )

                if (
                    train_sampling_batch is not None
                    and cfg.training.sample_every is not None
                    and self.epoch % cfg.training.sample_every == 0
                ):
                    with torch.no_grad():
                        self._add_sampling_log(
                            policy,
                            task_cfg,
                            train_sampling_batch,
                            step_log,
                        )

                # Persist completed progress, while keeping the zero-based epoch
                # used by checkpoint schedules, filenames and training logs.
                completed_epoch = self.epoch
                self.global_step += 1
                self.epoch += 1

                if (
                    cfg.training.checkpoint_every is not None
                    and completed_epoch % cfg.training.checkpoint_every == 0
                ):
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    metric_dict = {
                        key.replace("/", "_"): value
                        for key, value in step_log.items()
                        if isinstance(value, (int, float, np.floating))
                    }
                    monitor_key = cfg.checkpoint.topk.monitor_key
                    if monitor_key in metric_dict:
                        topk_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_path is not None:
                            self.save_checkpoint(path=topk_path)

                # Periodic EMA checkpoints are independent of top-k retention.
                # Keep every numbered checkpoint for evaluation and deployment;
                # latest.ckpt remains the full state used to resume training.
                if (
                    ema_checkpoint_every is not None
                    and completed_epoch > 0
                    and completed_epoch % ema_checkpoint_every == 0
                ):
                    self.save_ema_checkpoint(tag=f"epoch={completed_epoch:04d}-ema")

                self.model.train()
                wandb_run.log(
                    step_log,
                    step=step_log["global_step"],
                )
                json_logger.log(step_log)
                if fixed_training_steps is not None and self.global_step >= fixed_training_steps:
                    break

        if fixed_training_steps is not None and self.global_step != fixed_training_steps:
            raise RuntimeError(
                "Training stopped before the requested fixed update budget: "
                f"{self.global_step} != {fixed_training_steps}."
            )
        if cfg.checkpoint.get("save_final_ema_ckpt", False):
            self.save_ema_checkpoint(
                path=os.path.join(self.output_dir, "checkpoints", "final-ema.ckpt"),
                use_thread=False,
            )
