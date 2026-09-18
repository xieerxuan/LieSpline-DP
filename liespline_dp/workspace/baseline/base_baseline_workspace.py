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
from liespline_dp.workspace.scene_workspace import SceneWorkspace
from liespline_dp.workspace.step_rollout_util import (
    StepRolloutSchedule,
    isolated_rollout_rng,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class BaseBaselineWorkspace(SceneWorkspace):
    """Shared baseline optimization, validation, rollout and checkpoint loop."""

    algorithm = "baseline"
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.ema = (
            hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            if self.ema_model is not None else None
        )

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def save_ema_checkpoint(self, path=None, tag=None, use_thread=True):
        if self.ema_model is None:
            raise RuntimeError("Cannot save an EMA checkpoint when training.use_ema is false.")
        if path is None:
            if tag is None:
                tag = f"epoch={self.epoch:04d}-ema"
            path = os.path.join(self.output_dir, "checkpoints", f"{tag}.ckpt")
        return self.save_checkpoint(
            path=path,
            exclude_keys=("model", "optimizer", "ema"),
            use_thread=use_thread,
        )

    def _add_sampling_log(self, policy, task_cfg, batch, step_log):
        pred_action = policy.predict_action(batch["obs"])["action_pred"]
        gt_action = batch["action"]
        if "valid_mask" in batch:
            valid_mask = batch["valid_mask"].bool().unsqueeze(-1)
            squared_error = (pred_action - gt_action).square()
            denominator = valid_mask.sum().clamp_min(1) * int(gt_action.shape[-1])
            mse = (squared_error * valid_mask).sum() / denominator
        else:
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        step_log["train_action_mse_error"] = mse.item()

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        self._check_dataset(dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
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
            lr_num_epochs = int(cfg.training.get("lr_num_epochs") or cfg.training.num_epochs)
            lr_num_training_steps = (
                len(train_dataloader) * lr_num_epochs
            ) // cfg.training.gradient_accumulate_every
        else:
            lr_num_training_steps = int(lr_num_training_steps)
        if lr_num_training_steps <= 0:
            raise ValueError("training.lr_num_training_steps must be positive.")

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=lr_num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # Reuse the updater whose schedule and counter are restored with the model.
        ema = self.ema

        # configure env runner
        env_runner = None
        if (
            cfg.training.rollout_every is not None or step_rollout_schedule is not None
        ) and cfg.task.get("env_runner") is not None:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
            self._check_env_runner(env_runner)
        if step_rollout_schedule is not None:
            if env_runner is None:
                raise ValueError("Step-based rollout scheduling requires task.env_runner.")
            if not env_runner.supports_step_rollout_context:
                raise ValueError(
                    f"{type(env_runner).__name__} does not support step-based "
                    "rollout evaluation context."
                )

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        ema_checkpoint_every = cfg.checkpoint.get("ema_checkpoint_every", None)
        if ema_checkpoint_every is not None:
            ema_checkpoint_every = int(ema_checkpoint_every)
            if ema_checkpoint_every <= 0:
                raise ValueError("checkpoint.ema_checkpoint_every must be positive.")
            if not cfg.training.use_ema:
                raise ValueError("Periodic EMA checkpoints require training.use_ema=true.")

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
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

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(loop_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

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
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                        else:
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (
                    env_runner is not None
                    and cfg.training.rollout_every is not None
                    and self.epoch % cfg.training.rollout_every == 0
                ):
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if cfg.training.val_every is not None and self.epoch % cfg.training.val_every == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                    cfg.training.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (
                    cfg.training.sample_every is not None
                    and self.epoch % cfg.training.sample_every == 0
                ):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch, lambda x: x.to(device, non_blocking=True)
                        )
                        self._add_sampling_log(policy, cfg.task, batch, step_log)

                # Persist completed progress, while keeping the zero-based epoch
                # used by checkpoint schedules, filenames and training logs.
                completed_epoch = self.epoch
                self.global_step += 1
                self.epoch += 1

                # checkpoint
                if (
                    cfg.training.checkpoint_every is not None
                    and completed_epoch % cfg.training.checkpoint_every == 0
                ):
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = None
                    if cfg.checkpoint.topk.monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                if (
                    ema_checkpoint_every is not None
                    and completed_epoch > 0
                    and completed_epoch % ema_checkpoint_every == 0
                ):
                    self.save_ema_checkpoint(tag=f"epoch={completed_epoch:04d}-ema")
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=step_log["global_step"])
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
