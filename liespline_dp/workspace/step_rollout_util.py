from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_EVALUATION_SEED_BASE = 1_000_000_000


@dataclass(frozen=True)
class StepRolloutSchedule:
    first_step: int
    every_steps: int
    num_points: int
    epoch_stride: int

    @classmethod
    def from_training_cfg(cls, training_cfg) -> Optional["StepRolloutSchedule"]:
        values = {
            "first_step": training_cfg.get("rollout_first_step", None),
            "every_steps": training_cfg.get("rollout_every_steps", None),
            "num_points": training_cfg.get("rollout_num_points", None),
            "epoch_stride": training_cfg.get("rollout_epoch_stride", None),
        }
        if all(value is None for value in values.values()):
            return None
        if any(value is None for value in values.values()):
            missing = [key for key, value in values.items() if value is None]
            raise ValueError(
                "Step-based rollout configuration is incomplete; missing " + ", ".join(missing)
            )
        if training_cfg.get("rollout_every", None) is not None:
            raise ValueError(
                "training.rollout_every and step-based rollout scheduling cannot both be enabled."
            )
        parsed = {key: int(value) for key, value in values.items()}
        invalid = [key for key, value in parsed.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Step-based rollout values must be positive; invalid " + ", ".join(invalid)
            )
        return cls(**parsed)

    @property
    def last_step(self) -> int:
        return self.first_step + self.every_steps * (self.num_points - 1)

    def context_for_step(self, completed_optimizer_step: int, train_seed: int):
        delta = int(completed_optimizer_step) - self.first_step
        if delta < 0 or delta % self.every_steps != 0:
            return None
        checkpoint_index = delta // self.every_steps
        if checkpoint_index >= self.num_points:
            return None
        return {
            "checkpoint_index": checkpoint_index,
            "baseline_epoch": checkpoint_index * self.epoch_stride,
            "optimizer_step": int(completed_optimizer_step),
            "train_seed": int(train_seed),
            "evaluation_seed": (
                _EVALUATION_SEED_BASE + int(train_seed) * self.num_points + checkpoint_index
            ),
        }


@contextlib.contextmanager
def isolated_rollout_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()

    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
