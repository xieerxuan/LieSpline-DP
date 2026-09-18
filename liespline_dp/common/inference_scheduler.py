from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np


class PredictionAlignment(str, Enum):
    """Time origin used by a policy prediction."""

    OBSERVATION = "observation"
    PLAN_BOUNDARY = "plan_boundary"


def resolve_n_infer_steps(
    n_action_steps: int,
    n_infer_steps: Optional[int] = None,
    n_latency_steps: Optional[int] = None,
) -> int:
    """Resolve the trigger step, accepting latency only for old snapshots."""
    n_action_steps = int(n_action_steps)
    if n_infer_steps is None:
        if n_latency_steps is None:
            return n_action_steps
        return n_action_steps - int(n_latency_steps)
    n_infer_steps = int(n_infer_steps)
    if n_latency_steps is not None and n_action_steps - n_infer_steps != int(n_latency_steps):
        raise ValueError(
            "n_latency_steps must equal n_action_steps - n_infer_steps, got "
            f"{n_latency_steps} != {n_action_steps} - {n_infer_steps}"
        )
    return n_infer_steps


def compute_inference_trigger_monotonic(
    local_queue_deadline_monotonic: float,
    n_action_steps: int,
    n_infer_steps: int,
    frequency: float,
    trigger_advance_s: float = 0.0,
) -> float:
    latency_steps = int(n_action_steps) - int(n_infer_steps)
    if latency_steps < 0:
        raise ValueError("n_infer_steps cannot exceed n_action_steps")
    if float(frequency) <= 0.0:
        raise ValueError("frequency must be positive")
    trigger_advance_s = float(trigger_advance_s)
    if not np.isfinite(trigger_advance_s) or trigger_advance_s < 0.0:
        raise ValueError("trigger_advance_s must be non-negative and finite")
    return (
        float(local_queue_deadline_monotonic) - latency_steps / float(frequency) - trigger_advance_s
    )


def build_boundary_offsets(
    n_action_steps: int,
    frequency: float,
) -> np.ndarray:
    if int(n_action_steps) < 1:
        raise ValueError("n_action_steps must be positive")
    if float(frequency) <= 0.0:
        raise ValueError("frequency must be positive")
    dt = 1.0 / float(frequency)
    return dt * np.arange(
        1,
        int(n_action_steps) + 1,
        dtype=np.float64,
    )


@dataclass(frozen=True)
class EarlyInferenceScheduler:
    """Build boundary-aligned plans while inference overlaps the active plan.

    The control loop only executes and joins plans containing exactly
    ``n_action_steps`` actions. Policies whose predictions start at the
    observation time are adapted by dropping the actions that elapse during
    inference. B-spline policies already predict from the next plan boundary,
    so their complete prediction is joined without trimming.
    """

    n_action_steps: int
    n_infer_steps: int
    prediction_alignment: PredictionAlignment = PredictionAlignment.OBSERVATION

    def __post_init__(self):
        n_action_steps = int(self.n_action_steps)
        n_infer_steps = int(self.n_infer_steps)
        if n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {self.n_action_steps}")
        if not 1 <= n_infer_steps <= n_action_steps:
            raise ValueError(
                "n_infer_steps must satisfy "
                f"1 <= n_infer_steps <= n_action_steps, got "
                f"{self.n_infer_steps} and {self.n_action_steps}"
            )
        try:
            alignment = PredictionAlignment(self.prediction_alignment)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in PredictionAlignment)
            raise ValueError(
                f"prediction_alignment must be one of {choices}, got {self.prediction_alignment!r}"
            ) from exc
        object.__setattr__(self, "n_action_steps", n_action_steps)
        object.__setattr__(self, "n_infer_steps", n_infer_steps)
        object.__setattr__(self, "prediction_alignment", alignment)

    @property
    def n_latency_steps(self) -> int:
        return int(self.n_action_steps) - int(self.n_infer_steps)

    @property
    def required_prediction_steps(self) -> int:
        if self.prediction_alignment is PredictionAlignment.OBSERVATION:
            return int(self.n_action_steps) + self.n_latency_steps
        return int(self.n_action_steps)

    def _validate_prediction(
        self,
        prediction: np.ndarray,
        required_steps: int,
    ) -> np.ndarray:
        prediction = np.asarray(prediction)
        if prediction.ndim < 2:
            raise ValueError(
                "Predicted action must have batch and time dimensions, got "
                f"shape {prediction.shape}"
            )
        if prediction.shape[1] < required_steps:
            raise ValueError(
                "Policy prediction must contain at least "
                f"{required_steps} action steps for "
                f"{self.prediction_alignment.value!r} alignment, got "
                f"{prediction.shape[1]}"
            )
        return prediction

    def bootstrap(self, prediction: np.ndarray) -> np.ndarray:
        """Create the first plan before an overlapping plan exists."""
        prediction = self._validate_prediction(
            prediction,
            required_steps=int(self.n_action_steps),
        )
        return prediction[:, : self.n_action_steps]

    def split_active(self, active_plan: np.ndarray):
        """Split a plan at the observation/inference launch boundary."""
        active_plan = np.asarray(active_plan)
        if active_plan.ndim < 2 or active_plan.shape[1] != self.n_action_steps:
            raise ValueError(
                f"Active plan must contain exactly {self.n_action_steps} steps, "
                f"got shape {active_plan.shape}"
            )
        split = int(self.n_infer_steps)
        return active_plan[:, :split], active_plan[:, split:]

    def prepare_pending(self, prediction: np.ndarray) -> np.ndarray:
        """Adapt a raw policy prediction into a directly joinable plan."""
        prediction = self._validate_prediction(
            prediction,
            required_steps=self.required_prediction_steps,
        )
        start = 0
        if self.prediction_alignment is PredictionAlignment.OBSERVATION:
            start = self.n_latency_steps
        end = start + int(self.n_action_steps)
        return prediction[:, start:end]


def run_early_inference_control_loop(
    scheduler: EarlyInferenceScheduler,
    initial_observation,
    predict_plan: Callable,
    step_action: Callable,
    action_history_steps: int = 1,
) -> Tuple[object, Optional[np.ndarray]]:
    """Run a discrete-time early-inference loop.

    Inference is synchronous in elapsed monotonic time, but it only observes state at
    the launch boundary. The old plan tail is then executed before the pending,
    boundary-aligned plan is joined.
    """
    if int(action_history_steps) < 0:
        raise ValueError(f"action_history_steps must be non-negative, got {action_history_steps}")

    observation = initial_observation
    action_history = None

    def append_history(action):
        nonlocal action_history
        if action.shape[1] == 0 or action_history_steps == 0:
            return
        if action_history is None:
            action_history = action
        else:
            action_history = np.concatenate([action_history, action], axis=1)
        action_history = action_history[:, -action_history_steps:]

    initial_prediction = predict_plan(observation, action_history)
    active_plan = scheduler.bootstrap(initial_prediction)

    done = False
    while not done:
        before_inference, during_inference = scheduler.split_active(active_plan)

        if before_inference.shape[1] > 0:
            observation, done = step_action(before_inference)
            append_history(before_inference)
            if done:
                break

        pending_prediction = predict_plan(observation, action_history)
        pending_plan = scheduler.prepare_pending(pending_prediction)

        if during_inference.shape[1] > 0:
            observation, done = step_action(during_inference)
            append_history(during_inference)
            if done:
                break

        active_plan = pending_plan

    return observation, action_history
