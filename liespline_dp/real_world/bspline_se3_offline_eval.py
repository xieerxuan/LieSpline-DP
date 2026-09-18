"""Offline validation for real-robot B-spline SE(3) policies."""

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from liespline_dp.common.bspline_se3_util import se3_pose_errors
from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.common.se3_util import se3_exp
from liespline_dp.policy.bspline_se3.bspline_se3_policy import (
    CONTROL_POINT_GRIPPER_REPRESENTATIONS,
)


def _metric_value(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def run_offline(cfg, policy, device, max_batches):
    has_gripper = bool(getattr(policy, "has_gripper", True))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation_dataset = dataset.get_validation_dataset()
    loader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
    loader_cfg["shuffle"] = False
    loader = DataLoader(validation_dataset, **loader_cfg)

    losses = []
    pose_losses = []
    gripper_losses = []
    gripper_counts = {}
    translation_errors = []
    rotation_errors = []
    latencies = []
    sampled_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = dict_apply(
                batch,
                lambda value: value.to(device, non_blocking=True),
            )
            loss = policy.compute_loss(batch)
            losses.append(_metric_value(loss))
            components = policy._last_loss_components
            pose_losses.append(_metric_value(components["pose_loss"]))
            if has_gripper:
                gripper_losses.append(_metric_value(components["gripper_loss"]))

            policy.reset()
            prefix_steps = int(cfg.task.bspline.prefix_cp_steps)
            prediction_input = {
                "obs": batch["obs"],
                "prefix_cp": batch["condition_data"][:, :prefix_steps],
            }
            if (
                has_gripper
                and policy.gripper_representation in CONTROL_POINT_GRIPPER_REPRESENTATIONS
            ):
                prediction_input["prefix_gripper_cp"] = batch["gripper_control_point"][
                    :, :prefix_steps
                ]
            started_at = time.perf_counter()
            prediction = policy.predict_action(prediction_input)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            latencies.append(time.perf_counter() - started_at)
            policy.reset()

            if has_gripper:
                metrics = policy.compute_gripper_validation_metrics(batch, prediction=prediction)
                for key, value in metrics.items():
                    gripper_counts[key] = gripper_counts.get(key, 0.0) + (_metric_value(value))

            predicted_local = prediction["pose_waypoint_local"].detach().cpu().numpy()
            target_local = batch["pose_waypoint"].detach().cpu().numpy()
            predicted_pose = se3_exp(predicted_local)
            target_pose = se3_exp(target_local)
            trans, rot = se3_pose_errors(target_pose, predicted_pose)
            translation_errors.append(trans.reshape(-1))
            rotation_errors.append(rot.reshape(-1))
            sampled_batches += 1

    if not losses:
        raise RuntimeError("Offline validation dataset produced no batches")
    count = gripper_counts.get("count", 0.0)
    result = {
        "event": "offline_summary",
        "num_batches": len(losses),
        "num_sampled_batches": sampled_batches,
        "val_loss": float(np.mean(losses)),
        "val_pose_loss": float(np.mean(pose_losses)),
        "inference_latency_cold": float(latencies[0]),
        "inference_latency_median": float(np.median(latencies)),
        "inference_latency_p95": float(np.percentile(latencies, 95)),
        "dataset_path": str(cfg.task.dataset.zarr_path),
        "train_samples": int(len(dataset)),
        "validation_samples": int(len(validation_dataset)),
    }
    if has_gripper and gripper_losses:
        result["val_gripper_loss"] = float(np.mean(gripper_losses))
    if count > 0 and "absolute_error_sum" in gripper_counts:
        normalized_mae = gripper_counts["absolute_error_sum"] / count
        normalized_rmse = np.sqrt(gripper_counts["squared_error_sum"] / count)
        gripper_range = float(cfg.task.se3.gripper_bounds[1] - cfg.task.se3.gripper_bounds[0])
        result.update(
            {
                "val_gripper_mae": float(normalized_mae),
                "val_gripper_rmse": float(normalized_rmse),
                "val_gripper_physical_mae": float(normalized_mae * gripper_range / 2.0),
                "val_gripper_physical_rmse": float(normalized_rmse * gripper_range / 2.0),
            }
        )
    if translation_errors:
        trans = np.concatenate(translation_errors)
        rot_deg = np.rad2deg(np.concatenate(rotation_errors))
        result.update(
            {
                "waypoint_translation_mean": float(np.mean(trans)),
                "waypoint_translation_p95": float(np.percentile(trans, 95)),
                "waypoint_rotation_deg_mean": float(np.mean(rot_deg)),
                "waypoint_rotation_deg_p95": float(np.percentile(rot_deg, 95)),
            }
        )
    return result
