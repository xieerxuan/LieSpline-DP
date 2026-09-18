"""Validation loss and absolute-action errors for the native DP baseline."""

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from liespline_dp.common.pytorch_util import dict_apply
from liespline_dp.common.se3_util import rot6d_to_mat, so3_log


def run_offline(cfg, policy, device, max_batches=0):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    loader = DataLoader(validation, **OmegaConf.to_container(cfg.val_dataloader, resolve=True))
    losses, translations, rotations, gripper_errors, latencies = [], [], [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches > 0 and index >= max_batches:
                break
            batch = dict_apply(batch, lambda value: value.to(device))
            losses.append(float(policy.compute_loss(batch)))
            policy.reset()
            started = time.perf_counter()
            prediction = policy.predict_action(batch["obs"])["action_pred"]
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - started)
            target = batch["action"]
            valid = batch.get(
                "valid_mask", torch.ones_like(target[..., 0], dtype=torch.bool)
            ).bool()
            # Only compare real demonstration labels, excluding padded targets.
            translations.append(
                torch.linalg.norm(prediction[..., :3] - target[..., :3], dim=-1)[valid]
                .cpu()
                .numpy()
            )
            target_rotation = rot6d_to_mat(target[..., 3:9])
            predicted_rotation = rot6d_to_mat(prediction[..., 3:9])
            angular_error = so3_log(target_rotation.transpose(-1, -2) @ predicted_rotation)
            rotations.append(torch.linalg.norm(angular_error, dim=-1)[valid].cpu().numpy())
            if target.shape[-1] == 10:
                gripper_errors.append(
                    (prediction[..., 9] - target[..., 9]).abs()[valid].cpu().numpy()
                )
    policy.reset()
    if not losses:
        raise RuntimeError("Offline validation dataset produced no batches")
    translation = np.concatenate(translations)
    rotation = np.rad2deg(np.concatenate(rotations))
    result = {
        "event": "offline_summary",
        "num_batches": len(losses),
        "train_samples": len(dataset),
        "validation_samples": len(validation),
        "dataset_path": str(cfg.task.dataset.cache_path),
        "val_loss": float(np.mean(losses)),
        "waypoint_translation_mean": float(translation.mean()),
        "waypoint_translation_p95": float(np.percentile(translation, 95)),
        "waypoint_rotation_deg_mean": float(rotation.mean()),
        "waypoint_rotation_deg_p95": float(np.percentile(rotation, 95)),
        "inference_latency_cold": latencies[0],
        "inference_latency_median": float(np.median(latencies)),
        "inference_latency_p95": float(np.percentile(latencies, 95)),
    }
    if gripper_errors:
        result["val_gripper_physical_mae"] = float(np.concatenate(gripper_errors).mean())
    return result
