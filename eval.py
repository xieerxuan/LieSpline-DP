"""Evaluate a trained DP or LieSpline-DP checkpoint in Robomimic."""

import copy
import inspect
import json
import os
import pathlib
import sys

import click
import dill
import hydra
import torch
import wandb
from omegaconf import OmegaConf

from liespline_dp.workspace.base_workspace import BaseWorkspace


def _configure_line_buffering():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass


def _policy_label(cfg) -> str:
    # Both algorithms live under liespline_dp; identify the workspace class,
    # independently of the package namespace.
    workspace_class = hydra.utils.get_class(cfg._target_)
    return workspace_class.algorithm


def _apply_runner_overrides(
    cfg,
    *,
    n_train=None,
    n_train_vis=None,
    test_start_seed=None,
    n_test=None,
    n_test_vis=None,
    n_envs=None,
    n_latency_steps=None,
    measure_eef_jerk=False,
    save_eef_trajectories=False,
):
    cfg = copy.deepcopy(cfg)
    runner_cfg = cfg.task.env_runner
    overrides = {
        "n_train": n_train,
        "n_train_vis": n_train_vis,
        "test_start_seed": test_start_seed,
        "n_test": n_test,
        "n_test_vis": n_test_vis,
        "n_envs": n_envs,
        "n_latency_steps": n_latency_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            OmegaConf.update(
                runner_cfg,
                key,
                value,
                merge=False,
                force_add=True,
            )
    # Optional trajectory diagnostics are supported by the lowdim runner.
    # Forward only arguments accepted by the selected runner implementation.
    runner_cls = hydra.utils.get_class(runner_cfg._target_)
    supported = inspect.signature(runner_cls.__init__).parameters
    diagnostics = {
        "measure_eef_jerk": bool(measure_eef_jerk or save_eef_trajectories),
        "save_eef_trajectories": bool(save_eef_trajectories),
    }
    for key, value in diagnostics.items():
        if key in supported:
            OmegaConf.update(runner_cfg, key, value, force_add=True)
        elif value:
            raise click.ClickException(f"{runner_cls.__name__} does not support {key}")
    context = {
        "task_name": str(cfg.task.task_name),
        "method_name": _policy_label(cfg),
        "train_seed": int(cfg.training.seed),
    }
    for key, value in context.items():
        if key in supported:
            OmegaConf.update(runner_cfg, key, value, force_add=True)
    return cfg


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--n-train", type=click.IntRange(min=0), default=None)
@click.option("--n-train-vis", type=click.IntRange(min=0), default=None)
@click.option("--test-start-seed", type=int, default=None)
@click.option("--n-test", type=click.IntRange(min=0), default=None)
@click.option("--n-test-vis", type=click.IntRange(min=0), default=None)
@click.option("--n-envs", type=click.IntRange(min=1), default=None)
@click.option("--n-latency-steps", type=click.IntRange(min=0), default=None)
@click.option("--measure-eef-jerk", is_flag=True, default=False)
@click.option("--save-eef-trajectories", is_flag=True, default=False)
def main(
    checkpoint,
    output_dir,
    device,
    n_train,
    n_train_vis,
    test_start_seed,
    n_test,
    n_test_vis,
    n_envs,
    n_latency_steps,
    measure_eef_jerk,
    save_eef_trajectories,
):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    with open(checkpoint, "rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    cfg = _apply_runner_overrides(
        payload["cfg"],
        n_train=n_train,
        n_train_vis=n_train_vis,
        test_start_seed=test_start_seed,
        n_test=n_test,
        n_test_vis=n_test_vis,
        n_envs=n_envs,
        n_latency_steps=n_latency_steps,
        measure_eef_jerk=measure_eef_jerk,
        save_eef_trajectories=save_eef_trajectories,
    )
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    include_keys = tuple(key for key in payload.get("pickles", {}) if key != "_output_dir")
    workspace.load_payload(
        payload,
        exclude_keys=None,
        include_keys=include_keys,
    )

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    resolved_runner_cfg = OmegaConf.to_container(
        cfg.task.env_runner,
        resolve=True,
    )
    with open(os.path.join(output_dir, "eval_config.json"), "w") as stream:
        json.dump(
            {
                "checkpoint": str(pathlib.Path(checkpoint).resolve()),
                "runner": resolved_runner_cfg,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    # run eval
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif hasattr(value, "item"):
            json_log[key] = value.item()
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    with open(out_path, "w") as stream:
        json.dump(json_log, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    _configure_line_buffering()
    main()
