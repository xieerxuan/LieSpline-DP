import collections
import multiprocessing
import pathlib
import pickle
import shutil

import click
import h5py
import numpy as np
from tqdm import tqdm

from liespline_dp.common.robomimic_util import RobomimicAbsoluteActionConverter

OBJECT_OBS_REFRESH_MARKER = "robosuite_1_5_forced_second_read"
DATASET_ROOT = pathlib.Path("data/robomimic/datasets")
MODALITY_FILENAMES = {
    "lowdim": ("low_dim_v15.hdf5", "low_dim_abs.hdf5"),
    "image": ("image.hdf5", "image_abs.hdf5"),
}


def _demo_keys(dataset):
    return sorted(dataset["data"].keys(), key=lambda key: int(key.split("_")[-1]))


def _is_robomimic04_lowdim_dataset(input_path):
    with h5py.File(input_path, "r") as dataset:
        data = dataset["data"]
        demo_key = _demo_keys(dataset)[0]
        obs = data[f"{demo_key}/obs"]
        has_image_obs = any(key.endswith("_image") for key in obs.keys())
        return not has_image_obs and "object" in obs


def _active_object_observation_keys(robosuite_env):
    return [
        key
        for key, observable in robosuite_env._observables.items()
        if observable.modality == "object" and observable.is_enabled() and observable.is_active()
    ]


def _object_observation_from_components(env):
    robosuite_obs = env.env._get_observations(force_update=True)
    parts = []
    for key in _active_object_observation_keys(env.env):
        value = np.asarray(robosuite_obs[key])
        if value.shape == ():
            value = value.reshape(1)
        parts.append(value)
    return np.concatenate(parts, axis=-1)


def _refresh_demo_object_observations(converter, idx):
    demo = converter.file[f"data/demo_{idx}"]
    states = demo["states"][:]
    actions = demo["actions"][:]

    objects = []
    for step_idx, state in enumerate(states):
        reset_state = {"states": state}
        if step_idx == 0:
            if "model_file" in demo.attrs:
                reset_state["model"] = demo.attrs["model_file"]
            if "ep_meta" in demo.attrs:
                reset_state["ep_meta"] = demo.attrs["ep_meta"]
        converter.env.reset_to(reset_state)
        objects.append(_object_observation_from_components(converter.env))
    objects = np.asarray(objects)

    converter.env.reset_to({"states": states[-1]})
    converter.env.step(actions[-1])
    final_next_object = _object_observation_from_components(converter.env)
    next_objects = np.concatenate([objects[1:], final_next_object[None]], axis=0)
    return objects, next_objects


def worker(x):
    path, idx, do_eval, refresh_object_obs = x
    converter = RobomimicAbsoluteActionConverter(path)
    if do_eval:
        abs_actions, info = converter.convert_and_eval_idx(idx)
    else:
        abs_actions = converter.convert_idx(idx)
        info = dict()
    object_obs = None
    if refresh_object_obs:
        object_obs = _refresh_demo_object_observations(converter, idx)
    return abs_actions, object_obs, info


def _default_paths(task, dataset_type, modality):
    input_name, output_name = MODALITY_FILENAMES[modality]
    dataset_dir = DATASET_ROOT / task / dataset_type
    return dataset_dir / input_name, dataset_dir / output_name


def _default_eval_dir(task, dataset_type, modality):
    dataset_dir = DATASET_ROOT / task / dataset_type
    if modality == "lowdim":
        return dataset_dir / "abs_action_eval"
    return dataset_dir / f"{modality}_abs_action_eval"


def convert_robomimic_dataset(input, output, eval_dir=None, num_workers=None):
    # process inputs
    input = pathlib.Path(input).expanduser()
    if not input.is_file():
        raise FileNotFoundError(f"Input hdf5 does not exist: {input}")
    output = pathlib.Path(output).expanduser()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output.parent}")
    if output.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {output}")

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        if not eval_dir.parent.exists():
            raise FileNotFoundError(f"Eval parent directory does not exist: {eval_dir.parent}")
        do_eval = True

    refresh_object_obs = _is_robomimic04_lowdim_dataset(input)
    converter = RobomimicAbsoluteActionConverter(input)

    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(
            worker,
            [(input, i, do_eval, refresh_object_obs) for i in range(len(converter))],
        )

    # save output
    print("Copying hdf5")
    shutil.copy(str(input), str(output))

    # modify action
    with h5py.File(output, "r+") as out_file:
        for i in tqdm(range(len(converter)), desc="Writing to output"):
            abs_actions, object_obs, info = results[i]
            demo = out_file[f"data/demo_{i}"]
            demo["actions"][:] = abs_actions
            if object_obs is not None:
                objects, next_objects = object_obs
                if demo["obs/object"].shape != objects.shape:
                    raise ValueError(
                        f"data/demo_{i}/obs/object has shape "
                        f"{demo['obs/object'].shape}, rebuilt shape is "
                        f"{objects.shape}"
                    )
                demo["obs/object"][:] = objects.astype(demo["obs/object"].dtype, copy=False)
                if "next_obs/object" in demo:
                    if demo["next_obs/object"].shape != next_objects.shape:
                        raise ValueError(
                            f"data/demo_{i}/next_obs/object has shape "
                            f"{demo['next_obs/object'].shape}, rebuilt shape "
                            f"is {next_objects.shape}"
                        )
                    demo["next_obs/object"][:] = next_objects.astype(
                        demo["next_obs/object"].dtype, copy=False
                    )
        if refresh_object_obs:
            out_file["data"].attrs["object_obs_refresh"] = OBJECT_OBS_REFRESH_MARKER

    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, _, info in results]
        pickle.dump(infos, eval_dir.joinpath("error_stats.pkl").open("wb"))

        print("Generating visualization")
        metrics = ["pos", "rot"]
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt

        plt.switch_backend("PDF")

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10, 4)
        fig.savefig(str(eval_dir.joinpath("error_stats.pdf")))
        fig.savefig(str(eval_dir.joinpath("error_stats.png")))


@click.command()
@click.option(
    "--task",
    "tasks",
    required=True,
    multiple=True,
    help=('Robomimic task name. Accepts "--task lift can square" or repeated "--task" options.'),
)
@click.argument("extra_tasks", nargs=-1)
@click.option("--dataset_type", default="ph", show_default=True)
@click.option(
    "--modality",
    "modalities",
    type=click.Choice(sorted(MODALITY_FILENAMES.keys())),
    multiple=True,
    default=("lowdim",),
    show_default=True,
    help="Observation modality to convert. Repeat this option for multiple modalities.",
)
@click.option(
    "--eval",
    "do_eval",
    is_flag=True,
    default=False,
    help="Write conversion evaluation metrics to the default per-task directory.",
)
@click.option("-n", "--num_workers", default=None, type=int)
def main(tasks, extra_tasks, dataset_type, modalities, do_eval, num_workers):
    tasks = tuple(tasks) + tuple(extra_tasks)
    for task in tasks:
        for modality in modalities:
            input_path, output_path = _default_paths(task, dataset_type, modality)
            eval_dir = _default_eval_dir(task, dataset_type, modality) if do_eval else None
            print(
                f"Converting robomimic {task}/{dataset_type}/{modality}: "
                f"{input_path} -> {output_path}"
            )
            convert_robomimic_dataset(
                input=input_path,
                output=output_path,
                eval_dir=eval_dir,
                num_workers=num_workers,
            )
    print("robomimic absolute action dataset conversion finished.")


if __name__ == "__main__":
    main()
