import os

import numpy as np
import zarr


def _read_meta_scalar(meta, key):
    if key not in meta:
        raise RuntimeError(f"B-spline SE(3) dataset is missing conversion parameter meta/{key}.")
    values = np.asarray(meta[key][:]).reshape(-1)
    if len(values) != 1:
        raise RuntimeError(f"B-spline SE(3) dataset meta/{key} must contain one value.")
    value = values[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _open_dataset_root(dataset):
    if isinstance(dataset, (str, os.PathLike)):
        return zarr.open(os.path.expanduser(str(dataset)), mode="r")
    return dataset


def _contract_value(contract, key):
    if contract is None:
        raise ValueError("anchor_contract is required")
    try:
        return contract[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"anchor_contract is missing {key!r}") from exc


def _anchor_contract_values(contract):
    family = str(_contract_value(contract, "family"))
    adapter_target = str(_contract_value(contract, "adapter_target"))
    if not adapter_target:
        raise ValueError("anchor_contract.adapter_target must be non-empty")

    values = {
        "simulation_anchor_contract_family": family,
        "anchor_pose_source": "latest_stored_observation",
        "anchor_observation_history_index": -1,
        "anchor_pose_adapter_target": adapter_target,
    }
    compose_side = str(_contract_value(contract, "compose_side"))
    if compose_side not in {"left", "right"}:
        raise ValueError("anchor_contract.compose_side must be 'left' or 'right'")
    rotvec = np.asarray(_contract_value(contract, "rotvec"), dtype=np.float64).reshape(-1)
    if rotvec.shape != (3,) or not np.all(np.isfinite(rotvec)):
        raise ValueError("anchor_contract.rotvec must contain three finite values")
    num_robots = int(_contract_value(contract, "num_robots"))
    if num_robots < 1:
        raise ValueError("anchor_contract.num_robots must be positive")
    values.update(
        {
            "robomimic_observation_quaternion_order": "xyzw",
            "robomimic_observation_eef_frame": "robot_hand_body",
            "robomimic_action_eef_frame": "absolute_osc_grip_site",
            "robomimic_obs_eef_to_action_compose_side": compose_side,
            "robomimic_obs_eef_to_action_rotvec": rotvec,
            "robomimic_anchor_num_robots": num_robots,
        }
    )
    return values


def build_bspline_se3_anchor_contract_metadata(anchor_contract):
    metadata = {}
    for key, value in _anchor_contract_values(anchor_contract).items():
        if isinstance(value, np.ndarray):
            metadata[key] = value
        else:
            metadata[key] = [value]
    return metadata


def _decoded_strings(values):
    decoded = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded.append(str(value))
    return decoded


def validate_bspline_se3_anchor_contract(dataset, anchor_contract) -> None:
    root = _open_dataset_root(dataset)
    if "meta" not in root:
        raise RuntimeError("B-spline SE(3) dataset has no conversion metadata.")
    meta = root["meta"]
    mismatches = []
    for key, expected in _anchor_contract_values(anchor_contract).items():
        if key not in meta:
            mismatches.append(f"meta/{key} is missing")
            continue
        actual = np.asarray(meta[key][:]).reshape(-1)
        expected_array = np.asarray(expected).reshape(-1)
        if actual.shape != expected_array.shape:
            mismatches.append(
                f"meta/{key}: dataset shape={actual.shape}, task shape={expected_array.shape}"
            )
            continue
        if expected_array.dtype.kind in "iufb":
            try:
                matches = np.allclose(
                    actual.astype(np.float64),
                    expected_array.astype(np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = _decoded_strings(actual) == _decoded_strings(expected_array)
        if not matches:
            mismatches.append(
                f"meta/{key}: dataset={actual.tolist()}, task={expected_array.tolist()}"
            )

    if mismatches:
        raise RuntimeError(
            "B-spline SE(3) dataset anchor contract does not match the "
            "active task config: "
            + "; ".join(mismatches)
            + ". Regenerate the dataset with the active task config."
        )


def validate_bspline_se3_dataset_parameters(
    dataset,
    *,
    degree,
    prefix_cp_steps,
    control_point_stride,
    exec_horizon,
    n_action_steps,
    n_latency_steps,
) -> None:
    """Check parameters that must agree between conversion and rollout."""
    root = _open_dataset_root(dataset)
    if "meta" not in root:
        raise RuntimeError("B-spline SE(3) dataset has no conversion metadata.")

    expected = {
        "bspline_degree": int(degree),
        "prefix_cp_steps": int(prefix_cp_steps),
        "control_point_stride": int(control_point_stride),
        "exec_horizon": int(exec_horizon),
        "n_action_steps": int(n_action_steps),
        "n_latency_steps": int(n_latency_steps),
    }
    meta = root["meta"]
    mismatches = []
    for key, expected_value in expected.items():
        stored_value = int(_read_meta_scalar(meta, key))
        if stored_value != expected_value:
            mismatches.append(f"{key}: dataset={stored_value}, task={expected_value}")

    if mismatches:
        raise RuntimeError(
            "B-spline SE(3) dataset conversion parameters do not match the "
            "training task config: "
            + "; ".join(mismatches)
            + ". Regenerate the dataset with the active task config."
        )
