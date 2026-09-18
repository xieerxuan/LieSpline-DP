"""JSON plus raw NumPy multipart protocol.

No pickle is used. The first frame is UTF-8 JSON; every later frame contains one
C-contiguous array described by the JSON header.
"""

import json
import math

import numpy as np

PROTOCOL_NAME = "ur5e_robot_service"
MAX_HEADER_BYTES = 1_000_000
MAX_ARRAY_BYTES = 512 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pack_message(header, arrays=None):
    document = _json_safe(dict(header))
    document["protocol_name"] = PROTOCOL_NAME
    # Fixed wire field required by the deployed Robot Service envelope.
    document["protocol_version"] = 4
    payload = []
    descriptors = []
    for name, value in (arrays or {}).items():
        # np.ascontiguousarray() promotes a zero-dimensional scalar to shape
        # (1,), which violates scalar field schemas on get_latest/PUB messages.
        array = np.asarray(value)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if array.dtype.hasobject:
            raise ProtocolError(f"Object dtype is forbidden for array {name!r}")
        if array.nbytes > MAX_ARRAY_BYTES:
            raise ProtocolError(f"Array {name!r} exceeds the byte limit")
        descriptors.append(
            {
                "name": str(name),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": int(array.nbytes),
            }
        )
        payload.append(memoryview(array))
    document["arrays"] = descriptors
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ProtocolError("JSON header exceeds the byte limit")
    return [encoded, *payload]


def unpack_message(frames):
    if not frames:
        raise ProtocolError("Empty multipart message")
    if len(frames[0]) > MAX_HEADER_BYTES:
        raise ProtocolError("JSON header exceeds the byte limit")
    try:
        header = json.loads(bytes(frames[0]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Invalid JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ProtocolError("JSON header root must be an object")
    if header.get("protocol_name") != PROTOCOL_NAME:
        raise ProtocolError("Protocol name mismatch")
    descriptors = header.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) != len(frames) - 1:
        raise ProtocolError("Array descriptor/frame count mismatch")
    arrays = {}
    for descriptor, frame in zip(descriptors, frames[1:]):
        try:
            name = str(descriptor["name"])
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(int(item) for item in descriptor["shape"])
            declared_nbytes = int(descriptor["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Invalid array descriptor") from exc
        if not name or name in arrays or dtype.hasobject or any(item < 0 for item in shape):
            raise ProtocolError(f"Invalid array descriptor for {name!r}")
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        actual_nbytes = memoryview(frame).nbytes
        if (
            expected_nbytes != declared_nbytes
            or actual_nbytes != declared_nbytes
            or actual_nbytes > MAX_ARRAY_BYTES
        ):
            raise ProtocolError(f"Array byte count mismatch for {name!r}")
        arrays[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
    return header, arrays
