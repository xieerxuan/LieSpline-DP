"""Rotation conversion using the same tensor geometry as the spline policy."""

from liespline_dp.common.se3_util import (
    mat_to_quat,
    mat_to_rot6d,
    quat_to_mat,
    rot6d_to_mat,
    so3_exp,
    so3_log,
)


def _quaternion_to_matrix(quaternion):
    # Dataset action quaternions use wxyz; observation quaternions use xyzw.
    return quat_to_mat(quaternion[..., [1, 2, 3, 0]])


def _matrix_to_quaternion(matrix):
    return mat_to_quat(matrix)[..., [3, 0, 1, 2]]


class RotationTransformer:
    """Convert NumPy arrays or tensors through rotation matrices.

    The six-dimensional representation stores the first two matrix rows,
    matching the action representation used throughout training and rollout.
    """

    _to_matrix = {
        "axis_angle": so3_exp,
        "rotation_6d": rot6d_to_mat,
        "quaternion": _quaternion_to_matrix,
        "matrix": lambda value: value,
    }
    _from_matrix = {
        "axis_angle": so3_log,
        "rotation_6d": mat_to_rot6d,
        "quaternion": _matrix_to_quaternion,
        "matrix": lambda value: value,
    }

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        if from_rep not in self._to_matrix or to_rep not in self._from_matrix:
            raise ValueError(f"Unsupported rotation conversion: {from_rep} -> {to_rep}")
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, value):
        return self._from_matrix[self.to_rep](self._to_matrix[self.from_rep](value))

    def inverse(self, value):
        return self._from_matrix[self.from_rep](self._to_matrix[self.to_rep](value))
