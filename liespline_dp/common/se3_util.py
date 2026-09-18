from typing import Tuple

import numpy as np
import torch


def _as_float_tensor(x) -> Tuple[torch.Tensor, bool]:
    is_numpy = isinstance(x, np.ndarray)
    y = torch.from_numpy(x) if is_numpy else x
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if not torch.is_floating_point(y):
        y = y.float()
    return y, is_numpy


def _restore_type(x: torch.Tensor, is_numpy: bool):
    if is_numpy:
        return x.detach().cpu().numpy()
    return x


def _eye_like(x: torch.Tensor, n: int) -> torch.Tensor:
    return torch.eye(n, dtype=x.dtype, device=x.device)


def _safe_norm(x: torch.Tensor, dim=-1, keepdim=False, eps=1e-12) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=keepdim).clamp_min(eps)


def so3_hat(phi):
    phi_t, is_numpy = _as_float_tensor(phi)
    x, y, z = phi_t.unbind(dim=-1)
    z0 = torch.zeros_like(x)
    mat = torch.stack(
        [
            z0,
            -z,
            y,
            z,
            z0,
            -x,
            -y,
            x,
            z0,
        ],
        dim=-1,
    ).reshape(phi_t.shape[:-1] + (3, 3))
    return _restore_type(mat, is_numpy)


def so3_vee(mat):
    mat_t, is_numpy = _as_float_tensor(mat)
    vec = torch.stack(
        [
            mat_t[..., 2, 1],
            mat_t[..., 0, 2],
            mat_t[..., 1, 0],
        ],
        dim=-1,
    )
    return _restore_type(vec, is_numpy)


def so3_exp(phi):
    phi_t, is_numpy = _as_float_tensor(phi)
    k = _as_float_tensor(so3_hat(phi_t))[0]
    k2 = k @ k
    theta2 = (phi_t * phi_t).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    small = theta2 < 1e-12

    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-12),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    eye = _eye_like(phi_t, 3).expand(phi_t.shape[:-1] + (3, 3))
    rot = eye + a[..., None] * k + b[..., None] * k2
    return _restore_type(rot, is_numpy)


def quat_to_mat(quat):
    quat_t, is_numpy = _as_float_tensor(quat)
    quat_t = quat_t / _safe_norm(quat_t, dim=-1, keepdim=True)
    x, y, z, w = quat_t.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    ).reshape(quat_t.shape[:-1] + (3, 3))
    return _restore_type(rot, is_numpy)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def mat_to_quat(mat):
    mat_t, is_numpy = _as_float_tensor(mat)
    m00 = mat_t[..., 0, 0]
    m01 = mat_t[..., 0, 1]
    m02 = mat_t[..., 0, 2]
    m10 = mat_t[..., 1, 0]
    m11 = mat_t[..., 1, 1]
    m12 = mat_t[..., 1, 2]
    m20 = mat_t[..., 2, 0]
    m21 = mat_t[..., 2, 1]
    m22 = mat_t[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp_min(0.1))
    idx = q_abs.argmax(dim=-1)
    gather_idx = idx[..., None, None].expand(idx.shape + (1, 4))
    quat_wxyz = torch.gather(quat_candidates, dim=-2, index=gather_idx).squeeze(-2)
    quat_xyzw = torch.stack(
        [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
        dim=-1,
    )
    quat_xyzw = quat_xyzw / _safe_norm(quat_xyzw, dim=-1, keepdim=True)
    quat_xyzw = torch.where(quat_xyzw[..., 3:4] < 0.0, -quat_xyzw, quat_xyzw)
    return _restore_type(quat_xyzw, is_numpy)


def so3_log(rot):
    rot_t, is_numpy = _as_float_tensor(rot)
    quat = _as_float_tensor(mat_to_quat(rot_t))[0]
    vec = quat[..., :3]
    w = quat[..., 3:4].clamp(-1.0, 1.0)
    vec_norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vec_norm, w)
    scale = torch.where(
        vec_norm < 1e-12,
        2.0 / w.clamp_min(1e-12) - (2.0 * vec_norm * vec_norm) / (3.0 * w.clamp_min(1e-12) ** 3),
        angle / vec_norm.clamp_min(1e-12),
    )
    phi = scale * vec
    return _restore_type(phi, is_numpy)


def se3_hat(xi):
    xi_t, is_numpy = _as_float_tensor(xi)
    mat = torch.zeros(xi_t.shape[:-1] + (4, 4), dtype=xi_t.dtype, device=xi_t.device)
    mat[..., :3, :3] = _as_float_tensor(so3_hat(xi_t[..., 3:6]))[0]
    mat[..., :3, 3] = xi_t[..., :3]
    return _restore_type(mat, is_numpy)


def se3_vee(mat):
    mat_t, is_numpy = _as_float_tensor(mat)
    rho = mat_t[..., :3, 3]
    phi = _as_float_tensor(so3_vee(mat_t[..., :3, :3]))[0]
    xi = torch.cat([rho, phi], dim=-1)
    return _restore_type(xi, is_numpy)


def se3_exp(xi):
    xi_t, is_numpy = _as_float_tensor(xi)
    rho = xi_t[..., :3]
    phi = xi_t[..., 3:6]
    rot = _as_float_tensor(so3_exp(phi))[0]
    k = _as_float_tensor(so3_hat(phi))[0]
    k2 = k @ k
    theta2 = (phi * phi).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    small = theta2 < 1e-12
    a = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    b = torch.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(1e-12),
    )
    eye3 = _eye_like(xi_t, 3).expand(xi_t.shape[:-1] + (3, 3))
    v = eye3 + a[..., None] * k + b[..., None] * k2
    trans = (v @ rho[..., None]).squeeze(-1)

    transform = _eye_like(xi_t, 4).expand(xi_t.shape[:-1] + (4, 4)).clone()
    transform[..., :3, :3] = rot
    transform[..., :3, 3] = trans
    return _restore_type(transform, is_numpy)


def se3_log(transform):
    transform_t, is_numpy = _as_float_tensor(transform)
    rot = transform_t[..., :3, :3]
    trans = transform_t[..., :3, 3]
    phi = _as_float_tensor(so3_log(rot))[0]
    k = _as_float_tensor(so3_hat(phi))[0]
    k2 = k @ k
    theta2 = (phi * phi).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    small = theta2 < 1e-12
    c = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0,
        (1.0 / theta2.clamp_min(1e-12))
        - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(1e-12)),
    )
    eye3 = _eye_like(transform_t, 3).expand(transform_t.shape[:-2] + (3, 3))
    v_inv = eye3 - 0.5 * k + c[..., None] * k2
    rho = (v_inv @ trans[..., None]).squeeze(-1)
    xi = torch.cat([rho, phi], dim=-1)
    return _restore_type(xi, is_numpy)


def se3_inv(transform):
    transform_t, is_numpy = _as_float_tensor(transform)
    rot = transform_t[..., :3, :3]
    trans = transform_t[..., :3, 3]
    inv_transform = _eye_like(transform_t, 4).expand(transform_t.shape[:-2] + (4, 4)).clone()
    rot_t = rot.transpose(-1, -2)
    inv_transform[..., :3, :3] = rot_t
    inv_transform[..., :3, 3] = -(rot_t @ trans[..., None]).squeeze(-1)
    return _restore_type(inv_transform, is_numpy)


def se3_mul(a, b):
    a_t, a_is_numpy = _as_float_tensor(a)
    b_t, b_is_numpy = _as_float_tensor(b)
    out = a_t @ b_t
    return _restore_type(out, a_is_numpy and b_is_numpy)


def project_to_so3(mat):
    mat_t, is_numpy = _as_float_tensor(mat)
    u, _, vh = torch.linalg.svd(mat_t)
    rot = u @ vh
    det = torch.linalg.det(rot)
    fix = torch.ones(rot.shape[:-2] + (3,), dtype=rot.dtype, device=rot.device)
    fix[..., -1] = torch.where(det < 0.0, -1.0, 1.0)
    rot = u @ torch.diag_embed(fix) @ vh
    return _restore_type(rot, is_numpy)


def rot6d_to_mat(rot6d):
    rot6d_t, is_numpy = _as_float_tensor(rot6d)
    a1 = rot6d_t[..., 0:3]
    a2 = rot6d_t[..., 3:6]
    b1 = a1 / _safe_norm(a1, dim=-1, keepdim=True)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / _safe_norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    rot = torch.stack([b1, b2, b3], dim=-2)
    return _restore_type(rot, is_numpy)


def mat_to_rot6d(mat):
    mat_t, is_numpy = _as_float_tensor(mat)
    rot6d = mat_t[..., :2, :].clone().reshape(mat_t.shape[:-2] + (6,))
    return _restore_type(rot6d, is_numpy)
