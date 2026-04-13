from __future__ import annotations

from dataclasses import dataclass

import torch

from rsl_rl.diffusion.sampler import SMPDiffusionSampler


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _normalize_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return _normalize(quat_wxyz)


def _quat_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_conj = quat_wxyz.clone()
    quat_conj[..., 1:] = -quat_conj[..., 1:]
    return quat_conj


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _matrix_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = _normalize_quat(quat_wxyz)
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat_wxyz.shape[:-1], 3, 3)


def _build_heading_frame_rotation(root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    root_rotation = _matrix_from_quat(root_quat_wxyz)
    forward_w = root_rotation[..., :, 0]
    fallback_w = root_rotation[..., :, 1]
    up_w = torch.zeros_like(forward_w)
    up_w[..., 2] = 1.0

    forward_proj = forward_w - (forward_w * up_w).sum(dim=-1, keepdim=True) * up_w
    fallback_proj = fallback_w - (fallback_w * up_w).sum(dim=-1, keepdim=True) * up_w
    use_fallback = forward_proj.norm(dim=-1, keepdim=True) < 1.0e-6
    x_axis_w = _normalize(torch.where(use_fallback, fallback_proj, forward_proj))
    y_axis_w = up_w
    z_axis_w = _normalize(torch.cross(x_axis_w, y_axis_w, dim=-1))
    x_axis_w = _normalize(torch.cross(y_axis_w, z_axis_w, dim=-1))
    return torch.stack((x_axis_w, y_axis_w, z_axis_w), dim=-1)


def _world_to_local_frame(rotation_world_from_local: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vec_w.unsqueeze(-2), rotation_world_from_local).squeeze(-2)


def _local_to_world_frame(rotation_world_from_local: torch.Tensor, vec_local: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vec_local.unsqueeze(-2), rotation_world_from_local.transpose(-1, -2)).squeeze(-2)


def _expand_batch_tensor(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0).expand(batch_size, -1).clone()
    if tensor.ndim == 2 and tensor.shape[0] == batch_size:
        return tensor.clone()
    raise ValueError(f"Expected {name} shape ({batch_size}, D) or (D,), got {tuple(tensor.shape)}")


def _joint_angle_offsets_to_rot6d(joint_angle_offsets: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    normalized_axes = _normalize(joint_axes.to(device=joint_angle_offsets.device, dtype=joint_angle_offsets.dtype))
    axis_shape = (1,) * (joint_angle_offsets.ndim - 1) + normalized_axes.shape
    expanded_axes = normalized_axes.view(axis_shape)
    half_angle = 0.5 * joint_angle_offsets
    quat_wxyz = torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            expanded_axes * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )
    return _quat_to_rot6d(quat_wxyz)


def _quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return _matrix_from_quat(quat_wxyz)[..., :2].reshape(*quat_wxyz.shape[:-1], 6)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    col_1 = torch.stack((rot6d[..., 0], rot6d[..., 2], rot6d[..., 4]), dim=-1)
    col_2 = torch.stack((rot6d[..., 1], rot6d[..., 3], rot6d[..., 5]), dim=-1)
    basis_1 = _normalize(col_1)
    basis_2 = _normalize(col_2 - (basis_1 * col_2).sum(dim=-1, keepdim=True) * basis_1)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def _joint_rot6d_to_angle_offsets(joint_rot6d: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    rotmat = _rot6d_to_matrix(joint_rot6d)
    normalized_axes = _normalize(joint_axes.to(device=joint_rot6d.device, dtype=joint_rot6d.dtype))
    axis_shape = (1,) * (joint_rot6d.ndim - 2) + normalized_axes.shape
    expanded_axes = normalized_axes.view(axis_shape)
    cos_theta = ((torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vec = torch.stack(
        (
            rotmat[..., 2, 1] - rotmat[..., 1, 2],
            rotmat[..., 0, 2] - rotmat[..., 2, 0],
            rotmat[..., 1, 0] - rotmat[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * (skew_vec * expanded_axes).sum(dim=-1)
    return torch.atan2(sin_theta, cos_theta)


@dataclass(frozen=True)
class SMPFeatureLayout:
    feature_dim: int
    base_lin_vel_b: tuple[int, int]
    base_ang_vel_b: tuple[int, int]
    joint_pos_rel: tuple[int, int] | None = None
    joint_rot6d_rel: tuple[int, int] | None = None
    ee_pos_b: tuple[int, int] | None = None
    key_body_rot6d: tuple[int, int] | None = None

    @classmethod
    def from_feature_block_offsets(cls, feature_block_offsets: dict[str, tuple[int, int]]):
        required_keys = ("base_lin_vel_b", "base_ang_vel_b")
        missing_keys = [key for key in required_keys if key not in feature_block_offsets]
        if missing_keys:
            raise KeyError(f"Missing required feature block offsets: {missing_keys}")
        if "joint_rot6d_rel" not in feature_block_offsets and "joint_pos_rel" not in feature_block_offsets:
            raise KeyError("Missing required feature block offsets: ['joint_rot6d_rel' or 'joint_pos_rel']")

        feature_dim = max(int(stop) for _, stop in feature_block_offsets.values())
        return cls(
            feature_dim=feature_dim,
            base_lin_vel_b=tuple(map(int, feature_block_offsets["base_lin_vel_b"])),
            base_ang_vel_b=tuple(map(int, feature_block_offsets["base_ang_vel_b"])),
            joint_pos_rel=(
                tuple(map(int, feature_block_offsets["joint_pos_rel"]))
                if "joint_pos_rel" in feature_block_offsets
                else None
            ),
            joint_rot6d_rel=(
                tuple(map(int, feature_block_offsets["joint_rot6d_rel"]))
                if "joint_rot6d_rel" in feature_block_offsets
                else None
            ),
            ee_pos_b=(
                tuple(map(int, feature_block_offsets["ee_pos_b"]))
                if "ee_pos_b" in feature_block_offsets
                else None
            ),
            key_body_rot6d=(
                tuple(map(int, feature_block_offsets["key_body_rot6d"]))
                if "key_body_rot6d" in feature_block_offsets
                else None
            ),
        )

    @property
    def joint_block_name(self) -> str:
        if self.joint_rot6d_rel is not None:
            return "joint_rot6d_rel"
        if self.joint_pos_rel is not None:
            return "joint_pos_rel"
        raise ValueError("Feature layout does not define any joint feature block")

    @property
    def joint_block(self) -> tuple[int, int]:
        if self.joint_rot6d_rel is not None:
            return self.joint_rot6d_rel
        if self.joint_pos_rel is not None:
            return self.joint_pos_rel
        raise ValueError("Feature layout does not define any joint feature block")

    @property
    def joint_dim(self) -> int:
        start, stop = self.joint_block
        if self.joint_rot6d_rel is not None:
            width = stop - start
            if width % 6 != 0:
                raise ValueError("joint_rot6d_rel block width must be divisible by 6")
            return width // 6
        return stop - start


@dataclass
class SMPResetReference:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor | None = None


@dataclass
class SMPResetState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass
class SMPGSIDecodeResult:
    state: SMPResetState
    supports_reset_state: bool
    reconstruction_mse: float
    unrecoverable_feature_blocks: tuple[str, ...]


class SMPGSIDecoder:
    """把采样得到的 SMP 窗口解码成 reset state，并显式报告可恢复块的误差。"""

    def __init__(
        self,
        feature_layout: SMPFeatureLayout,
        joint_axes: torch.Tensor | None = None,
        error_threshold: float = 1.0e-6,
    ):
        self.feature_layout = feature_layout
        self.error_threshold = float(error_threshold)
        if self.feature_layout.joint_rot6d_rel is not None and joint_axes is None:
            raise ValueError("joint_axes is required when decoding joint_rot6d_rel features")
        if joint_axes is not None:
            joint_axes = joint_axes.to(dtype=torch.float32)
            if joint_axes.shape != (self.feature_layout.joint_dim, 3):
                raise ValueError(
                    f"Expected joint_axes shape ({self.feature_layout.joint_dim}, 3), got {tuple(joint_axes.shape)}"
                )
        self.joint_axes = joint_axes

    def _unrecoverable_feature_blocks(self) -> tuple[str, ...]:
        missing = []
        if self.feature_layout.ee_pos_b is not None:
            missing.append("ee_pos_b")
        if self.feature_layout.key_body_rot6d is not None:
            missing.append("key_body_rot6d")
        return tuple(missing)

    def _recoverable_target(self, last_frame: torch.Tensor) -> torch.Tensor:
        joint_start, joint_stop = self.feature_layout.joint_block
        return torch.cat(
            (
                last_frame[:, self.feature_layout.base_lin_vel_b[0] : self.feature_layout.base_lin_vel_b[1]],
                last_frame[:, self.feature_layout.base_ang_vel_b[0] : self.feature_layout.base_ang_vel_b[1]],
                last_frame[:, joint_start:joint_stop],
            ),
            dim=-1,
        )

    def _recoverable_reencode(self, state: SMPResetState, reference_state: SMPResetReference) -> torch.Tensor:
        joint_pos_default = _expand_batch_tensor(reference_state.joint_pos, state.joint_pos.shape[0], "joint_pos")
        heading_rotation = _build_heading_frame_rotation(state.root_quat_w)
        reencoded = [
            _world_to_local_frame(heading_rotation, state.root_lin_vel_w),
            _world_to_local_frame(heading_rotation, state.root_ang_vel_w),
        ]
        if self.feature_layout.joint_rot6d_rel is not None:
            joint_rot6d = _joint_angle_offsets_to_rot6d(state.joint_pos - joint_pos_default, self.joint_axes)
            reencoded.append(joint_rot6d.reshape(state.joint_pos.shape[0], -1))
        else:
            reencoded.append(state.joint_pos - joint_pos_default)
        return torch.cat(reencoded, dim=-1)

    def decode(self, window: torch.Tensor, reference_state: SMPResetReference) -> SMPGSIDecodeResult:
        if window.ndim != 3:
            raise ValueError(f"Expected motion window shape (batch, window, feature), got {tuple(window.shape)}")
        if window.shape[-1] != self.feature_layout.feature_dim:
            raise ValueError(
                f"Expected motion window feature dim {self.feature_layout.feature_dim}, got {window.shape[-1]}"
            )
        batch_size = int(window.shape[0])
        last_frame = window[:, -1].to(dtype=torch.float32)

        root_pos_w = _expand_batch_tensor(reference_state.root_pos_w, batch_size, "root_pos_w")
        root_quat_w = _normalize_quat(_expand_batch_tensor(reference_state.root_quat_w, batch_size, "root_quat_w"))
        joint_pos_default = _expand_batch_tensor(reference_state.joint_pos, batch_size, "joint_pos")
        if reference_state.joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos_default)
        else:
            joint_vel = _expand_batch_tensor(reference_state.joint_vel, batch_size, "joint_vel")

        base_lin_vel_b = last_frame[:, self.feature_layout.base_lin_vel_b[0] : self.feature_layout.base_lin_vel_b[1]]
        base_ang_vel_b = last_frame[:, self.feature_layout.base_ang_vel_b[0] : self.feature_layout.base_ang_vel_b[1]]
        heading_rotation = _build_heading_frame_rotation(root_quat_w)

        joint_start, joint_stop = self.feature_layout.joint_block
        if self.feature_layout.joint_rot6d_rel is not None:
            joint_rot6d_rel = last_frame[:, joint_start:joint_stop].view(batch_size, self.feature_layout.joint_dim, 6)
            joint_angle_offsets = _joint_rot6d_to_angle_offsets(joint_rot6d_rel, self.joint_axes)
        else:
            joint_angle_offsets = last_frame[:, joint_start:joint_stop]

        state = SMPResetState(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            root_lin_vel_w=_local_to_world_frame(heading_rotation, base_lin_vel_b),
            root_ang_vel_w=_local_to_world_frame(heading_rotation, base_ang_vel_b),
            joint_pos=joint_pos_default + joint_angle_offsets,
            joint_vel=joint_vel,
        )
        target = self._recoverable_target(last_frame)
        reconstructed = self._recoverable_reencode(state, reference_state)
        reconstruction_mse = float((reconstructed - target).pow(2).mean().item())
        return SMPGSIDecodeResult(
            state=state,
            supports_reset_state=reconstruction_mse <= self.error_threshold,
            reconstruction_mse=reconstruction_mse,
            unrecoverable_feature_blocks=self._unrecoverable_feature_blocks(),
        )


class SMPGSISampler:
    """串联扩散采样器与 reset-state 解码器，供 runner 在 reset 时直接调用。"""

    def __init__(self, sampler: SMPDiffusionSampler, decoder: SMPGSIDecoder):
        self.sampler = sampler
        self.decoder = decoder

    def sample_reset_state(
        self,
        batch_size: int,
        reference_state: SMPResetReference,
        *,
        style_program: dict[str, object] | None = None,
        style_id: torch.Tensor | int | None = None,
        guidance_scale: float = 1.0,
    ) -> SMPGSIDecodeResult:
        sampled_window = self.sampler.sample(
            batch_size=batch_size,
            style_id=style_id,
            style_program=style_program,
            guidance_scale=guidance_scale,
        )
        return self.decoder.decode(sampled_window, reference_state=reference_state)
