from __future__ import annotations

from dataclasses import dataclass

import torch

from rsl_rl.diffusion.sampler import SMPDiffusionSampler


def _normalize_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


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


def _quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_wxyz = _normalize_quat(quat_wxyz)
    zeros = torch.zeros(*vec.shape[:-1], 1, device=vec.device, dtype=vec.dtype)
    vec_quat = torch.cat((zeros, vec), dim=-1)
    return _quat_multiply(_quat_multiply(quat_wxyz, vec_quat), _quat_conjugate(quat_wxyz))[..., 1:]


def _quat_apply_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_conjugate(_normalize_quat(quat_wxyz)), vec)


def _expand_batch_tensor(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0).expand(batch_size, -1).clone()
    if tensor.ndim == 2 and tensor.shape[0] == batch_size:
        return tensor.clone()
    raise ValueError(f"Expected {name} shape ({batch_size}, D) or (D,), got {tuple(tensor.shape)}")


@dataclass(frozen=True)
class SMPFeatureLayout:
    feature_dim: int
    base_lin_vel_b: tuple[int, int]
    base_ang_vel_b: tuple[int, int]
    joint_pos_rel: tuple[int, int]
    ee_pos_b: tuple[int, int] | None = None
    key_body_rot6d: tuple[int, int] | None = None

    @classmethod
    def from_feature_block_offsets(cls, feature_block_offsets: dict[str, tuple[int, int]]):
        required_keys = ("base_lin_vel_b", "base_ang_vel_b", "joint_pos_rel")
        missing_keys = [key for key in required_keys if key not in feature_block_offsets]
        if missing_keys:
            raise KeyError(f"Missing required feature block offsets: {missing_keys}")
        feature_dim = max(int(stop) for _, stop in feature_block_offsets.values())
        return cls(
            feature_dim=feature_dim,
            base_lin_vel_b=tuple(map(int, feature_block_offsets["base_lin_vel_b"])),
            base_ang_vel_b=tuple(map(int, feature_block_offsets["base_ang_vel_b"])),
            joint_pos_rel=tuple(map(int, feature_block_offsets["joint_pos_rel"])),
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
    def joint_dim(self) -> int:
        return self.joint_pos_rel[1] - self.joint_pos_rel[0]


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

    def __init__(self, feature_layout: SMPFeatureLayout, error_threshold: float = 1.0e-6):
        self.feature_layout = feature_layout
        self.error_threshold = float(error_threshold)

    def _unrecoverable_feature_blocks(self) -> tuple[str, ...]:
        missing = []
        if self.feature_layout.ee_pos_b is not None:
            missing.append("ee_pos_b")
        if self.feature_layout.key_body_rot6d is not None:
            missing.append("key_body_rot6d")
        return tuple(missing)

    def _recoverable_target(self, last_frame: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                last_frame[:, self.feature_layout.base_lin_vel_b[0] : self.feature_layout.base_lin_vel_b[1]],
                last_frame[:, self.feature_layout.base_ang_vel_b[0] : self.feature_layout.base_ang_vel_b[1]],
                last_frame[:, self.feature_layout.joint_pos_rel[0] : self.feature_layout.joint_pos_rel[1]],
            ),
            dim=-1,
        )

    def _recoverable_reencode(self, state: SMPResetState, reference_state: SMPResetReference) -> torch.Tensor:
        joint_pos_default = _expand_batch_tensor(reference_state.joint_pos, state.joint_pos.shape[0], "joint_pos")
        return torch.cat(
            (
                _quat_apply_inverse(state.root_quat_w, state.root_lin_vel_w),
                _quat_apply_inverse(state.root_quat_w, state.root_ang_vel_w),
                state.joint_pos - joint_pos_default,
            ),
            dim=-1,
        )

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
        root_quat_w = _expand_batch_tensor(reference_state.root_quat_w, batch_size, "root_quat_w")
        joint_pos_default = _expand_batch_tensor(reference_state.joint_pos, batch_size, "joint_pos")
        if reference_state.joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos_default)
        else:
            joint_vel = _expand_batch_tensor(reference_state.joint_vel, batch_size, "joint_vel")

        base_lin_vel_b = last_frame[:, self.feature_layout.base_lin_vel_b[0] : self.feature_layout.base_lin_vel_b[1]]
        base_ang_vel_b = last_frame[:, self.feature_layout.base_ang_vel_b[0] : self.feature_layout.base_ang_vel_b[1]]
        joint_pos_rel = last_frame[:, self.feature_layout.joint_pos_rel[0] : self.feature_layout.joint_pos_rel[1]]

        state = SMPResetState(
            root_pos_w=root_pos_w,
            root_quat_w=_normalize_quat(root_quat_w),
            root_lin_vel_w=_quat_apply(root_quat_w, base_lin_vel_b),
            root_ang_vel_w=_quat_apply(root_quat_w, base_ang_vel_b),
            joint_pos=joint_pos_default + joint_pos_rel,
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
