from __future__ import annotations

import torch
import torch.nn as nn


NULL_STYLE_ID = -1


def maybe_drop_style(
    style_id: torch.Tensor | None,
    drop_prob: float,
    null_style_id: int = NULL_STYLE_ID,
) -> torch.Tensor | None:
    """按照给定概率把条件样本改写为 null-style。"""
    if style_id is None:
        return None
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
    if drop_prob == 0.0:
        return style_id
    dropped = style_id.clone()
    if drop_prob == 1.0:
        dropped.fill_(null_style_id)
        return dropped
    drop_mask = torch.rand(style_id.shape, device=style_id.device) < drop_prob
    dropped[drop_mask] = null_style_id
    return dropped


def apply_classifier_free_guidance(
    eps_uncond: torch.Tensor,
    eps_cond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """按标准 CFG 公式组合无条件与有条件预测。"""
    if eps_uncond.shape != eps_cond.shape:
        raise ValueError(f"CFG inputs must have the same shape, got {eps_uncond.shape} and {eps_cond.shape}")
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


class StyleConditioner(nn.Module):
    """把离散风格标签映射到模型隐空间。"""

    def __init__(
        self,
        num_styles: int,
        hidden_dim: int,
        null_style_id: int = NULL_STYLE_ID,
    ):
        super().__init__()
        if num_styles <= 0:
            raise ValueError(f"num_styles must be positive, got {num_styles}")
        self.num_styles = num_styles
        self.hidden_dim = hidden_dim
        self.null_style_id = null_style_id
        self.null_embedding_index = num_styles
        # 最后一行 embedding 专门保留给 null-style。
        self.embedding = nn.Embedding(num_styles + 1, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def _style_id_to_indices(
        self,
        style_id: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if style_id is None:
            return torch.full((batch_size,), self.null_embedding_index, device=device, dtype=torch.long)
        style_id = style_id.to(device=device, dtype=torch.long)
        if style_id.ndim != 1 or style_id.shape[0] != batch_size:
            raise ValueError(f"Expected style_id shape ({batch_size},), got {tuple(style_id.shape)}")

        style_indices = style_id.clone()
        null_mask = style_indices == self.null_style_id
        invalid_mask = (style_indices < 0) & ~null_mask
        if torch.any(invalid_mask):
            raise ValueError("style_id contains negative values other than null_style_id")
        if torch.any(style_indices[~null_mask] >= self.num_styles):
            raise ValueError(f"style_id must be smaller than num_styles={self.num_styles}")
        style_indices[null_mask] = self.null_embedding_index
        return style_indices

    def forward(
        self,
        style_id: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        style_indices = self._style_id_to_indices(style_id, batch_size=batch_size, device=device)
        return self.embedding(style_indices)
