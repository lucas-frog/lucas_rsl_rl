from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.diffusion.composition import compose_style_predictions_with_body_masks
from rsl_rl.diffusion.conditioning import NULL_STYLE_ID, apply_classifier_free_guidance
from rsl_rl.diffusion.scheduler import DiffusionScheduler


def _infer_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class SMPDiffusionSampler:
    """执行 SMP 先验的反向扩散采样，并复用在线阶段的 style program 语义。"""

    def __init__(
        self,
        model: nn.Module,
        num_diffusion_steps: int,
        feature_dim: int,
        window_size: int,
        device: str | torch.device | None = None,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ):
        self.model = model
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.device = torch.device(device) if device is not None else _infer_model_device(model)
        self.scheduler = DiffusionScheduler(
            num_steps=self.num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    @property
    def num_styles(self) -> int:
        return int(getattr(self.model, "num_styles", 0))

    def _full_style_id(self, batch_size: int, style_id: int) -> torch.Tensor:
        return torch.full((batch_size,), int(style_id), device=self.device, dtype=torch.long)

    def _forward_model(self, xt: torch.Tensor, t: torch.Tensor, style_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_styles <= 0:
            return self.model(xt, t)
        return self.model(xt, t, style_id=style_id)

    def _resolve_style_program(
        self,
        batch_size: int,
        style_program: dict[str, object] | None = None,
        style_id: torch.Tensor | int | None = None,
        guidance_scale: float = 1.0,
    ) -> dict[str, object]:
        if style_program is not None:
            return style_program
        if style_id is None or self.num_styles <= 0:
            return {"mode": "unconditional", "guidance_scale": 1.0}
        if isinstance(style_id, torch.Tensor):
            style_id_tensor = style_id.to(device=self.device, dtype=torch.long)
            if style_id_tensor.ndim != 1 or style_id_tensor.shape[0] != batch_size:
                raise ValueError(f"Expected style_id shape ({batch_size},), got {tuple(style_id_tensor.shape)}")
            return {
                "mode": "single_style_batch",
                "guidance_scale": float(guidance_scale),
                "target_style_id": style_id_tensor,
            }
        return {
            "mode": "single_style",
            "guidance_scale": float(guidance_scale),
            "target_style_id": int(style_id),
        }

    def predict_eps(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        style_program: dict[str, object] | None = None,
        style_id: torch.Tensor | int | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """根据单风格或身体掩码组合程序预测 epsilon。"""
        if xt.ndim != 3:
            raise ValueError(f"Expected xt shape (batch, window, feature), got {tuple(xt.shape)}")
        if xt.shape[1] != self.window_size or xt.shape[2] != self.feature_dim:
            raise ValueError(
                f"Expected xt shape (*, {self.window_size}, {self.feature_dim}), got {tuple(xt.shape)}"
            )
        t = t.to(device=self.device, dtype=torch.long)
        xt = xt.to(self.device)
        program = self._resolve_style_program(
            batch_size=xt.shape[0],
            style_program=style_program,
            style_id=style_id,
            guidance_scale=guidance_scale,
        )

        mode = str(program["mode"])
        if mode == "unconditional":
            return self._forward_model(xt, t)

        eps_uncond = self._forward_model(xt, t, style_id=self._full_style_id(xt.shape[0], NULL_STYLE_ID))
        if mode in {"single_style", "single_style_batch"}:
            target_style_id = program["target_style_id"]
            if isinstance(target_style_id, torch.Tensor):
                target_style_id_tensor = target_style_id.to(device=self.device, dtype=torch.long)
            else:
                target_style_id_tensor = self._full_style_id(xt.shape[0], int(target_style_id))
            eps_cond = self._forward_model(xt, t, style_id=target_style_id_tensor)
            return apply_classifier_free_guidance(eps_uncond, eps_cond, float(program["guidance_scale"]))

        if mode != "body_mask":
            raise ValueError(f"Unsupported style program mode: {mode}")

        part_to_eps: dict[str, torch.Tensor] = {}
        eps_cache: dict[int, torch.Tensor] = {}
        for part_name, part_style_id in dict(program["part_style_ids"]).items():
            part_style_id = int(part_style_id)
            if part_style_id not in eps_cache:
                eps_cache[part_style_id] = self._forward_model(
                    xt,
                    t,
                    style_id=self._full_style_id(xt.shape[0], part_style_id),
                )
            part_to_eps[str(part_name)] = eps_cache[part_style_id]
        eps_cond_comp = compose_style_predictions_with_body_masks(
            part_to_eps,
            dict(program["feature_masks"]),
        )
        guidance_scale = float(program.get("guidance_scale", 1.0))
        if guidance_scale == 1.0:
            return eps_cond_comp
        return apply_classifier_free_guidance(eps_uncond, eps_cond_comp, guidance_scale)

    def p_sample(self, xt: torch.Tensor, timestep: int, eps_hat: torch.Tensor) -> torch.Tensor:
        """执行一步 DDPM 反向采样。"""
        if eps_hat.shape != xt.shape:
            raise ValueError(f"eps_hat must match xt shape, got {tuple(eps_hat.shape)} and {tuple(xt.shape)}")
        beta_t = self.scheduler.beta.to(device=xt.device, dtype=xt.dtype)[timestep]
        alpha_t = self.scheduler.alpha.to(device=xt.device, dtype=xt.dtype)[timestep]
        alpha_bar_t = self.scheduler.alpha_bar.to(device=xt.device, dtype=xt.dtype)[timestep]

        mean = (xt - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps_hat) / alpha_t.sqrt()
        if timestep == 0:
            return mean

        alpha_bar_prev = self.scheduler.alpha_bar.to(device=xt.device, dtype=xt.dtype)[timestep - 1]
        posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        noise = torch.randn_like(xt)
        return mean + posterior_var.sqrt() * noise

    def sample(
        self,
        batch_size: int,
        style_id: torch.Tensor | int | None = None,
        *,
        style_program: dict[str, object] | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """从标准高斯噪声开始反向采样一个 motion window。"""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        xt = torch.randn(batch_size, self.window_size, self.feature_dim, device=self.device)
        with torch.inference_mode():
            for timestep in reversed(range(self.num_diffusion_steps)):
                t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
                eps_hat = self.predict_eps(
                    xt,
                    t,
                    style_program=style_program,
                    style_id=style_id,
                    guidance_scale=guidance_scale,
                )
                xt = self.p_sample(xt, timestep, eps_hat)
        return xt
