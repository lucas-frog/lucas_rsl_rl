from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.diffusion.conditioning import NULL_STYLE_ID, maybe_drop_style
from rsl_rl.diffusion.ema import ExponentialMovingAverage
from rsl_rl.diffusion.logging import log_smp_pretrain_metrics
from rsl_rl.diffusion.model import MotionEpsilonTransformer
from rsl_rl.diffusion.scheduler import DiffusionScheduler
from rsl_rl.motion import SMPMotionWindowDataset


def _collate_smp_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    """将带元数据的 sample 聚合成 batch。"""
    motions = torch.stack([sample["motion"] for sample in samples], dim=0)
    style_ids = [sample["style_id"] for sample in samples]
    return {
        "motion": motions,
        "style_id": None if any(style_id is None for style_id in style_ids) else torch.tensor(style_ids, dtype=torch.long),
        "style_name": [sample["style_name"] for sample in samples],
        "clip_id": torch.tensor([int(sample["clip_id"]) for sample in samples], dtype=torch.long),
        "source_name": [str(sample["source_name"]) for sample in samples],
    }


class SMPDiffusionTrainer:
    """负责离线预训练 SMP diffusion prior，并记录日志与保存检查点。"""

    def __init__(
        self,
        dataset_path: str | Path,
        log_dir: str | Path,
        batch_size: int = 32,
        max_iters: int = 1000,
        window_size: int = 10,
        stride: int = 1,
        num_diffusion_steps: int = 50,
        timesteps_k: list[int] | tuple[int, ...] = (22, 15, 8),
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        learning_rate: float = 3.0e-4,
        ema_decay: float = 0.999,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        num_styles: int | None = None,
        style_drop_prob: float = 0.0,
        device: str | torch.device | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.log_dir = Path(log_dir)
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.timesteps_k = list(timesteps_k)
        self.style_drop_prob = style_drop_prob

        self.dataset = SMPMotionWindowDataset(self.dataset_path, window_size=window_size, stride=stride)
        inferred_num_styles = len(self.dataset.style_to_id)
        self.num_styles = inferred_num_styles if num_styles is None else num_styles
        if self.num_styles < inferred_num_styles:
            raise ValueError(
                f"num_styles={self.num_styles} is smaller than dataset styles={inferred_num_styles}"
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_smp_samples,
        )
        self._dataloader_iter = iter(self.dataloader)

        sample_window = self.dataset[0]["motion"]
        self.feature_dim = int(sample_window.shape[-1])
        self.window_size = int(sample_window.shape[0])

        self.scheduler = DiffusionScheduler(
            num_steps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.model = MotionEpsilonTransformer(
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            num_diffusion_steps=num_diffusion_steps,
            num_styles=self.num_styles,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ).to(self.device)
        self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)

    def _next_batch(self) -> dict[str, object]:
        """获取下一批数据，并把张量字段移动到目标设备。"""
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self.dataloader)
            batch = next(self._dataloader_iter)
        batch["motion"] = batch["motion"].to(self.device)
        if isinstance(batch["style_id"], torch.Tensor):
            batch["style_id"] = batch["style_id"].to(self.device)
        return batch

    def _compute_loss(
        self,
        batch: dict[str, object],
    ) -> tuple[
        torch.Tensor,
        dict[int, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        """执行一次前向扩散与噪声预测，返回损失和日志统计。"""
        x0 = batch["motion"]
        style_id = batch["style_id"] if isinstance(batch["style_id"], torch.Tensor) else None

        t = self.scheduler.sample_timesteps(x0.shape[0], device=self.device, timesteps_k=self.timesteps_k)
        eps = torch.randn_like(x0)
        xt = self.scheduler.q_sample(x0, t, eps)
        dropped_style_id = maybe_drop_style(style_id, self.style_drop_prob, null_style_id=NULL_STYLE_ID)
        eps_hat = self.model(xt, t, style_id=dropped_style_id)

        sample_mse = (eps_hat - eps).pow(2).flatten(start_dim=1).mean(dim=1)
        loss = sample_mse.mean()

        per_timestep_mse = {}
        for timestep in self.timesteps_k:
            timestep_mask = t == timestep
            if torch.any(timestep_mask):
                per_timestep_mse[int(timestep)] = sample_mse[timestep_mask].mean().detach()

        loss_cond = None
        loss_uncond = None
        if dropped_style_id is not None:
            uncond_mask = dropped_style_id == NULL_STYLE_ID
            cond_mask = ~uncond_mask
            if torch.any(cond_mask):
                loss_cond = sample_mse[cond_mask].mean().detach()
            if torch.any(uncond_mask):
                loss_uncond = sample_mse[uncond_mask].mean().detach()

        per_style_mse: dict[str, torch.Tensor] = {}
        style_names = batch["style_name"]
        if style_id is not None and isinstance(style_names, list):
            for index, style_name in enumerate(style_names):
                if style_name is None:
                    continue
                per_style_mse.setdefault(str(style_name), []).append(sample_mse[index].detach())
            per_style_mse = {
                style_name: torch.stack(style_losses).mean()
                for style_name, style_losses in per_style_mse.items()
            }

        return loss, per_timestep_mse, eps, eps_hat, loss_cond, loss_uncond, per_style_mse

    def save_checkpoint(self, output_path: str | Path | None = None) -> Path:
        """保存模型、EMA、优化器状态及关键配置。"""
        checkpoint_path = Path(output_path) if output_path is not None else self.log_dir / "model_latest.pt"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_dim": self.feature_dim,
            "window_size": self.window_size,
            "timesteps_k": self.timesteps_k,
            "model_cfg": {
                "feature_dim": self.feature_dim,
                "window_size": self.window_size,
                "num_diffusion_steps": self.scheduler.num_steps,
                "num_styles": self.num_styles,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": len(self.model.encoder.layers),
                "num_heads": self.model.encoder.layers[0].self_attn.num_heads,
            },
            "style_cfg": {
                "style_names": list(self.dataset.style_names),
                "style_to_id": dict(self.dataset.style_to_id),
                "null_style_id": NULL_STYLE_ID,
                "drop_prob": self.style_drop_prob,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def train(self) -> dict[str, object]:
        """执行离线预训练循环，并返回最终损失与产物路径。"""
        final_loss = None
        for global_step in range(1, self.max_iters + 1):
            batch = self._next_batch()
            loss, per_timestep_mse, eps, eps_hat, loss_cond, loss_uncond, per_style_mse = self._compute_loss(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.ema.update(self.model)

            final_loss = float(loss.detach().cpu().item())

            log_smp_pretrain_metrics(
                self.writer,
                global_step=global_step,
                loss=loss.detach(),
                per_timestep_mse=per_timestep_mse,
                eps=eps.detach(),
                eps_hat=eps_hat.detach(),
                loss_cond=loss_cond,
                loss_uncond=loss_uncond,
                per_style_mse=per_style_mse,
            )

        checkpoint_path = self.save_checkpoint()
        self.writer.flush()
        self.writer.close()
        return {
            "checkpoint_path": checkpoint_path,
            "final_loss": final_loss,
            "feature_dim": self.feature_dim,
            "window_size": self.window_size,
        }
