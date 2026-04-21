from __future__ import annotations

import torch


def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


def _to_histogram_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().reshape(-1)


def log_smp_noise_metrics(
    writer,
    global_step: int,
    noise_mse,
    per_timestep_mse: dict[int, float | torch.Tensor],
    eps,
    eps_hat,
    prefix: str = "SMP",
    write_histograms: bool = True,
):
    """记录噪声误差标量与直方图。"""
    writer.add_scalar(f"{prefix}/noise_mse", _to_scalar(noise_mse), global_step)
    for timestep, mse in sorted(per_timestep_mse.items()):
        writer.add_scalar(f"{prefix}/t{timestep}/noise_mse", _to_scalar(mse), global_step)

    if not write_histograms:
        return

    if isinstance(eps, dict):
        if not isinstance(eps_hat, dict):
            raise TypeError("eps_hat must also be a dict when eps is a dict")
        for timestep in sorted(eps.keys()):
            writer.add_histogram(f"{prefix}/eps_true_t{timestep}", _to_histogram_tensor(eps[timestep]), global_step)
            writer.add_histogram(f"{prefix}/eps_pred_t{timestep}", _to_histogram_tensor(eps_hat[timestep]), global_step)
            writer.add_histogram(
                f"{prefix}/eps_gap_t{timestep}",
                _to_histogram_tensor(eps_hat[timestep] - eps[timestep]),
                global_step,
            )
        return

    writer.add_histogram(f"{prefix}/eps", _to_histogram_tensor(eps), global_step)
    writer.add_histogram(f"{prefix}/eps_hat", _to_histogram_tensor(eps_hat), global_step)
    writer.add_histogram(f"{prefix}/eps_gap", _to_histogram_tensor(eps_hat - eps), global_step)


def log_smp_pretrain_metrics(
    writer,
    global_step: int,
    loss,
    per_timestep_mse: dict[int, float | torch.Tensor],
    eps: torch.Tensor,
    eps_hat: torch.Tensor,
    loss_cond: float | torch.Tensor | None = None,
    loss_uncond: float | torch.Tensor | None = None,
    per_style_mse: dict[str, float | torch.Tensor] | None = None,
    write_histograms: bool = False,
):
    """记录离线预训练阶段的噪声误差指标。"""
    writer.add_scalar("SMPPretrain/loss", _to_scalar(loss), global_step)
    writer.add_scalar("SMPPretrain/loss_total", _to_scalar(loss), global_step)
    if loss_cond is not None:
        writer.add_scalar("SMPPretrain/loss_cond", _to_scalar(loss_cond), global_step)
    if loss_uncond is not None:
        writer.add_scalar("SMPPretrain/loss_uncond", _to_scalar(loss_uncond), global_step)
    if per_style_mse:
        for style_name, mse in sorted(per_style_mse.items()):
            writer.add_scalar(f"SMPPretrain/style/{style_name}/noise_mse", _to_scalar(mse), global_step)
    log_smp_noise_metrics(
        writer,
        global_step=global_step,
        noise_mse=loss,
        per_timestep_mse=per_timestep_mse,
        eps=eps,
        eps_hat=eps_hat,
        prefix="SMPPretrain",
        write_histograms=write_histograms,
    )
