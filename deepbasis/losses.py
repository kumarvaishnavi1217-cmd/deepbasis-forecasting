"""Scale-free training losses (the paper ensembles models trained on each of them)."""
from __future__ import annotations

import torch


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # torch.where keeps everything on-device (boolean-index assignment forces a sync on MPS)
    out = a / b
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def smape_loss(forecast, target, mask, insample=None, m=1):
    return 200 * torch.mean(_safe_div((forecast - target).abs(), forecast.detach().abs() + target.abs()) * mask)


def mape_loss(forecast, target, mask, insample=None, m=1):
    return torch.mean((forecast - target).abs() * _safe_div(mask, target.abs()))


def mase_loss(forecast, target, mask, insample, m=1, insample_mask=None):
    """MASE with the scale computed on the (masked) lookback window of each sample."""
    diffs = (insample[:, m:] - insample[:, :-m]).abs()
    if insample_mask is not None:
        diffs = diffs * insample_mask[:, m:] * insample_mask[:, :-m]
        count = (insample_mask[:, m:] * insample_mask[:, :-m]).sum(dim=1).clamp(min=1)
        scale = diffs.sum(dim=1) / count
    else:
        scale = diffs.mean(dim=1)
    return torch.mean((forecast - target).abs() * _safe_div(mask, scale[:, None]))


LOSSES = {"smape": smape_loss, "mape": mape_loss, "mase": mase_loss}
