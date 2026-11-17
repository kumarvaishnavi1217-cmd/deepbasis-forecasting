"""N-BEATS architecture (Section 3 of the paper).

A block maps a lookback window to expansion coefficients theta through 4 FC+ReLU layers,
then projects theta onto a backcast basis and a forecast basis. Blocks are chained by
doubly residual stacking: x_{l} = x_{l-1} - backcast_{l-1};  forecast = sum_l forecast_l.

* Generic: the basis is a learned linear projection (N-BEATS-G).
* Interpretable: a trend stack with a polynomial basis and a seasonality stack with a
  Fourier basis; weights are shared within each stack (N-BEATS-I).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, input_size: int, theta_size: int, basis: nn.Module, layers: int, width: int):
        super().__init__()
        fc: list[nn.Module] = [nn.Linear(input_size, width), nn.ReLU()]
        for _ in range(layers - 1):
            fc += [nn.Linear(width, width), nn.ReLU()]
        self.fc = nn.Sequential(*fc)
        self.theta = nn.Linear(width, theta_size)
        self.basis = basis

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.basis(self.theta(self.fc(x)))


class GenericBasis(nn.Module):
    """theta = [backcast coefficients | forecast coefficients]; the FC theta layer is the basis."""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size, self.forecast_size = backcast_size, forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    """Polynomial basis of degree p over normalised time t in [0, 1) (Eq. 2)."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.p = degree + 1
        bt = np.stack([(np.arange(backcast_size) / backcast_size) ** i for i in range(self.p)])
        ft = np.stack([(np.arange(forecast_size) / forecast_size) ** i for i in range(self.p)])
        self.register_buffer("backcast_time", torch.tensor(bt, dtype=torch.float32))
        self.register_buffer("forecast_time", torch.tensor(ft, dtype=torch.float32))

    def forward(self, theta: torch.Tensor):
        backcast = theta[:, self.p:] @ self.backcast_time
        forecast = theta[:, : self.p] @ self.forecast_time
        return backcast, forecast

    @staticmethod
    def theta_size(degree: int) -> int:
        return 2 * (degree + 1)


class SeasonalityBasis(nn.Module):
    """Fourier basis with harmonics i = 0..floor(H/2 - 1) (Eq. 3)."""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        freqs = np.arange(0, self.n_harmonics(forecast_size))[None, :]
        bgrid = -2 * np.pi * (np.arange(backcast_size)[:, None] / forecast_size) * freqs
        fgrid = 2 * np.pi * (np.arange(forecast_size)[:, None] / forecast_size) * freqs
        for name, grid in (("b", bgrid), ("f", fgrid)):
            self.register_buffer(f"{name}_cos", torch.tensor(np.cos(grid).T, dtype=torch.float32))
            self.register_buffer(f"{name}_sin", torch.tensor(np.sin(grid).T, dtype=torch.float32))
        self.k = freqs.shape[1]

    @staticmethod
    def n_harmonics(forecast_size: int) -> int:
        return max(1, forecast_size // 2)

    @classmethod
    def theta_size(cls, forecast_size: int) -> int:
        return 4 * cls.n_harmonics(forecast_size)

    def forward(self, theta: torch.Tensor):
        k = self.k
        forecast = theta[:, :k] @ self.f_cos + theta[:, k: 2 * k] @ self.f_sin
        backcast = theta[:, 2 * k: 3 * k] @ self.b_cos + theta[:, 3 * k:] @ self.b_sin
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Input is reversed so position 0 is always the most recent observation (handles
        # zero-padded short series consistently); forecast starts from the last level.
        residuals = x.flip(dims=(1,))
        mask = mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * mask
            forecast = forecast + block_forecast
        return forecast


    @torch.no_grad()
    def decompose(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the level term and each block's forecast contribution (they sum to forward())."""
        residuals = x.flip(dims=(1,))
        mask = mask.flip(dims=(1,))
        level = x[:, -1:].expand(-1, self._horizon(x, mask))
        parts = []
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * mask
            parts.append(block_forecast)
        return level, parts

    def _horizon(self, x, mask) -> int:
        return self.blocks[0](x.flip(dims=(1,)))[1].shape[1]


def generic(input_size: int, output_size: int, stacks: int = 30, layers: int = 4, width: int = 512) -> NBeats:
    """N-BEATS-G: 30 stacks x 1 block, 4 layers of width 512, no sharing (Table 18)."""
    return NBeats([
        Block(input_size, input_size + output_size, GenericBasis(input_size, output_size), layers, width)
        for _ in range(stacks)
    ])


def interpretable(
    input_size: int,
    output_size: int,
    trend_blocks: int = 3, trend_layers: int = 4, trend_width: int = 256, degree: int = 2,
    season_blocks: int = 3, season_layers: int = 4, season_width: int = 2048,
) -> NBeats:
    """N-BEATS-I: trend stack then seasonality stack, weights shared within each stack (Table 18)."""
    trend = Block(input_size, TrendBasis.theta_size(degree),
                  TrendBasis(degree, input_size, output_size), trend_layers, trend_width)
    season = Block(input_size, SeasonalityBasis.theta_size(output_size),
                   SeasonalityBasis(input_size, output_size), season_layers, season_width)
    # Sharing at stack level = the same module object repeated.
    return NBeats([trend] * trend_blocks + [season] * season_blocks)
