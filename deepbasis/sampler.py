"""Training-window sampler (Section 5.2 / Appendix D.1).

Each batch: pick `batch_size` series uniformly with replacement; for each, pick a random
anchor among the last L_H * H points of its training part. Points before the anchor form the
input (lookback = k*H), the anchor and the following H-1 points form the target. Missing
points (series too short) are zero-filled and masked.
"""
from __future__ import annotations

import numpy as np


class WindowSampler:
    def __init__(self, series: list[np.ndarray], input_size: int, horizon: int,
                 history_size: float, batch_size: int, seed: int = 0):
        self.series = series
        self.input_size, self.horizon, self.batch_size = input_size, horizon, batch_size
        self.window_limit = int(history_size * horizon)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        b, n_in, h = self.batch_size, self.input_size, self.horizon
        x = np.zeros((b, n_in), np.float32)
        x_mask = np.zeros((b, n_in), np.float32)
        y = np.zeros((b, h), np.float32)
        y_mask = np.zeros((b, h), np.float32)
        for i, idx in enumerate(self.rng.integers(0, len(self.series), size=b)):
            s = self.series[idx]
            cut = self.rng.integers(max(1, len(s) - self.window_limit), len(s))
            ins = s[max(0, cut - n_in): cut]
            x[i, -len(ins):] = ins
            x_mask[i, -len(ins):] = 1.0
            out = s[cut: min(len(s), cut + h)]
            y[i, : len(out)] = out
            y_mask[i, : len(out)] = 1.0
        return x, x_mask, y, y_mask


def last_window(series: list[np.ndarray], input_size: int):
    """Most recent lookback window of every series, for producing the test forecast."""
    x = np.zeros((len(series), input_size), np.float32)
    mask = np.zeros_like(x)
    for i, s in enumerate(series):
        ins = s[-input_size:]
        x[i, -len(ins):] = ins
        mask[i, -len(ins):] = 1.0
    return x, mask
