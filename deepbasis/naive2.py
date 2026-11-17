"""Naive2 benchmark of the M4 competition: seasonally adjusted naive forecast.

1. 90% autocorrelation test at lag m decides if a series is seasonal.
2. If seasonal: classical multiplicative decomposition (centred moving average) gives
   seasonal indices; the series is deseasonalised.
3. Forecast = last (deseasonalised) value, repeated H times, then reseasonalised.

Mirrors the reference implementation in the M4-methods repository (benchmarks).
"""
from __future__ import annotations

import numpy as np


def acf(x: np.ndarray, k: int) -> float:
    mean = x.mean()
    denom = np.sum((x - mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x[: len(x) - k] - mean) * (x[k:] - mean)) / denom)


def seasonality_test(x: np.ndarray, m: int) -> bool:
    """True if the series has significant autocorrelation at lag m (90% one-sided)."""
    if m <= 1 or len(x) < 3 * m:
        return False
    tcrit = 1.645
    acfs = np.array([acf(x, k) for k in range(1, m + 1)])
    limit = tcrit * np.sqrt((1 + 2 * np.sum(acfs[:-1] ** 2)) / len(x))
    return abs(acfs[-1]) > limit


def seasonal_indices(x: np.ndarray, m: int) -> np.ndarray:
    """Multiplicative seasonal indices (length m, aligned so index 0 = first observation)."""
    n = len(x)
    if m % 2 == 0:  # 2 x m centred moving average
        kernel = np.r_[0.5, np.ones(m - 1), 0.5] / m
    else:
        kernel = np.ones(m) / m
    trend = np.convolve(x, kernel, mode="valid")
    offset = (len(kernel) - 1) // 2
    ratios = np.full(n, np.nan)
    ratios[offset: offset + len(trend)] = x[offset: offset + len(trend)] / trend
    idx = np.array([np.nanmean(ratios[i::m]) for i in range(m)])
    return idx / idx.mean()


def forecast(x: np.ndarray, h: int, m: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if seasonality_test(x, m):
        si = seasonal_indices(x, m)
        in_idx = np.resize(si, len(x))
        last = x[-1] / in_idx[-1]
        out_idx = si[(np.arange(len(x), len(x) + h)) % m]
        return last * out_idx
    return np.full(h, x[-1])


def forecast_all(train: list[np.ndarray], h: int, m: int) -> np.ndarray:
    return np.stack([forecast(x, h, m) for x in train])
