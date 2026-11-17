"""M4 evaluation metrics, following the official M4 competition definitions.

* sMAPE = 200/H * sum |y - yhat| / (|y| + |yhat|)
* MASE  = mean|y - yhat| / mean_{t=m+1..n} |x_t - x_{t-m}|, scale from the *in-sample* series
* OWA   = 0.5 * (sMAPE / sMAPE_Naive2 + MASE / MASE_Naive2), where both ratios use the
  metric averaged over all series of the group (not per-series ratios).

Note: the paper's printed MASE sums the scale term to T+H; the official M4 evaluation (and
the published Naive2 reference values we check against) uses the in-sample part only.
"""
from __future__ import annotations

import numpy as np


def smape(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Per-series sMAPE in percent. y, yhat: (n, H)."""
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(np.abs(y - yhat), denom, out=np.zeros_like(denom, dtype=np.float64), where=denom != 0)
    return 200.0 * ratio.mean(axis=1)


def mase_scale(insample: np.ndarray, m: int) -> float:
    """Mean absolute seasonal-naive in-sample error. Falls back to m=1 for very short series."""
    if len(insample) <= m:
        m = 1
    return float(np.mean(np.abs(insample[m:] - insample[:-m])))


def mase(insample: list[np.ndarray], y: np.ndarray, yhat: np.ndarray, m: int) -> np.ndarray:
    scales = np.array([mase_scale(x, m) for x in insample])
    return np.abs(y - yhat).mean(axis=1) / scales


def owa(smape_model: float, mase_model: float, smape_naive2: float, mase_naive2: float) -> float:
    return 0.5 * (smape_model / smape_naive2 + mase_model / mase_naive2)
