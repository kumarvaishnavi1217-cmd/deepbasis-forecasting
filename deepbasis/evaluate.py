"""Ensemble member forecasts (median) and score them with sMAPE / MASE / OWA per M4 group."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data, metrics, naive2

# Published N-BEATS results, Oreshkin et al. (2020) Tables 11 (sMAPE) and 12 (OWA): 180-model ensembles.
PAPER = {
    "N-BEATS-G":   {"smape": [13.023, 9.212, 12.048, 3.574, 11.168], "owa": [0.765, 0.800, 0.820, 0.822, 0.797]},
    "N-BEATS-I":   {"smape": [12.924, 9.287, 12.059, 3.684, 11.174], "owa": [0.758, 0.807, 0.824, 0.849, 0.798]},
    "N-BEATS-I+G": {"smape": [12.913, 9.213, 12.024, 3.643, 11.135], "owa": [0.758, 0.800, 0.819, 0.840, 0.795]},
    "M4 winner (Smyl)": {"smape": [13.176, 9.679, 12.126, 4.014, 11.374], "owa": [0.778, 0.847, 0.836, 0.920, 0.821]},
}
GROUP_ORDER = ["Yearly", "Quarterly", "Monthly", "Others", "Average"]


def ensemble(freq: str, root: Path, iteration: int, pattern: str = "*.npy") -> tuple[np.ndarray | None, int]:
    files = sorted((root / "forecasts" / freq / f"it{iteration}").glob(pattern))
    if not files:
        return None, 0
    return np.median(np.stack([np.load(f) for f in files]), axis=0), len(files)


def per_series_scores(freq: str, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = data.load(freq)
    return metrics.smape(s.test, forecast), metrics.mase(s.train, s.test, forecast, s.freq.seasonality)


_NAIVE2_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def naive2_scores(freq: str) -> tuple[np.ndarray, np.ndarray]:
    if freq not in _NAIVE2_CACHE:
        s = data.load(freq)
        fc = naive2.forecast_all(s.train, s.freq.horizon, s.freq.seasonality)
        _NAIVE2_CACHE[freq] = per_series_scores(freq, fc)
    return _NAIVE2_CACHE[freq]


def score_groups(per_freq: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Aggregate per-series scores into the paper's groups. A group is scored only if all its
    frequencies are present."""
    out = {}
    for group, members in data.GROUPS.items():
        if not all(m in per_freq for m in members):
            continue
        sm = np.concatenate([per_freq[m][0] for m in members]).mean()
        ma = np.concatenate([per_freq[m][1] for m in members]).mean()
        n_sm = np.concatenate([naive2_scores(m)[0] for m in members]).mean()
        n_ma = np.concatenate([naive2_scores(m)[1] for m in members]).mean()
        out[group] = {"smape": float(sm), "mase": float(ma), "owa": metrics.owa(sm, ma, n_sm, n_ma)}
    return out
