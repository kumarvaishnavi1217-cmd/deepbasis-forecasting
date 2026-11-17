"""M4 dataset loading.

The raw CSVs (https://github.com/Mcompetitions/M4-methods/tree/master/Dataset) store one
series per row, right-padded with NaN. We load them into ragged lists of float32 arrays and
cache them as .npz so later runs skip parsing (~250 MB of CSV).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"


@dataclass(frozen=True)
class Frequency:
    name: str
    horizon: int        # H, fixed by the competition
    seasonality: int    # m, used by MASE and Naive2
    history_size: float  # L_H from Table 18 of the paper
    iterations: int      # training iterations from Table 18


FREQUENCIES: dict[str, Frequency] = {
    "Yearly":    Frequency("Yearly",    6,  1,  1.5, 15_000),
    "Quarterly": Frequency("Quarterly", 8,  4,  1.5, 15_000),
    "Monthly":   Frequency("Monthly",   18, 12, 1.5, 15_000),
    "Weekly":    Frequency("Weekly",    13, 1,  10,  5_000),
    "Daily":     Frequency("Daily",     14, 1,  10,  5_000),
    "Hourly":    Frequency("Hourly",    48, 24, 10,  5_000),
}
# The paper reports Weekly+Daily+Hourly together as "Others".
GROUPS = {
    "Yearly": ["Yearly"],
    "Quarterly": ["Quarterly"],
    "Monthly": ["Monthly"],
    "Others": ["Weekly", "Daily", "Hourly"],
    "Average": list(FREQUENCIES),
}


@dataclass
class M4Split:
    freq: Frequency
    ids: np.ndarray
    train: list[np.ndarray]
    test: np.ndarray  # (n_series, H)


def _read_ragged(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].to_numpy()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    series = [row[~np.isnan(row)] for row in values]
    return ids, series


def download(raw_dir: Path = DATA_DIR / "raw") -> None:
    """Download the raw M4 CSVs if missing."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in FREQUENCIES:
        for part in ("Train", "Test"):
            dest = raw_dir / f"{f}-{part.lower()}.csv"
            if not dest.exists():
                print(f"downloading {dest.name}")
                urllib.request.urlretrieve(f"{RAW_URL}/{part}/{dest.name}", dest)


def load(freq_name: str, data_dir: Path = DATA_DIR) -> M4Split:
    freq = FREQUENCIES[freq_name]
    cache = data_dir / "cache" / f"{freq_name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return M4Split(freq, z["ids"], list(z["train"]), z["test"])

    raw = data_dir / "raw"
    if not (raw / f"{freq_name}-train.csv").exists():
        download(raw)
    ids, train = _read_ragged(raw / f"{freq_name}-train.csv")
    test_ids, test = _read_ragged(raw / f"{freq_name}-test.csv")
    assert (ids == test_ids).all(), "train/test series order mismatch"
    test_arr = np.stack(test)
    assert test_arr.shape[1] == freq.horizon

    cache.parent.mkdir(parents=True, exist_ok=True)
    train_obj = np.empty(len(train), dtype=object)
    train_obj[:] = train
    np.savez(cache, ids=ids, train=train_obj, test=test_arr)
    return M4Split(freq, ids, train, test_arr)
