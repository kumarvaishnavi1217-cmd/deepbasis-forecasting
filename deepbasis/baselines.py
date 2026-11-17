"""Classical baselines: Theta (M4's strongest simple benchmark) and SES via statsforecast.

Seasonal series are handled by statsforecast's built-in classical multiplicative
decomposition (season_length = M4 seasonality), mirroring the M4 benchmark protocol.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run(freq: str, n_jobs: int = 4, root: Path = RESULTS) -> dict[str, np.ndarray]:
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalExponentialSmoothingOptimized, Theta  # noqa: F401
    from statsforecast.models import SimpleExponentialSmoothingOptimized as SES

    split = data.load(freq)
    fr = split.freq
    frames = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(s)), "y": s.astype(np.float64)})
              for i, s in zip(split.ids, split.train)]
    df = pd.concat(frames, ignore_index=True)
    models = [Theta(season_length=fr.seasonality, alias="Theta"), SES(alias="SES")]
    t = time.time()
    sf = StatsForecast(models=models, freq=1, n_jobs=n_jobs)
    fc = sf.forecast(df=df, h=fr.horizon)
    out = {}
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    # each series' ds continues from its own length, so align by position within a series
    fc = fc.sort_values(["unique_id", "ds"])
    pos = {uid: k for k, uid in enumerate(split.ids)}
    for name in ("Theta", "SES"):
        arr = np.empty((len(split.ids), fr.horizon))
        for uid, grp in fc.groupby("unique_id", sort=False):
            arr[pos[uid]] = grp[name].to_numpy()
        path = root / "forecasts" / freq / "baselines" / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        out[name] = arr
    print(f"{freq}: baselines done in {time.time() - t:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or list(data.FREQUENCIES):
        if not (RESULTS / "forecasts" / f / "baselines" / "Theta.npy").exists():
            run(f)
