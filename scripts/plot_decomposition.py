"""Figure: DeepBasis-I forecasts split into trend and seasonality stacks for real M4 series.

Trains (or loads) one interpretable model on Monthly and plots, for a few test series,
history + truth + forecast, and the trend vs. seasonality stack outputs.
Usage: python scripts/plot_decomposition.py [--iterations 1000] [--series M1 M100 ...]
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import torch

from deepbasis import data
from deepbasis.sampler import last_window
from deepbasis.train import RESULTS, RunConfig, build, device, train

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="Monthly")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--series", nargs="*", default=None)
    args = ap.parse_args()

    cfg = RunConfig(args.freq, "interpretable", "smape", 3, iterations=args.iterations, save_model=True)
    split = data.load(args.freq)
    H = split.freq.horizon
    mp = RESULTS / "models" / args.freq / f"{cfg.name}-it{args.iterations}.pt"
    if not mp.exists():
        train(cfg, log_every=250, split=split)
    dev = device()
    net = build(cfg, H).to(dev)
    net.load_state_dict(torch.load(mp, map_location=dev))
    net.eval()

    ids = list(split.ids)
    if args.series:
        idx = [ids.index(s) for s in args.series]
    else:  # pick clearly seasonal, reasonably long series deterministically
        from deepbasis.naive2 import seasonality_test
        cands = [i for i, s in enumerate(split.train) if len(s) > 120 and seasonality_test(s, split.freq.seasonality)]
        idx = list(np.random.default_rng(3).choice(cands, 4, replace=False))

    x, m = last_window([split.train[i] for i in idx], cfg.lookback * H)
    xt, mt = torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)
    level, parts = net.decompose(xt, mt)
    level = level.cpu().numpy()
    trend = level + sum(p.cpu().numpy() for p in parts[:3])
    season = sum(p.cpu().numpy() for p in parts[3:])
    fc = trend + season

    fig, axes = plt.subplots(len(idx), 2, figsize=(11, 2.6 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        hist = split.train[i][-4 * H:]
        th = np.arange(len(hist))
        tf = np.arange(len(hist), len(hist) + H)
        ax = axes[r, 0]
        ax.plot(th, hist, color="0.3", lw=1.2, label="history")
        ax.plot(tf, split.test[i], color="0.3", ls="--", lw=1.2, label="actual")
        ax.plot(tf, fc[r], color="C0", lw=2, label="DeepBasis-I forecast")
        ax.set_title(f"{split.ids[i]}: forecast", fontsize=10)
        ax2 = axes[r, 1]
        ax2.plot(tf, trend[r], color="C1", lw=2, label="trend stack (+ level)")
        ax2.plot(tf, season[r], color="C2", lw=2, label="seasonality stack")
        ax2.axhline(0, color="0.8", lw=0.8)
        ax2.set_title(f"{split.ids[i]}: stack outputs", fontsize=10)
        if r == 0:
            ax.legend(frameon=False, fontsize=8)
            ax2.legend(frameon=False, fontsize=8)
    fig.suptitle(f"DeepBasis-I (interpretable) on M4 {args.freq} ({args.iterations} iterations, single model)", fontsize=11)
    fig.tight_layout()
    out = RESULTS / "figures" / f"decomposition_{args.freq.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
