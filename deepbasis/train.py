"""Train one ensemble member and write its test-set forecast to disk."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import data
from . import model as nbeats
from .losses import LOSSES
from .sampler import WindowSampler, last_window

RESULTS = Path(__file__).resolve().parent.parent / "results"


@dataclass
class RunConfig:
    freq: str
    arch: str          # "generic" | "interpretable"
    loss: str          # "smape" | "mape" | "mase"
    lookback: int      # input size = lookback * H, lookback in 2..7
    seed: int = 0
    iterations: int | None = None   # None -> paper value from Table 18
    batch_size: int = 1024
    lr: float = 1e-3
    lr_decay: bool = True           # x0.5 every third of training (assumption; paper: "initial lr 0.001")
    grad_clip: float = 1.0
    checkpoints: tuple[int, ...] = ()  # also save test forecasts at these iterations
    save_model: bool = False           # write the final state_dict (used for interpretability plots)

    @property
    def name(self) -> str:
        return f"{self.arch}-{self.loss}-L{self.lookback}-s{self.seed}"

    def out_path(self, root: Path = RESULTS, iteration: int | None = None) -> Path:
        it = iteration or self.iterations
        return root / "forecasts" / self.freq / f"it{it}" / f"{self.name}.npy"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build(cfg: RunConfig, horizon: int) -> torch.nn.Module:
    n_in = cfg.lookback * horizon
    if cfg.arch == "generic":
        return nbeats.generic(n_in, horizon)
    if cfg.arch == "interpretable":
        return nbeats.interpretable(n_in, horizon)
    raise ValueError(cfg.arch)


def train(cfg: RunConfig, root: Path = RESULTS, log_every: int = 1000, split: data.M4Split | None = None) -> np.ndarray:
    split = split or data.load(cfg.freq)
    freq = split.freq
    iters = cfg.iterations = cfg.iterations or freq.iterations
    save_at = {c for c in cfg.checkpoints if c < iters} | {iters}
    torch.manual_seed(cfg.seed)
    dev = device()

    net = build(cfg, freq.horizon).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = LOSSES[cfg.loss]
    sampler = WindowSampler(split.train, cfg.lookback * freq.horizon, freq.horizon,
                            freq.history_size, cfg.batch_size, seed=cfg.seed)
    decay_step = max(1, iters // 3)

    t0 = time.time()
    net.train()
    for it in range(1, iters + 1):
        if cfg.lr_decay:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * 0.5 ** ((it - 1) // decay_step)
        x, xm, y, ym = (torch.from_numpy(a).to(dev) for a in sampler.sample())
        forecast = net(x, xm)
        loss = loss_fn(forecast, y, ym, insample=x, m=freq.seasonality, **({"insample_mask": xm} if cfg.loss == "mase" else {}))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        if log_every and it % log_every == 0:
            print(f"[{cfg.freq} {cfg.name}] it {it}/{iters} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if it in save_at:
            fc = predict(net, split.train, cfg.lookback * freq.horizon, dev)
            net.train()
            out = cfg.out_path(root, it)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, fc)
            out.with_suffix(".json").write_text(json.dumps({**asdict(cfg), "saved_at": it,
                                                            "seconds": round(time.time() - t0, 1),
                                                            "device": str(dev)}, indent=2))
    if cfg.save_model:
        mp = root / "models" / cfg.freq / f"{cfg.name}-it{iters}.pt"
        mp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), mp)
    return fc


@torch.no_grad()
def predict(net: torch.nn.Module, series: list[np.ndarray], input_size: int, dev: torch.device,
            batch: int = 4096) -> np.ndarray:
    net.eval()
    x, m = last_window(series, input_size)
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i: i + batch]).to(dev)
        mb = torch.from_numpy(m[i: i + batch]).to(dev)
        outs.append(net(xb, mb).cpu().numpy())
    return np.concatenate(outs)
