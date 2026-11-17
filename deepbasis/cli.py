"""Command line entry point.

  deepbasis download
  deepbasis grid --profile laptop            # train the ensemble (resumable)
  deepbasis evaluate                         # results/RESULTS.md + results/results.json
"""
from __future__ import annotations

import argparse
import itertools
import time

from . import data
from .train import RESULTS, RunConfig, train

# Ensemble profiles. "paper" is the published N-BEATS protocol (180 models/frequency = 3 losses x
# 6 lookbacks x 10 seeds, Table 18 iterations). "laptop" is what we could afford on an
# Apple M1 (8 GB): 6 members/frequency, 1/7.5 (Y/Q/M) or 1/5 (W/D/H) of the iterations
# -> 54k model-iterations vs 10.8M for the published N-BEATS-G ensemble (0.5%).
PROFILES = {
    "paper": {
        "losses": ["smape", "mape", "mase"], "lookbacks": [2, 3, 4, 5, 6, 7], "seeds": list(range(10)),
        "iterations": {f: fr.iterations for f, fr in data.FREQUENCIES.items()}, "checkpoints": [],
    },
    "laptop": {
        "losses": ["smape", "mape", "mase"], "lookbacks": [3, 5], "seeds": [0],
        "iterations": {"Yearly": 2000, "Quarterly": 2000, "Monthly": 2000,
                       "Weekly": 1000, "Daily": 1000, "Hourly": 1000},
        "checkpoints": [250, 500, 1000],
    },
    "smoke": {
        "losses": ["smape"], "lookbacks": [3], "seeds": [0],
        "iterations": {f: 20 for f in data.FREQUENCIES}, "checkpoints": [10],
    },
}


def cmd_grid(args):
    prof = PROFILES[args.profile]
    freqs = args.freqs or list(data.FREQUENCIES)
    jobs = [RunConfig(f, args.arch, loss, lb, seed, iterations=prof["iterations"][f],
                      checkpoints=tuple(prof["checkpoints"]))
            for f in freqs
            for loss, lb, seed in itertools.product(prof["losses"], prof["lookbacks"], prof["seeds"])]
    todo = [j for j in jobs if not j.out_path(RESULTS).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    for i, job in enumerate(todo, 1):
        t = time.time()
        print(f"=== [{i}/{len(todo)}] {job.freq} {job.name} ({job.iterations} it)", flush=True)
        train(job, log_every=500)
        print(f"=== done in {time.time() - t:.0f}s", flush=True)


def cmd_train(args):
    train(RunConfig(args.freq, args.arch, args.loss, args.lookback, args.seed, iterations=args.iterations))


def cmd_evaluate(args):
    from .report import write_report
    write_report(RESULTS)


def main(argv=None):
    p = argparse.ArgumentParser(prog="deepbasis")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=lambda a: data.download())

    g = sub.add_parser("grid", help="train an ensemble grid (skips finished members)")
    g.add_argument("--profile", choices=PROFILES, default="laptop")
    g.add_argument("--arch", choices=["generic", "interpretable"], default="generic")
    g.add_argument("--freqs", nargs="*", choices=list(data.FREQUENCIES))
    g.set_defaults(fn=cmd_grid)

    t = sub.add_parser("train", help="train a single ensemble member")
    t.add_argument("--freq", required=True, choices=list(data.FREQUENCIES))
    t.add_argument("--arch", choices=["generic", "interpretable"], default="generic")
    t.add_argument("--loss", choices=["smape", "mape", "mase"], default="smape")
    t.add_argument("--lookback", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--iterations", type=int)
    t.set_defaults(fn=cmd_train)

    sub.add_parser("evaluate", help="score ensembles and write results/RESULTS.md").set_defaults(fn=cmd_evaluate)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
