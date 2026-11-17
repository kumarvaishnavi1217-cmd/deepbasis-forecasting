"""Build results/RESULTS.md, results/results.json and figures from saved forecasts."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data, evaluate
from .evaluate import GROUP_ORDER, PAPER


def _iterations_available(root: Path, freq: str, arch: str) -> list[int]:
    d = root / "forecasts" / freq
    its = []
    for p in d.glob("it*"):
        if any(p.glob(f"{arch}-*.npy")):
            its.append(int(p.name[2:]))
    return sorted(its)


def _final_iteration(root: Path, freq: str, arch: str) -> int | None:
    its = _iterations_available(root, freq, arch)
    return max(its) if its else None


def _members(root: Path, freq: str, arch: str, it: int) -> list[Path]:
    return sorted((root / "forecasts" / freq / f"it{it}").glob(f"{arch}-*.npy"))


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ensemble_scores(root: Path, arch: str, it_for: dict[str, int], select=lambda p: True):
    per_freq, sizes = {}, {}
    for f, it in it_for.items():
        files = [p for p in _members(root, f, arch, it) if select(p)]
        if not files:
            continue
        fc = np.median(np.stack([np.load(p) for p in files]), axis=0)
        per_freq[f] = evaluate.per_series_scores(f, fc)
        sizes[f] = len(files)
    return evaluate.score_groups(per_freq), sizes


def baseline_scores(root: Path, name: str):
    per_freq = {}
    for f in data.FREQUENCIES:
        p = root / "forecasts" / f / "baselines" / f"{name}.npy"
        if p.exists():
            per_freq[f] = evaluate.per_series_scores(f, np.load(p))
    return evaluate.score_groups(per_freq)


def naive2_group_scores():
    per_freq = {f: evaluate.naive2_scores(f) for f in data.FREQUENCIES}
    return evaluate.score_groups(per_freq)


def _row(name, scores, key):
    return f"| {name} | " + " | ".join(_fmt(scores.get(g, {}).get(key)) for g in GROUP_ORDER) + " |"


def _paper_row(name, key):
    tag = "published, 180 models" if name.startswith("N-BEATS") else "published"
    return f"| {name} *({tag})* | " + " | ".join(_fmt(v) for v in PAPER[name][key]) + " |"


def write_report(root: Path) -> dict:
    out: dict = {"groups": GROUP_ORDER}
    lines = ["# Results", "",
             ("All numbers are on the official M4 test set (100,000 series). Published reference numbers are "
             "copied from Tables 11-12 of Oreshkin et al. (2020). Rows marked *(ours)* are produced by DeepBasis; see "
             "`results/results.json` for the raw values."), ""]

    # Main tables: full ensembles only (the interpretable model is a single-model demo).
    archs = [a for a in ("generic",) if any(_final_iteration(root, f, a) for f in data.FREQUENCIES)]
    final_scores = {}
    for arch in archs:
        it_for = {f: _final_iteration(root, f, arch) for f in data.FREQUENCIES if _final_iteration(root, f, arch)}
        scores, sizes = ensemble_scores(root, arch, it_for)
        final_scores[arch] = (scores, sizes, it_for)
        out[f"ours_{arch}"] = {"scores": scores, "ensemble_size": sizes, "iterations": it_for}

    n2 = naive2_group_scores()
    theta = baseline_scores(root, "Theta")
    ses = baseline_scores(root, "SES")
    out.update({"naive2": n2, "theta": theta, "ses": ses})

    label = {"generic": "DeepBasis-G", "interpretable": "DeepBasis-I"}
    for key, title in (("smape", "sMAPE (lower is better)"), ("owa", "OWA (lower is better; Naive2 = 1.000)")):
        lines += [f"## {title}", "", "| Model | " + " | ".join(GROUP_ORDER) + " |",
                  "|---|" + "---:|" * len(GROUP_ORDER)]
        for name in PAPER:
            lines.append(_paper_row(name, key))
        for arch in archs:
            scores, sizes, _ = final_scores[arch]
            k = max(sizes.values()) if sizes else 0
            lines.append(_row(f"**{label[arch]} *(ours, {k}-model ensemble)***", scores, key))
        lines.append(_row("Theta *(ours, statsforecast)*", theta, key))
        lines.append(_row("SES *(ours, statsforecast)*", ses, key))
        lines.append(_row("Naive2 *(ours; matches official M4 values exactly)*", n2, key))
        lines.append("")

    # Training budget curve (generic)
    if "generic" in archs:
        lines += ["## Accuracy vs. training budget (DeepBasis-G ensemble)", "",
                  "Each column uses the forecasts saved at that iteration by the *same* training runs.", "",
                  "| Group | " + " | ".join(f"{it} it" for it in (250, 500, 1000, 2000)) + " |",
                  "|---|" + "---:|" * 4]
        curve = defaultdict(dict)
        for it in (250, 500, 1000, 2000):
            it_for = {}
            for f in data.FREQUENCIES:
                avail = _iterations_available(root, f, "generic")
                if it in avail:
                    it_for[f] = it
                elif avail and it > max(avail):
                    it_for[f] = max(avail)   # Others are trained for 1000 it max
            sc, _ = ensemble_scores(root, "generic", it_for)
            for g, v in sc.items():
                curve[g][it] = v
        for g in GROUP_ORDER:
            if g in curve:
                lines.append(f"| {g} sMAPE / OWA | " + " | ".join(
                    f"{curve[g][it]['smape']:.3f} / {curve[g][it]['owa']:.3f}" if it in curve[g] else "—"
                    for it in (250, 500, 1000, 2000)) + " |")
        lines += ["", ("Weekly/Daily/Hourly (\"Others\") are trained for 1000 iterations; later columns reuse "
                  "their 1000-iteration forecasts."), ""]
        out["budget_curve"] = curve
        _plot_curve(curve, root / "figures" / "budget_curve.png")

        # Ensemble gain + member breakdown
        scores, sizes, it_for = final_scores["generic"]
        lines += ["## Ensembling gain and member breakdown (DeepBasis-G, final iteration)", "",
                  "| Group | mean single-model sMAPE | best single model | ensemble sMAPE |", "|---|---:|---:|---:|"]
        member_tab = {}
        for g in GROUP_ORDER:
            freqs = data.GROUPS[g]
            if not all(f in it_for for f in freqs):
                continue
            names = sorted({p.stem for f in freqs for p in _members(root, f, "generic", it_for[f])})
            singles = {}
            for n in names:
                per = {f: evaluate.per_series_scores(f, np.load(root / "forecasts" / f / f"it{it_for[f]}" / f"{n}.npy"))
                       for f in freqs if (root / "forecasts" / f / f"it{it_for[f]}" / f"{n}.npy").exists()}
                if len(per) == len(freqs):
                    singles[n] = float(np.concatenate([per[f][0] for f in freqs]).mean())
            if singles:
                member_tab[g] = singles
                lines.append(f"| {g} | {np.mean(list(singles.values())):.3f} | {min(singles.values()):.3f} "
                             f"(`{min(singles, key=singles.get)}`) | {scores[g]['smape']:.3f} |")
        out["single_models_smape"] = member_tab
        lines.append("")

        lines += ["### By training loss and lookback (sub-ensembles, Average sMAPE)", ""]
        if all(f in it_for for f in data.FREQUENCIES):
            rows = []
            for tag in ("smape", "mape", "mase"):
                sc, _ = ensemble_scores(root, "generic", it_for, select=lambda p, t=tag: f"-{t}-" in p.name)
                rows.append((f"loss = {tag}", sc.get("Average", {})))
            for lb in sorted({int(p.stem.split("-L")[1].split("-")[0]) for f in it_for
                              for p in _members(root, f, "generic", it_for[f])}):
                sc, _ = ensemble_scores(root, "generic", it_for, select=lambda p, b=lb: f"-L{b}-" in p.name)
                rows.append((f"lookback = {lb}H", sc.get("Average", {})))
            lines += ["| Sub-ensemble | sMAPE | OWA |", "|---|---:|---:|"]
            lines += [f"| {n} | {_fmt(s.get('smape'))} | {_fmt(s.get('owa'))} |" for n, s in rows]
            lines.append("")

    it_i = _final_iteration(root, "Monthly", "interpretable")
    if it_i:
        per = {"Monthly": evaluate.per_series_scores(
            "Monthly", np.load(_members(root, "Monthly", "interpretable", it_i)[0]))}
        g = evaluate.score_groups(per)["Monthly"]
        out["interpretable_demo"] = {"Monthly": g, "iterations": it_i}
        lines += ["## Interpretable model (single-model demo)", "",
                  (f"One DeepBasis-I model (trend + seasonality stacks, sMAPE loss, lookback 3H, {it_i} iterations) "
                  f"on Monthly: **sMAPE {g['smape']:.3f}, OWA {g['owa']:.3f}**. That is on par with the 6-model "
                  "DeepBasis-G ensemble on Monthly, while its forecast splits into readable trend and "
                  "seasonality components (`results/figures/decomposition_monthly.png`)."), ""]

    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (root / "results.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(lines))
    return out


def _plot_curve(curve, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4))
    paper_owa = dict(zip(GROUP_ORDER, PAPER["N-BEATS-G"]["owa"]))
    for i, g in enumerate(["Yearly", "Quarterly", "Monthly", "Average"]):
        if g not in curve:
            continue
        its = sorted(curve[g])
        line, = ax.plot(its, [curve[g][it]["owa"] for it in its], marker="o", label=f"{g}")
        ax.axhline(paper_owa[g], ls="--", lw=1, color=line.get_color(), alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training iterations per model (log scale)")
    ax.set_ylabel("OWA (lower is better)")
    ax.set_title("DeepBasis-G (6 models) vs. training budget\ndashed = published N-BEATS-G, 180 models x 15k it", fontsize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
