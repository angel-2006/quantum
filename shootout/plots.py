"""All figures. Uses the non-interactive Agg backend so it works anywhere."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .strategies import LABELS, STRATEGIES

COLORS = {"none": "#8c8c8c", "rem": "#2a9d8f", "zne": "#e9a23b", "rem+zne": "#3b5bdb"}


def fig_main_errors(df, path):
    """Grouped bars: mean absolute error per benchmark and strategy."""
    benches = list(dict.fromkeys(df.benchmark))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.2
    x = np.arange(len(benches) + 1)
    labels = benches + ["Average"]
    for i, s in enumerate(STRATEGIES):
        means, stds = [], []
        for b in benches:
            e = df[(df.benchmark == b) & (df.strategy == s)].abs_error
            means.append(e.mean()); stds.append(e.std())
        means.append(np.mean(means)); stds.append(0)
        ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=2, label=LABELS[s], color=COLORS[s])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean absolute error  |estimate - ideal|")
    ax.set_title("Error of each mitigation strategy (lower is better)\nerror bars = std over trials")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_distributions(df, path):
    """Box plots of the raw estimates: shows bias (distance from ideal) vs spread."""
    benches = list(dict.fromkeys(df.benchmark))
    fig, axes = plt.subplots(1, len(benches), figsize=(3.2 * len(benches), 4.4), sharey=False)
    for ax, b in zip(np.atleast_1d(axes), benches):
        sub = df[df.benchmark == b]
        data = [sub[sub.strategy == s].estimate.values for s in STRATEGIES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, s in zip(bp["boxes"], STRATEGIES):
            patch.set_facecolor(COLORS[s]); patch.set_alpha(0.8)
        ax.axhline(sub.ideal.iloc[0], color="k", ls="--", lw=1, label="ideal")
        ax.set_xticklabels(["none", "REM", "ZNE", "REM+\nZNE"])
        ax.set_title(b); ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Estimated expectation value")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle("Spread of estimates over trials: dashed line = exact answer", y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_sweeps(sweep_df, path):
    """Average error vs gate-noise multiplier (left) and readout-noise multiplier (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, exp, xl in ((axes[0], "sweep_gate", "gate-noise multiplier (x default)"),
                        (axes[1], "sweep_readout", "readout-noise multiplier (x default)")):
        sub = sweep_df[sweep_df.experiment == exp]
        for s in STRATEGIES:
            per_bench = (sub[sub.strategy == s].groupby(["factor", "benchmark"]).abs_error.mean()
                         .groupby("factor").mean())
            ax.plot(per_bench.index, per_bench.values, "o-", color=COLORS[s], label=LABELS[s])
        ax.set_xlabel(xl); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean absolute error (avg over 4 circuits)")
    axes[0].legend()
    fig.suptitle("Which strategy wins as noise grows?")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_zne(illustration, ideal, bench_name, path):
    """Expectation value vs noise scale factor, with the Richardson extrapolation to 0."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for (label, (scales, vals, limit)), col in zip(illustration.items(), ("#e9a23b", "#3b5bdb")):
        scales, vals = np.array(scales, float), np.array(vals, float)
        xs = np.linspace(0, scales.max(), 100)
        coef = np.polyfit(scales, vals, len(scales) - 1)
        ax.plot(xs, np.polyval(coef, xs), "--", color=col, alpha=0.7)
        ax.plot(scales, vals, "o", color=col, label=f"{label}: measured")
        ax.plot([0], [limit], "*", ms=15, color=col, label=f"{label}: extrapolated = {limit:.3f}")
    ax.axhline(ideal, color="k", lw=1, label=f"ideal = {ideal:.3f}")
    ax.set_xlabel("noise scale factor (1 = original circuit, 3 = U U† U folding, ...)")
    ax.set_ylabel("expectation value")
    ax.set_title(f"Zero-noise extrapolation on {bench_name}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
