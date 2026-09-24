"""Accuracy on all classes seen so far after each task, one panel per benchmark.

    python scripts/plot_curves.py --out figures/accuracy_curves.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TITLES = {"mnist": "MNIST (5/2)", "svhn": "Balanced SVHN (5/2)",
          "cifar10": "CIFAR-10 (5/2)", "cifar100": "CIFAR-100 (10/10)"}
# categorical slots 1-3 of the reference palette (validated all-pairs), neutrals for references
SERIES = [("ahr", "AHR", "#2a78d6"), ("icarl", "iCaRL", "#eb6834"), ("ft_e", "FT-E", "#1baf7a")]
NEUTRAL = "#8a8985"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(results):
    runs = defaultdict(list)
    for p in glob.glob(os.path.join(results, "*", "*.json")):
        r = json.load(open(p))
        if not r.get("tag"):
            runs[(r["dataset"], r["method"])].append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = load(a.results)
    datasets = [d for d in TITLES if any(k[0] == d and k[1] in ("ahr", "icarl", "ft_e") for k in runs)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False)
    plt.rcParams.update({"font.size": 9})
    for ax, d in zip(axes[0], datasets):
        ax.set_facecolor("#fcfcfb")
        for key in ("ft",):
            if (d, key) in runs:
                accs = np.mean([r["accs"] for r in runs[(d, key)]], 0)
                t = np.arange(1, len(accs) + 1)
                ax.plot(t, accs, color=NEUTRAL, lw=1.5, ls=":", marker="o", ms=4, zorder=2)
                ax.annotate("FT", (t[-1], accs[-1]), xytext=(4, 0), textcoords="offset points",
                            color=TEXT2, va="center", fontsize=8)
        n = max(len(r["accs"]) for k, v in runs.items() if k[0] == d for r in v)
        if (d, "joint") in runs:
            j = np.mean([r["final_acc"] for r in runs[(d, "joint")]])
            ax.axhline(j, color=NEUTRAL, lw=1.2, ls="--", zorder=1)
            ax.annotate(f"Joint {j:.1f}", (n + 0.85, j), xytext=(0, 3), textcoords="offset points",
                        color=TEXT2, fontsize=8, ha="right", va="bottom")
        for key, label, color in SERIES:
            if (d, key) not in runs:
                continue
            accs = np.mean([r["accs"] for r in runs[(d, key)]], 0)
            t = np.arange(1, len(accs) + 1)
            ax.plot(t, accs, color=color, lw=2, marker="o", ms=5, zorder=3,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
            ax.annotate(f"{label} {accs[-1]:.1f}", (t[-1], accs[-1]), xytext=(5, 0), textcoords="offset points",
                        color=TEXT, va="center", fontsize=8)
        ax.set_title(TITLES[d], color=TEXT, fontsize=10, loc="left")
        ax.set_xticks(range(1, n + 1))
        ax.set_xlim(0.7, n + 1.3)
        ax.set_ylim(0, 101)
        ax.set_xlabel("task", color=TEXT2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT2, length=0)
    axes[0][0].set_ylabel("accuracy on all seen classes (%)", color=TEXT2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
    fig.patch.set_facecolor("#fcfcfb")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", a.out)


if __name__ == "__main__":
    main()
