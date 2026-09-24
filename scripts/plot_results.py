"""Figures for results/RESULTS.md: reproduced vs. reported numbers.

    python scripts/plot_results.py --summary results/summary.json --out results/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = json.load(open(os.path.join(ROOT, "configs", "paper_results.json")))

OURS, THEIRS = "#2a78d6", "#eb6834"          # categorical slots 1-2 of the reference palette
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"hr": "HR", "hr_mini": "HR-mini", "hr_no_latent": "w/o latent ex.",
         "hr_perfect": "w perfect ex.", "hr_no_kd": "w/o KD", "hr_no_global": "w/o global replay",
         "hr_rfa": "w RFA", "fedavg_ft": "FedAvg-FT"}
ORDER = list(NAMES)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def curve_plot(rows, profile, bench, out):
    r = next((r for r in rows if r["profile"] == profile and r["benchmark"] == bench
              and r["variant"] == "hr" and r["curve"]), None)
    pc = PAPER["figure3"]["accuracy_per_task"].get(bench, {}).get("HR")
    if r is None:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    style(ax)
    xs = range(1, len(r["curve"]) + 1)
    if pc:
        ax.plot(range(1, len(pc) + 1), pc, color=THEIRS, lw=2, marker="o", ms=5, ls="--",
                label="HR, paper (Fig. 3)")
        ax.annotate(f"{pc[-1]:.1f}", (len(pc), pc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.plot(xs, r["curve"], color=OURS, lw=2, marker="o", ms=5,
            label=f"HR, reproduced ({r['n_seeds']} seed{'s' if r['n_seeds'] != 1 else ''})")
    ax.annotate(f"{r['curve'][-1]:.1f}", (len(r["curve"]), r["curve"][-1]), xytext=(6, 0),
                textcoords="offset points", color=INK2, fontsize=9, va="center")
    ax.set_xlabel("Task", color=INK2, fontsize=9)
    ax.set_ylabel("Accuracy on all seen classes (%)", color=INK2, fontsize=9)
    ax.set_title(f"{bench} ({profile})", color=INK, fontsize=10, loc="left")
    ax.set_xticks(list(xs))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    path = os.path.join(out, f"curve_{profile}_{bench}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def bar_plot(rows, profile, bench, out):
    sel = [r for r in rows if r["profile"] == profile and r["benchmark"] == bench
           and r["final_acc"] is not None]
    sel.sort(key=lambda r: ORDER.index(r["variant"]) if r["variant"] in ORDER else 99)
    if not sel:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    style(ax)
    w = 0.38
    for i, r in enumerate(sel):
        if r["paper_final"] is not None:
            ax.bar(i - w / 2 - 0.01, r["paper_final"], w, color=THEIRS, label="Paper (Table 2)" if i == 0 else None)
            ax.text(i - w / 2, r["paper_final"] + 0.8, f"{r['paper_final']:.1f}", ha="center",
                    fontsize=7.5, color=INK2)
        ax.bar(i + w / 2 + 0.01, r["final_acc"], w, color=OURS, label="Reproduced" if i == 0 else None,
               yerr=r["final_sem"] if r["final_sem"] == r["final_sem"] else None,
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.text(i + w / 2, r["final_acc"] + 0.8, f"{r['final_acc']:.1f}", ha="center", fontsize=7.5,
                color=INK2)
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([NAMES.get(r["variant"], r["variant"]) for r in sel], rotation=20,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("Final accuracy (%)", color=INK2, fontsize=9)
    ax.set_title(f"Table 2 rows, {bench} ({profile})", color=INK, fontsize=10, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    path = os.path.join(out, f"bars_{profile}_{bench}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="json written by aggregate.py --json")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    args = ap.parse_args()
    rows = json.load(open(args.summary))
    os.makedirs(args.out, exist_ok=True)
    made = []
    for profile, bench in sorted({(r["profile"], r["benchmark"]) for r in rows}):
        made += [p for p in (curve_plot(rows, profile, bench, args.out),
                             bar_plot(rows, profile, bench, args.out)) if p]
    print("\n".join(made))


if __name__ == "__main__":
    main()
