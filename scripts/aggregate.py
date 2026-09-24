"""Collect result.json files and compare them with the numbers reported in the paper.

    python scripts/aggregate.py --results_root results [--markdown out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = json.load(open(os.path.join(ROOT, "configs", "paper_results.json")))

VARIANT_NAMES = {
    "hr": "HR",
    "hr_mini": "HR-mini w 10x less memory",
    "hr_no_latent": "HR w/o Latent Exemplars",
    "hr_perfect": "HR w Perfect Exemplars",
    "hr_no_kd": "HR w/o KD",
    "hr_no_global": "HR w/o Global Replay",
    "hr_rfa": "HR w RFA",
    "fedavg_ft": "FedAvg fine-tuning (lower bound, not in paper)",
}


def mean_sem(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(xs) == 0:
        return None, None
    sem = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return float(xs.mean()), float(sem)


def collect(root):
    groups = defaultdict(list)
    for p in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
        rel = os.path.relpath(p, root).split(os.sep)
        if len(rel) < 4:
            continue
        profile, bench, run = rel[-4], rel[-3], rel[-2]
        variant = run.rsplit("_seed", 1)[0]
        try:
            r = json.load(open(p))
        except Exception:
            continue
        r["_seed"] = int(run.rsplit("_seed", 1)[1])
        groups[(profile, bench, variant)].append(r)
    return groups


def summarise(groups):
    rows = []
    for (profile, bench, variant), runs in sorted(groups.items()):
        done = [r for r in runs if r.get("finished")]
        partial = [r for r in runs if not r.get("finished")]
        fm, fs = mean_sem([r["final_acc"] for r in done])
        am, as_ = mean_sem([r["avg_acc"] for r in done])
        gm, gs = mean_sem([r["avg_forgetting"] for r in done])
        curve = None
        if done:
            curve = np.mean([r["acc_per_task"] for r in done], axis=0).round(2).tolist()
        paper = PAPER["table2"].get(variant, {}).get(bench)
        rows.append(dict(profile=profile, benchmark=bench, variant=variant, n_seeds=len(done),
                         seeds=sorted(r["_seed"] for r in done),
                         partial=[dict(seed=r["_seed"], tasks_done=r["tasks_done"],
                                       acc_per_task=[round(a, 2) for a in r["acc_per_task"]])
                                  for r in partial],
                         final_acc=fm, final_sem=fs, avg_acc=am, avg_sem=as_,
                         forgetting=gm, forgetting_sem=gs, curve=curve,
                         minutes=float(np.mean([r["elapsed_min"] for r in done])) if done else None,
                         paper_final=paper["mean"] if paper else None,
                         paper_sem=paper["sem"] if paper else None))
    return rows


def fmt(m, s):
    if m is None:
        return "–"
    return f"{m:.2f}" + (f" ± {s:.2f}" if s is not None and not np.isnan(s) else "")


def to_markdown(rows):
    out = []
    for profile in sorted({r["profile"] for r in rows}):
        for bench in sorted({r["benchmark"] for r in rows if r["profile"] == profile}):
            out.append(f"\n### {profile} / {bench}\n")
            out.append("| Method | Paper (final acc.) | Ours: final acc. | Ours − paper | Avg. inc. acc. | "
                       "Avg. forgetting | seeds | min/run |")
            out.append("|---|---|---|---|---|---|---|---|")
            sel = [r for r in rows if r["profile"] == profile and r["benchmark"] == bench]
            order = list(VARIANT_NAMES)
            sel.sort(key=lambda r: order.index(r["variant"]) if r["variant"] in order else 99)
            for r in sel:
                diff = (f"{r['final_acc'] - r['paper_final']:+.2f}"
                        if r["final_acc"] is not None and r["paper_final"] is not None else "–")
                name = VARIANT_NAMES.get(r["variant"], r["variant"])
                part = ""
                if r["partial"]:
                    part = " (running: " + ", ".join(
                        f"seed {p['seed']} at task {p['tasks_done']}" for p in r["partial"]) + ")"
                out.append(f"| {name}{part} | {fmt(r['paper_final'], r['paper_sem'])} | "
                           f"{fmt(r['final_acc'], r['final_sem'])} | {diff} | "
                           f"{fmt(r['avg_acc'], r['avg_sem'])} | {fmt(r['forgetting'], r['forgetting_sem'])} | "
                           f"{r['n_seeds']} | {r['minutes']:.0f} |" if r["minutes"] else
                           f"| {name}{part} | {fmt(r['paper_final'], r['paper_sem'])} | – | – | – | – | 0 | – |")
            curves = [r for r in sel if r["curve"]]
            if curves:
                out.append("\nAccuracy (%) on all seen classes after each task:\n")
                pc = PAPER["figure3"]["accuracy_per_task"].get(bench, {}).get("HR")
                T = max([len(r["curve"]) for r in curves] + [len(pc) if pc else 0])
                out.append("| Method | " + " | ".join(str(i + 1) for i in range(T)) + " |")
                out.append("|---|" + "---|" * T)
                if pc:
                    out.append("| HR (paper, Fig. 3) | " + " | ".join(f"{v:.1f}" for v in pc) + " |")
                for r in curves:
                    out.append(f"| {VARIANT_NAMES.get(r['variant'], r['variant'])} | "
                               + " | ".join(f"{v:.1f}" for v in r["curve"]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=os.path.join(ROOT, "results"))
    ap.add_argument("--markdown", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = summarise(collect(args.results_root))
    md = to_markdown(rows)
    print(md)
    if args.markdown:
        open(args.markdown, "w").write(md + "\n")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
