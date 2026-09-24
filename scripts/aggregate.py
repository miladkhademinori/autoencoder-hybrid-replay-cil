"""Collect ``results/<dataset>/<method>_s<seed>[_tag].json`` into a Markdown table
(mean +- SEM over seeds) next to the numbers reported in the paper (Table 2).

    python scripts/aggregate.py --results results --tag "" > results/summary.md
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

DATASETS = ["mnist", "svhn", "cifar10", "cifar100"]
HEADER = {"mnist": "MNIST (5/2)", "svhn": "Balanced SVHN (5/2)",
          "cifar10": "CIFAR-10 (5/2)", "cifar100": "CIFAR-100 (10/10)"}
METHOD_NAMES = {"ft": "FT", "ft_e": "FT-E", "joint": "Joint", "icarl": "iCaRL", "ahr": "AHR",
                "ahr_lossy_mini": "AHR-lossy-mini", "ahr_lossless_mini": "AHR-lossless-mini",
                "ahr_lossless": "AHR-lossless"}
ORDER = ["ft", "ft_e", "joint", "icarl", "ahr", "ahr_lossy_mini", "ahr_lossless_mini", "ahr_lossless"]

# Table 2 of the paper: (mean, SEM) of the final accuracy.
PAPER = {
    "ft": {"mnist": (19.93, .03), "svhn": (19.19, .04), "cifar10": (18.72, .30), "cifar100": (8.91, .12)},
    "ft_e": {"mnist": (92.17, .16), "svhn": (87.13, .37), "cifar10": (72.17, .84), "cifar100": (48.47, .83)},
    "joint": {"mnist": (98.48, .06), "svhn": (95.88, .04), "cifar10": (92.37, .09), "cifar100": (73.87, .10)},
    "icarl": {"mnist": (93.06, .33), "svhn": (89.63, .61), "cifar10": (73.29, .73), "cifar100": (49.38, .62)},
    "ahr": {"mnist": (97.53, .32), "svhn": (93.02, .65), "cifar10": (77.12, .75), "cifar100": (54.43, .93)},
    "ahr_lossy_mini": {"mnist": (93.35, .32), "svhn": (90.40, .58), "cifar10": (73.28, .47), "cifar100": (50.29, .90)},
    "ahr_lossless_mini": {"mnist": (93.76, .26), "svhn": (90.88, .50), "cifar10": (73.68, .41), "cifar100": (50.85, .81)},
    "ahr_lossless": {"mnist": (98.12, .08), "svhn": (94.21, .23), "cifar10": (78.35, .37), "cifar100": (56.71, .57)},
}


def mean_sem(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def load(results, tag):
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(results, "*", "*.json")):
        r = json.load(open(path))
        if (r.get("tag") or "") != tag:
            continue
        runs[(r["dataset"], r["method"])].append(r)
    return runs


def fmt(ms, n=None):
    m, s = ms
    if m is None:
        return "-"
    txt = f"{m:.2f}" + ("" if math.isnan(s) else f" ± {s:.2f}")
    return txt + (f" (n={n})" if n else "")


ABLATION_TAGS = [
    ("ahr", "", "AHR (final configuration)"),
    ("ahr", "recon_new", "+ latent loss on reconstructions of new samples"),
    ("ahr", "no_memorize", "- decoder memorisation"),
    ("ahr", "literal_herding", "- frozen codes, - memorisation = literal Alg. 1-4 (herding selection)"),
    ("ahr", "literal_rank", "literal Alg. 1-4, Rank selection"),
    ("ft_e", "balanced", "FT-E with AHR's balanced minibatches"),
]


def ablation_table(results, dataset):
    rows = []
    for method, tag, label in ABLATION_TAGS:
        rs = [r for (d, m), lst in load(results, tag).items() if d == dataset and m == method for r in lst]
        if rs:
            rows.append(f"| {label} | {fmt(mean_sem([r['final_acc'] for r in rs]), len(rs))} |")
    if rows:
        print(f"\n{dataset} ablations (final accuracy %, mean ± SEM):\n")
        print("| Variant | Final acc. |\n|---|---|")
        print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default="final_acc", choices=["final_acc", "avg_inc_acc"])
    ap.add_argument("--ablations", default="", help="dataset for which to print the ablation table")
    args = ap.parse_args()
    if args.ablations:
        ablation_table(args.results, args.ablations)
        return
    runs = load(args.results, args.tag)
    datasets = [d for d in DATASETS if any(k[0] == d for k in runs)]
    print(f"Final accuracy (%) after the last task, mean ± SEM over seeds "
          f"(metric: `{args.metric}`, tag: `{args.tag or 'main'}`).\n")
    print("| Method | " + " | ".join(f"{HEADER[d]} ours | paper" for d in datasets) + " |")
    print("|---|" + "---|---|" * len(datasets))
    for m in ORDER:
        if not any((d, m) in runs for d in datasets):
            continue
        cells = []
        for d in datasets:
            rs = runs.get((d, m), [])
            ours = fmt(mean_sem([r[args.metric] for r in rs]), len(rs)) if rs else "-"
            paper = fmt(PAPER[m][d]) if m in PAPER and d in PAPER[m] else "-"
            cells += [ours, paper]
        print(f"| {METHOD_NAMES.get(m, m)} | " + " | ".join(cells) + " |")
    print()
    print("Epochs / exemplars / wall-clock per run:\n")
    print("| Dataset | Method | epochs | stored exemplars | memory (scalars) | minutes/run |")
    print("|---|---|---|---|---|---|")
    for d in datasets:
        for m in ORDER:
            rs = runs.get((d, m), [])
            if not rs:
                continue
            r = rs[0]
            print(f"| {d} | {METHOD_NAMES[m]} | {r['args']['epochs']} | {r['n_exemplars']} | "
                  f"{r.get('memory_scalars', 0):,} | {sum(x['time_s'] for x in rs) / len(rs) / 60:.0f} |")


if __name__ == "__main__":
    main()
