"""Run HR (or one of its ablations) on an FCIL benchmark.

Examples
--------
python run.py --benchmark cifar100_10_10_50_5 --variant hr --seed 0
python run.py --benchmark cifar100_10_10_50_5 --variant hr_no_kd --seed 1 --amp
python run.py --benchmark cifar100_10_10_50_5 --variant hr --set rounds_per_task=20 local_epochs=2
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from hr_fcil.config import BENCHMARKS, VARIANTS, Config, build_config
from hr_fcil.hr import HRTrainer


def parse_value(field: dataclasses.Field, raw: str):
    t = field.type if not isinstance(field.type, str) else {"int": int, "float": float,
                                                               "str": str, "bool": bool}[field.type]
    if t is bool:
        return raw.lower() in ("1", "true", "yes")
    return t(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="cifar100_10_10_50_5", choices=sorted(BENCHMARKS))
    ap.add_argument("--variant", default="hr", choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--results_root", default="./results")
    ap.add_argument("--profile", default="paper",
                    help="name of the compute profile (used in the output path)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any Config field")
    args = ap.parse_args()

    fields = {f.name: f for f in dataclasses.fields(Config)}
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(fields[k], v)
    out_dir = os.path.join(args.results_root, args.profile, args.benchmark,
                           f"{args.variant}_seed{args.seed}")
    cfg = build_config(args.benchmark, args.variant, seed=args.seed, data_root=args.data_root,
                       out_dir=out_dir, amp=args.amp, device=args.device,
                       tag=f"{args.profile}/{args.benchmark}/{args.variant}", **overrides)
    res = HRTrainer(cfg).run()
    print(f"FINAL accuracy {res['final_acc']:.2f} | average accuracy {res['avg_acc']:.2f} | "
          f"forgetting {res['avg_forgetting']}")


if __name__ == "__main__":
    main()
