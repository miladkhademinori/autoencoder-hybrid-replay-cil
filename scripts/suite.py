"""Run a grid of experiments with a small local job queue (skips finished runs).

    python scripts/suite.py --datasets mnist --methods ahr ft ft_e icarl joint \\
        --seeds 0 1 2 3 4 --jobs 4 --threads 1
    python scripts/suite.py --datasets cifar10 --methods ahr --seeds 0 --jobs 1 --threads 4 \\
        --extra "--epochs 50"

Any ``--extra`` arguments are forwarded to ``scripts/run.py``.
"""
import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cmds = []
    for d, m, s in itertools.product(args.datasets, args.methods, args.seeds):
        name = f"{m}_s{s}{('_' + args.tag) if args.tag else ''}"
        if os.path.exists(os.path.join(args.out, d, name + ".json")):
            print(f"skip {d}/{name} (done)")
            continue
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run.py"), "--dataset", d, "--method", m,
               "--seed", str(s), "--threads", str(args.threads), "--out", args.out]
        if args.tag:
            cmd += ["--tag", args.tag]
        cmd += shlex.split(args.extra)
        cmds.append((f"{d}/{name}", cmd))

    def run(item):
        name, cmd = item
        t0 = time.time()
        print(f"start {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            return
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, cwd=ROOT)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode}): {r.stderr[-2000:]}"
        print(f"done  {name} in {(time.time() - t0) / 60:.1f} min: {status}", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(run, cmds))


if __name__ == "__main__":
    main()
