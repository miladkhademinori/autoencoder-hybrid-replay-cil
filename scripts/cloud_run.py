"""Run one experiment on a fresh cloud machine and push its results to a git branch.

Installs missing dependencies, downloads the dataset, runs ``scripts/run.py`` with
all cores, resumes from the newest checkpoint if the machine was restarted, and
commits + pushes the run's log every ``--sync-min`` minutes and its JSON/log/sample
files when it finishes.

    python scripts/cloud_run.py --branch claude/ahr-exp-cifar10-ahr-s2 -- \\
        --dataset cifar10 --method ahr --seed 2
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=ROOT, check=check)


def ensure_deps():
    try:
        import numpy, pyarrow, PIL, matplotlib, torch  # noqa: F401
    except ImportError:
        sh(f"{sys.executable} -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu", check=False)
        sh(f"{sys.executable} -m pip install -q numpy pyarrow pillow matplotlib")


def run_name(args):
    def get(flag, default=""):
        return args[args.index(flag) + 1] if flag in args else default
    tag = get("--tag")
    name = f"{get('--method')}_s{get('--seed', '0')}{('_' + tag) if tag else ''}"
    return get("--dataset"), os.path.join(get("--out", "results"), get("--dataset")), name


def push(branch, files, msg):
    files = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
    if not files:
        return
    sh("git add -f " + " ".join(files), check=False)
    if subprocess.run("git diff --cached --quiet", shell=True, cwd=ROOT).returncode == 0:
        return
    sh(f"git commit -q -m '{msg}' -m 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'", check=False)
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        if sh(f"git push -q -u origin HEAD:refs/heads/{branch}", check=False).returncode == 0:
            return
    print("push failed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sync-min", type=float, default=30)
    ap.add_argument("run_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = [x for x in a.run_args if x != "--"]
    dataset, out_dir, name = run_name(args)
    ensure_deps()
    sh(f"{sys.executable} scripts/prepare_data.py --root data --datasets {dataset}")
    cmd = [sys.executable, "scripts/run.py", *args, "--threads", str(os.cpu_count() or 4)]
    method = args[args.index("--method") + 1]
    if method.startswith("ahr") and "--save-ckpt" not in args:
        cmd += ["--save-ckpt", "1"]
    ckpts = sorted(glob.glob(os.path.join(ROOT, out_dir, f"{name}_ckpt_t*.pt")),
                   key=lambda f: int(re.search(r"_ckpt_t(\d+)\.pt$", f).group(1)))
    if os.path.exists(os.path.join(ROOT, out_dir, name + ".json")):
        print("result already exists", flush=True)
    else:
        if ckpts:
            cmd += ["--resume", os.path.relpath(ckpts[-1], ROOT)]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT)
        last = time.time()
        while proc.poll() is None:
            time.sleep(30)
            if time.time() - last > a.sync_min * 60:
                push(a.branch, [os.path.join(out_dir, name + ".log")], f"{dataset} {name}: progress log")
                last = time.time()
        if proc.returncode != 0:
            push(a.branch, [os.path.join(out_dir, name + ".log")], f"{dataset} {name}: FAILED")
            sys.exit(proc.returncode)
    files = [os.path.join(out_dir, name + ext) for ext in (".json", ".log")]
    files += [os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(ROOT, out_dir, f"{name}_samples_t*.pt"))]
    push(a.branch, files, f"{dataset} {name}: result")
    print("DONE", open(os.path.join(ROOT, out_dir, name + ".log")).read().strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
