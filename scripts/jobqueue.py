"""Thread-budget job queue for long experiment campaigns on one machine.

Reads jobs from a text file (one per line: ``<threads> <arguments for run.py>``),
and every ``--poll`` seconds starts the next job whose thread count fits into the
budget left by *all* running ``scripts/run.py`` processes (including ones started
elsewhere, matched by result file). Jobs whose result JSON already exists are skipped;
``--save-ckpt 1`` jobs resume from their newest checkpoint. The file is re-read
on every poll, so jobs can be appended while the queue runs.

    python scripts/jobqueue.py --jobs results/jobs.txt --budget 4
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def running_runs():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines()
            if "scripts/run.py" in line and line.split()[0].endswith(("python3", "python"))]


def running_threads():
    used = 0
    for line in running_runs():
        m = re.search(r"--threads (\d+)", line)
        used += int(m.group(1)) if m else 4
    return used


def result_path(args):
    def get(flag, default=""):
        return args[args.index(flag) + 1] if flag in args else default
    tag = get("--tag")
    name = f"{get('--method')}_s{get('--seed', '0')}{('_' + tag) if tag else ''}"
    return os.path.join(ROOT, get("--out", "results"), get("--dataset"), name + ".json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    started = set()
    while True:
        jobs = []
        for line in open(a.jobs):
            line = line.strip()
            if line and not line.startswith("#"):
                threads, rest = line.split(maxsplit=1)
                jobs.append((int(threads), rest))
        # skip finished jobs and jobs already running (started by this or another queue, or by
        # hand with extra arguments such as --resume): a running run.py with the same result
        # file counts as the same job
        running = {result_path(shlex.split(line.split("scripts/run.py", 1)[1])) for line in running_runs()}
        pending = [(t, r) for t, r in jobs if r not in started
                   and not os.path.exists(result_path(shlex.split(r)))
                   and result_path(shlex.split(r)) not in running]
        if not pending:
            print("queue empty", flush=True)
            return
        t, rest = pending[0]  # strictly in order, so long jobs are not starved
        if running_threads() + t <= a.budget:
            args = shlex.split(rest)
            if "--save-ckpt" in args and "--resume" not in args:
                # continue from the newest checkpoint if an earlier attempt was killed
                base = result_path(args)[:-len(".json")]
                ckpts = sorted(glob.glob(base + "_ckpt_t*.pt"),
                               key=lambda f: int(re.search(r"_ckpt_t(\d+)\.pt$", f).group(1)))
                if ckpts:
                    args += ["--resume", os.path.relpath(ckpts[-1], ROOT)]
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "run.py"), *args, "--threads", str(t)]
            os.makedirs(os.path.dirname(result_path(args)), exist_ok=True)
            log = open(result_path(args).replace(".json", ".stdout"), "w")
            subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started.add(rest)
            print(time.strftime("%H:%M"), "started:", " ".join(cmd), flush=True)
            time.sleep(5)
            continue
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
