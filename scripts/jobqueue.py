"""Thread-budget job queue for long experiment campaigns on one machine.

Reads jobs from a text file (one per line: ``<threads> <arguments for run.py>``),
and every ``--poll`` seconds starts the next job whose thread count fits into the
budget left by *all* running ``scripts/run.py`` processes (including ones started
elsewhere). Jobs whose result JSON already exists are skipped. The file is re-read
on every poll, so jobs can be appended while the queue runs.

    python scripts/jobqueue.py --jobs results/jobs.txt --budget 4
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def running_threads():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    used = 0
    for line in out.splitlines():
        if "scripts/run.py" in line and line.split()[0].endswith(("python3", "python")):
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
        pending = [(t, r) for t, r in jobs if r not in started
                   and not os.path.exists(result_path(shlex.split(r)))]
        if not pending:
            print("queue empty", flush=True)
            return
        t, rest = pending[0]  # strictly in order, so long jobs are not starved
        if running_threads() + t <= a.budget:
            args = shlex.split(rest)
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
