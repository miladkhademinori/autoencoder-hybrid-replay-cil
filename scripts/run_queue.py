"""Run a queue of experiments, resumably, with optional concurrency on one GPU.

Each job is one line of a queue file:  <profile> <benchmark> <variant> <seed> [KEY=VALUE ...]
Jobs whose result.json is marked finished are skipped; interrupted jobs resume from their
checkpoint. A heartbeat lock file lets several workers (or Colab sessions sharing one
Google Drive folder) cooperate on the same queue without running a job twice.

    python scripts/run_queue.py --queue configs/queue_colab.txt \
        --results_root /content/drive/MyDrive/hr_fcil_results --data_root /content/data --amp
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALE_SEC = 15 * 60


def read_queue(path):
    jobs = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        jobs.append(dict(profile=parts[0], benchmark=parts[1], variant=parts[2],
                         seed=int(parts[3]), overrides=parts[4:]))
    return jobs


def job_dir(root, j):
    return os.path.join(root, j["profile"], j["benchmark"], f"{j['variant']}_seed{j['seed']}")


def finished(d):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return False
    try:
        return bool(json.load(open(p)).get("finished"))
    except Exception:
        return False


def try_claim(d, me):
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, "running.lock")
    if os.path.exists(lock):
        try:
            info = json.load(open(lock))
            if info.get("owner") != me and time.time() - info.get("heartbeat", 0) < STALE_SEC:
                return False
        except Exception:
            pass
    with open(lock, "w") as f:
        json.dump(dict(owner=me, host=socket.gethostname(), heartbeat=time.time()), f)
    return True


def heartbeat(d, me):
    lock = os.path.join(d, "running.lock")
    with open(lock, "w") as f:
        json.dump(dict(owner=me, host=socket.gethostname(), heartbeat=time.time()), f)


def release(d):
    try:
        os.remove(os.path.join(d, "running.lock"))
    except FileNotFoundError:
        pass


def write_status(root, jobs, me, failures=None):
    rows = []
    for j in jobs:
        d = job_dir(root, j)
        st = "pending"
        extra = {}
        if finished(d):
            st = "finished"
            r = json.load(open(os.path.join(d, "result.json")))
            extra = dict(final_acc=r["final_acc"], avg_acc=r["avg_acc"], minutes=r["elapsed_min"])
        elif os.path.exists(os.path.join(d, "running.lock")):
            st = "running"
            if os.path.exists(os.path.join(d, "result.json")):
                r = json.load(open(os.path.join(d, "result.json")))
                extra = dict(tasks_done=r["tasks_done"], acc_per_task=r["acc_per_task"],
                             minutes=r["elapsed_min"])
        if st == "pending" and failures and failures.get(d):
            st = f"failed x{failures[d]} (see stdout.txt)"
        rows.append(dict(job=f"{j['profile']}/{j['benchmark']}/{j['variant']}_seed{j['seed']}",
                         status=st, **extra))
    tmp = os.path.join(root, f".queue_status.{me}.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), jobs=rows), f, indent=1)
    os.replace(tmp, os.path.join(root, "queue_status.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent jobs on this machine")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--keep_checkpoints", action="store_true")
    ap.add_argument("--max_hours", type=float, default=1e9)
    ap.add_argument("--git_pull_minutes", type=float, default=0,
                    help=">0: git pull the repo this often so queue/code updates are picked up")
    ap.add_argument("--idle_exit_minutes", type=float, default=0,
                    help="when the queue is exhausted, keep polling for new jobs this long")
    args = ap.parse_args()

    me = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    os.makedirs(args.results_root, exist_ok=True)
    t_start = time.time()
    running = {}  # dir -> (Popen, logfile)
    failures = {}  # dir -> number of crashes in this session (give up after 2)
    last_pull = 0.0
    idle_since = None
    while True:
        if args.git_pull_minutes > 0 and time.time() - last_pull > args.git_pull_minutes * 60:
            r = subprocess.run(["git", "-C", ROOT, "pull", "--ff-only", "-q"], capture_output=True,
                               text=True)
            if r.returncode != 0:
                print(f"[queue] git pull failed: {r.stderr.strip()}", flush=True)
            last_pull = time.time()
        jobs = read_queue(args.queue)
        wanted = {job_dir(args.results_root, j) for j in jobs}
        # reap finished processes; stop jobs that were removed from the queue (checkpoint is kept)
        for d, (p, lf) in list(running.items()):
            if p.poll() is None and d not in wanted:
                p.terminate()
                p.wait()
                print(f"[queue] cancelled (removed from queue): {d}", flush=True)
            if p.poll() is not None:
                lf.close()
                release(d)
                if finished(d) and not args.keep_checkpoints:
                    ck = os.path.join(d, "checkpoint.pt")
                    if os.path.exists(ck):
                        os.remove(ck)
                if not finished(d) and d in wanted:
                    failures[d] = failures.get(d, 0) + 1
                print(f"[queue] {'done' if finished(d) else 'stopped (exit %d)' % p.returncode}: {d}",
                      flush=True)
                del running[d]
            else:
                heartbeat(d, me)
        write_status(args.results_root, jobs, me, failures)
        out_of_time = time.time() - t_start > args.max_hours * 3600
        # launch new jobs
        while len(running) < args.parallel and not out_of_time:
            nxt = None
            for j in jobs:
                d = job_dir(args.results_root, j)
                if d in running or finished(d) or failures.get(d, 0) >= 2:
                    continue
                if try_claim(d, me):
                    nxt = (j, d)
                    break
            if nxt is None:
                break
            j, d = nxt
            cmd = [sys.executable, os.path.join(ROOT, "run.py"), "--benchmark", j["benchmark"],
                   "--variant", j["variant"], "--seed", str(j["seed"]),
                   "--profile", j["profile"], "--results_root", args.results_root,
                   "--data_root", args.data_root]
            if args.amp:
                cmd.append("--amp")
            if j["overrides"]:
                cmd += ["--set", *j["overrides"]]
            lf = open(os.path.join(d, "stdout.txt"), "a")
            print(f"[queue] start: {' '.join(cmd[2:])}", flush=True)
            running[d] = (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT), lf)
        if not running:
            write_status(args.results_root, jobs, me, failures)
            idle_since = idle_since or time.time()
            if out_of_time or time.time() - idle_since > args.idle_exit_minutes * 60:
                print("[queue] nothing left to run", flush=True)
                break
        else:
            idle_since = None
        time.sleep(30)


if __name__ == "__main__":
    main()
