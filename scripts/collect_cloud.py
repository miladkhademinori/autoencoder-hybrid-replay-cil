"""Copy the results pushed by cloud experiment sessions (scripts/cloud_run.py) into
this checkout's results/ directory.

    python scripts/collect_cloud.py --prefix claude/ahr-exp-
"""
import argparse
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="claude/ahr-exp-")
    ap.add_argument("--base", default="HEAD", help="files identical to this revision are skipped")
    a = ap.parse_args()
    git("fetch", "-q", "origin", f"+refs/heads/{a.prefix}*:refs/remotes/origin/{a.prefix}*")
    branches = git("for-each-ref", "--format=%(refname:short)", f"refs/remotes/origin/{a.prefix}").stdout.split()
    done, running = [], []
    for b in sorted(branches):
        files = git("diff", "--name-only", f"{a.base}...{b}", "--", "results/").stdout.split()
        if not files:
            running.append((b, "no results yet"))
            continue
        for f in files:
            blob = subprocess.run(["git", "show", f"{b}:{f}"], cwd=ROOT, capture_output=True).stdout
            os.makedirs(os.path.dirname(os.path.join(ROOT, f)), exist_ok=True)
            with open(os.path.join(ROOT, f), "wb") as fh:
                fh.write(blob)
        has_json = any(f.endswith(".json") for f in files)
        last = [f for f in files if f.endswith(".log")]
        tail = ""
        if last:
            lines = open(os.path.join(ROOT, last[0])).read().strip().splitlines()
            tail = next((l for l in reversed(lines) if l.startswith(("after task", "FINAL", "joint"))), lines[-1])
        (done if has_json else running).append((b, tail[:140]))
    print(f"finished ({len(done)}):")
    for b, t in done:
        print(f"  {b}: {t}")
    print(f"running ({len(running)}):")
    for b, t in running:
        print(f"  {b}: {t}")


if __name__ == "__main__":
    main()
