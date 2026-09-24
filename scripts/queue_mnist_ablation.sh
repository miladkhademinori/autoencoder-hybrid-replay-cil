#!/bin/sh
# Ablation of the implementation choices of REPRODUCTION.md §3 on MNIST(5/2), 3 seeds.
# Run from the repository root.
python3 - <<'PY'
# FT-E runs made before the baselines switched to FACIL-style union sampling
# used AHR's balanced minibatches: keep them under the tag "balanced".
import glob, json, os
for f in glob.glob("results/mnist/ft_e_s[0-9].json"):
    r = json.load(open(f))
    if r["args"].get("replay_sampling", "balanced") == "balanced":
        r["tag"] = "balanced"
        json.dump(r, open(f[:-5] + "_balanced.json", "w"), indent=1)
        os.remove(f)
        os.rename(f[:-5] + ".log", f[:-5] + "_balanced.log")
PY
python3 scripts/suite.py --datasets mnist --methods ft_e --seeds 0 1 2 --jobs 1 --threads 1
S="python3 scripts/suite.py --datasets mnist --methods ahr --seeds 0 1 2 --jobs 1 --threads 1"
$S --tag no_memorize --extra "--memorize-epochs 0"
$S --tag no_recon_new --extra "--lam-recon-new 0"
$S --tag reencode --extra "--memory-mode reencode --alpha-mem 0 --memorize-epochs 0"
$S --tag literal_herding --extra "--memory-mode reencode --alpha-mem 0 --memorize-epochs 0 --lam-recon-new 0"
$S --tag literal_rank --extra "--memory-mode reencode --alpha-mem 0 --memorize-epochs 0 --lam-recon-new 0 --selection rank"
