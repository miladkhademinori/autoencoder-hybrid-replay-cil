"""Measure local-training throughput (encoder + decoder + KD teacher, batch 32) on this device
and estimate the wall-clock time of one full CIFAR-100 (10/10/50/5) HR run."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import socket
import time

import torch

from hr_fcil.models import HybridAutoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and dev.type == "cuda"
    dtype = (torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
    m = HybridAutoencoder("cifar100", 32, 307).to(dev)
    old = HybridAutoencoder("cifar100", 32, 307).to(dev).eval()
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    x = torch.rand(32, 3, 32, 32, device=dev)
    n_it = 200 if dev.type == "cuda" else 5
    for it in range(n_it + 10):
        if it == 10:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
        with torch.autocast(dev.type, dtype=dtype, enabled=use_amp):
            mu, lv = m.encode(x)
            z = mu.float() + torch.exp(0.5 * lv.float()) * torch.randn_like(mu.float())
            xr = m.decode(z).float()
            with torch.no_grad():
                mo, _ = old.encode(x)
                xo = old.decode(mo).float()
            xd = m.decode(mu).float()
        loss = (xr - x).pow(2).mean() + (mu.float() - mo.float()).pow(2).mean() + (xd - xo).pow(2).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sps = 32 * n_it / (time.time() - t)
    passes = 100 * 5 * 5 * 100 * 55  # rounds x clients x epochs x samples/client/task x sum_h h
    res = dict(host=socket.gethostname(), device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
               amp=use_amp, samples_per_sec=sps,
               est_hours_full_cifar100_run=passes / sps / 3600)
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
