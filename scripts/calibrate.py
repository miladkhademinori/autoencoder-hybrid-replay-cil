"""Centralised sanity check of the HAE objective on the first task (used to choose lambda).

Trains the hybrid autoencoder on task 1 (10 classes) without federation and reports the
nearest-centroid test accuracy and the reconstruction MSE, next to a cross-entropy ResNet-18
trained with the same budget as a reference point.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from hr_fcil.alignment import lj_align
from hr_fcil.config import Config
from hr_fcil.data import TaskStream
from hr_fcil.hr import augment
from hr_fcil.models import HybridAutoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--lams", default="1,10,100")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--ce", action="store_true")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(data_root=args.data_root).resolved()
    data = TaskStream(cfg, np.random.default_rng(0))
    tr = np.where(data.y_train < 10)[0][: args.n_train]
    te = np.where(data.y_test < 10)[0]
    xtr = torch.from_numpy(data.x_train[tr]).float().div(255).to(dev)
    ytr = torch.from_numpy(data.y_train[tr]).to(dev)
    xte = torch.from_numpy(data.x_test[te]).float().div(255).to(dev)
    yte = torch.from_numpy(data.y_test[te]).to(dev)
    gen = torch.Generator().manual_seed(0)
    results = {}

    def lr_at(step, total):
        return 0.1 * (0.01 / 0.1) ** (step / max(1, total - 1))

    runs = [("ce", None)] if args.ce else []
    runs += [(f"hae_lam{l}", float(l)) for l in args.lams.split(",")]
    for name, lam in runs:
        torch.manual_seed(0)
        m = HybridAutoencoder("cifar100", 32, 307).to(dev)
        head = torch.nn.Linear(307, 10).to(dev)
        params = list(m.parameters()) + (list(head.parameters()) if lam is None else [])
        opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
        with torch.no_grad():
            m.eval()
            mu0 = m.encode(xtr[:1000])[0]
            p0 = torch.stack([mu0[ytr[:1000] == k].mean(0) for k in range(10)])
        P = lj_align(p0.cpu(), torch.zeros(0, 307), 1.0, 5.0, 1.0, 3000, 0.05, gen).to(dev)
        steps_per_epoch = math.ceil(len(xtr) / 32)
        total = steps_per_epoch * args.epochs
        step = 0
        t0 = time.time()
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(len(xtr), generator=gen).to(dev)
            for b in range(0, len(xtr), 32):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total)
                bi = perm[b:b + 32]
                x = augment(xtr[bi], gen)
                y = ytr[bi]
                mu, lv = m.encode(x)
                if lam is None:
                    loss = F.cross_entropy(head(mu), y)
                else:
                    lv = lv.clamp(-10, 10)
                    z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
                    xr = m.decode(z)
                    rec = (xr - x).pow(2).flatten(1).sum(1)
                    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)
                    clu = (z - P[y]).pow(2).sum(1)
                    loss = (rec + kl + lam * clu).mean() / 3072
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
                opt.step()
                step += 1
            m.eval()
            with torch.no_grad():
                mu = torch.cat([m.encode(xte[i:i + 500])[0] for i in range(0, len(xte), 500)])
                if lam is None:
                    acc = (head(mu).argmax(1) == yte).float().mean().item() * 100
                    mse = float("nan")
                else:
                    acc = (torch.cdist(mu, P).argmin(1) == yte).float().mean().item() * 100
                    mse = (m.decode(mu[:500]) - xte[:500]).pow(2).mean().item()
            print(f"{name} epoch {ep + 1}: acc {acc:.2f} rec_mse {mse:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            results[name] = dict(acc=acc, mse=mse)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
