import contextlib
import math
import random
import time

import numpy as np
import torch

from .data import to_float


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(enabled):
    """bf16 autocast on CPU (uses AMX/AVX512-bf16 when available) or CUDA."""
    if not enabled:
        return contextlib.nullcontext()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.autocast(device, dtype=torch.bfloat16)


@torch.no_grad()
def batched(fn, x, batch_size=1000, uint8=True, device="cpu"):
    """Apply ``fn`` to ``x`` in chunks and concatenate the (float32) outputs."""
    outs = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        xb = to_float(xb) if uint8 else xb
        outs.append(fn(xb.to(device)).float().cpu())
    return torch.cat(outs) if outs else torch.zeros(0)


def make_optimizer(params, args):
    return torch.optim.Adam(params, lr=args.lr, betas=(args.momentum, 0.999),
                            weight_decay=args.weight_decay)


def make_scheduler(opt, args, total_steps):
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    return None


def accuracy_report(pred, y, classes_per_task, n_tasks_seen):
    """Overall accuracy and per-task accuracies (task-agnostic predictions)."""
    acc = float((pred == y).float().mean()) * 100
    per_task = []
    for t in range(n_tasks_seen):
        m = (y >= t * classes_per_task) & (y < (t + 1) * classes_per_task)
        per_task.append(float((pred[m] == y[m]).float().mean()) * 100)
    return acc, per_task


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
