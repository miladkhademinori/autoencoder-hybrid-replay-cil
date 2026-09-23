"""Class-incremental benchmark construction.

A benchmark ``D(T/C)`` splits dataset ``D`` into ``T`` tasks of ``C`` disjoint
classes (Masana et al., 2022). Images are kept in memory as uint8 tensors and
converted to float ``[0, 1]`` per minibatch.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# name -> (npz file, default number of tasks, horizontal flip allowed)
BENCHMARKS = {
    "mnist": ("mnist", 5, False),
    "svhn": ("svhn", 5, False),        # "Balanced SVHN": classes subsampled to equal size
    "cifar10": ("cifar10", 5, True),
    "cifar100": ("cifar100", 10, True),
}


@dataclass
class Task:
    classes: list
    x_train: torch.Tensor  # uint8, N x C x H x W
    y_train: torch.Tensor  # int64, labels already remapped to incremental order
    x_test: torch.Tensor
    y_test: torch.Tensor


def _balance(x, y, rng):
    """Subsample every class to the size of the smallest one."""
    classes, counts = np.unique(y, return_counts=True)
    n = counts.min()
    keep = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in classes])
    keep.sort()
    return x[keep], y[keep]


class CILBenchmark:
    def __init__(self, name, root="data", n_tasks=None, class_order="natural", seed=0,
                 train_fraction=1.0):
        file, default_tasks, flip = BENCHMARKS[name]
        self.name = name
        self.flip = flip
        path = os.path.join(root, f"{file}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run scripts/prepare_data.py first")
        d = np.load(path)
        x_tr, y_tr, x_te, y_te = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
        rng = np.random.RandomState(1993)
        if name == "svhn":  # Balanced SVHN
            x_tr, y_tr = _balance(x_tr, y_tr, rng)
            x_te, y_te = _balance(x_te, y_te, rng)
        if train_fraction < 1.0:  # optional per-class subsampling for quick experiments
            keep = []
            for c in np.unique(y_tr):
                idx = np.where(y_tr == c)[0]
                keep.append(rng.choice(idx, int(len(idx) * train_fraction), replace=False))
            keep = np.sort(np.concatenate(keep))
            x_tr, y_tr = x_tr[keep], y_tr[keep]

        n_classes = int(y_tr.max()) + 1
        self.n_tasks = n_tasks or default_tasks
        assert n_classes % self.n_tasks == 0
        self.classes_per_task = n_classes // self.n_tasks
        self.n_classes = n_classes
        if class_order == "natural":
            order = np.arange(n_classes)
        elif class_order == "icarl":  # the class order used by iCaRL / FACIL (seed 1993)
            order = np.random.RandomState(1993).permutation(n_classes)
        else:
            order = np.random.RandomState(seed).permutation(n_classes)
        self.class_order = order.tolist()
        remap = np.empty(n_classes, dtype=np.int64)
        remap[order] = np.arange(n_classes)
        y_tr, y_te = remap[y_tr], remap[y_te]

        self.input_shape = tuple(x_tr.shape[1:])
        self.tasks = []
        C = self.classes_per_task
        for t in range(self.n_tasks):
            cls = list(range(t * C, (t + 1) * C))
            mtr = (y_tr >= t * C) & (y_tr < (t + 1) * C)
            mte = (y_te >= t * C) & (y_te < (t + 1) * C)
            self.tasks.append(Task(cls, torch.from_numpy(x_tr[mtr]), torch.from_numpy(y_tr[mtr]),
                                   torch.from_numpy(x_te[mte]), torch.from_numpy(y_te[mte])))

    @property
    def input_size(self):
        c, h, w = self.input_shape
        return c * h * w

    def test_upto(self, t):
        """Test images/labels of all classes seen in tasks 0..t."""
        xs = torch.cat([self.tasks[i].x_test for i in range(t + 1)])
        ys = torch.cat([self.tasks[i].y_test for i in range(t + 1)])
        return xs, ys

    def train_upto(self, t):
        xs = torch.cat([self.tasks[i].x_train for i in range(t + 1)])
        ys = torch.cat([self.tasks[i].y_train for i in range(t + 1)])
        return xs, ys


def to_float(x_uint8):
    return x_uint8.float().div_(255.0)


def augment(x, pad=4, flip=True):
    """Random crop with zero padding + optional horizontal flip on a float batch."""
    if pad <= 0 and not flip:
        return x
    B, C, H, W = x.shape
    if pad > 0:
        xp = F.pad(x, (pad, pad, pad, pad))
        i = torch.randint(0, 2 * pad + 1, (B,))
        j = torch.randint(0, 2 * pad + 1, (B,))
        rows = (i[:, None] + torch.arange(H)[None])[:, None, :, None]
        cols = (j[:, None] + torch.arange(W)[None])[:, None, None, :]
        x = xp[torch.arange(B)[:, None, None, None], torch.arange(C)[None, :, None, None], rows, cols]
    if flip:
        m = torch.rand(B) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(3), x)
    return x
