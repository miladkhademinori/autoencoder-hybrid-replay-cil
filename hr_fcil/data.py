"""Datasets, class-incremental task splits and LDA (Dirichlet) client partitions."""
from __future__ import annotations

import io
import os
import pickle
import urllib.request

import numpy as np

# Datasets are fetched from the Hugging Face CDN (fast everywhere, incl. Colab). The HF copy of
# CIFAR-100 is bit-identical to the original release (same pixels, order and fine labels).
# The original hosts (cs.toronto.edu, cs231n.stanford.edu) are deliberately NOT used: they
# serve at ~100 KB/s.
HF = "https://huggingface.co/datasets"
CIFAR100_HF = {
    "train": f"{HF}/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
    "test": f"{HF}/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
}
TINY_HF = {
    "train": f"{HF}/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet",
    "test": f"{HF}/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet",
}


def _download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _read_parquet_images(path: str, image_col: str, label_col: str):
    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(path, columns=[image_col, label_col]).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"), dtype=np.uint8)
                  .transpose(2, 0, 1) for d in t[image_col]])
    return x, np.asarray(t[label_col], dtype=np.int64)


def _load_hf(root: str, name: str, urls: dict, image_col: str, label_col: str):
    cache = os.path.join(root, f"{name}_uint8.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    out = {}
    for split, key in (("train", "tr"), ("test", "te")):
        pq_path = os.path.join(root, f"{name}_{split}.parquet")
        _download(urls[split], pq_path)
        out["x" + key], out["y" + key] = _read_parquet_images(pq_path, image_col, label_col)
    tmp = cache + ".part.npz"
    np.savez(tmp, **out)
    os.replace(tmp, cache)
    return out["xtr"], out["ytr"], out["xte"], out["yte"]


def load_cifar100(root: str):
    """Returns (x_train uint8 [N,3,32,32], y_train int64, x_test, y_test)."""
    base = os.path.join(root, "cifar-100-python")
    if os.path.isdir(base):  # an existing copy of the original pickles is used as is
        def read(split):
            with open(os.path.join(base, split), "rb") as f:
                d = pickle.load(f, encoding="latin1")
            return (np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32),
                    np.asarray(d["fine_labels"], dtype=np.int64))
        return (*read("train"), *read("test"))
    return _load_hf(root, "cifar100", CIFAR100_HF, "img", "fine_label")


def load_tinyimagenet(root: str):
    """TinyImageNet-200 (64x64); the official validation split is the test set."""
    return _load_hf(root, "tinyimagenet200", TINY_HF, "image", "label")


def load_dataset(name: str, root: str):
    if name == "cifar100":
        return load_cifar100(root)
    if name == "tinyimagenet":
        return load_tinyimagenet(root)
    raise ValueError(f"unknown dataset {name}")


def lda_partition(labels: np.ndarray, num_clients: int, alpha: float,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Latent Dirichlet Allocation partition (Hsu et al. 2019; FedML / MFCL implementation).

    For every class, client proportions are drawn from Dir(alpha * 1); clients that already
    hold more than the average share receive no further samples. Resampled until every client
    holds at least `min_size` samples.
    """
    n = len(labels)
    classes = np.unique(labels)
    min_require = max(1, min(10, n // (2 * num_clients)))
    for _ in range(1000):
        buckets = [[] for _ in range(num_clients)]
        for k in classes:
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            prop = rng.dirichlet(np.repeat(alpha, num_clients))
            prop = np.array([p * (len(b) < n / num_clients) for p, b in zip(prop, buckets)])
            prop = prop / prop.sum()
            cuts = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            for b, part in zip(buckets, np.split(idx_k, cuts)):
                b.extend(part.tolist())
        if min(len(b) for b in buckets) >= min_require:
            break
    return [np.array(sorted(b), dtype=np.int64) for b in buckets]


class TaskStream:
    """Class-incremental stream: a seeded class order split into `num_tasks` disjoint tasks.

    Labels are remapped to their position in the class order, so task t owns labels
    [t*C, (t+1)*C).
    """

    def __init__(self, cfg, rng: np.random.Generator):
        xtr, ytr, xte, yte = load_dataset(cfg.dataset, cfg.data_root)
        n_classes = cfg.num_tasks * cfg.classes_per_task
        order = np.random.RandomState(cfg.class_order_seed).permutation(len(np.unique(ytr)))
        order = order[:n_classes]
        remap = -np.ones(int(ytr.max()) + 1, dtype=np.int64)
        remap[order] = np.arange(n_classes)
        ytr, yte = remap[ytr], remap[yte]
        keep_tr, keep_te = ytr >= 0, yte >= 0
        self.x_train, self.y_train = xtr[keep_tr], ytr[keep_tr]
        self.x_test, self.y_test = xte[keep_te], yte[keep_te]
        self.class_order = order
        self.num_tasks = cfg.num_tasks
        self.cpt = cfg.classes_per_task
        self.image_shape = self.x_train.shape[1:]

        if cfg.max_test_per_class > 0:
            sel = np.concatenate([np.where(self.y_test == c)[0][:cfg.max_test_per_class]
                                  for c in range(n_classes)])
            self.x_test, self.y_test = self.x_test[sel], self.y_test[sel]

        # Per task: indices of the task's training samples, split across clients by LDA.
        self.client_indices = []  # [task][client] -> np.ndarray of indices into x_train
        for t in range(cfg.num_tasks):
            idx = np.where((self.y_train >= t * self.cpt) & (self.y_train < (t + 1) * self.cpt))[0]
            if cfg.max_train_per_task > 0:
                idx = rng.permutation(idx)[:cfg.max_train_per_task]
            parts = lda_partition(self.y_train[idx], cfg.num_clients, cfg.lda_alpha, rng)
            self.client_indices.append([idx[p] for p in parts])

    def task_classes(self, t: int) -> range:
        return range(t * self.cpt, (t + 1) * self.cpt)
