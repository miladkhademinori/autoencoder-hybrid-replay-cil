"""Download the CIL benchmark datasets and cache them as uint8 ``.npz`` arrays.

Images are stored as ``N x C x H x W`` uint8 arrays with integer labels, which
keeps loading instantaneous and lets the whole training set live in memory.

Sources are the Hugging Face mirrors of the original datasets (MNIST, CIFAR-10,
CIFAR-100 and the cropped-digit SVHN), which are identical to the torchvision
versions but are much faster to fetch.

    python scripts/prepare_data.py --root data --datasets mnist svhn cifar10 cifar100
"""
import argparse
import io
import os
import urllib.request

import numpy as np

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "mnist": ("ylecun/mnist", "mnist/{split}-00000-of-00001.parquet", "image", "label"),
    "cifar10": ("uoft-cs/cifar10", "plain_text/{split}-00000-of-00001.parquet", "img", "label"),
    "cifar100": ("uoft-cs/cifar100", "cifar100/{split}-00000-of-00001.parquet", "img", "fine_label"),
    "svhn": ("ufldl-stanford/svhn", "cropped_digits/{split}-00000-of-00001.parquet", "image", "label"),
}


def _download(url, dst):
    if os.path.exists(dst):
        return
    tmp = dst + ".part"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _decode(table, img_col, label_col):
    from PIL import Image

    imgs = table.column(img_col).to_pylist()
    labels = np.asarray(table.column(label_col).to_pylist(), dtype=np.int64)
    arr = []
    for rec in imgs:
        im = np.asarray(Image.open(io.BytesIO(rec["bytes"])))
        if im.ndim == 2:
            im = im[None]
        else:
            im = im.transpose(2, 0, 1)
        arr.append(im)
    return np.stack(arr).astype(np.uint8), labels


def prepare(name, root):
    import pyarrow.parquet as pq

    out = os.path.join(root, f"{name}.npz")
    if os.path.exists(out):
        print(f"{out} exists")
        return
    repo, pattern, img_col, label_col = SOURCES[name]
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    arrays = {}
    for split in ("train", "test"):
        dst = os.path.join(raw_dir, f"{name}_{split}.parquet")
        _download(HF.format(repo=repo, path=pattern.format(split=split)), dst)
        x, y = _decode(pq.read_table(dst), img_col, label_col)
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x, y
        print(f"{name} {split}: x={x.shape} y={y.shape} classes={len(np.unique(y))}")
    np.savez(out, **arrays)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES))
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    for d in args.datasets:
        prepare(d, args.root)
