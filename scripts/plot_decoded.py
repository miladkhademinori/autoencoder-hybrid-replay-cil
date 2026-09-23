"""Plot original vs. decoded exemplars (cf. Fig. 4 of the paper).

    python scripts/plot_decoded.py results/cifar10/ahr_s0 --out figures/decoded_cifar10.png
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="run prefix, e.g. results/cifar10/ahr_s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    files = sorted(glob.glob(args.prefix + "_samples_t*.pt"),
                   key=lambda f: int(f.rsplit("_t", 1)[1][:-3]))
    rows = []
    for f in files:
        s = torch.load(f)
        if s["orig"] is None:
            continue
        rows.append((os.path.basename(f).rsplit("_t", 1)[1][:-3], s["orig"][:args.n], s["dec"][:args.n]))
    if not rows:
        raise SystemExit("no sample files found")
    n = min(args.n, rows[0][1].shape[0])
    fig, axes = plt.subplots(2 * len(rows), n, figsize=(n * 0.9, 2 * len(rows) * 0.95))
    for r, (task, orig, dec) in enumerate(rows):
        for k in range(n):
            for j, (img, label) in enumerate(((orig[k], "original"), (dec[k], "decoded"))):
                ax = axes[2 * r + j, k]
                im = img.permute(1, 2, 0).numpy()
                ax.imshow(im.squeeze(), cmap="gray" if im.shape[2] == 1 else None, vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel(f"t{task}\n{label}", fontsize=7)
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
