"""Exemplar memories with a fixed total budget (``fixed exemplar memory``).

* ``RawMemory`` stores uint8 images (exemplar replay baselines and the
  AHR-lossless ablations).
* ``LatentMemory`` stores latent codes ``z = phi(x)`` (AHR). Codes can optionally
  be quantised to 16 or 8 bits to make the byte-level footprint match the
  scalar-count footprint the paper uses for its memory accounting.

Both keep ``per_class = budget // n_seen_classes`` exemplars per class.
"""
import torch


class _Memory:
    def __init__(self, budget):
        self.budget = int(budget)
        self.data = None
        self.labels = torch.zeros(0, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def per_class(self, n_classes):
        return self.budget // n_classes

    def sample_indices(self, n, generator=None):
        return torch.randint(0, len(self), (n,), generator=generator)


class RawMemory(_Memory):
    def set(self, x_uint8, y):
        self.data, self.labels = x_uint8.clone(), y.clone()

    def get(self, idx):
        return self.data[idx], self.labels[idx]

    def scalars(self):
        return 0 if self.data is None else self.data[0].numel() * len(self)

    def nbytes(self):
        return 0 if self.data is None else self.data.numel() * self.data.element_size()


class LatentMemory(_Memory):
    def __init__(self, budget, bits=32):
        super().__init__(budget)
        assert bits in (8, 16, 32)
        self.bits = bits
        self.lo = self.scale = None

    def set(self, z, y):
        z = z.detach().float()
        self.labels = y.clone()
        if self.bits == 32:
            self.data = z.clone()
        elif self.bits == 16:
            self.data = z.half()
        else:  # per-dimension affine uint8 quantisation
            self.lo = z.min(0).values
            self.scale = (z.max(0).values - self.lo).clamp_min(1e-8) / 255.0
            self.data = ((z - self.lo) / self.scale).round().clamp(0, 255).to(torch.uint8)

    def get(self, idx):
        d = self.data[idx]
        if self.bits == 8:
            z = d.float() * self.scale + self.lo
        else:
            z = d.float()
        return z, self.labels[idx]

    def scalars(self):
        return 0 if self.data is None else self.data.shape[1] * len(self)

    def nbytes(self):
        if self.data is None:
            return 0
        extra = 0 if self.lo is None else 2 * self.lo.numel() * 4
        return self.data.numel() * self.data.element_size() + extra


# --------------------------------------------------------------------------- #
# Exemplar selection
# --------------------------------------------------------------------------- #
def select_rank(scores, labels, per_class):
    """AHR's selection (Algorithm 4): per class, keep the ``per_class`` samples
    with the smallest latent loss ``||phi(x) - p_y||^2`` (``Rank`` ascending)."""
    keep = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        order = torch.argsort(scores[idx])
        keep.append(idx[order[:per_class]])
    return torch.cat(keep)


def truncate_per_class(labels, per_class):
    """Indices of the first ``per_class`` stored exemplars of every class (stable)."""
    keep = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        keep.append(idx[:per_class])
    return torch.cat(keep)


def select_random(labels, per_class, generator=None):
    keep = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        keep.append(idx[torch.randperm(len(idx), generator=generator)[:per_class]])
    return torch.cat(keep)


def select_herding(feats, labels, per_class):
    """iCaRL herding on L2-normalised features."""
    keep = []
    feats = torch.nn.functional.normalize(feats.float(), dim=1)
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        f = feats[idx]
        mu = f.mean(0)
        chosen, running = [], torch.zeros_like(mu)
        avail = torch.ones(len(idx), dtype=torch.bool)
        for k in range(min(per_class, len(idx))):
            cand = (running[None] + f) / (k + 1)
            dist = (cand - mu[None]).norm(dim=1)
            dist[~avail] = float("inf")
            i = int(dist.argmin())
            chosen.append(i)
            avail[i] = False
            running += f[i]
        keep.append(idx[torch.tensor(chosen, dtype=torch.long)])
    return torch.cat(keep)
