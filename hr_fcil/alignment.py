"""Server-side placement of new class-centroid embeddings (CCEs) in the latent space.

`lj_align` minimises the Lennard-Jones potential of the paper (Eq. 8) w.r.t. the new
centroids while the centroids of previous tasks stay fixed (CCEs never move once placed).

Note on Eq. 9 of the paper: the printed update
    p <- p - eta * sum 24 eps [2 s^12/r^13 - s^6/r^7] (p - p')
moves p *towards* p' when the pair is repulsive, i.e. it ascends the potential, and the
(p - p') direction is not normalised. We implement plain gradient descent on U:
    p <- p - eta * dU/dp,  dU/dp = sum 24 eps [-2 s^12/r^14 + s^6/r^8] (p - p'),
with the per-step displacement capped for numerical safety (the r^-13 term explodes when
the unaligned centroids of a randomly initialised encoder almost coincide).

`rfa_align` is the Repulsive Force Algorithm (Coulomb repulsion + Newtonian dynamics,
Algorithm 2 of the companion ICML 2025 AHR paper), used for the "HR w RFA" ablation. Pure
repulsion never reaches an equilibrium, so the simulation stops once every new centroid
is at least the LJ equilibrium distance 2^(1/6) sigma away from all other centroids.
"""
from __future__ import annotations

import torch


def _pairwise(p: torch.Tensor, allp: torch.Tensor, n_old: int):
    diff = p[:, None, :] - allp[None, :, :]                 # [n_new, n_all, d]
    r = diff.norm(dim=-1)
    idx = torch.arange(p.shape[0], device=p.device)
    r[idx, n_old + idx] = float("inf")                     # exclude self-pairs
    return diff, r


def lj_energy(p_new: torch.Tensor, p_old: torch.Tensor, eps: float, sigma: float) -> float:
    allp = torch.cat([p_old, p_new])
    _, r = _pairwise(p_new, allp, p_old.shape[0])
    sr6 = (sigma / r) ** 6
    return float((4 * eps * (sr6 ** 2 - sr6)).sum())


def lj_align(p_init: torch.Tensor, p_old: torch.Tensor, eps: float, sigma: float,
             lr: float, steps: int, max_disp: float, gen: torch.Generator | None = None):
    p = p_init.double().clone()
    p_old = p_old.double()
    # break exact ties (e.g. a class whose unaligned centroid coincides with another one)
    p += 1e-3 * sigma * torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
    n_old = p_old.shape[0]
    cap = max_disp * sigma
    for _ in range(steps):
        allp = torch.cat([p_old, p])
        diff, r = _pairwise(p, allp, n_old)
        coef = 24 * eps * (-2 * sigma ** 12 / r ** 14 + sigma ** 6 / r ** 8)   # [n_new, n_all]
        grad = (coef[..., None] * diff).sum(1)
        step = -lr * grad
        norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        step = step * torch.clamp(cap / norm, max=1.0)
        p = p + step
        if step.norm(dim=-1).max() < 1e-6 * sigma:
            break
    return p.float()


def rfa_align(p_init: torch.Tensor, p_old: torch.Tensor, zeta: float, mass: float, dt: float,
              steps: int, target_dist: float, gen: torch.Generator | None = None):
    p = p_init.double().clone()
    p_old = p_old.double()
    p += 1e-3 * target_dist * torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
    v = torch.zeros_like(p)
    n_old = p_old.shape[0]
    for _ in range(steps):
        allp = torch.cat([p_old, p])
        diff, r = _pairwise(p, allp, n_old)
        if r.min() >= target_dist:
            break
        rr = r.clamp_min(1e-3 * target_dist)
        force = (zeta / rr ** 3)[..., None] * diff               # zeta / |d|^2 * d/|d|
        force = force.sum(1)
        v = v + force / mass * dt
        # cap the velocity so that near-coincident particles cannot jump to infinity
        vmax = target_dist / dt * 0.05
        vn = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        v = v * torch.clamp(vmax / vn, max=1.0)
        p = p + v * dt
    return p.float()


def align_centroids(cfg, p_init: torch.Tensor, p_old: torch.Tensor, gen=None) -> torch.Tensor:
    if cfg.alignment == "lj":
        return lj_align(p_init, p_old, cfg.lj_epsilon, cfg.lj_sigma, cfg.lj_lr, cfg.lj_steps,
                        cfg.lj_max_disp, gen)
    if cfg.alignment == "rfa":
        return rfa_align(p_init, p_old, cfg.rfa_zeta, cfg.rfa_mass, cfg.rfa_dt, cfg.rfa_steps,
                         2 ** (1 / 6) * cfg.lj_sigma, gen)
    if cfg.alignment == "none":
        return p_init.clone()
    raise ValueError(cfg.alignment)
