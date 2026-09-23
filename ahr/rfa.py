"""Class-centroid-embedding (CCE) placement with the repulsive force algorithm.

Implements Algorithm 2 of the paper (``CCE_Placement``). The CCEs are treated as
charged particles with Coulomb interaction energy

    U = sum_{ij} q^2/2 sum_{i'j' != ij} 1 / ||p_{i'}^{j'} - p_i^j||

and the new task's CCEs are moved by integrating the Euler-Lagrange equations of
motion (m dv/dt = F, dp/dt = v) under the repulsive force
``f = zeta / ||d||^2 * d / ||d||`` exerted by every other CCE. CCEs of previous
tasks are fixed (they never change once placed); only the new ones move.

Two optional additions over the bare algorithm, both off by default:

* ``damping`` - velocity friction ``v <- (1 - damping) v`` so that the kinetic
  energy is dissipated and the system settles instead of coasting;
* ``softening`` - Plummer softening ``||d||^2 + eps^2`` which keeps the force
  finite when two initial positions nearly coincide (e.g. the class means of a
  randomly initialised encoder on the first task).

The simulation duration ``tau`` is a hyper-parameter of Algorithm 2. Because the
Coulomb potential has no minimum at finite distance, the final spacing depends on
``tau`` and on how close the initial positions are, which differs a lot between
the first task (class means of a random encoder, almost coincident) and later
tasks (class means of a trained encoder). ``target_dist`` therefore optionally
ends the simulation as soon as every new CCE is at least ``target_dist`` away from
every other CCE (``steps`` then acts as the maximum duration), which yields a
uniform spacing across tasks with the least displacement from the initial
(natural) positions.
"""
import torch


@torch.no_grad()
def repulsive_force(p_new, p_all, self_index, zeta, softening):
    """Total Coulomb force on each new particle from all other particles."""
    d = p_new[:, None, :] - p_all[None, :, :]                 # J x N x m
    r2 = (d * d).sum(-1) + softening ** 2                      # J x N
    inv_r3 = r2.pow(-1.5)
    inv_r3[torch.arange(len(p_new)), self_index] = 0.0         # no self-interaction
    return zeta * (d * inv_r3[..., None]).sum(1)               # J x m


@torch.no_grad()
def place_cces(init_new, old_cces=None, zeta=1.0, mass=1.0, dt=0.01, steps=100,
               damping=0.0, softening=1e-3, sequential=True, target_dist=None):
    """Return the positions of the new CCEs after ``steps`` RFA time steps.

    Args:
        init_new: J x m initial positions (class means of the new task under the
            previous encoder, Algorithm 2 line 3).
        old_cces: N x m fixed CCEs of the previous tasks (or None).
        sequential: update the particles one after another inside a time step,
            exactly as the loops of Algorithm 2 do (Gauss-Seidel); otherwise all
            particles are updated simultaneously (Jacobi).
    """
    p = init_new.clone().double()
    old = (old_cces.double() if old_cces is not None and len(old_cces) else
           torch.zeros(0, p.shape[1], dtype=p.dtype))
    J = len(p)
    v = torch.zeros_like(p)
    n_old = len(old)
    for step in range(steps):
        if target_dist is not None and _min_dist_new(p, old) >= target_dist:
            break
        if sequential:
            for j in range(J):
                allp = torch.cat([old, p])
                f = repulsive_force(p[j:j + 1], allp, torch.tensor([n_old + j]), zeta, softening)[0]
                v[j] = (1.0 - damping) * v[j] + f / mass * dt
                p[j] = p[j] + v[j] * dt
        else:
            allp = torch.cat([old, p])
            f = repulsive_force(p, allp, torch.arange(n_old, n_old + J), zeta, softening)
            v = (1.0 - damping) * v + f / mass * dt
            p = p + v * dt
    return p.to(init_new.dtype), step + 1 if steps else 0


def _min_dist_new(p, old):
    """Smallest distance between a new CCE and any other CCE."""
    d = torch.cdist(p, torch.cat([old, p]))
    d[torch.arange(len(p)), len(old) + torch.arange(len(p))] = float("inf")
    return float(d.min())


def coulomb_energy(p, q=1.0):
    """Potential energy U of a set of CCEs (for diagnostics)."""
    d = torch.cdist(p.double(), p.double())
    iu = torch.triu_indices(len(p), len(p), 1)
    return float((q ** 2) * (1.0 / d[iu[0], iu[1]]).sum())


def spacing_stats(p):
    """Minimum / mean pairwise distance between CCEs (for diagnostics)."""
    if len(p) < 2:
        return 0.0, 0.0
    d = torch.cdist(p.double(), p.double())
    iu = torch.triu_indices(len(p), len(p), 1)
    dd = d[iu[0], iu[1]]
    return float(dd.min()), float(dd.mean())
