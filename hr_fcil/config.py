"""Experiment configuration for Hybrid Replay (HR) in Federated Class-Incremental Learning.

Values marked "paper" are stated in Khademi Nori et al., ICLR 2025 (arXiv:2501.15356).
Values marked "MFCL" come from the FCIL simulation protocol HR follows
(Babakniya et al., NeurIPS 2023). Everything else is not specified in the paper and
was chosen by us; see REPRODUCTION.md for the rationale.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- benchmark (paper: A/B/C/D = tasks/classes-per-task/clients/active) --------
    dataset: str = "cifar100"            # cifar100 | tinyimagenet
    data_root: str = "./data"
    num_tasks: int = 10                  # paper
    classes_per_task: int = 10           # paper
    num_clients: int = 50                # paper
    clients_per_round: int = 5           # paper
    lda_alpha: float = 1.0               # paper (LDA, alpha = 1)
    class_order_seed: int = -1           # -1 -> use `seed`

    # ---------------- federated optimisation -------------------------------------------------
    rounds_per_task: int = 100           # MFCL
    local_epochs: int = 5                # 100 rounds x 10% participation x 5 = 50 epochs (paper Tab. 3)
    batch_size: int = 32                 # MFCL
    lr_start: float = 0.1                # MFCL: 0.1 decayed exponentially to 0.01 within each task
    lr_end: float = 0.01                 # MFCL
    momentum: float = 0.9
    weight_decay: float = 5e-4
    grad_clip: float = 10.0

    # ---------------- hybrid autoencoder -----------------------------------------------------
    latent_dim: int = 0                  # 0 -> input_dim / 10 (307 for 32x32x3, as in AHR Tab. 4)
    encoder: str = "resnet18"            # paper
    decoder_channels: int = 64           # 4-layer CNN decoder, ~1.4M params for CIFAR (paper Tab. 3)
    lam: float = 10.0                    # lambda: weight of ||z - p||^2 (paper Eq. 7)
    beta_kl: float = 1.0                 # ELBO KL weight (standard VAE = 1)
    kd_z: float = 1.0                    # latent distillation  ||f_{h-1}(x) - f_h(x)||^2
    kd_x: float = 1.0                    # decoder distillation ||g_{h-1}(f_{h-1}(x)) - g_h(f_h(x))||^2
    logvar_clip: float = 10.0

    # ---------------- centroid (class embedding) placement on the server ---------------------
    alignment: str = "lj"                # lj (paper) | rfa (ablation "HR w RFA") | none
    lj_epsilon: float = 1.0
    lj_sigma: float = 5.0
    lj_lr: float = 1.0
    lj_steps: int = 3000
    lj_max_disp: float = 0.05            # max displacement per step, in units of sigma
    rfa_zeta: float = 1.0
    rfa_mass: float = 1.0
    rfa_dt: float = 0.05
    rfa_steps: int = 20000

    # ---------------- replay ------------------------------------------------------------------
    memory: str = "latent"               # latent (HR) | raw ("perfect exemplars") | none
    memory_size: int = 20000             # per-client fixed memory; 200 latent/class at 100 classes
    global_replay: bool = True           # decode p_j + N(0, s^2) for classes absent from memory
    synth_noise: float = -1.0            # s; -1 -> 1/sqrt(1 + 2*lam) (posterior std of the HAE)
    synth_per_class: int = -1            # -1 -> balanced with the client's memory / local data

    # ---------------- bookkeeping -------------------------------------------------------------
    seed: int = 0
    device: str = "auto"
    amp: bool = False
    eval_batch_size: int = 1000
    out_dir: str = "results/run"
    ckpt_every: int = 10                 # rounds
    log_every: int = 10                  # rounds
    max_train_per_task: int = 0          # >0 subsamples each task (smoke tests only)
    max_test_per_class: int = 0          # >0 subsamples the test set (smoke tests only)
    tag: str = ""

    def resolved(self) -> "Config":
        c = dataclasses.replace(self)
        if c.class_order_seed < 0:
            c.class_order_seed = c.seed
        return c

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Benchmarks of Table 2 (A/B/C/D = tasks / classes per task / clients / active clients).
BENCHMARKS = {
    "cifar100_10_10_50_5": dict(dataset="cifar100", num_tasks=10, classes_per_task=10,
                                num_clients=50, clients_per_round=5),
    "cifar100_20_5_50_5": dict(dataset="cifar100", num_tasks=20, classes_per_task=5,
                               num_clients=50, clients_per_round=5),
    "tinyimagenet_10_5_300_30": dict(dataset="tinyimagenet", num_tasks=10, classes_per_task=5,
                                     num_clients=300, clients_per_round=30),
}

# HR and the ablations of Table 2. memory_size is per client; 20000 latent = 200/class
# at the end of a 100-class stream, "10x less memory" = 2000 latent = 20/class.
VARIANTS = {
    "hr": dict(),
    "hr_mini": dict(memory_size=2000),
    "hr_no_latent": dict(memory="none"),
    "hr_perfect": dict(memory="raw"),
    "hr_no_kd": dict(kd_z=0.0, kd_x=0.0),
    "hr_no_global": dict(global_replay=False),
    "hr_rfa": dict(alignment="rfa"),
    # extra reference points (not rows of Table 2)
    "fedavg_ft": dict(memory="none", global_replay=False, kd_z=0.0, kd_x=0.0),
}


def build_config(benchmark: str, variant: str, **overrides) -> Config:
    kw = {}
    kw.update(BENCHMARKS[benchmark])
    kw.update(VARIANTS[variant])
    kw.update(overrides)
    return Config(**kw)
