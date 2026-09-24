"""Hybrid Replay (HR) for Federated Class-Incremental Learning.

Implements Algorithm 1 (server) and Algorithm 2 (clients) of
"Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and
Data-Free Techniques to Address Local and Global Forgetting" (ICLR 2025).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import align_centroids, lj_energy
from .data import TaskStream
from .models import HybridAutoencoder, count_params


# ----------------------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------------------
def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def augment(x: torch.Tensor, gen: torch.Generator | None, pad: int = 4) -> torch.Tensor:
    """Random crop (zero padding) + horizontal flip, batched on the tensor's device."""
    b, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device="cpu").to(dev)
    j = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device="cpu").to(dev)
    flip = (torch.rand(b, generator=gen, device="cpu") < 0.5).to(dev)
    rows = i[:, None] + torch.arange(h, device=dev)[None]
    cols = j[:, None] + torch.arange(w, device=dev)[None]
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    return xp[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]


def fedavg(states: list[dict]) -> dict:
    """theta <- 1/I sum_c theta^c (Algorithm 1, line 13); BN statistics are averaged too."""
    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = torch.stack([s[k] for s in states]).mean(0)
        else:
            out[k] = states[0][k].clone()
    return out


def rng_state(np_rng, torch_gen):
    return dict(np=np_rng.bit_generator.state, torch_gen=torch_gen.get_state(),
                torch=torch.get_rng_state(), py=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(st, np_rng, torch_gen):
    np_rng.bit_generator.state = st["np"]
    torch_gen.set_state(st["torch_gen"])
    torch.set_rng_state(st["torch"])
    random.setstate(st["py"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ----------------------------------------------------------------------------------------
# client memory (Algorithm 2, lines 5-9 and 23-25)
# ----------------------------------------------------------------------------------------
class ClientMemory:
    """Per-client exemplar memory: latent codes (HR), raw indices (perfect) or nothing."""

    def __init__(self, kind: str):
        self.kind = kind
        self.labels = torch.zeros(0, dtype=torch.long)
        self.codes = torch.zeros(0)            # latent: [n, d] float32
        self.indices = torch.zeros(0, dtype=torch.long)  # raw: indices into x_train

    def __len__(self):
        return len(self.labels)

    def state(self):
        return dict(kind=self.kind, labels=self.labels, codes=self.codes, indices=self.indices)

    @classmethod
    def from_state(cls, st):
        m = cls(st["kind"])
        m.labels, m.codes, m.indices = st["labels"], st["codes"], st["indices"]
        return m

    def classes(self) -> set:
        return set(self.labels.tolist())

    def reduce(self, quota: np.ndarray):
        """Fixed-memory policy (iCaRL): keep at most `quota[k]` exemplars of class k."""
        if len(self) == 0:
            return
        keep = []
        for c in self.labels.unique().tolist():
            keep.append(torch.where(self.labels == c)[0][:int(quota[c])])
        keep = torch.cat(keep).sort().values
        self.labels = self.labels[keep]
        if self.kind == "latent":
            self.codes = self.codes[keep]
        else:
            self.indices = self.indices[keep]


# ----------------------------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------------------------
class HRTrainer:
    def __init__(self, cfg):
        self.cfg = cfg = cfg.resolved()
        self.dev = pick_device(cfg.device)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)

        self.data = TaskStream(cfg, np.random.default_rng(cfg.seed + 12345))
        c, h, w = self.data.image_shape
        self.input_dim = c * h * w
        self.latent_dim = cfg.latent_dim or int(round(self.input_dim / 10))
        self.n_classes = cfg.num_tasks * cfg.classes_per_task
        if cfg.synth_noise < 0:
            cfg.synth_noise = 1.0 / math.sqrt(1.0 + 2.0 * cfg.lam)

        self.x_train = torch.from_numpy(self.data.x_train).to(self.dev)     # uint8
        self.y_train = torch.from_numpy(self.data.y_train).to(self.dev)
        self.x_test = torch.from_numpy(self.data.x_test).to(self.dev)
        self.y_test = torch.from_numpy(self.data.y_test).to(self.dev)

        self.model = HybridAutoencoder(cfg.dataset, h, self.latent_dim, cfg.decoder_channels).to(self.dev)
        self.local = copy.deepcopy(self.model)
        self.old = None                                           # theta_{h-1} (frozen)
        self.centroids = torch.zeros(self.n_classes, self.latent_dim, device=self.dev)
        self.memories = [ClientMemory(cfg.memory) for _ in range(cfg.num_clients)]
        # persistent per-(client, class) uniforms that assign the fractional part of the
        # per-class quota (needed when the budget is below one exemplar per class)
        self.slot_u = np.random.default_rng(cfg.seed + 777).random((cfg.num_clients, self.n_classes))
        self.use_amp = cfg.amp and self.dev.type == "cuda"
        # bf16 only on Ampere or newer (T4/V100 would emulate it slowly); fp16 + GradScaler otherwise
        self.amp_dtype = (torch.bfloat16 if self.use_amp and torch.cuda.get_device_capability(self.dev)[0] >= 8
                          else torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.task, self.round = 0, 0
        self.acc_matrix = []          # acc_matrix[h][t] = accuracy on task t after task h
        self.history = []
        self.log_lines = []
        self.elapsed = 0.0
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.ckpt_path = os.path.join(cfg.out_dir, "checkpoint.pt")
        self.result_path = os.path.join(cfg.out_dir, "result.json")
        cfg.save(os.path.join(cfg.out_dir, "config.json"))

    # ------------------------------------------------------------------ logging / ckpt
    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(self.cfg.out_dir, "log.txt"), "a") as f:
            f.write(line + "\n")

    def save_checkpoint(self):
        st = dict(model=self.model.state_dict(), old=self.old.state_dict() if self.old else None,
                  centroids=self.centroids, memories=[m.state() for m in self.memories],
                  task=self.task, round=self.round, acc_matrix=self.acc_matrix,
                  history=self.history, elapsed=self.elapsed,
                  rng=rng_state(self.np_rng, self.gen), cfg=self.cfg.to_dict())
        tmp = self.ckpt_path + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, self.ckpt_path)

    def try_resume(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            return False
        st = torch.load(self.ckpt_path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(st["model"])
        if st["old"] is not None:
            self.old = copy.deepcopy(self.model)
            self.old.load_state_dict(st["old"])
            self.old.eval().requires_grad_(False)
        self.centroids = st["centroids"].to(self.dev)
        self.memories = [ClientMemory.from_state(m) for m in st["memories"]]
        self.task, self.round = st["task"], st["round"]
        self.acc_matrix, self.history, self.elapsed = st["acc_matrix"], st["history"], st["elapsed"]
        set_rng_state(st["rng"], self.np_rng, self.gen)
        self.log(f"resumed from checkpoint at task {self.task + 1} round {self.round}")
        return True

    # ------------------------------------------------------------------ helpers
    def to_float(self, x_uint8):
        return x_uint8.float().div_(255.0)

    @torch.no_grad()
    def encode_mu(self, model, x, bs=500):
        model.eval()
        out = []
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            xb = self.to_float(xb) if xb.dtype == torch.uint8 else xb.float()
            with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                mu, _ = model.encode(xb)
            out.append(mu.float())
        return torch.cat(out) if out else torch.zeros(0, self.latent_dim, device=self.dev)

    @torch.no_grad()
    def decode(self, model, z, bs=500):
        model.eval()
        out = []
        for i in range(0, len(z), bs):
            with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out.append(model.decode(z[i:i + bs]).float())
        c, h, w = self.data.image_shape
        return torch.cat(out) if out else torch.zeros(0, c, h, w, device=self.dev)

    def lr_at(self, r: int) -> float:
        """Exponential decay from lr_start to lr_end over the rounds of a task (MFCL)."""
        cfg = self.cfg
        if cfg.rounds_per_task <= 1:
            return cfg.lr_start
        return cfg.lr_start * (cfg.lr_end / cfg.lr_start) ** (r / (cfg.rounds_per_task - 1))

    # ------------------------------------------------------------------ server: centroids
    def place_new_centroids(self, t: int):
        """Algorithm 1 lines 5-8 / Algorithm 2 lines 12-13."""
        cfg = self.cfg
        cls = list(self.data.task_classes(t))
        encoder_model = self.model  # = theta_{h-1} at the start of task h
        sums = torch.zeros(len(cls), self.latent_dim, device=self.dev)
        counts = torch.zeros(len(cls), device=self.dev)
        order = self.np_rng.permutation(cfg.num_clients)
        reporting = list(order[:cfg.clients_per_round])
        extra = list(order[cfg.clients_per_round:])
        while True:
            for c in reporting:
                idx = self.data.client_indices[t][c]
                if len(idx) == 0:
                    continue
                idx_t = torch.from_numpy(idx).to(self.dev)
                mu = self.encode_mu(encoder_model, self.x_train[idx_t])
                y = self.y_train[idx_t] - t * cfg.classes_per_task
                sums.index_add_(0, y, mu)
                counts += torch.bincount(y, minlength=len(cls)).float()
            if (counts > 0).all() or not extra:
                break
            reporting = [extra.pop(0)]  # a class is missing: ask one more client
        p_init = sums / counts.clamp_min(1)[:, None]
        p_old = self.centroids[: t * cfg.classes_per_task]
        p_new = align_centroids(cfg, p_init.cpu(), p_old.cpu(), self.gen).to(self.dev)
        self.centroids[t * cfg.classes_per_task:(t + 1) * cfg.classes_per_task] = p_new
        allp = self.centroids[: (t + 1) * cfg.classes_per_task]
        d = torch.cdist(allp, allp) + torch.eye(len(allp), device=self.dev) * 1e9
        shift = (p_new - p_init).norm(dim=1).mean().item()
        self.log(f"task {t + 1}: centroids placed ({cfg.alignment}); min pairwise dist "
                 f"{d.min().item():.3f}, mean shift from unaligned {shift:.3f}, "
                 f"LJ energy {lj_energy(p_new.cpu().double(), p_old.cpu().double(), cfg.lj_epsilon, cfg.lj_sigma):.2f}")

    # ------------------------------------------------------------------ client: replay data
    def build_replay(self, c: int, t: int, n_local: int):
        """Algorithm 2 lines 5-9: decode latent exemplars; synthesise absent classes."""
        cfg = self.cfg
        if t == 0:
            return None, None
        mem = self.memories[c]
        xs, ys = [], []
        if len(mem) > 0:
            if mem.kind == "latent":
                xs.append(self.decode(self.old, mem.codes.to(self.dev)))
            else:
                xs.append(self.to_float(self.x_train[mem.indices.to(self.dev)]))
            ys.append(mem.labels.to(self.dev))
        if cfg.global_replay:
            present = mem.classes()
            missing = [k for k in range(t * cfg.classes_per_task) if k not in present]
            if missing:
                if cfg.synth_per_class > 0:
                    n_syn = cfg.synth_per_class
                elif len(mem) > 0:
                    n_syn = max(1, round(len(mem) / len(present)))
                else:
                    n_syn = max(1, round(n_local / cfg.classes_per_task))
                lab = torch.tensor(missing, device=self.dev).repeat_interleave(n_syn)
                z = self.centroids[lab]
                z = z + cfg.synth_noise * torch.randn(z.shape, generator=self.gen).to(self.dev)
                xs.append(self.decode(self.old, z))
                ys.append(lab)
        if not xs:
            return None, None
        return torch.cat(xs), torch.cat(ys)

    # ------------------------------------------------------------------ client: local update
    def local_update(self, c: int, t: int, lr: float) -> tuple[dict, int, dict]:
        cfg = self.cfg
        idx = torch.from_numpy(self.data.client_indices[t][c]).to(self.dev)
        x_new = self.to_float(self.x_train[idx])
        y_new = self.y_train[idx]
        x_rep, y_rep = self.build_replay(c, t, len(idx))
        if x_rep is not None:
            x_all, y_all = torch.cat([x_new, x_rep]), torch.cat([y_new, y_rep])
        else:
            x_all, y_all = x_new, y_new
        n = len(x_all)
        stats = dict(n=n, n_new=len(idx), loss=0.0, rec=0.0, kl=0.0, clu=0.0, kd=0.0, steps=0)
        if n < 2:
            return None, 0, stats
        acc = torch.zeros(5, device=self.dev)  # running sums of loss terms (no per-step sync)

        local = self.local
        local.load_state_dict(self.model.state_dict())
        local.train()
        opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
        use_kd = self.old is not None and (cfg.kd_z > 0 or cfg.kd_x > 0)
        for _ in range(cfg.local_epochs):
            perm = torch.randperm(n, generator=self.gen).to(self.dev)
            for b in range(0, n, cfg.batch_size):
                bi = perm[b:b + cfg.batch_size]
                if len(bi) < 2:
                    continue  # BatchNorm needs >1 sample
                x = augment(x_all[bi], self.gen)
                y = y_all[bi]
                eps = torch.randn((len(bi), self.latent_dim), generator=self.gen).to(self.dev)
                with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    mu, logvar = local.encode(x)
                    mu, logvar = mu.float(), logvar.float().clamp(-cfg.logvar_clip, cfg.logvar_clip)
                    zs = mu + torch.exp(0.5 * logvar) * eps
                    x_rec = local.decode(zs).float()
                    if use_kd:
                        with torch.no_grad():
                            mu_o, _ = self.old.encode(x)
                            x_o = self.old.decode(mu_o).float()
                            mu_o = mu_o.float()
                        x_d = local.decode(mu).float() if cfg.kd_x > 0 else None
                # ELBO (Gaussian decoder, SSE) + lambda ||z - p_y||^2  (paper Eq. 7)
                rec = (x_rec - x).pow(2).flatten(1).sum(1)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
                clu = (zs - self.centroids[y]).pow(2).sum(1)
                loss = rec + cfg.beta_kl * kl + cfg.lam * clu
                kd = torch.zeros_like(loss)
                if use_kd:  # Algorithm 2 lines 20-21
                    if cfg.kd_z > 0:
                        kd = kd + cfg.kd_z * (mu - mu_o).pow(2).sum(1)
                    if cfg.kd_x > 0:
                        kd = kd + cfg.kd_x * (x_d - x_o).pow(2).flatten(1).sum(1)
                    loss = loss + kd
                # per-pixel normalisation keeps SGD with lr=0.1 stable; relative weights unchanged
                loss = loss.mean() / self.input_dim
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(local.parameters(), cfg.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
                acc += torch.stack([loss.detach(), rec.mean().detach(), kl.mean().detach(),
                                    clu.mean().detach(), kd.mean().detach()])
                stats["steps"] += 1
        for k, v in zip(("loss", "rec", "kl", "clu", "kd"), acc.tolist()):
            stats[k] = v
        return {k: v.detach().clone() for k, v in local.state_dict().items()}, n, stats

    # ------------------------------------------------------------------ memory update
    @torch.no_grad()
    def update_memories(self, t: int):
        """Algorithm 2 lines 23-25 with the fixed-memory (iCaRL) budget."""
        cfg = self.cfg
        if cfg.memory == "none":
            return
        seen = (t + 1) * cfg.classes_per_task
        q = cfg.memory_size / seen                      # per-class quota m = K / #classes
        for c in range(cfg.num_clients):
            mem = self.memories[c]
            quota = np.floor(q) + (self.slot_u[c] < q - np.floor(q))
            # re-encode old latent exemplars: M <- f_h(g_{h-1}(M))
            if mem.kind == "latent" and len(mem) > 0 and self.old is not None:
                x_dec = self.decode(self.old, mem.codes.to(self.dev))
                mem.codes = self.encode_mu(self.model, x_dec).cpu()
            mem.reduce(quota)
            # add randomly sampled exemplars of the current task
            idx = self.data.client_indices[t][c]
            if len(idx) == 0:
                continue
            idx = idx[self.np_rng.permutation(len(idx))]
            labels = self.data.y_train[idx]
            sel = np.concatenate([np.where(labels == k)[0][:int(quota[k])] for k in np.unique(labels)])
            if len(sel) == 0:
                continue
            idx, labels = idx[sel], labels[sel]
            if mem.kind == "latent":
                codes = self.encode_mu(self.model, self.x_train[torch.from_numpy(idx).to(self.dev)]).cpu()
                mem.codes = torch.cat([mem.codes.view(-1, self.latent_dim), codes])
            else:
                mem.indices = torch.cat([mem.indices, torch.from_numpy(idx)])
            mem.labels = torch.cat([mem.labels, torch.from_numpy(labels)])

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, t: int):
        cfg = self.cfg
        seen = (t + 1) * cfg.classes_per_task
        mask = self.y_test < seen
        x, y = self.x_test[mask], self.y_test[mask]
        mu = self.encode_mu(self.model, x, bs=cfg.eval_batch_size)
        d = torch.cdist(mu, self.centroids[:seen])
        pred = d.argmin(1)
        correct = (pred == y).float()
        per_task = []
        for k in range(t + 1):
            m = (y >= k * cfg.classes_per_task) & (y < (k + 1) * cfg.classes_per_task)
            per_task.append(correct[m].mean().item() * 100)
        # reconstruction quality of the decoder on seen test data (diagnostic)
        n_rec = min(1000, len(x))
        xr = self.decode(self.model, mu[:n_rec])
        mse = (xr - self.to_float(x[:n_rec])).pow(2).mean().item()
        return correct.mean().item() * 100, per_task, mse

    # ------------------------------------------------------------------ main loop
    def run(self):
        cfg = self.cfg
        if os.path.exists(self.result_path):
            with open(self.result_path) as f:
                res = json.load(f)
            if res.get("finished"):
                self.log("already finished; skipping")
                return res
        resumed = self.try_resume()
        if not resumed:
            self.log(f"config: {json.dumps(cfg.to_dict())}")
            self.log(f"device {self.dev}; encoder {count_params(self.model.encoder) / 1e6:.2f}M params, "
                     f"decoder {count_params(self.model.decoder) / 1e6:.2f}M params, latent {self.latent_dim}")
            sizes = [[len(i) for i in task] for task in self.data.client_indices]
            self.log(f"client samples per task (task 1): min {min(sizes[0])} max {max(sizes[0])} "
                     f"mean {np.mean(sizes[0]):.1f}")
        t0 = time.time() - self.elapsed
        while self.task < cfg.num_tasks:
            t = self.task
            if self.round == 0:
                self.place_new_centroids(t)
            while self.round < cfg.rounds_per_task:
                r = self.round
                lr = self.lr_at(r)
                clients = self.np_rng.choice(cfg.num_clients, cfg.clients_per_round, replace=False)
                states, agg = [], dict(n=0, n_new=0, loss=0.0, steps=0, rec=0.0, clu=0.0, kd=0.0, kl=0.0)
                for c in clients:
                    st, n, s = self.local_update(int(c), t, lr)
                    for k in agg:
                        agg[k] += s[k]
                    if st is not None:
                        states.append(st)
                if states:
                    self.model.load_state_dict(fedavg(states))
                self.round += 1
                self.elapsed = time.time() - t0
                if self.round % cfg.log_every == 0 or self.round == cfg.rounds_per_task:
                    k = max(1, agg["steps"])
                    self.log(f"task {t + 1} round {self.round}/{cfg.rounds_per_task} lr {lr:.4f} "
                             f"samples {agg['n']} (new {agg['n_new']}) loss {agg['loss'] / k:.4f} "
                             f"rec {agg['rec'] / k:.1f} kl {agg['kl'] / k:.1f} clu {agg['clu'] / k:.2f} "
                             f"kd {agg['kd'] / k:.1f} | {self.elapsed / 60:.1f} min")
                if self.round % cfg.ckpt_every == 0 and self.round < cfg.rounds_per_task:
                    self.save_checkpoint()
            # end of task: memory update, evaluation, teacher hand-over
            self.update_memories(t)
            acc, per_task, mse = self.evaluate(t)
            self.acc_matrix.append(per_task)
            mem_sizes = [len(m) for m in self.memories]
            self.history.append(dict(task=t + 1, acc=acc, per_task=per_task, rec_mse=mse,
                                     mem_mean=float(np.mean(mem_sizes)), elapsed_min=self.elapsed / 60))
            self.log(f"== task {t + 1}: accuracy on {(t + 1) * cfg.classes_per_task} classes {acc:.2f} | "
                     f"per task {[round(a, 1) for a in per_task]} | rec MSE {mse:.4f} | "
                     f"mean memory {np.mean(mem_sizes):.0f}")
            self.old = copy.deepcopy(self.model).eval().requires_grad_(False)
            self.task += 1
            self.round = 0
            self.save_checkpoint()
            self.write_result(finished=self.task >= cfg.num_tasks)
        return self.write_result(finished=True)

    def write_result(self, finished: bool):
        accs = [h["acc"] for h in self.history]
        T = len(self.acc_matrix)
        forgetting = None
        if T > 1:
            f = [max(self.acc_matrix[h][k] for h in range(k, T - 1)) - self.acc_matrix[T - 1][k]
                 for k in range(T - 1)]
            forgetting = float(np.mean(f))
        res = dict(config=self.cfg.to_dict(), finished=finished, tasks_done=T,
                   final_acc=accs[-1] if accs else None,
                   avg_acc=float(np.mean(accs)) if accs else None,
                   avg_forgetting=forgetting, acc_per_task=accs, acc_matrix=self.acc_matrix,
                   history=self.history, elapsed_min=self.elapsed / 60,
                   latent_dim=self.latent_dim,
                   decoder_params=count_params(self.model.decoder),
                   encoder_params=count_params(self.model.encoder))
        tmp = self.result_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, self.result_path)
        return res
