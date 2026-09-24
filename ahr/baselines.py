"""Reference CIL baselines sharing AHR's backbone, optimiser and minibatch scheme.

* ``ft``    - fine-tuning on the new task only (lower bound).
* ``ft_e``  - fine-tuning with raw exemplar replay (fixed memory, random selection).
* ``icarl`` - iCaRL (Rebuffi et al., 2017) as in FACIL: exemplar replay + knowledge
  distillation (T=2) on the old logits, herding selection, nearest-mean-of-exemplars
  classifier at test time.
* ``joint`` - one model trained on all tasks jointly (upper bound).

By default (``replay_sampling="union"``) the replay baselines train on the shuffled
union of the new data and the exemplars, as in FACIL (Masana et al., 2022).
``replay_sampling="balanced"`` uses AHR's task-balanced minibatches instead
(``B / l`` new samples, ``B (l-1) / l`` exemplars).
"""
import copy
import math

import torch
import torch.nn.functional as F

from . import memory as mem
from .common import autocast, batched, make_optimizer, make_scheduler
from .data import augment, to_float
from .models import IncrementalClassifier, make_backbone, n_params


class SoftmaxLearner:
    def __init__(self, bench, args, method, log=print):
        self.bench, self.args, self.method, self.log = bench, args, method, log
        self.model = IncrementalClassifier(make_backbone(bench.name, bench.input_shape, pool=1))
        self.use_memory = method in ("ft_e", "icarl")
        self.memory = mem.RawMemory(args.n_exemplars) if self.use_memory else None
        self.n_seen = 0
        self.class_means = None
        self.log(f"[{method}] backbone params={n_params(self.model.backbone):,}"
                 + (f" memory=raw x {args.n_exemplars}" if self.use_memory else ""))

    # ------------------------------------------------------------------ #
    def predict(self, x_uint8):
        self.model.eval()
        if self.method == "icarl":
            f = F.normalize(batched(self.model.features, x_uint8, self.args.eval_batch), dim=1)
            return torch.cdist(f, self.class_means).argmin(1)
        return batched(self.model, x_uint8, self.args.eval_batch).argmax(1)

    # ------------------------------------------------------------------ #
    def _train(self, t, x_new, y_new, old):
        a = self.args
        model = self.model
        opt = make_optimizer(model.parameters(), a)
        replay = self.use_memory and len(self.memory) > 0
        if replay and a.replay_sampling == "union":
            # FACIL-style: shuffle the union of the new data and the exemplars
            x_new = torch.cat([x_new, self.memory.data])
            y_new = torch.cat([y_new, self.memory.labels])
            replay = False
        n_new = len(y_new)
        b_new = max(1, round(a.batch_size / (t + 1))) if replay else a.batch_size
        b_mem = a.batch_size - b_new if replay else 0
        iters = math.ceil(n_new / b_new)
        sched = make_scheduler(opt, a, a.epochs * iters)
        n_old = 0 if old is None else old.fc.out_features
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(n_new)
            tot_ce = tot_kd = 0.0
            for it in range(iters):
                idx = perm[it * b_new:(it + 1) * b_new]
                x, y = x_new[idx], y_new[idx]
                if b_mem > 0:
                    xm, ym = self.memory.get(self.memory.sample_indices(b_mem))
                    x, y = torch.cat([x, xm]), torch.cat([y, ym])
                x = to_float(x)
                if a.augment:
                    x = augment(x, pad=a.crop_pad, flip=self.bench.flip)
                with autocast(a.bf16):
                    out = model(x)
                    if self.method == "icarl" and old is not None:
                        with torch.no_grad():
                            out_old = old(x)
                out = out.float()
                loss = F.cross_entropy(out, y)
                tot_ce += loss.item()
                if self.method == "icarl" and old is not None:
                    T = a.kd_temperature
                    kd = F.kl_div(F.log_softmax(out[:, :n_old] / T, 1),
                                  F.softmax(out_old.float() / T, 1), reduction="batchmean") * T * T
                    loss = loss + a.kd_lambda * kd
                    tot_kd += kd.item()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
            if ep == 0 or (ep + 1) % a.log_every == 0 or ep + 1 == a.epochs:
                self.log(f"  task {t + 1} epoch {ep + 1}/{a.epochs} ({iters} it) "
                         f"ce={tot_ce / iters:.3f} kd={tot_kd / iters:.3f}")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_memory(self, task):
        per_class = self.memory.per_class(self.n_seen)
        xs, ys = [task.x_train], [task.y_train]
        if len(self.memory) > 0:
            xs.append(self.memory.data)
            ys.append(self.memory.labels)
        x, y = torch.cat(xs), torch.cat(ys)
        if self.method == "icarl":
            self.model.eval()
            feats = batched(self.model.features, x, self.args.eval_batch)
            keep = mem.select_herding(feats, y, per_class)
        else:
            keep = mem.select_random(y, per_class)
        self.memory.set(x[keep], y[keep])

    @torch.no_grad()
    def _compute_class_means(self):
        """iCaRL NME: mean of L2-normalised exemplar features (plus flipped copies)."""
        self.model.eval()
        x, y = self.memory.data, self.memory.labels
        f = F.normalize(batched(self.model.features, x, self.args.eval_batch), dim=1)
        if self.bench.flip:
            f2 = F.normalize(batched(lambda b: self.model.features(b.flip(3)), x,
                                     self.args.eval_batch), dim=1)
            f = F.normalize(f + f2, dim=1)
        self.class_means = F.normalize(
            torch.stack([f[y == c].mean(0) for c in range(self.n_seen)]), dim=1)

    # ------------------------------------------------------------------ #
    def learn_task(self, t, task):
        old = None
        if self.method == "icarl" and t > 0:
            old = copy.deepcopy(self.model).eval()
            for p in old.parameters():
                p.requires_grad_(False)
        self.model.add_classes(len(task.classes))
        self.n_seen += len(task.classes)
        self._train(t, task.x_train, task.y_train, old)
        if self.use_memory:
            self._update_memory(task)
        if self.method == "icarl":
            self._compute_class_means()


class JointLearner(SoftmaxLearner):
    """Upper bound: a single model trained on the union of all tasks' data."""

    def __init__(self, bench, args, log=print):
        super().__init__(bench, args, "joint", log)

    def learn_all(self):
        n_tasks = self.bench.n_tasks
        x, y = self.bench.train_upto(n_tasks - 1)
        self.model.add_classes(self.bench.n_classes)
        self.n_seen = self.bench.n_classes
        self._train(0, x, y, None)
