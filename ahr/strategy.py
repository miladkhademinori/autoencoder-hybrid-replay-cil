"""Autoencoder-Based Hybrid Replay (AHR), Algorithms 1-4 of the paper.

For every new task ``l``:

1. ``CCE_Placement`` (Alg. 2): the new classes' centroid embeddings are
   initialised at their class means under the current encoder and pushed away
   from all other CCEs by the repulsive force algorithm. CCEs never move again.
2. ``HAE_Train`` (Alg. 3): the previous HAE is frozen and copied; the copy is
   trained on the new data plus exemplars decoded on the fly from the latent
   memory. Each minibatch holds ``B / l`` new samples and ``B (l-1) / l``
   decoded exemplars drawn uniformly (class-balanced) from memory. The loss is

       ||x - psi(phi(x))||^2 + lambda ||phi(x) - p_y||^2                 (Eq. 1)
       + a_z ||phi_old(x) - phi(x)|| + a_x ||psi_old(phi_old(x)) - psi(phi(x))||

3. ``Memory_Population`` (Alg. 4): exemplars are selected per class
   (``M / #classes`` each) and stored as latent codes.

Inference: ``argmin_{c} ||phi(x) - p_c||`` over all classes seen so far.

Implementation options (see REPRODUCTION.md for why they exist):

* ``memory_mode="reencode"``: every task, the old exemplars are decoded with the
  previous decoder and re-encoded with the new encoder (literal reading of
  Alg. 4, line 4). ``memory_mode="frozen"``: a code ``phi(w_i, x)`` is stored once
  at task ``i`` and never re-encoded (Alg. 4, line 11); ``alpha_mem`` then applies
  the decoder distillation directly to the stored codes,
  ``||psi(m) - psi_old(m)||^2``, so that they stay decodable.
* ``memorize_epochs``: after selection, the decoder alone is fitted to map the
  new exemplars' codes to their original images (the "memorisation" the paper
  relies on) while old codes keep their previous decoding.
* ``lam_recon_new``: the latent loss is also applied to the previous HAE's
  reconstructions of the new samples, so that "looks decoded" is not a cue for
  "belongs to an old task".
"""
import copy
import math

import torch

from . import memory as mem
from .common import autocast, batched, make_optimizer, make_scheduler
from .data import augment, to_float
from .models import make_hae, n_params
from .rfa import place_cces, spacing_stats


def _sq(a, b):
    return ((a - b) ** 2).flatten(1).sum(1)


class AHR:
    def __init__(self, bench, args, log=print):
        self.bench, self.args, self.log = bench, args, log
        self.model = make_hae(bench.name, bench.input_shape, args.latent_dim,
                              decoder_width=args.decoder_width, enc_pool=args.enc_pool,
                              input_blur=args.input_blur, latent_kind=args.latent_kind)
        if args.latent_kind == "spatial" and bench.name != "mnist":
            args.latent_dim = self.model.encoder.head.conv.out_channels * 64
        self.cces = torch.zeros(0, args.latent_dim)
        self.lossless = args.memory_kind == "raw"
        if self.lossless:
            self.memory = mem.RawMemory(args.n_exemplars)
        else:
            self.memory = mem.LatentMemory(args.n_exemplars, bits=args.latent_bits)
        self.n_seen = 0
        from .models import GaussianBlur
        self._hf_blur = GaussianBlur(bench.input_shape[0], args.hf_sigma)
        self.mem_src = torch.zeros(0, dtype=torch.long)  # (task * 1e6 + index) of each exemplar
        self.log(f"[AHR] encoder params={n_params(self.model.encoder):,} "
                 f"decoder params={n_params(self.model.decoder):,} "
                 f"memory={'raw' if self.lossless else 'latent'} x {args.n_exemplars} exemplars")

    # ------------------------------------------------------------------ #
    def encode(self, x_uint8, model=None):
        model = model or self.model
        model.eval()
        return batched(model.encoder, x_uint8, self.args.eval_batch)

    def class_features(self, x_uint8):
        """Latent features used for classification: phi(x) (paper), or phi(psi(phi(x)))
        when classification operates in the decoder's output domain (--latent-domain recon)."""
        if self.args.latent_domain != "recon":
            return self.encode(x_uint8)
        m = self.model
        m.eval()
        return batched(lambda b: m.encoder(m.decoder(m.encoder(b))), x_uint8, self.args.eval_batch)

    def predict(self, x_uint8):
        z = self.class_features(x_uint8)
        return torch.cdist(z, self.cces).argmin(1)

    def decode_memory(self, idx, decoder):
        """Replay samples: stored latents decoded with the (frozen) previous decoder."""
        data, y = self.memory.get(idx)
        if self.lossless:
            return to_float(data), y
        with torch.no_grad(), autocast(self.args.bf16):
            x = decoder(data).float()
        return x, y

    # ------------------------------------------------------------------ #
    def cce_placement(self, task):
        a = self.args
        z = self.class_features(task.x_train)
        init = torch.stack([z[task.y_train == c].mean(0) for c in task.classes])
        new, n_steps = place_cces(init, self.cces, zeta=a.rfa_zeta, mass=a.rfa_mass, dt=a.rfa_dt,
                                  steps=a.rfa_steps, damping=a.rfa_damping,
                                  softening=a.rfa_softening, target_dist=a.rfa_target)
        shift = (new - init).norm(dim=1).mean()
        self.cces = torch.cat([self.cces, new.float()])
        mn, mean = spacing_stats(self.cces)
        self.log(f"  CCE placement ({n_steps} RFA steps): |init spread|={spacing_stats(init)[1]:.3f} "
                 f"mean shift={shift:.3f} min/mean CCE distance={mn:.3f}/{mean:.3f} "
                 f"|p| mean={self.cces.norm(dim=1).mean():.3f}")

    # ------------------------------------------------------------------ #
    def train_task(self, t, task, old):
        a = self.args
        model = self.model
        opt = make_optimizer(model.parameters(), a)
        n_new = len(task.y_train)
        replay = old is not None and len(self.memory) > 0
        b_new = max(1, round(a.batch_size / (t + 1))) if replay else a.batch_size
        b_mem = a.batch_size - b_new if replay else 0
        iters = math.ceil(n_new / b_new)
        sched = make_scheduler(opt, a, a.epochs * iters)
        cces = self.cces
        flip = self.bench.flip
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(n_new)
            tot = {"rec": 0.0, "lat": 0.0, "dz": 0.0, "dx": 0.0, "mem": 0.0}
            for it in range(iters):
                idx = perm[it * b_new:(it + 1) * b_new]
                x = to_float(task.x_train[idx])
                y = task.y_train[idx]
                codes = None
                if b_mem > 0:
                    midx = self.memory.sample_indices(b_mem)
                    xm, ym = self.decode_memory(midx, old.decoder)
                    if not self.lossless and a.alpha_mem > 0:
                        codes, xm_target = self.memory.get(midx)[0], xm
                    x, y = torch.cat([x, xm]), torch.cat([y, ym])
                n = len(idx)
                if b_mem > 0 and a.hf_transplant > 0 and not self.lossless:
                    # give decoded exemplars the high-frequency residual of random real
                    # new-task images, so that "sharp" is not a cue for "new class"
                    with torch.no_grad():
                        donors = x[torch.randint(0, n, (len(x) - n,))]
                        hf = donors - self._hf_blur(donors)
                        use = (torch.rand(len(x) - n, 1, 1, 1) < a.hf_transplant).float()
                        x = torch.cat([x[:n], (x[n:] + use * hf).clamp(0, 1)])
                if a.augment:
                    x = augment(x, pad=a.crop_pad, flip=flip)
                use_rn = a.lam_recon_new > 0 and (old is not None or a.recon_new_source == "current")
                x_in = x
                if use_rn:
                    # new-task samples are also presented as reconstructions, so that "looks
                    # decoded" is not a cue for "belongs to an old class". They go through the
                    # encoder in the *same* forward pass as the rest of the minibatch: a separate
                    # all-reconstruction batch would get its own BatchNorm statistics in train
                    # mode, which the network can exploit and which disappears in eval mode.
                    with torch.no_grad(), autocast(a.bf16):
                        if a.recon_new_source == "current":
                            model.eval()
                            x_rn = model(x[:n])[1].float()
                            model.train()
                        else:
                            x_rn = old(x[:n])[1].float()
                    x_in = torch.cat([x, x_rn])
                with autocast(a.bf16):
                    z_all = model.encoder(x_in)
                    z = z_all[:len(x)]
                    xh = model.decoder(z)
                    if old is not None:
                        with torch.no_grad():
                            z_old = old.encoder(x)
                            x_old = old.decoder(z_old)
                z, xh = z.float(), xh.float()
                l_rec = _sq(xh, x).mean()
                if a.latent_domain == "recon":
                    # the latent (classification) loss only sees decoder outputs: decoded
                    # exemplars here and reconstructions of the new samples below
                    l_lat = _sq(z[n:], cces[y[n:]]).mean() if len(x) > n else z.new_zeros(())
                else:
                    l_lat = _sq(z, cces[y]).mean()
                loss = l_rec + a.lam * l_lat
                if old is not None:
                    dz, dx = _sq(z, z_old.float()), _sq(xh, x_old.float())
                    if a.distill_norm == "l2":
                        dz, dx = dz.clamp_min(1e-12).sqrt(), dx.clamp_min(1e-12).sqrt()
                    l_dz, l_dx = dz.mean(), dx.mean()
                    loss = loss + a.alpha_z * l_dz + a.alpha_x * l_dx
                    tot["dz"] += l_dz.item()
                    tot["dx"] += l_dx.item()
                if old is not None and a.alpha_kd > 0:
                    # knowledge distillation on the relative distances to the old CCEs
                    # (softmax over -||z - p_c||^2 of the old classes), in the spirit of
                    # iCaRL/LwF: preserves how the old encoder related each (real or decoded)
                    # image to the old classes without pinning new samples to their old codes
                    n_old = self.n_seen - len(task.classes)
                    p_old = cces[:n_old]
                    scale = a.kd_scale * a.rfa_target ** 2
                    lo = -torch.cdist(z_old.float(), p_old) ** 2 / scale
                    ln = -torch.cdist(z, p_old) ** 2 / scale
                    l_kd = torch.nn.functional.kl_div(ln.log_softmax(1), lo.softmax(1),
                                                      reduction="batchmean")
                    loss = loss + a.alpha_kd * l_kd
                    tot["kd"] = tot.get("kd", 0.0) + l_kd.item()
                if use_rn:
                    z_rn = z_all[len(x):].float()
                    loss = loss + a.lam * a.lam_recon_new * _sq(z_rn, cces[y[:n]]).mean()
                if codes is not None:
                    # decoder distillation on the stored codes: psi(m) must keep decoding
                    # every stored latent into the same exemplar as psi_old(m)
                    with autocast(a.bf16):
                        xm_new = model.decoder(codes)
                    l_mem = _sq(xm_new.float(), xm_target).mean()
                    loss = loss + a.alpha_mem * l_mem
                    tot["mem"] += l_mem.item()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                tot["rec"] += l_rec.item()
                tot["lat"] += l_lat.item()
            if ep == 0 or (ep + 1) % a.log_every == 0 or ep + 1 == a.epochs:
                msg = " ".join(f"{k}={v / iters:.3f}" for k, v in tot.items())
                self.log(f"  task {t + 1} epoch {ep + 1}/{a.epochs} ({iters} it, b_new={b_new}) {msg}")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _select(self, z, y, scores, per_class):
        a = self.args
        if a.selection == "rank":
            return mem.select_rank(scores, y, per_class)
        if a.selection == "herding":
            return mem.select_herding(z, y, per_class)
        return mem.select_random(y, per_class)

    @torch.no_grad()
    def populate_memory(self, t, task, old):
        if self.args.memory_mode == "frozen":
            return self.populate_memory_frozen(t, task)
        a = self.args
        n_classes = self.n_seen
        per_class = self.memory.per_class(n_classes)
        # D <- D_l  U  psi_{l-1}(M): new data plus the old exemplars decoded by the old decoder
        xs, ys = [to_float(task.x_train)], [task.y_train]
        src = [t * 10**6 + torch.arange(len(task.y_train)), self.mem_src]
        if len(self.memory) > 0:
            all_idx = torch.arange(len(self.memory))
            for i in range(0, len(all_idx), a.eval_batch):
                xm, ym = self.decode_memory(all_idx[i:i + a.eval_batch], old.decoder)
                xs.append(xm)
                ys.append(ym)
        x, y, src = torch.cat(xs), torch.cat(ys), torch.cat(src)
        self.model.eval()
        z = batched(self.model.encoder, x, a.eval_batch, uint8=False)
        scores = _sq(z, self.cces[y])
        keep = self._select(z, y, scores, per_class)
        if self.lossless:
            self.memory.set((x[keep] * 255).round().to(torch.uint8), y[keep])
        else:
            self.memory.set(z[keep], y[keep])
        self.mem_src = src[keep]
        self.log(f"  memory: {len(self.memory)} exemplars ({per_class}/class), "
                 f"{self.memory.scalars():,} scalars, {self.memory.nbytes():,} bytes")

    @torch.no_grad()
    def populate_memory_frozen(self, t, task):
        """Alg. 4 with codes frozen at storage time: exemplars of task i keep the code
        phi(w_i, x) computed by the encoder of task i; old classes are only reduced to
        the new per-class quota (keeping the first-selected ones, as in iCaRL)."""
        a = self.args
        per_class = self.memory.per_class(self.n_seen)
        self.model.eval()
        z = self.encode(task.x_train)
        keep_new = self._select(z, task.y_train, _sq(z, self.cces[task.y_train]), per_class)
        new_data = task.x_train[keep_new] if self.lossless else z[keep_new]
        new_src = t * 10**6 + keep_new
        if len(self.memory) > 0:
            keep_old = mem.truncate_per_class(self.memory.labels, per_class)
            old_data, old_y = self.memory.get(keep_old)
            data = torch.cat([old_data, new_data])
            ys = torch.cat([old_y, task.y_train[keep_new]])
            src = torch.cat([self.mem_src[keep_old], new_src])
        else:
            data, ys, src = new_data, task.y_train[keep_new], new_src
        self.memory.set(data, ys)
        self.mem_src = src
        self.log(f"  memory (frozen codes): {len(self.memory)} exemplars ({per_class}/class), "
                 f"{self.memory.scalars():,} scalars, {self.memory.nbytes():,} bytes")

    # ------------------------------------------------------------------ #
    def memorize(self, t, task, old):
        """Decoder-only memorisation of the stored exemplars.

        New exemplars are fitted to their original images (still available at the end
        of the task); old exemplars are fitted to what the previous decoder produced
        for them, so that their decoding does not drift. Only used with frozen codes.
        """
        a = self.args
        if (a.memorize_epochs <= 0 or self.lossless or len(self.memory) == 0
                or a.memory_mode != "frozen"):
            return
        codes, _ = self.memory.get(torch.arange(len(self.memory)))
        is_new = (self.mem_src // 10**6) == t
        targets = torch.empty(len(codes), *self.bench.input_shape)
        new_idx = is_new.nonzero(as_tuple=True)[0]
        targets[new_idx] = to_float(task.x_train[self.mem_src[new_idx] % 10**6])
        old_idx = (~is_new).nonzero(as_tuple=True)[0]
        if len(old_idx):
            old.eval()
            targets[old_idx] = batched(old.decoder, codes[old_idx], a.eval_batch, uint8=False)
        dec = self.model.decoder
        dec.train()
        params = list(dec.parameters())
        if a.memorize_codes:
            # the stored codes are memory contents: refine them jointly with the decoder
            # (auto-decoder style) so that they decode to their exemplars more faithfully
            codes = codes.clone().requires_grad_(True)
            params.append(codes)
        opt = torch.optim.Adam(params, lr=a.memorize_lr)
        n, B = len(codes), a.batch_size
        iters = math.ceil(n / B)
        epochs = a.memorize_epochs
        if a.memorize_steps:  # a fixed number of optimisation steps instead of epochs
            epochs = max(1, math.ceil(a.memorize_steps / iters))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * iters)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for it in range(iters):
                idx = perm[it * B:(it + 1) * B]
                with autocast(a.bf16):
                    out = dec(codes[idx])
                loss = _sq(out.float(), targets[idx]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
        if a.memorize_codes:
            self.memory.set(codes.detach(), self.memory.labels)
        self.log(f"  memorisation: {epochs} decoder epochs ({epochs * iters} steps) over {n} exemplars, "
                 f"final loss={tot / iters:.3f}")

    # ------------------------------------------------------------------ #
    def learn_task(self, t, task):
        self.n_seen += len(task.classes)
        old = None
        if t > 0:
            old = copy.deepcopy(self.model).eval()
            for p in old.parameters():
                p.requires_grad_(False)
        self.cce_placement(task)              # Alg. 2 (uses phi_{l-1})
        self.train_task(t, task, old)         # Alg. 3
        self.populate_memory(t, task, old)    # Alg. 4
        self.memorize(t, task, old)
        del old                               # Alg. 1, line 7

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reconstruction_mse(self, x_uint8, n=500):
        self.model.eval()
        x = to_float(x_uint8[:n])
        z = self.model.encoder(x)
        return float(((self.model.decoder(z) - x) ** 2).mean())

    @torch.no_grad()
    def diagnostics(self, x_uint8, y):
        """Accuracy on AE-reconstructed test images and on the decoded memory."""
        self.model.eval()
        rec = batched(lambda b: self.model.decoder(self.model.encoder(b)), x_uint8, self.args.eval_batch)
        z = batched(self.model.encoder, rec, self.args.eval_batch, uint8=False)
        acc_rec = float((torch.cdist(z, self.cces).argmin(1) == y).float().mean()) * 100
        acc_mem = None
        if len(self.memory):
            idx = torch.arange(len(self.memory))
            xm, ym = self.decode_memory(idx, self.model.decoder)
            zm = batched(self.model.encoder, xm, self.args.eval_batch, uint8=False)
            acc_mem = float((torch.cdist(zm, self.cces).argmin(1) == ym).float().mean()) * 100
        return acc_rec, acc_mem

    @torch.no_grad()
    def memory_fidelity(self, n=None):
        """Diagnostic only: PSNR (dB) of the exemplars decoded from memory against the
        original training images they were encoded from, and a sample of both."""
        if len(self.memory) == 0:
            return None, None, None
        idx = torch.arange(len(self.memory)) if n is None else torch.randperm(len(self.memory))[:n]
        self.model.eval()
        dec, _ = self.decode_memory(idx, self.model.decoder)
        orig = to_float(torch.stack([self.bench.tasks[int(s) // 10**6].x_train[int(s) % 10**6]
                                     for s in self.mem_src[idx]]))
        mse = ((dec - orig) ** 2).flatten(1).mean(1).clamp_min(1e-10)
        return float((10 * torch.log10(1.0 / mse)).mean()), orig[:16].clone(), dec[:16].clone()
