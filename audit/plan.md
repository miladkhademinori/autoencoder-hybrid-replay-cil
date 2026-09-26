# AHR reproduction: action plan for one 4-core CPU

Nothing has been run for this plan, and every accuracy change below is an estimate. The estimates come from the adversarial verdicts on K01-K22, not from the original claims. Timings come from the `[Ns]` stamps in `results/*/*.log`.

## 0. Starting point

**Jobs running now** (checked with ps and the log tails at 21:24 UTC). ETAs are extrapolated from earlier tasks' durations.

| PID | Run | Progress | ETA | Recommendation |
|---|---|---|---|---|
| 7683 | svhn ahr_lossless s0 (input domain) | task 4 of 5, epoch 40 | about 4.5 h | Keep. It fills the SVHN AHR-lossless cell. |
| 4981 | cifar100 ahr_lossless s0 | task 9 of 10 (53.5 after task 8) | about 7 h | Keep. It fills the CIFAR-100 AHR-lossless cell. |
| 7306 | svhn ahr split s0 (zero padding) | task 4 of 5 (87.8 after task 3) | about 12 h | Keep. It settles split vs spatial on the same seed (K10), and its `ckpt_t1` seeds action A9. |
| 7676 | cifar10 ahr s1 (old config, zero padding) | task 3 of 5 | about 13 h | **Stop it if the user agrees.** It is a second seed of a configuration this plan replaces (K03). The cloud `cifar10 s2` run already gives a seed replicate of that configuration. Stopping it frees one core now, worth about 13 CPU-h. |

**Cloud results to collect with `python scripts/collect_cloud.py`** when they finish (`results/cloud_jobs.txt`):
- The reflect-padding runs (cifar10 s0, cifar100 s0, svhn split s0). They measure the effect of K03 on its own.
- svhn split s1/s2 and cifar10 s2.
- The ft_e bft30 runs.

**Rule for all code changes:**
- Every change sits behind a new flag whose default reproduces today's behaviour exactly.
- Every new variant is also identical to today's code on task 1.
- So the running jobs, `jobqueue.py`/`cloud_run.py`, and `--resume` from the existing `*_ckpt_t1.pt` all stay valid. Resuming from a task-1 checkpoint therefore gives a clean A/B against the logged runs.

## 1. Ranked plan (by expected Table-2 gain per CPU-hour)

| # | Action | Findings | CPU-h | Expected effect (estimate) | Priority |
|---|---|---|---|---|---|
| A0 | REPRODUCTION.md corrections (section 5) | K05 K07 K08 K11 K16 K22, plus the doc parts of K01-K03, K06 and the rest | 0 | Changes no numbers. Corrects how the FT-E/iCaRL/Joint gaps are explained. | P0 |
| A1 | One patch adding 4 flags (section 2) | K01 K02 K12 K15 K17 | 0 | Enables A2-A11 | P0 |
| A2 | CIFAR-100 task-1 probe `--lat-real-first 1` | K06 | 1 | Task 1 from 63.2 up to about 70-77. Worth +1-5 on the final number if it carries over. | P0 (gate) |
| A3 | CIFAR-100 AHR with the round-trip fix, resumed from the existing task-1 checkpoint, review gates at t3/t5 | K02 K01 K03 K17 | 5.5 to t3; about 40 to t10 (about 27 h wall on 2 threads) | **+10-20** (15.3 to about 25-35). This is the largest single gap. | P0 |
| A4 | MNIST `--ahr-epoch union`, 3 seeds | K12 | 0.5 | MNIST AHR ±1-2 (94.6), sign unknown. If it does not hurt, it cuts CIFAR-10/SVHN AHR steps 2.6x. | P1 (gate) |
| A5 | MNIST recon-domain sandbox (`single` vs `roundtrip`), 3 seeds each | K01 K02 | 2.5 | No headline cell changes. Separates the causes cheaply: 91.45 now vs 94.57 in the input domain. | P1 |
| A6 | MNIST FT-E/iCaRL budget sweep | K05 | 0.8 | No headline change. Shows which budget the paper's 92.2/93.1 imply (expected about 1000-2000). | P1 |
| A7 | pathgap2 probe on `results/svhn/ahr_s0_split_ckpt_t3.pt` | K02 | 0.005 | Tells whether K02 matters on SVHN before spending 28 CPU-h | P1 |
| A8 | CIFAR-10 AHR with the fix, resumed from `results/cifar10/ahr_s0_ckpt_t1.pt` | K02 K03 K01 K17 | about 28 (about 11 with union epochs) | +3-10 (55.8 to about 59-66) | P1 |
| A9 | SVHN AHR, split latent plus the fix, resumed from `results/svhn/ahr_s0_split_ckpt_t1.pt` | K10 K03 K02 K17 | about 28 (about 12 with union epochs) | +3-12 (74.4 to about 78-86) | P1 |
| A10 | CIFAR-100 incremental Joint (`ft_e --n-exemplars 50000`) | K09 K08 | 11 | Joint +1-5 (59.4 to about 60-64). The ResNet-32 ceiling is about 70 (K07). | P2 |
| A11 | iCaRL with FACIL's BCE distillation: MNIST, then CIFAR-100 | K15 | 0.1 + 4.2 | CIFAR-100 iCaRL ±0-3, sign unknown. About 0 on the 5/2 benchmarks. | P2 |
| A12 | Seeds 1 and 2 of whichever fixed configurations are adopted | none | about 190 | Gives SEMs like the paper's; the means do not change | P3 |
| — | Not recommended: memorisation length (K22, ≤1 dB); RFA confinement (K16, the lossless control shows the inflation is not what limits accuracy); SGD recipe (K08 verdict says the repo's recipe is the paper's); CIFAR-10 incremental Joint (about 8 CPU-h for about ±1) | K22 K16 K08 K09 | — | — | skip |

## 2. Code changes (A1): one patch, all flags default to current behaviour

### C1. `--recon-latent {split,single,roundtrip}` (K01, K02; K03's latent-loss asymmetry is removed as a side effect)

In `scripts/run.py`, after line 142 (`--recon-new-source`):
```python
ap.add_argument("--recon-latent", default="split", choices=["split", "single", "roundtrip"],
    help="latent-domain=recon only. split: separate means over decoded replays and new-sample "
         "reconstructions (current). single: one per-sample mean over both (K01). roundtrip: one "
         "mean over current-AE reconstructions of EVERY sample, decoded replays included (K02)")
ap.add_argument("--herd-space", default="encoder", choices=["encoder", "class"])   # C2
ap.add_argument("--ahr-epoch", default="new", choices=["new", "union"])            # C3
```
At the end of `parse_args` (after line 180), add a guard:
```python
assert args.recon_latent == "split" or (args.latent_domain == "recon" and args.lam_recon_new > 0)
```
Without it, `roundtrip` with `lam_recon_new=0` would silently drop every latent loss on replays.

In `ahr/strategy.py`, `AHR.train_task`:
- **Lines 172-178 (reconstruction source):**
  ```python
  src = x if a.recon_latent == "roundtrip" else x[:n]
  if a.recon_new_source == "current":
      model.eval(); x_rn = model(src)[1].float(); model.train()
  else:
      x_rn = old(src)[1].float()
  ```
  `x` has already been augmented at line 163, so each replay goes through aug(psi_old(m)) and then the current AE. That is the same path as the new samples and the test rule (strategy.py:81-88).
- **Lines 190-193:** keep the existing `l_lat` only when `a.recon_latent == "split"`. Otherwise set `l_lat = z.new_zeros(())`, because the term is folded into the single mean below.
- **Lines 219-223:**
  ```python
  if use_rn:
      z_rn = z_all[len(x):].float()
      if a.latent_domain == "recon" and a.recon_latent != "split":
          if a.recon_latent == "single":
              f, tgt = torch.cat([C(z[n:]), C(z_rn)]), torch.cat([y[n:], y[:n]])
          else:  # roundtrip
              f, tgt = C(z_rn), y
          l_rn = _sq(f, cces[tgt]).mean()
          # same total latent weight as "split" (K01 verdict: do not halve lambda)
          w = a.lam * (a.lam_recon_new + (1.0 if len(x) > n else 0.0))
      else:
          l_rn = _sq(C(z_rn), cces[y[:n]]).mean()
          w = a.lam * a.lam_recon_new
      loss = loss + w * l_rn
      tot["rn"] = tot.get("rn", 0.0) + l_rn.item()
  ```
- **Task 1:** there `len(x) == n`, so all three modes compute the same loss, which is why resuming from the existing task-1 checkpoints is a clean comparison.
- **Cost:** about 1.35-1.5x per step on late tasks (one extra no-grad AE pass over B samples, and the encoder runs forward and backward on 2B samples instead of B+n). Read the real overhead from the first epoch's `[Ns]` stamps.

### C2. `--herd-space {encoder,class}` (K17)

`ahr/strategy.py`, in `populate_memory_frozen` at line 291:
```python
zc = self.class_features(task.x_train) if a.herd_space == "class" else self.model.cls(z)
```
The memory still stores `z = phi(x)` (line 293). This costs one extra triple pass over the task's training set, about 1 min.

### C3. `--ahr-epoch {new,union}` (K12)

`ahr/strategy.py`, `train_task`, line 133:
```python
iters = math.ceil(n_new / b_new)
if replay and a.ahr_epoch == "union":      # one epoch = |D_l U M| / B steps (Alg. 3, tex:266)
    iters = math.ceil((n_new + len(self.memory)) / a.batch_size)
```
At line 140, after `perm = torch.randperm(n_new)`:
```python
if a.ahr_epoch == "union" and iters * b_new > n_new:   # e.g. CIFAR-100 task 2 needs 5,120 > 5,000
    perm = torch.cat([perm] + [torch.randperm(n_new)
                               for _ in range(math.ceil(iters * b_new / n_new) - 1)])
```
The wrap is guarded so the default mode's batches and RNG use do not change.

### C4. `--kd-form {kl,bce}` (K15)

In `scripts/run.py`, after line 170, add `ap.add_argument("--kd-form", default="kl", choices=["kl", "bce"])`.

In `ahr/baselines.py:84-89`:
```python
if a.kd_form == "bce":   # FACIL src/approach/icarl.py criterion: sum over old classes of per-class BCE
    kd = F.binary_cross_entropy_with_logits(out[:, :n_old], torch.sigmoid(out_old.float()),
                                            reduction="none").mean(0).sum()
else:
    T = a.kd_temperature
    kd = F.kl_div(...) * T * T      # unchanged
```
Also fix the docstring at `baselines.py:5-7`. Today it says "as in FACIL". It should say "LwF-style softmax KD, T=2; FACIL's iCaRL uses per-class sigmoid BCE (`--kd-form bce`)".

**No code change is needed** for K03 (`--pad-mode reflect`), K06 (`--lat-real-first 1`), K09 (`ft_e --n-exemplars 50000`) or K10 (`--latent-kind split --lam 1 --rfa-jitter 0.25`).

## 3. Actions

All commands run from `/home/user/autoencoder-hybrid-replay-cil`. Launch them with `nohup ... > results/<ds>/<name>.stdout 2>&1 &`, as the existing runs were.

### A2. CIFAR-100 task-1 probe (K06). 1 CPU-h. P0 gate.
```
python scripts/run.py --dataset cifar100 --method ahr --seed 0 --lat-real-first 1 --pad-mode reflect \
  --stop-after 1 --save-ckpt 1 --threads 1 --tag lrf
```
- **Why:** the controlled pair shows that recon-only training costs 14 points on task 1 (63.2 vs 77.5). The same checkpoint scores 63.7 with phi(x), so the loss is in the encoder, not in the test rule. With the spatial latent at λ=0.3, `lat-real-first` raised task 1 from 12.6 to 17.8 (`results/scratch/c100/cifar100/ahr_s0_j25*.json` vs `ahr_s0_j25_real1.json`). It has never been tried with the split latent.
- **Compare with:** 63.2 (`results/scratch/c100/cifar100/ahr_s0_split_l1.log:13`).
- **Gate:** if task 1 is ≥ 67, start A3' (below) from `results/cifar100/ahr_s0_lrf_ckpt_t1.pt`.
- **Caveat:** task 2 in the real run reaches only about 57% on its own replays, so a better task 1 alone is likely worth only +1-5 at t10 (K06 verdict).

### A3. CIFAR-100 AHR with the round-trip fix (K02 + K01 + K03 + K17). P0.
```
python scripts/run.py --dataset cifar100 --method ahr --seed 0 \
  --resume results/scratch/c100/cifar100/ahr_s0_split_l1_ckpt_t1.pt \
  --recon-latent roundtrip --herd-space class --pad-mode reflect --save-ckpt 1 --threads 1 --tag rt
```
- **Setup:** `split_l1` is exactly the first two tasks of the reported run, with the same defaults (split, λ 1, jitter 0.25, recon, lam_recon_new 1). Task 1 is identical under C1, so this is a clean A/B.
- **Gate at t3 (about 5.5 h):**
  - Compare with 47.45 [49.9, 45.0] after task 2 and 38.63 [32.0, 29.2, 54.7] after task 3.
  - Compare acc(memory) with 57.0 / 49.8.
  - Run `python <scratchpad>/chk/pathgap2.py cifar100 10 3 results/cifar100/ahr_s0_rt_ckpt_t3.pt split` (13 s). Replay-path and test-path accuracy should be equal and the newest-task share should fall. On the old run at t3 they were about 36 vs 33.
- **Gate at t5 (about 14 h):** compare with 27.68 [18.3, 13.1, 25.6, 20.2, 61.2]. Abort if it is below 30 with no fall in the newest-task share. The cue grows with task age (t6 replay/test 30.5 vs 16.2; t10 39 vs 12), so t5 is the first reliable read.
- **After the t3 gate:** raise to `--threads 2` by resuming in place from `results/cifar100/ahr_s0_rt_ckpt_t<k>.pt` with the same `--tag rt`, which keeps the log. The JSON `accs` will then lack task 1 (63.2); `final_acc` is unaffected.
- **Expected:** +8-15 from K02 (the replay-path ceiling is about 40%), +1-6 from K01 and +3-10 from K03. These overlap, so the combined estimate is **+10-20 (15.3 to about 25-35)**. It will not reach 54.4, because the recon-domain classifier fits only about 40% of its own replays (K06).
- **Cost:** tasks 4-10 took 19.3 h on 2 threads. With the ×1.4 overhead that is about 27 h wall on 2 threads, or about 40 CPU-h in total.
- **A3' (only if A2 passes):** run the same command with `--resume results/cifar100/ahr_s0_lrf_ckpt_t1.pt --tag lrf_rt` on another core. Keep whichever run is ahead at t3/t5 and kill the other.

### A4. MNIST union epochs (K12). 0.5 CPU-h. P1 gate.
```
for s in 0 1 2; do python scripts/run.py --dataset mnist --method ahr --seed $s --ahr-epoch union --threads 1 --tag union; done
```
- **Compare with:** 95.18 / 93.35 / 95.19 (mean 94.57).
- **Why:** the paper claims fixed compute (tex:123, tex:410), but the repo gives AHR 2.9-4x the baselines' optimiser steps. Section 3.10 saw more forgetting with more steps. The sign of the effect is unknown.
- **Gate:** if the mean is ≥ 94.1, A8/A9 may use `--ahr-epoch union` (about 11-12 CPU-h each instead of about 28). Report steps and wall-clock next to accuracy. Otherwise keep `new`.

### A5. MNIST recon-domain sandbox (K01, K02). 2.5 CPU-h. P1.
```
for s in 0 1 2; do
  python scripts/run.py --dataset mnist --method ahr --seed $s --latent-domain recon --lam-recon-new 1 --recon-latent single    --threads 1 --tag recon_single
  python scripts/run.py --dataset mnist --method ahr --seed $s --latent-domain recon --lam-recon-new 1 --recon-latent roundtrip --threads 1 --tag recon_rt
done
```
- **Compare with:** `recon_domain` 90.41 / 92.22 / 91.73 (mean 91.45) and input-domain AHR 94.57.
- **Reading:** if `recon_rt` gets close to 94.6, the path asymmetry is the cost of the recon domain. MNIST has no augmentation, so K03 plays no part here.
- **Headline:** the reported MNIST configuration stays in the input domain.

### A6. MNIST budget sweep (K05). 0.8 CPU-h. P1.
```
for N in 500 1000 2000; do for m in ft_e icarl; do for s in 0 1 2; do
  python scripts/run.py --dataset mnist --method $m --seed $s --n-exemplars $N --threads 1 --tag bud$N; done; done; done
```
- **Expected, from van de Ven 2019 (same 2x400 MLP, Adam):** iCaRL about 93 and replay about 92 at about 1000. The paper's 93.06/92.17 would then correspond to 5-10x the stated 200.
- **Reporting:** a supplementary column only. Do not replace the 200-exemplar headline, and do not tune the baselines toward Table 2.

### A7. SVHN K02 probe. Seconds. P1.
```
OMP_NUM_THREADS=1 python <scratchpad>/chk/pathgap2.py svhn 2 3 results/svhn/ahr_s0_split_ckpt_t3.pt split
```
If the replay-path vs test-path gap is small, as on CIFAR-10 (99 vs 93), A9 can run with only `--pad-mode reflect` plus split, and skip `roundtrip`'s ×1.4 cost.

### A8. CIFAR-10 AHR with the fix (K02, K03, K01, K17). About 28 CPU-h. P1.
```
python scripts/run.py --dataset cifar10 --method ahr --seed 0 --resume results/cifar10/ahr_s0_ckpt_t1.pt \
  --recon-latent roundtrip --herd-space class --pad-mode reflect --save-ckpt 1 --threads 1 --tag rt
```
Add `--ahr-epoch union` only if A4 passes.
- **Compare with:** the s0 trajectory 77.0 [79.4, 74.7], 61.78, 59.18, 55.79 [59.6, 33.6, 44.7, 68.7, 72.4]. Also compare with the cloud `reflect` run, which isolates K03.
- **Expected:**
  - K03 is the main lever: the newest-task deficit against lossless is -22.6, and the border cue's per-task profile matches it.
  - K02 adds about 0-5 and K01 about ±2.
  - Combined: **+3-10 (to about 59-66)**; paper 77.1.
- **Seed noise:** about 3 points between seeds after task 2 (77.0 vs 80.2), so judge on the final number, not on task 2.

### A9. SVHN AHR: split latent plus the fix (K10, K03, K02, K17). About 28 CPU-h. P1.
```
python scripts/run.py --dataset svhn --method ahr --seed 0 --resume results/svhn/ahr_s0_split_ckpt_t1.pt \
  --latent-kind split --lam 1 --rfa-jitter 0.25 --recon-latent roundtrip --herd-space class \
  --pad-mode reflect --save-ckpt 1 --threads 1 --tag split_rt
```
- **Setup:** this puts SVHN on the same configuration as CIFAR, which the paper also shares across them (Table 4).
- **Compare with:** 74.43 (spatial), the running split zero-pad run (93.01 after task 2, 87.83 after task 3), and the cloud `split_reflect` run.
- **Expected:** K10 +0-5 (the lead fell from +6.7 to +3.7 after task 3), K03 +2-8, K02 unknown (see A7). Combined: **+3-12 (to about 78-86)**; paper 93.0.

### A10. CIFAR-100 incremental Joint (K09). 11 CPU-h. P2.
```
python scripts/run.py --dataset cifar100 --method ft_e --seed 0 --n-exemplars 50000 --threads 1 --tag incjoint
```
- **What it runs:** `memory.py:108-113` keeps every sample when per_class ≥ 500. So this is FACIL-style warm-started Joint on all tasks seen so far, which is what the paper's citation implies (tex:332). It takes 5.5x the steps of the one-shot run, whose training CE is still 0.98.
- **Expected:** +1-5 (to about 60-64). About 3 points of the 14.4-point gap are unreachable with a 0.47M ResNet-32 (K07).
- **Reporting:** report it as a tagged Joint variant.

### A11. iCaRL with BCE distillation (K15). P2.
```
for s in 0 1 2; do python scripts/run.py --dataset mnist --method icarl --seed $s --kd-form bce --threads 1 --tag bce; done   # 0.1 h
python scripts/run.py --dataset cifar100 --method icarl --seed 0 --kd-form bce --threads 1 --tag bce              # 4.2 h
```
- **Why:** at 90 old classes FACIL's term has 4-17x the gradient of the repo's KL, while at 2-8 old classes the two are within about 0.6-1.7x.
- **Expected:** CIFAR-100 ±0-3 (38.0 now, paper 49.4); MNIST about 0 (88.6).
- **Weight decay:** leave it at 0. That is FACIL's effective CLI default too, so the proposed 1e-5 has no basis.

## 4. Schedule on the 4 cores (wall-clock from now, approximate)

| Time | Core A | Core B | Core C | Core D |
|---|---|---|---|---|
| 0 h | apply the C1-C4 patch; if the user agrees, stop PID 7676; then A2 (1 h) → A3 | svhn lossless (running) | c100 lossless (running) | svhn split (running) |
| about 4.5 h | A3 continues | A7 (seconds), A4 (0.5 h), A5 (2.5 h), A6 (0.8 h), A11-MNIST (0.1 h) | ″ | ″ |
| about 5.5 h | **A3 t3 gate**. If A2 passed: A3' starts on the first free core | ″ | ″ | ″ |
| about 7 h | A3 | ″ | A10 (11 h) | ″ |
| about 8.5 h | A3 | A8 (with union epochs if A4 passes) | A10 | ″ |
| about 12 h | A3 | A8 | A10 | A9 |
| about 14 h | **A3 t5 gate**; kill the losing run between A3 and A3' | A8 | A10 | A9 |
| about 18 h | A3 on `--threads 2` (resume in place) | A8 | A11-CIFAR-100 (4 h), then idle or seeds | A9 |
| about 40-45 h | all P0-P2 runs done. Total about 120 CPU-h (about 80 if union epochs are adopted for A8/A9) | | | |

If cloud sessions are still available, A8 and A9 are the best candidates to offload through `scripts/cloud_run.py`. A3 should stay local because its gates need watching.

## 5. Documentation-only findings, to record in REPRODUCTION.md

**Documentation-only (no run is recommended):**
- **K05 (Summary lines 106-108).** State that at the stated 200/2000 budgets, published FT-E/iCaRL values match the repo, not the paper:
  - DER S-CIFAR-10 at 200: ER 44.79, iCaRL 49.02, against the repo's 44.0 / 62.8.
  - van de Ven split-MNIST at 200: replay about 80, iCaRL about 89.
  - GDumb at 200/1000 on CIFAR-10: 35.0 / 61.3, against the paper's 72.79.
  - The paper's Fig. 1 claims about 73% at 16 exemplars per class, so this is a paper-vs-literature mismatch, not a hidden larger budget.
  - Exception: CIFAR-100, where FT-E (27.1) and Joint (59.4) sit 11 and 7 points below FACIL at the same budget. That part is a repo recipe gap.
- **K07.** The Joint row is close to PEC's ResNet-18 "Joint, 50 epochs" row. Only SVHN 95.88 is identical. CIFAR-100 73.87 is about 3 points above what a typical ResNet-32 reaches. Correct Summary line 97 ("FT and Joint match within 1-3 points on every benchmark"): that is false for CIFAR-100 (14.4) and CIFAR-10 (3.4).
- **K08.** The Adam / 1e-3 / 50 epochs / batch 256 recipe is what the paper states for all 19 strategies (tex:347, tex:504, Table 4). Weight decay 0 and the cosine schedule are the repo's own choices where the paper is silent. Only CIFAR-100 Joint is under-trained. FT-E and iCaRL on tasks 2-10 are not, since their gaps match those on the fully converged benchmarks.
- **K11 (lines 121-124 and 343).** AHR and AHR-lossless differ in 5 effective settings on CIFAR-10 (domain, lam_recon_new, latent kind, λ, jitter) and in 2 on SVHN and CIFAR-100. The gap measures the recon-domain workaround, not replay fidelity: replays are classified 99-100% correctly at about 25 dB, and the oldest task ends the same (59.6 vs 59.4). Label the lossless rows "input-domain rule, raw exemplars".
- **K16 (line 126 and section 3.12, lines 359-363).** "Fixed by jittering" holds only in simulation. In the real CIFAR-100 run the per-task |p| grows from 14.8 to 61.6 and the mean CCE distance reaches 50.9. It is not the main cause: lossless has the same inflation and still reaches 53.5 after task 8, and AHR is already 14 points behind on task 1.
- **K22.** Memorisation over a fixed 1500 steps gives CIFAR-100 5x fewer passes per exemplar. The measured benefit is +0.1 dB (newest task) vs +1.6 dB on CIFAR-10, and it decays to about 0 on both within 2-4 tasks. Not worth compute.

**Documentation plus the actions above:**
- **K01, K02 and K06.** Add a new section 3.15 on the train/test path asymmetry in the recon domain, and a section on the controlled task-1 deficit, with the probe numbers from the verdicts.
- **K03 (section 3.14).**
  - Add the SVHN (t5) and CIFAR-100 (t10) border measurements.
  - State that `--pad-mode` still defaults to `constant` (run.py:155).
  - State which runs are zero-padded (all reported AHR runs, cifar10 s1/s2 and svhn split s0/s1/s2), so they are not the final configuration.
- **Section 2 table rows to add:**
  - K01: per-group latent means in recon mode.
  - K09: one-shot vs incremental Joint.
  - K10: SVHN uses the spatial latent while CIFAR uses split, although the paper gives SVHN and CIFAR-10 the same configuration (Table 4).
  - K12: iterations per epoch and the resulting 2.9-4x steps and 5-7x wall-clock, which conflicts with the fixed-compute claim.
  - K15: the iCaRL KD form, and correct the docstring.
  - K17: herding in cls(phi(x)), not in the classification space.

## 6. Expected end state vs Table 2 (estimates)

| Method | MNIST ours → plan / paper | SVHN | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| FT | 19.8 / 19.9 | 19.6 / 19.2 | 19.6 / 18.7 | 8.9 / 8.9 |
| FT-E | 72.2 (unchanged; documented, K05) / 92.2 | 55.6 / 87.1 | 44.0 / 72.2 | 27.1 / 48.5 |
| Joint | 98.5 / 98.5 | 95.4 / 95.9 | 89.0 / 92.4 | 59.4 → 60-64 (A10) / 73.9 |
| iCaRL | 88.6 / 93.1 | 71.1 / 89.6 | 62.8 / 73.3 | 38.0 ± 3 (A11) / 49.4 |
| AHR | 94.6 ± 1-2 (A4) / 97.5 | 74.4 → 78-86 (A9) / 93.0 | 55.8 → 59-66 (A8) / 77.1 | 15.3 → 25-35 (A3) / 54.4 |
| AHR-lossless | 95.1 / 98.1 | running / 94.2 | 68.8 / 78.4 | running (53.5 after task 8) / 56.7 |

**Gaps this plan cannot close:**
- **FT-E and iCaRL at the stated budgets.** Code cannot fix a mismatch between the paper and the published literature (K05).
- **About 3 points of CIFAR-100 Joint.** This is the ResNet-32 capacity limit (K07).
- **MNIST AHR-lossy-mini and AHR-lossless-mini (67-69 vs 93.4/93.8).** No verified finding explains them.
- **miniImageNet.** It is not run on this hardware.
- **Most of the remaining CIFAR-100 AHR gap.** The recon-domain classifier fits only about 40% of its own decoded replays (K06).