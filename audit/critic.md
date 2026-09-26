# Completeness critique of the AHR action plan

I only read files and ran short probes. Nothing in the repo was changed. One scratch probe exists: `/tmp/claude-0/-home-user-autoencoder-hybrid-replay-cil/ec600478-b228-5e5a-8217-aae3ac14b66f/scratchpad/ncm_t10.py`.

## A. Paper details nobody checked

**A1. Table 2 says FT-E uses implicit bias correction. The repo's FT-E has none.** This is the largest overlooked item.
- **Paper:**
  - tex:360 reads `FT-E & I & N`, where "I" is the bias-correction column.
  - tex:450 defines it: "Implicit bias-correction, as seen in \citep{Castro} [EEIL], relies on data equalization".
- **Repo:**
  - The headline FT-E trains on the plain shuffled union of new data and exemplars (`ahr/baselines.py:52-56`).
  - `--balanced-ft-epochs` defaults to 0 (`scripts/run.py:162`).
  - REPRODUCTION.md never mentions bias correction.
- **Evidence the effect is large:** `results/mnist/ft_e_s{0,1,2}_bft30.json` (EEIL-style balanced fine-tuning) score 84.21 / 85.12 / 84.86. The mean is 84.7, against 72.18 for the headline FT-E.
- **What this changes in the plan:**
  - It closes 12.5 of the 20-point MNIST FT-E gap.
  - The plan's end state still lists FT-E as "unchanged (K05)".
  - K05 compares FT-E to DER's ER at 200 exemplars. DER's ER has no bias correction, so the comparison overstates the "paper vs literature" mismatch.
- **Action:**
  - Report FT-E with bias correction (`--balanced-ft-epochs 30`) as the paper-faithful FT-E row.
  - Pull the SVHN, CIFAR-10 and CIFAR-100 bft30 cloud runs into section 6.
- **Affects:** the FT-E row. MNIST is measured at +12.5. The other benchmarks are unknown until the cloud runs finish.

**A2. The paper claims a "matched parameter count" (tex:410). Nobody has addressed it.**
- **Parameter counts:**
  - AHR in the repo has 470,676 encoder and 1,210,883 decoder parameters, about 1.68M (`results/cifar10/ahr_s0.json`).
  - The paper's AHR would be 0.47M + 1.4M (Table 3, tex:433).
  - The baselines and Joint use a 0.47M ResNet-32 (`baselines.py:30`).
- **Two readings, and the paper is silent on which holds:**
  - Either the paper's claim is false.
  - Or the baselines were enlarged to about 1.9M parameters, for example a ResNet-32 at twice the width (about 1.85M). tex:347 says only "ResNet-32 is used".
- **Why it matters:** under the matched reading, K07's "about 3 points unreachable with a 0.47M ResNet-32" and part of the CIFAR-100 FT-E/iCaRL "recipe gap" become protocol gaps instead.
- **Test:** CIFAR-100 Joint with a 2× wide ResNet-32. This needs a new width flag and costs about 4× the FLOPs, roughly 8 CPU-h.
- **Estimate:** Joint CIFAR-100 +3-8. This is unverified.

**A3. Table 3 (tex:430-442) contradicts K08, and changes C3.**
- **Epochs:**
  - Table 3 gives BiC, IL2M and EEIL 60 epochs on CIFAR-100 and 80 on miniImageNet. AHR gets 50.
  - Table 4 is captioned "…for our AHR strategy" (tex:509). Its values are epochs 40/50/50/50/70 and batch 128/128/128/256/256 (tex:522-523).
  - So K08's "Adam/1e-3/50 epochs/batch 256 for all 19 strategies" is wrong.
  - BiC also scores 52.12 in Table 3 but 51.41 in Table 2, so Table 2's baselines come from a different configuration.
  - Estimated effect: +0-2 on the baselines. This is mainly a documentation fix.
- **Wall-clock:**
  - The paper's AHR takes 462 min, against 455-478 min for the 60-epoch baselines.
  - The repo's AHR takes 1242 min, against 249 min for iCaRL.
  - Proposed C3 `union` still gives AHR about (5,000 + 19,200)/256 = 95 iterations per epoch on CIFAR-100. BiC gets about 28, so union is still about 3.4× BiC.
  - Table 3 therefore implies something closer to `iters = ceil(n_new / B)`. Add that as a third C3 mode, and test it on a CIFAR benchmark rather than gating it only on MNIST (A4). MNIST has no augmentation and classifies in the input domain, so its result may not transfer.
- **Memory accounting:**
  - Table 3's 150 latent codes per class (4.6M) plus the 1.4M decoder equals 6.0M, which is 2000 × 3072. So the paper charges the decoder to the budget.
  - Table 4 contradicts this with 20,000 latent codes (tex:535).
  - The repo does not charge the decoder (`run.py:95`, default 0) and stores codes as float32 (`run.py:97`). AHR's memory is 2,457,600 bytes against iCaRL's 614,400 on CIFAR-10, i.e. 4×.
  - On the 5/2 benchmarks the decoder alone (1.21M parameters) is about twice the entire 614,400-scalar budget. The "same memory budget" claims (tex:123, tex:410) therefore cannot hold if the decoder is counted.
  - The repo favours AHR here, so this changes no number, but it must be documented.

**A4. Fig. 3 (do.jpg, tex:336-338; the plan calls it "Fig. 1"), digitised from `scratchpad/do_row1.png`, about ±0.5 points.**
- **Joint:** the CIFAR-10 Joint line sits at about 96.0, while Table 2 says 92.37 (tex:361). The CIFAR-100 and miniImageNet Joint lines match their tables (74.1 and 73.0).
- **At 16 exemplars per class:**
  - CIFAR-10: iCaRL 73.3, FT-E about 72.5, AHR 77.5. These equal Table 2, which supports K05's "no hidden larger budget".
  - CIFAR-100: AHR is 58.5, against 54.43 in Table 2.
- **Consequence:** these two paper cells carry about 4 points of internal inconsistency. Chasing the last points of CIFAR-10 Joint is pointless.

## B. Plan claims that are unverified or wrong

**B1. "Resuming from task-1 checkpoints gives a clean A/B" is false.**
- `run.py:242` reseeds after the model is built.
- Checkpoints store no RNG state (`run.py:284-286`).
- So a resumed task 2 draws different RFA jitter (`strategy.py:113`), batch permutations, memory samples and crops from the logged run.
- A3, A8 and A9 also add `--pad-mode reflect` on top of task-1 checkpoints that were trained with zero padding (`run.py:155`).
- The reported CIFAR-100 run is itself a resume (`results/cifar100/ahr_s0.log:4`).
- The plan's own seed noise is about 3 points after task 2, while A8/A9 expect +3-10. With one seed they cannot be separated.
- Fix: read A8/A9 mainly from the RNG-robust diagnostics (pathgap, acc(memory)), or run a resume with default flags to task 3 as a control.

**B2. A3 changes K02, K01, K03 and K17 together.** The plan's per-cause split ("+8-15 from K02, +1-6 from K01, +3-10 from K03") cannot be read from one run.

**B3. My probe on `results/cifar100/ahr_s0_ckpt_t10.pt` supports A3 and rules out a cheap alternative** (1,500 test images and 1,500 codes; not written to the repo).

| Measurement | Result |
|---|---|
| Replay accuracy, training path phi(psi(m)) | 41.4% |
| Replay accuracy, test path (round trip) | 16.7% |
| Test accuracy, nearest fixed centroid | 15.4 |
| Test accuracy, nearest re-estimated class mean (replay path) | 16.3 |
| Test accuracy, nearest re-estimated class mean (round trip) | 14.9 |

- The first two rows confirm K02 at task 10.
- Re-estimating class means (iCaRL-style nearest mean) instead of the fixed centroids does not recover the old classes; the information is gone from the features.
- So a post-hoc rebalancing or nearest-mean fix of the existing run will not help, and "balanced training (from EEIL)" (tex:320) is low priority for CIFAR-100.

**B4. "Evaluation metric" can be struck from the hypotheses.** tex:341 says "final test accuracy". The paper's FT values (19.93, 8.91) are final accuracies; average incremental accuracy for FT in the repo would be 44.9 and 23.6.

## C. Unexplained gaps and cheap experiments that would separate the explanations

1. **MNIST AHR-lossless 95.1 vs 98.1, and AHR 94.6 vs 97.5.** Decoding costs only 0.5 points and Joint matches.
   - Test: extend A6 with `--n-exemplars 7840` for ft_e and icarl (about 0.3 CPU-h).
   - Reading: at ≥97.5 the deficit is specific to AHR's training (epoch definition, λ, α_z). At about 95 it is shared protocol.
2. **CIFAR-10 AHR-lossless 68.8 vs 78.35.** In Fig. 3, the paper's FT-E at about 192 per class is about 79, i.e. roughly its AHR-lossless number.
   - Test: run `ft_e --n-exemplars 1920`, with and without `--balanced-ft-epochs 30`, on CIFAR-10 (about 3 CPU-h).
   - Reading: at about 70 the lossless gap is the shared baseline gap. At about 79 it is specific to AHR. This sets A8's realistic target before spending about 28 CPU-h.
3. **MNIST AHR-mini 67-69 vs 93.4/93.8.**
   - The per-task pattern (for example [70.8, 49.5, 46.5, 75.5, 98.6]) looks like FT-E without bias correction, while iCaRL with the same 200 exemplars reaches 88.6.
   - Hypothesis (a): α_z = 0.01 was tuned with 7,840 exemplars (`run.py:42`) and is too weak at 200. Test `--alpha-z 0.1` and `--alpha-z 1`.
   - Hypothesis (b): over-exposure of the memory. In task 5 each of the 200 stored exemplars is drawn about 9,300 times (102 per step × 454 steps × 40 epochs / 200), against 40 for union FT-E. Test `--epochs 8`, or C3.
   - Cost: 3 seeds each, 13-20 min per run, about 1.5 CPU-h in total.
4. **CIFAR-100 Joint.**
   - Training CE is still 0.981 at epoch 50 (`results/cifar100/joint_s0.log`), against 0.103 for CIFAR-10 and 0.006 for SVHN.
   - Before A10 (11 CPU-h, which mixes warm-starting with 5.5× the steps), run a one-shot Joint with `--epochs 100` (about 4 CPU-h). That separates "more steps" from "incremental protocol".
   - bf16 *training* was never tested for the baselines. REPRODUCTION.md §3.9 tested bf16 only at inference, on an AHR checkpoint.
5. **Class order.** The repo uses the natural order (`data.py:82`). K05's CIFAR-100 comparison is to FACIL numbers, which use the iCaRL seed-1993 order (`--class-order icarl` already exists). The paper is silent. Expect about ±1-2 on CIFAR-100 for all methods. Flag it in the documentation; no run is needed unless FACIL numbers are quoted as targets.