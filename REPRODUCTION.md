# Reproducing "Autoencoder-Based Hybrid Replay for Class-Incremental Learning" (ICML 2025)

This document describes the implementation in this repository, every detail that
had to be chosen because the paper does not specify it, what was run, and how the
numbers compare with Table 2 of the paper
([arXiv:2505.05926](https://arxiv.org/abs/2505.05926)).

<!-- RESULTS -->

## 1. What is implemented

| Paper | Code |
|---|---|
| HAE: encoder `phi: R^n -> R^m`, decoder `psi: R^m -> R^n` | `ahr/models.py` (`make_hae`) |
| Eq. 1: `||x - x_hat||^2 + lambda ||z - p_y||^2` | `ahr/strategy.py` (`AHR.train_task`) |
| Alg. 1 (AHR loop: placement -> training -> memory -> delete old model) | `AHR.learn_task` |
| Alg. 2 (CCE placement with the repulsive force algorithm, CPSEM) | `ahr/rfa.py` (`place_cces`) |
| Alg. 3 (HAE training on `D_l U psi_old(M)` with encoder and decoder distillation, `1/l` new vs `(l-1)/l` decoded samples per minibatch) | `AHR.train_task` |
| Alg. 4 (memory population, `M / #classes` exemplars per class, stored as latent codes) | `AHR.populate_memory*` |
| Inference `argmin_{ij} ||phi(x) - p_i^j||` | `AHR.predict` |
| Benchmarks MNIST(5/2), Balanced SVHN(5/2), CIFAR-10(5/2), CIFAR-100(10/10) | `ahr/data.py` |
| Dense 400-400 network + mirror decoder (MNIST); ResNet-32 + 3-layer CNN decoder (others) | `ahr/models.py` |
| Adam, lr 1e-3, momentum 0.9, epochs 40/50/50/50, batch 128/128/128/256, latent 20/307/307/307, 200/200/200/2000 raw exemplars (Table 4) | `scripts/run.py` (`PRESETS`) |
| Fixed exemplar memory; AHR stores `budget x input size / latent size` codes (7,840 / 2,001 / 2,001 / 20,013) | `scripts/run.py` (`exemplar_count`) |
| Ablations AHR-lossless, AHR-lossy-mini, AHR-lossless-mini | `--method ahr_lossless / ahr_lossy_mini / ahr_lossless_mini` |
| Baselines FT, FT-E, iCaRL, Joint | `ahr/baselines.py` |

The remaining baselines of Table 2 (SLDA, Gen-C, PEC, DGR, MeRGAN, BI-R-SI, GD,
GDumb, EEIL, BiC, LUCIR, IL2M, i-CTRL, REMIND, REMIND+) are not re-implemented;
their paper numbers are quoted for reference only.

## 2. Details the paper does not specify, and what was chosen

| Item | Choice | Why |
|---|---|---|
| `lambda` (Eq. 1) | 0.3 (MNIST), 1.0 (SVHN / CIFAR) | MNIST sweep over {0.1, 0.3, 1, 10, 100}; 0.3 best (93.0 vs 91.8 for 1.0) |
| Distillation weights `a_z`, `a_x` and form | `a_z = 0.01`, `a_x = 1`, squared L2 | Strong encoder distillation (`a_z >= 1`) keeps new-task samples away from their shifted CCEs (see §3.4) |
| RFA constants `zeta, m, dt` | 1, 1, 0.01, softening 1e-3, no damping | Scale-free once the duration is chosen as below |
| RFA duration `tau` | integrate Alg. 2 until every new CCE is `>= d` from every other CCE; `d = 5` (m = 20) and `d = 20` (m = 307), i.e. `d ~ sqrt(m)` | With a fixed `tau` the spacing depends on how close the initial class means are, which differs by an order of magnitude between the first task (random encoder) and later ones (§3.3) |
| Encoder latent head | ResNet-32 feature map pooled to 64x4x4, linear map to the 307-d latent; head initialised with std 1e-3 | Global pooling would cap the latent at 64 informative dimensions; the small init keeps the latent scale set by RFA rather than by the backbone's activation scale (~500) |
| Decoder (CIFAR / SVHN) | Linear(307 -> 192x4x4), ConvT 192->128->64->3 (k4 s2), ReLU, sigmoid; 1.47M params | "3 layers of CNNs", ~1.4M params (Table 3) |
| Minibatch composition | `round(B/l)` new samples + `B - round(B/l)` exemplars sampled uniformly from the class-balanced memory; one epoch = one pass over the new task's data | Sec. 2 ("1/l fraction ... (l-1)/l fraction"); gives the O(t) compute of Table 1 |
| Exemplar selection | herding in latent space | The text says Alg. 4 is "based on Herding as in iCaRL"; the literal `Rank` of `L_z` (smallest distance to the CCE first) was also implemented, see §3.1 |
| Learning-rate schedule | cosine annealing within each task (all methods) | not specified |
| Augmentation | random crop (pad 4) for SVHN/CIFAR, + horizontal flip for CIFAR; none for MNIST | standard |
| Class order | natural (0,1 / 2,3 / ...) | not specified |
| Precision | bf16 autocast for convolutions on CPU (AMX); latent maps in fp32 | compute budget |

## 3. Findings while implementing (what did *not* work as described)

### 3.1 `Rank`-based exemplar selection hurts
Algorithm 4 ranks samples by `L_z = ||phi(x) - p_y||^2` ascending and keeps the
first ones, i.e. the samples *closest to their class centroid*. These are the least
diverse samples of each class. With perfect (raw) exemplars on MNIST(5/2)
(10 epochs), `Rank` selection gives 77.1% final accuracy vs 96.3% with random
selection. Herding (which the paper's text names) is used instead.

### 3.2 Decoded exemplars create a "looks-decoded => old task" shortcut
Implemented literally (memory re-encoded every task, replay = decoded exemplars,
test on raw images), AHR on MNIST(5/2) at the paper's 40 epochs reaches only
71.8%. The encoder classifies the decoded exemplars almost perfectly (98-99%)
but forgets the *real* images of old classes: new classes are only ever seen as
sharp images and old classes only as decoded (blurry) ones, so blur becomes a
task cue. Evidence: classifying the AE reconstruction of each test image instead
of the image itself gives higher accuracy on old tasks (e.g. 93.5% vs 87.0% after
task 4). Presenting the new samples to the latent loss also through the HAE's
own reconstruction (`--lam-recon-new 1`) removes most of this shortcut
(75.5% -> 85.6% with frozen codes).

### 3.3 RFA needs a stopping criterion
With fixed `zeta, m, dt, tau`, first-task CCEs (class means of a *random* encoder,
~0.1 apart) are blown ~13 units apart while later CCEs (class means of a trained
encoder, several units from the old CCEs) move only ~1.5 units, so later classes
end up 1.6 units apart and get confused. Stopping the simulation once a target
separation `d` is reached keeps the spacing uniform over tasks.

### 3.4 Encoder distillation vs. CCE shift
The encoder-distillation term pulls new-class samples towards where the old
encoder put them, while `L_z` pulls them to their new CCEs, which RFA has moved
away by roughly `d`. For squared distillation the equilibrium sits a fraction
`a_z / (lambda + a_z)` of the way back, so `a_z` must be small relative to
`lambda`; for the (non-squared) norm the lag is `a_z / (2 lambda)` and must stay
well below `d / 2`. With `a_z = 1`, `lambda = 1`, `d = 10` on MNIST (10 epochs)
the second task is barely learned (27% on its classes, 99% on the first task).

### 3.5 Decoder memorisation
The paper motivates AHR by the decoder *memorising* the stored exemplars. Trained
only through Alg. 3, the mirror MLP decoder on MNIST reaches ~23 dB PSNR on the
first task's exemplars and ~20 dB averaged over all digits (a plain AE with the
same architecture trained 200 epochs on 8,000 digits reaches 25 dB train /
20 dB test). Storing codes once (`--memory-mode frozen`, Alg. 4's `phi(w_i, D)`),
applying the decoder distillation directly on the stored codes
(`||psi(m) - psi_old(m)||^2`) and a short decoder-only memorisation pass on the
new exemplars (`--memorize-epochs 20`) raises MNIST accuracy from 85.6% to 91.8%
(with `lambda = 1`) and to 93.7% (with `lambda = 0.3`).

