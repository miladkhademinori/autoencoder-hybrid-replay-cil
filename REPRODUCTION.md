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

### 3.6 CIFAR: the decoded-vs-real shortcut is total
On CIFAR-10(5/2) the decoder's output (17-18 dB PSNR for a 307-d code and the
3-layer decoder) is trivially distinguishable from real images. After task 2 the
encoder maps the *decoded* airplane/automobile exemplars to their CCEs (92% correct)
but maps *real* airplane/automobile test images to the bird/cat CCEs (0-7%
correct). Controls (2 tasks, 40% of the data, 20 epochs):

| Variant | task-1 acc. after task 2 | task-2 acc. |
|---|---|---|
| AHR (decoded exemplars) | 0.0 | 81.7 |
| AHR without the reconstruction term | 1.6 | 84.8 |
| AHR + KD on distances to the old CCEs (weight 1 / 10) | 0.1 / 0.0 | 82.1 / 82.7 |
| AHR-lossless (same method, raw exemplars) | **84.5** | 75.4 |
| AHR, classification only on decoder outputs (`--latent-domain recon`) | 72.2 | 24.3 |

With perfect exemplars the method works; with decoded ones every real image is
"new". Classifying only decoder outputs removes the shortcut but a classifier
trained only on blurry reconstructions learns the new classes poorly.

### 3.7 BatchNorm and separate forward passes
A first version passed the reconstructions of the new samples through the encoder
in a separate forward pass. In train mode that all-reconstruction batch gets its
own BatchNorm statistics, which the network can exploit and which vanish in eval
mode. All samples of a step now share one forward pass (§3.6 numbers are with the
fix).

### 3.8 How far can "memorisation" go?
Fidelity of the 2,000 stored CIFAR-10 exemplars after task 1 (40% data, 20 epochs),
continuing to train on the stored (code, image) pairs only:

| Steps (batch 128) | decoder only | decoder + stored codes |
|---|---|---|
| 0 | 15.1 dB | 15.1 dB |
| 1,500 | 16.6 dB | 20.3 dB |
| 3,000 | 17.1 dB | 22.4 dB |
| 6,000 | 18.2 dB | |

Refining the stored codes together with the decoder (auto-decoder style,
`--memorize-codes 1`) memorises far better than fitting the decoder alone; the
memory footprint is unchanged (the codes are the memory).

### 3.9 The latent must keep spatial layout (CIFAR)
With the latent formed by pooling the ResNet-32 feature map and a linear map to a
307-d vector, decoded CIFAR exemplars are colour blobs (17-18 dB): even after
memorisation they carry little class evidence, so no replay trick can transfer them
to real test images (§3.6). The paper only fixes the latent *size* (307) and a
"3-layer CNN" decoder, so the latent is instead taken as a 1x1 convolution of the
8x8x64 ResNet-32 feature map to 5 channels (8x8x5 = 320 numbers, 1,920 codes for
the CIFAR-10 budget) and decoded by Conv3x3(5->384), ConvT(384->192), ConvT(192->3)
(1.2M parameters). The encoder is then exactly ResNet-32 plus a 1x1 convolution
(467k parameters, the same as the baselines' ResNet-32). Decoded exemplars become
recognisable (20-22 dB), and for the first time real images of the first task
survive the second task (2 tasks, 40% data, 20 epochs):

| Latent | lambda | task-1 acc. after task 2 | task-2 acc. |
|---|---|---|---|
| vector (pooled -> linear 307) | 1 | 0.0 | 81.7 |
| spatial 8x8x5 | 1 | 19.9 | 71.6 |
| spatial 8x8x5 | 0.3 | **73.4** | 44.2 |
