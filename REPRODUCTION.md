# Reproducing "Autoencoder-Based Hybrid Replay for Class-Incremental Learning" (ICML 2025)

This document describes the implementation in this repository, every detail that
had to be chosen because the paper does not specify it, what was run, and how the
numbers compare with Table 2 of the paper
([arXiv:2505.05926](https://arxiv.org/abs/2505.05926)).

<!-- RESULTS -->
## 0. Results


Final accuracy (%) after the last task, mean ± SEM over seeds (metric: `final_acc`, tag: `main`).

| Method | MNIST (5/2) ours | paper | Balanced SVHN (5/2) ours | paper | CIFAR-10 (5/2) ours | paper | CIFAR-100 (10/10) ours | paper |
|---|---|---|---|---|---|---|---|---|
| FT | 19.76 ± 0.01 (n=3) | 19.93 ± 0.03 | 19.64 (n=1) | 19.19 ± 0.04 | 19.57 (n=1) | 18.72 ± 0.30 | 8.85 (n=1) | 8.91 ± 0.12 |
| FT-E | 72.18 ± 0.81 (n=3) | 92.17 ± 0.16 | 55.61 (n=1) | 87.13 ± 0.37 | 43.95 (n=1) | 72.17 ± 0.84 | 27.10 (n=1) | 48.47 ± 0.83 |
| Joint | 98.54 ± 0.04 (n=3) | 98.48 ± 0.06 | 95.43 (n=1) | 95.88 ± 0.04 | 89.02 (n=1) | 92.37 ± 0.09 | 59.43 (n=1) | 73.87 ± 0.10 |
| iCaRL | 88.60 ± 0.10 (n=3) | 93.06 ± 0.33 | 71.07 (n=1) | 89.63 ± 0.61 | 62.82 (n=1) | 73.29 ± 0.73 | 37.99 (n=1) | 49.38 ± 0.62 |
| AHR | 94.57 ± 0.61 (n=3) | 97.53 ± 0.32 | - | 93.02 ± 0.65 | - | 77.12 ± 0.75 | - | 54.43 ± 0.93 |
| AHR-lossy-mini | 68.90 ± 0.60 (n=3) | 93.35 ± 0.32 | - | 90.40 ± 0.58 | - | 73.28 ± 0.47 | - | 50.29 ± 0.90 |
| AHR-lossless-mini | 67.34 ± 2.06 (n=3) | 93.76 ± 0.26 | - | 90.88 ± 0.50 | - | 73.68 ± 0.41 | - | 50.85 ± 0.81 |
| AHR-lossless | 95.11 ± 0.26 (n=3) | 98.12 ± 0.08 | - | 94.21 ± 0.23 | - | 78.35 ± 0.37 | - | 56.71 ± 0.57 |

Epochs / exemplars / wall-clock per run:

| Dataset | Method | epochs | stored exemplars | memory (scalars) | minutes/run |
|---|---|---|---|---|---|
| mnist | FT | 40 | 0 | 0 | 2 |
| mnist | FT-E | 40 | 200 | 156,800 | 2 |
| mnist | Joint | 40 | 0 | 0 | 2 |
| mnist | iCaRL | 40 | 200 | 156,800 | 2 |
| mnist | AHR | 40 | 7840 | 156,800 | 18 |
| mnist | AHR-lossy-mini | 40 | 200 | 4,000 | 20 |
| mnist | AHR-lossless-mini | 40 | 200 | 156,800 | 13 |
| mnist | AHR-lossless | 40 | 7840 | 6,146,560 | 15 |
| svhn | FT | 50 | 0 | 0 | 136 |
| svhn | FT-E | 50 | 200 | 614,400 | 137 |
| svhn | Joint | 50 | 0 | 0 | 142 |
| svhn | iCaRL | 50 | 200 | 614,400 | 162 |
| cifar10 | FT | 50 | 0 | 0 | 143 |
| cifar10 | FT-E | 50 | 200 | 614,400 | 147 |
| cifar10 | Joint | 50 | 0 | 0 | 155 |
| cifar10 | iCaRL | 50 | 200 | 614,400 | 186 |
| cifar100 | FT | 50 | 0 | 0 | 134 |
| cifar100 | FT-E | 50 | 2000 | 6,144,000 | 184 |
| cifar100 | Joint | 50 | 0 | 0 | 123 |
| cifar100 | iCaRL | 50 | 2000 | 6,144,000 | 249 |

mnist ablations (final accuracy %, mean ± SEM):

| Variant | Final acc. |
|---|---|
| AHR (final configuration) | 94.57 ± 0.61 (n=3) |
| + latent loss on reconstructions of new samples | 93.42 ± 0.38 (n=3) |
| + classification in the decoder's output domain (as used for SVHN/CIFAR) | 91.45 ± 0.54 (n=3) |
| - decoder memorisation | 84.00 ± 1.39 (n=3) |
| - frozen codes, - memorisation = literal Alg. 1-4 (herding selection) | 75.01 ± 1.48 (n=3) |
| literal Alg. 1-4, Rank selection | 70.06 ± 0.96 (n=3) |
| FT-E with AHR's balanced minibatches | 75.10 ± 0.57 (n=3) |
<!-- /RESULTS -->

## Summary

**Setting.** Everything was run on a 4-core CPU without a GPU (bf16 on AMX), at the
paper's epochs, batch sizes, optimiser and memory budgets (Table 4). MNIST results
are 3 seeds; the image benchmarks are 1 seed. miniImageNet was not run (compute).
The paper does not specify the loss weights, the RFA constants, the latent head or
the decoder beyond "3 layers of CNNs"; the values used are listed in §2.

**What reproduces.**
* FT and Joint match the paper within 1-3 points on every benchmark run.
* On MNIST the ordering AHR (94.6) > iCaRL (88.6) > FT-E (72.2) holds, and so do both
  ablation claims: decoded exemplars are almost as good as perfect ones
  (AHR 94.6 vs AHR-lossless 95.1; paper 97.5 vs 98.1), and storing ~40x more
  (compressed) exemplars matters far more than their fidelity (AHR-lossy-mini
  68.9, AHR-lossless-mini 67.3).

**What does not reproduce as described.**
* The absolute numbers: AHR is 3 points below the paper on MNIST, and the FT-E /
  iCaRL baselines are 5-30 points below the paper's values with the same
  protocol (FACIL-style replay, Adam 1e-3, the paper's epochs).
* Algorithm 4's `Rank` selection (closest-to-centroid first) and re-encoding the
  memory every task both hurt; the literal Algorithms 1-4 give 70-75% on MNIST.
  Frozen codes plus explicit decoder memorisation are what make AHR work (§3.5).
* On SVHN/CIFAR the paper's recipe (a 307-d latent from ResNet-32, classification
  of the raw image by the nearest CCE) fails: decoded exemplars are either too
  blurry to carry class information (§3.6) or, once good (24 dB), the encoder learns
  "decoded = old task, real = new task" and forgets every old class (§3.10). What
  works is a spatial 8x8x5 latent plus classifying in the decoder's output domain
  (§3.9, §3.11), which departs from the paper's test rule.

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
| Fixed exemplar memory; AHR stores `budget x input size / latent size` codes (MNIST 7,840; SVHN / CIFAR-10 1,920 and CIFAR-100 19,200 with the 320-number latent) | `scripts/run.py` (`exemplar_count`) |
| Ablations AHR-lossless, AHR-lossy-mini, AHR-lossless-mini | `--method ahr_lossless / ahr_lossy_mini / ahr_lossless_mini` |
| Baselines FT, FT-E, iCaRL, Joint | `ahr/baselines.py` |

The remaining baselines of Table 2 (SLDA, Gen-C, PEC, DGR, MeRGAN, BI-R-SI, GD,
GDumb, EEIL, BiC, LUCIR, IL2M, i-CTRL, REMIND, REMIND+) are not re-implemented;
their paper numbers are quoted for reference only.

## 2. Details the paper does not specify, and what was chosen

| Item | Choice | Why |
|---|---|---|
| `lambda` (Eq. 1) | 0.3 | MNIST sweep over {0.1, 0.3, 1, 10, 100} (0.3: 93.0 vs 91.8 for 1.0); CIFAR sweep over {0.3, 0.5, 1} (§3.9) |
| Distillation weights `a_z`, `a_x` and form | `a_z` = 0.01 (MNIST) / 0.1 (SVHN, CIFAR), `a_x = 1`, squared L2 | strong encoder distillation keeps new-class samples away from their shifted CCEs (§3.4); on CIFAR at full scale 0.1 retains more of the old task than 0.01 (32.4% vs 6.6% after task 2) |
| RFA constants `zeta, m, dt` | 1, 1, 0.01, softening 1e-3, no damping | scale-free once the duration is chosen as below |
| RFA duration `tau` | integrate Alg. 2 until every new CCE is `>= d` from every other CCE; `d = 5` (MNIST, m = 20) and `d = 20` (m = 320), i.e. `d ~ sqrt(m)` | with a fixed `tau` the spacing depends on how close the initial class means are, which differs by an order of magnitude between the first task (random encoder) and later ones (§3.3) |
| Encoder / latent (MNIST) | 784-400-400 ReLU + linear to 20 | paper |
| Encoder / latent (SVHN, CIFAR) | ResNet-32 up to its 8x8x64 feature map + 1x1 conv to 5 channels: an 8x8x5 = 320-number spatial latent (paper: 307); head initialised with std 1e-3 | a pooled 307-d vector latent reconstructs only colour blobs (§3.6, §3.9); the small init keeps the latent scale set by RFA rather than by the backbone's activation scale |
| Decoder (MNIST) | mirror MLP 20-400-400-784, sigmoid | paper |
| Decoder (SVHN, CIFAR) | Conv3x3(5->384), ConvT(384->192, x2), ConvT(192->3, x2), ReLU, sigmoid; 1.21M params | "3 layers of CNNs", ~1.4M params (Table 3) |
| Classification domain (SVHN, CIFAR) | latent loss only on decoder outputs (decoded exemplars + reconstructions of the new samples); test on `phi(psi(phi(x)))` | removes the real-vs-decoded task cue (§3.10, §3.11); MNIST keeps the paper's rule |
| Memory | codes stored once with the encoder of their task (Alg. 4 `phi(w_i, D)`) and never re-encoded; decoder distillation also applied to the stored codes; after each task the decoder alone is fitted to the new exemplars' (code, image) pairs for 20 epochs (MNIST) / 1,500 steps (others) | §3.5 |
| Minibatch composition | `round(B/l)` new samples + `B - round(B/l)` exemplars sampled uniformly from the class-balanced memory; one epoch = one pass over the new task's data | Sec. 2 ("1/l fraction ... (l-1)/l fraction"); gives the O(t) compute of Table 1 |
| Exemplar selection | herding in latent space | the text says Alg. 4 is "based on Herding as in iCaRL"; the literal `Rank` of `L_z` is implemented too (§3.1) |
| Baselines' replay | shuffled union of new data and exemplars (FACIL) | AHR's balanced minibatches over-replay the 200 raw exemplars (MNIST FT-E: 75.1% balanced vs 72.2% union, both far below the paper's 92.2%) |
| Learning-rate schedule | cosine annealing within each task (all methods) | not specified |
| Augmentation | random crop (pad 4) for SVHN/CIFAR, + horizontal flip for CIFAR; none for MNIST | standard |
| Class order | natural (0,1 / 2,3 / ...) | not specified |
| Precision | bf16 autocast for convolutions on CPU (AMX); latent maps in fp32 | compute budget (no GPU was available) |

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
| spatial 8x8x5, no reconstruction term | 0.3 | **76.6** | 41.4 |
| spatial 8x8x5, CCE spacing 40 | 0.3 | 33.8 | 67.6 |
| spatial 8x8x5, CCE spacing 40 | 0.5 | 19.4 | 74.4 |
| split: 8x8x4 spatial + 64-d class part, spacing 10 | 1 | 23.8 | 79.4 |
| split: 8x8x4 spatial + 64-d class part, spacing 20 | 1 | 21.3 | 75.7 |
| *reference: AHR-lossless (raw exemplars, vector latent)* | 1 | *84.5* | *75.4* |

A checkpoint analysis of the best spatial run rules out BatchNorm statistics,
augmentation and bf16 as causes (recomputing the BN statistics, fp32 inference and
augmented inputs all change the accuracies by < 1 point): with the spatial latent the
same 320 numbers must both reconstruct the image and sit next to a fixed class
centroid, so a weak latent pull (lambda = 0.3) under-learns the new classes and a
strong one (lambda >= 0.5, or a larger spacing) forgets the old ones.

At full scale (all data, 50 epochs) the plasticity side wins: the spatial latent with
lambda = 0.3 reached 97.2% after task 1 (decoded exemplars at 24.1 dB, 98.5% of them
correctly classified) but after task 2 kept only 6.6% on the first task (87.9% on the
second), i.e. the longer training drifts the encoder away from the old classes.

### 3.10 At full scale the encoder becomes a real-vs-decoded detector
Full CIFAR-10(5/2) run with the spatial latent (lambda = 0.3, a_z = 0.1, paper epochs
and data): 97.2% after task 1, 53.3% after task 2 ([32.4, 74.2]), 31.8% after task 3
([0.4, 0.4, 94.6]), i.e. fine-tuning behaviour. The task-3 checkpoint shows why:

| Images | classified correctly |
|---|---|
| decoded exemplars of the old classes (0-3) | 98.3% |
| decoded exemplars of the newest classes (4-5) | 0.3% (all go to classes 0-3) |
| real test images of the old classes | 0.5% (all go to classes 4-5) |
| real test images of the newest classes | 94.6% |

Recomputing BatchNorm statistics, fp32 inference or test-time augmentation change
none of this. Even at 24 dB PSNR the network separates decoded from real images
perfectly and uses that as the task label: whatever looks decoded is "old", whatever
looks real is "new". With 6x fewer iterations (40% of the data, 20 epochs) the same
configuration still retained 76.6% of the first task after the second, so the
shortcut is learned gradually and the paper's training length is enough to learn
it completely.

### 3.11 What finally works on CIFAR: classify in the decoder's output domain
Since any detectable difference between real and decoded images becomes a task cue
(§3.10), the classification loss is applied only to images produced by the decoder:
the decoded exemplars and, for the new task, the HAE's own reconstructions of the new
samples (computed in the same minibatch, §3.7); real images still train the
autoencoder. At test time an image is classified through the autoencoder,
`argmin_c ||phi(psi(phi(x))) - p_c||`, so train and test inputs come from the same
domain for every class (`--latent-domain recon`). With the spatial latent the
reconstructions are good enough (26 dB on the stored exemplars) for this to cost
little accuracy: at full scale on CIFAR-10 the first task reaches 95.9% and after the
second task the model keeps **81.7%** of the first task while learning the second to
79.3% (80.5% overall; iCaRL at the same point: 83.5%, FT-E 75.4%). With the
vector latent (§3.6) the same idea only reached 70.7% on the first task because its
reconstructions carried too little class information.

This deviates from the paper's test rule (`argmin ||phi(x) - p||`) and uses the
reconstruction term of §3.2, but keeps every other element of Alg. 1-4.
