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
