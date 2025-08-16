# Autoencoder-Based Hybrid Replay for Class-Incremental Learning (ICML 2025)

> Official repository for the paper **“Autoencoder-Based Hybrid Replay for Class-Incremental Learning”** (ICML 2025).

**Status:** We’re preparing the camera-ready, research-grade implementation and documentation.  
**Code availability:** public release in ~2 months — **by October 16, 2025**.

---

## Overview

This repo will host the reference implementation of **Autoencoder-Based Hybrid Replay (AHR)** and the **Hybrid Autoencoder (HAE)** used for efficient class-incremental learning.  
AHR stores exemplars **in the latent space** (compressed by HAE) and decodes them on demand for replay, targeting strong accuracy with a much smaller memory footprint while keeping compute linear in the number of tasks.

Key ideas to expect in the release:
- **Hybrid Autoencoder (HAE):** supports both discriminative (classification) and generative (replay) roles.
- **Latent-space replay:** exemplars are kept as embeddings and decoded for training batches.
- **Repulsive-force placement:** incremental class-centroid embeddings are arranged via a charged-particle/repulsive-force scheme.
- **Benchmarks & baselines:** we will include scripts to reproduce the main tables/figures from the paper on standard CIL benchmarks.

---

## Timeline

- **Now:** camera-ready polishing and code cleanup.
- **By Oct 16, 2025:** open-source release (training & eval pipelines, configs, and reproduction scripts).
- **Post-release:** checkpoints and a small “smoke test” to validate the environment.

---

## Paper

- arXiv: https://arxiv.org/abs/2505.05926  
- DOI: https://doi.org/10.48550/arXiv.2505.05926

### BibTeX
```bibtex
@article{khademi2025ahr,
  title   = {Autoencoder-Based Hybrid Replay for Class-Incremental Learning},
  author  = {Milad Khademi Nori and Il-Min Kim and Guanghui Wang},
  journal = {arXiv preprint arXiv:2505.05926},
  year    = {2025},
  doi     = {10.48550/arXiv.2505.05926},
  url     = {https://arxiv.org/abs/2505.05926}
}
