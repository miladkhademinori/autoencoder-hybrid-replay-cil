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
```

---

## Reproduction: Hybrid Replay for Federated Class-Incremental Learning (ICLR 2025)

This branch also contains an independent re-implementation of the companion federated paper,
*Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free
Techniques to Address Local and Global Forgetting* (Khademi Nori, Kim, Wang, ICLR 2025,
[arXiv:2501.15356](https://arxiv.org/abs/2501.15356)), together with the scripts used to
reproduce its Table 2 and Figure 3.

- `hr_fcil/`: HR (ResNet-18 VAE encoder, 4-layer CNN decoder, Lennard-Jones centroid placement,
  latent-exemplar and centroid-based global replay, KD, FedAvg over LDA client splits) and all
  Table 2 ablations
- `run.py`: a single run; `scripts/run_queue.py`: resumable experiment queue;
  `scripts/aggregate.py`: comparison with the paper's numbers
- `notebooks/HR_FCIL_colab.ipynb`: runs the queue on a Colab GPU and saves results to Google Drive
- [`REPRODUCTION.md`](REPRODUCTION.md): protocol, every implementation choice, deviations,
  and issues found in the paper
- [`results/RESULTS.md`](results/RESULTS.md): reproduced numbers against the paper

```bash
pip install -r requirements.txt
python run.py --benchmark cifar100_10_10_50_5 --variant hr --seed 0 --amp
```
