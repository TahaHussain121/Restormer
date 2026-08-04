# Experiment Report — Noisy Restormer Baseline

**Experiment ID:** `Holo_Baseline_Restormer`
**Trained:** 2026-06-12 09:30 → 2026-06-15 00:23 (CEST) · chained SLURM jobs
**Evaluated:** locked test set (carved before training)
**Report generated:** 2026-07-22
**Config:** `Deraining_Holo/Options/Holo_Baseline_Restormer.yml` (as of commit d32ab74)

---

## Table 1 — Results

Reported checkpoint: **`net_g_224000.pth`** (best by validation PSNR).

| Split | Region | PSNR (dB) | SSIM |
|:--|:--|--:|--:|
| Test (n=338) | Full image | 33.499 ± 2.329 | 0.9458 ± 0.027 |
| Test (n=338) | Masked (foreground) | 29.312 ± 1.981 | 0.8739 ± 0.051 |
| Validation | Full image (best) | 33.473 | 0.9479 |
| Validation | Full image (final, 300k) | 33.405 | 0.9478 |

Mean foreground coverage: 39.5 %.

---

## Table 2 — Experimental settings

| | |
|:--|:--|
| **Task** | Holographic CFR heatmap denoising (single-channel, 256×256) |
| **Dataset** | `holo_image_dataset` |
| **Input (LQ) / Target (GT)** | noisy / clean |
| **Split (train / val / test)** | 6101 / 339 / 338 (seed 42; test locked before training) |
| **Image format** | uint16 PNG, normalized /65535 → [0,1] |
| **Data loader** | `Dataset_PairedImage_uint16` |
| **Geometric augmentation** | on (flip / rotate) |
| **Mixup** | **on** (beta 1.2, upstream deraining default) |
| **Architecture** | Restormer |
| **Parameters** | 26,124,052 |
| **inp / out channels** | 1 / 1 |
| **dim** | 48 |
| **num_blocks** | [4, 6, 6, 8] |
| **num_refinement_blocks** | 4 |
| **attention heads** | [1, 2, 4, 8] |
| **FFN expansion factor** | 2.66 |
| **bias** | False |
| **LayerNorm** | WithBias |
| **Loss function** | **L1 (mean reduction, weight 1)** |
| **Optimizer** | AdamW (β = [0.9, 0.999], weight decay 1e-4) |
| **Learning rate** | 3e-4 initial |
| **LR schedule** | CosineAnnealingRestartCyclicLR, periods [92000, 208000], eta_min [3e-4, 1e-6] |
| **Gradient clipping** | on |
| **Warmup** | none |
| **Total iterations** | 300,000 |
| **Progressive patch sizes** | [128, 160, 192, 256] over iters [92000, 64000, 48000, 96000] |
| **Progressive batch sizes** | [8, 5, 4, 2] per GPU |
| **Effective batch** | 8 (1 GPU; upstream recipe assumes 8 GPUs → 64) |
| **Random seed** | 100 |
| **GPUs** | 1 × NVIDIA V100-PCIE-32 GB |
| **Framework** | PyTorch 2.5.1 + CUDA 12.1 (BasicSR) |

---

## Notes

- **Clean held-out test set:** the test split was carved *before* this training
  run began, so the 338 test images influenced neither the weights nor checkpoint
  selection. This is a genuine held-out estimate (contrast Exp 2, whose split was
  carved after training).
- **Checkpoint selection:** best of the run by validation PSNR was 224k
  (33.473 dB); final 300k was 33.405 dB. A slight post-peak decline after 224k is
  visible in `figures/early_stop_knee.png` (mild over-training, not divergence).
- **Comparison to Exp 2:** not directly comparable — different dataset, noise
  level and mixup setting. See `../results.md`.
