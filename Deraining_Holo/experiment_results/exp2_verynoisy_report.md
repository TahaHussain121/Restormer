# Experiment Report — Verynoisy Restormer Baseline

**Experiment ID:** `Holo_Baseline_Restormer_verynoisy`
**Trained:** 2026-07-18 23:12 → 2026-07-21 13:59 (CEST) · 16 h 47 m compute
**Evaluated:** 2026-07-22 · 338-image held-out test set
**Report generated:** 2026-07-22
**Config:** `Deraining_Holo/Options/Holo_Baseline_Restormer.yml`

---

## Table 1 — Results

Reported checkpoint: **`net_g_292000.pth`** (best by validation PSNR).

| Split | Region | PSNR (dB) | SSIM |
|:--|:--|--:|--:|
| Test (n=338) | Full image | 22.405 ± 3.052 | 0.7999 ± 0.077 |
| Test (n=338) | Masked (foreground) | 18.313 ± 2.917 | 0.5438 ± 0.136 |
| Validation (n=677)† | Full image (best) | 22.446 | 0.8156 |

Degraded input baseline (verynoisy vs clean, no model): **12.354 dB / 0.3353**.
Improvement from denoising: **+10.050 dB / +0.4645 SSIM**.
Mean foreground coverage: 39.5 % (range 13.9–80.3 %).

† Validation used all 677 images during training; the 338 test images were
carved out afterwards (see Notes). Test 22.405 vs val 22.446 = 0.04 dB gap.

**Sharpness (high-frequency retention), test set:**

| Measure | Input | Prediction | Ground truth | Pred/GT |
|:--|--:|--:|--:|--:|
| Laplacian variance | 0.00083 | 0.00015 | 0.00052 | 0.283 |
| Sobel gradient | 0.19479 | 0.08010 | 0.10969 | 0.730 |
| HF energy fraction | 0.00078 | 0.00021 | 0.00099 | 0.216 |

The prediction retains ~22 % of the ground-truth high-frequency energy — the
model over-smooths, removing fine structure along with noise.

---

## Table 2 — Experimental settings

| | |
|:--|:--|
| **Task** | Holographic CFR heatmap denoising (single-channel, 256×256) |
| **Dataset** | `holographic_image_dataset` |
| **Input (LQ) / Target (GT)** | verynoisy / clean |
| **Split (train / val / test)** | 6101 / 339 / 338 (seed 42; test carved post-hoc) |
| **Image format** | uint16 PNG, normalized /65535 → [0,1] |
| **Data loader** | `Dataset_PairedImage_uint16` |
| **Geometric augmentation** | on (flip / rotate) |
| **Mixup** | **off** |
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
| **Training time** | 16 h 47 m (394 epochs; 3 chained SLURM jobs) |

---

## Notes

- **Checkpoint selection:** best of 151 checkpoints by validation PSNR was 292k
  (22.446 dB); final 300k was 22.437 dB (Δ 0.009 dB). No overfitting observed —
  training loss falls throughout while validation plateaus without decline.
- **Test-split caveat:** the test split was carved *after* training completed, so
  validation during training covered all 677 images and the 338 test images
  influenced checkpoint *selection* (not the weights — `train ∩ test = 0` is
  verified). Top-5 checkpoints span 0.016 dB, so the practical effect is
  negligible, but this test figure is not a fully clean held-out estimate.
- **Masked vs full:** the full-image metric is inflated by the ~60 % empty
  background; the object-region (masked) figure is the honest measure of
  reconstruction quality. Full-minus-masked PSNR gap 4.09 ± 1.24 dB.
- **Reproduce:** `sbatch Deraining_Holo/train_holo_chain_verynoisy.sh` (train),
  `sbatch Deraining_Holo/eval_test_verynoisy.sh` (test). Full write-up and
  figures in `experiment_results/results.md` and `exp2_verynoisy/`.
