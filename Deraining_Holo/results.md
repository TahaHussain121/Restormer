# Holo / Holographic Denoising — Experiment Results Log

Running log of every training experiment. Each entry records the **context**
(dataset, inputs, key hyperparameters, hardware), the **exact commands** used to
train and evaluate, and the **results**. Newest experiments at the top.

Metric conventions:
- **Val PSNR/SSIM** — full-image, computed on the validation set during training
  (basicsr `Validation ValSet` log lines).
- **Test full** — full-image PSNR/SSIM on the locked test set.
- **Test masked** — PSNR/SSIM restricted to the foreground object mask
  (`masked_metrics.py`), which better reflects object-region reconstruction than
  the full frame (large background lowers the denominator).

---

## Exp 2 — Baseline Restormer, VERYNOISY input  🔄 RUNNING

**Status:** in progress — **174,000 / 300,000 iters (58%)** as of 2026-07-20 08:31.
Final results TBD.

**Context**
- Dataset: `holographic_image_dataset/` (6778 pairs; split 6101 train / 677 val)
- GT (target): `train_clean` / `val_clean`
- LQ (input): `train_verynoisy` / `val_verynoisy`  ← *verynoisy*, not noisy
- Experiment name: `Holo_Baseline_Restormer_verynoisy` (separate dir; does **not**
  touch Exp 1)
- Key change vs Exp 1: **mixup disabled** (`mixing_augs.mixup: false`)
- Config: `Options/Holo_Baseline_Restormer.yml`
- Model: Restormer, `inp_channels=1`, `out_channels=1` (grayscale uint16 /65535)
- Total iters: 300000 · loss: L1Loss · effective batch 8 (1 GPU)
- Hardware: 1× V100

**Commands**
```bash
# from repo root: /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
# self-chaining resume driver (submit once; re-queues itself until 300k)
sbatch Deraining_Holo/train_holo_chain_verynoisy.sh
```
Outputs → `experiments/Holo_Baseline_Restormer_verynoisy/`
TensorBoard → `tb_logger/Holo_Baseline_Restormer_verynoisy`

**Run log**
- Submitted 2026-07-18 as job `1752760`; chain job #2 = `1752761` (running on
  `tg072`), successor `1753869` queued with `afterany` dependency.
- Throughput ~0.79 s/iter (~1000 iters / 13.5 min) on 1× V100.
- basicsr ETA at the 174k snapshot: ~1 day 2 h to reach 300k.

**Results — in-progress snapshot @ 174k (2026-07-20)**

| Metric | Value |
|---|---|
| Val PSNR / SSIM (best so far) | **21.996** / **0.7961** |
| Val PSNR / SSIM (@174k) | 21.588 / 0.7868 |
| Train loss `l_pix` @174k | ~4.4e-2 – 7.3e-2 (stable) |
| Test full PSNR / SSIM | _TBD — after run completes_ |
| Test masked PSNR / SSIM | _TBD — after run completes_ |

**Observations so far**
- Val PSNR has been oscillating in a **21.0–22.0** band for many evals — looks
  like a plateau, but no divergence and loss is stable.
- The LR schedule has a cosine restart at iter 208000
  (`periods: [92000, 208000]`), so a further improvement may still appear after
  208k. Worth re-checking before concluding the run has converged.
- Val PSNR ~21.6 here vs ~33.4 in Exp 1 is **expected**, not a regression: the
  verynoisy input is far harder *and* it is a different dataset. See note below —
  the two runs are not directly comparable.

---

## Exp 1 — Baseline Restormer, NOISY input  ✅ done

**Context**
- Dataset: `holo_image_dataset/` (locked train/val/test splits)
- GT (target): `train_clean` / `val_clean`
- LQ (input): `train_noisy` / `val_noisy`
- Experiment name: `Holo_Baseline_Restormer`
- mixup: **enabled** (`mixup_beta: 1.2`, upstream deraining recipe, faithful baseline)
- Config: `Options/Holo_Baseline_Restormer.yml` (as of commit d32ab74)
- Model: Restormer, `inp_channels=1`, `out_channels=1` (grayscale uint16 /65535)
- Loss: L1Loss · progressive patch schedule truncated at native 256px
- Hardware: 1× V100, effective batch 8 (upstream assumes 8 GPUs → batch 64)
- Best-val checkpoint selected: **iter 224000** (val PSNR plateaued; 300k not needed)

**Commands**
```bash
# Train (self-chaining resume driver)
sbatch Deraining_Holo/train_holo_chain.sh

# Evaluate best-val checkpoint (224k) on the LOCKED test set
python Deraining_Holo/test_holo.py \
  --input_dir  .../holo_image_dataset/test_noisy \
  --gt_dir     .../holo_image_dataset/test_clean \
  --weights    experiments/Holo_Baseline_Restormer/models/net_g_224000.pth \
  --result_dir Deraining_Holo/results/Holo_test_224k/

# Masked (foreground-object) metrics
python Deraining_Holo/masked_metrics.py \
  --pred_dir  Deraining_Holo/results/Holo_test_224k/raw \
  --gt_dir    .../holo_image_dataset/test_clean \
  --threshold 0.01 --dilate 3 \
  --csv       Deraining_Holo/results/Holo_test_224k/masked_metrics_per_image.csv
```

**Results** — test set, n=338 images

| Metric | Value |
|---|---|
| Val PSNR / SSIM (best, full-image) | **33.47** / **0.9479** |
| Val PSNR / SSIM (final @224k) | 33.40 / 0.9478 |
| Test full PSNR / SSIM | **33.50** / **0.9458** |
| Test masked PSNR / SSIM | **29.31** / **0.8739** |
| Mean mask fraction | 0.395 (range 0.14–0.80) |

Per-image data: `results/Holo_test_224k/masked_metrics_per_image.csv`
Figures: `results/Holo_test_224k/for_professor/` (training curves, typical &
blurry examples), `experiments/Holo_Baseline_Restormer/training_curves.png`

**Notes**
- Test set is the *noisy* variant of the old `holo_image_dataset`. Exp 2 switches
  to the newer `holographic_image_dataset` with the harder *verynoisy* input, so
  the two are **not directly comparable** — Exp 2 needs its own verynoisy test
  metrics before any cross-comparison.
