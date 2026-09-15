# Foreground-balanced refiner — result (2026-09-15)

Numbers of record: **DEVLOG.md Step 52**. A short summary; exploratory
follow-up on a previously inspected test split. Confidence intervals are over
images for one training run each, not over training seeds.

| | |
|---|---|
| identity | `Holo_E0_frozen_noisy_output_fgbalanced_refiner` |
| changed vs refiner_e0 | loss (0.5 fg + 0.5 bg region-mean L1, gt > 0.01) and selection (val foreground PSNR, background MAE/RMSE no worse than E0) |
| jobs | training 1813720 · evaluation 1813729 (failed, my bug, before test) · 1813745 |
| selected | update 8,500 (original rule on this run: 6,000) |

## Test, full256, n = 338 — E0 / refiner_e0 / fg-balanced

| | E0 | refiner_e0 | fg-balanced | fg-bal − E0 |
|---|---|---|---|---|
| whole PSNR | 21.873 | 22.065 | 22.154 | +0.282 [+0.241, +0.323] |
| foreground PSNR | 17.599 | 17.473 | 17.809 | **+0.210** [+0.175, +0.245] |
| foreground SSIM | 0.5598 | 0.5492 | 0.5636 | +0.0038 |
| whole SSIM | 0.7829 | 0.7898 | 0.7789 | −0.0040 (worse on 267/338) |
| background MAE | 0.0110 | 0.0076 | 0.0106 | −0.0004 (but worse on 225/338) |
| background RMSE | 0.0434 | 0.0346 | 0.0399 | −0.0035 |

## Verdict

* Foreground reconstruction: **better** (both splits, above +0.10 dB).
* Overall denoising: **mixed** (PSNR up, SSIM down).
* Recovery of previously missing structure: **no**. *This foreground-balanced
  refiner did not improve structure recovery under the tested setup.*
* Background: **uncertain** — the registered constraint is met, but low-level
  signal spreads (edge rims).
* Loss vs selection: at the matched update 6,000 (same init, schedule, data
  order), validation foreground PSNR is 17.812 (refiner_e0) vs 18.130 (this
  run). Most of the difference comes with the loss weighting. One seed.

Recommendation: record as a limited result and stop.
