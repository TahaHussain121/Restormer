# Residual refiner on the frozen E0 — result (2026-09-15)

Numbers of record: **DEVLOG.md Step 51**. This file is a short summary.
Proposal: a ChatGPT-assisted discussion, authorised by the author as one run.

| | |
|---|---|
| identity | `Holo_E0_frozen_noisy_output_residual_refiner` |
| frozen base | `Holo_E0_fixed128_baseline`, checkpoint 268,000 (validation-selected, md5 `a7d094b5…`) |
| refiner | fixed U-Net, **118,129** trainable parameters (0.45% of E0's 26,124,052 total trainable) |
| jobs | cache + smoke 1813080 · training 1813217 · evaluation 1813693 (all v100) |
| training | early stop at 8,500 updates, 5.5 min; **selected update 6,000** |
| checkpoints | `experiments/<EXP>/models/refiner_{step0,best,final}.pth` |
| outputs (gitignored) | `results/{predictions,metrics,figures}/` |

## Test, full256, n = 338 (E0 → E0 + refiner)

| metric | change | 95% CI (images, one run) | better / worse |
|---|---|---|---|
| whole-image PSNR | 21.873 → 22.065, **+0.193** | [+0.124, +0.264] | 189 / 149 |
| foreground PSNR | 17.599 → 17.473, **−0.127** | [−0.183, −0.071] | 124 / 214 |
| foreground SSIM | **−0.0106** | [−0.0121, −0.0091] | 61 / 277 |
| background MAE | 0.0110 → 0.0076 | | 338 / 0 |

Validation agrees in sign on every line (whole +0.139, foreground −0.167).

## Reading

* The correction darkens: 95.5% of foreground pixels get darker, and the
  foreground is darkened on net in every image. That cleans E0's background
  haze (whole-image gain) and adds error on the object (foreground loss). A
  trade-off, not an improvement.
* No sidelobes return: in the background the correction moves away from the
  noisy frame (correlation −0.17).
* **No recovery of weak, measurement-supported structure**: in the 228
  pre-declared recovery windows retention fell 0.42 → 0.35 (−0.63 dB), the
  same as in the 191 control windows (0.33 → 0.23, −0.66 dB).

Pre-registered guesses: 2 of 3 correct (gain size, no sidelobes); the
recovery-over-control guess was wrong. No follow-up run is started.
