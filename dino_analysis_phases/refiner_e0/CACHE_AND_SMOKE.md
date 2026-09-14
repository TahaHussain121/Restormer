# Refiner experiment — cache provenance and smoke checks (2026-09-14)

Job **1813080**, v100 (tg073, Tesla V100-PCIE-32GB), 16 min 54 s, exit 0.
Code at commit `bc0bf7c`. Written before training started; the
pre-registration (`PREREGISTRATION.md`) is not edited.

## Frozen E0

`experiments/Holo_E0_fixed128_baseline/models/net_g_268000.pth`, iteration
268,000 from `best_checkpoint.json`, md5 `a7d094b5e56bc13fc9660034a7bc0775` =
`KEPT_CHECKPOINTS.json`. Eval mode, `requires_grad=False` everywhere; parameter
digest `4058c5d38a02a97fe5f49bcd9d27851a` identical before and after caching
and after six real backward passes through it.

## Cache (`experiments/<EXP>/e0_cache/`)

| split | shape | bytes | sha256 (prefix) | time | reference reproduction |
|---|---|---|---|---|---|
| val | 339x256x256 float32 | 88,866,944 | `dbe023988e6a898c` | 45 s | **339/339 bit-identical** to E0's recorded full256 PNGs; mean PSNR 22.076654 = recorded |
| train | 6101x256x256 float32 | 1,599,340,672 | `5ec766fc78e5226e` | 686 s | no reference exists; 3 rows re-run live match exactly |

Preprocessing: `cv2.IMREAD_UNCHANGED` uint16 → float32 / 65535, [1,1,256,256],
batch 1, no padding / crop / augmentation, `torch.no_grad`. Stored before
clamping and quantisation.

**Out-of-range values (recorded, not hidden).** 16.6% of raw output pixels are
slightly negative (val min −0.020, train min −0.030), essentially all
background; above 1: 4e-6 (val) and 6e-6 (train) of pixels. Clamping at
evaluation zeroes the negatives for both A and B. The refiner receives and
corrects the raw values.

Test is cached only at evaluation, after selection.

## Smoke checks — 26/26 PASS (`results/smoke/smoke_results.json`)

| check | measured |
|---|---|
| parameters | **118,129** (2,624 / 13,888 / 55,424 / 36,928 / 9,248 / 17) |
| output conv | weight and bias exactly 0 |
| module types | Conv2d, ReLU, AvgPool2d, Sequential only |
| TF32 | disabled (matmul and cuDNN) |
| normalisation | identical to the training loader's `imfrombytes_uint16` (max diff 0) |
| cache row pairing | live E0 on train ids 0001, 1390, 6778 = cache, max diff 0 |
| val reproduction | 339/339 identical, max per-image PSNR diff 7e-15 dB |
| refiner input | first conv receives (16,2,256,256): ch0 == X, ch1 == Y0, bitwise |
| shapes | delta and Y (16,1,256,256) |
| initial state | max \|delta\| = 0; Y == Y0 bitwise; identity val mean 22.076654 = E0 |
| metric path | torch quantisation = predict_phase3 numpy path (0 pixels differ); own PSNR = skimage = torch (7e-15) |
| optimizer | 22 tensors, all refiner; no E0 parameter |
| learning, live E0, 6 AdamW steps | step 1: only the output conv has gradient (0.416), all earlier layers exactly 0, as zero-init requires; step 2 onwards every layer non-zero (8e-5 to 1.5e-4); max \|delta\| 6e-4 → 2.5e-3; all finite |
| E0 after backward | eval, no grad, no `.grad`, digest unchanged |
| batch 16 | fits physically: peak 1.01 GiB, 174 ms/update on v100 including host loading. No accumulation needed |

Training submitted as job **1813124** (rtx3080).
