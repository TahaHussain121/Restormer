# Residual refiner on the frozen E0 — ONE isolated curiosity experiment

`Holo_E0_frozen_noisy_output_residual_refiner`

**Pre-registration, written 2026-09-14, before any refiner training.** Proposed
in a **ChatGPT-assisted discussion** and authorised by the author as one bounded
run (implementation plus one training run, no sweep). It is **not** a supervisor
instruction, and it is **outside** the DINO injection study: no render, no DINO.
It does not touch the two running final experiments.

## Question

Can a small second stage, which sees the noisy measurement X **and** the frozen
radar-only Restormer output Y0, restore measurement-supported structures that
Restormer weakened or removed, **without reintroducing sidelobes**?

    delta   = Refiner(concat(X, Y0))
    Y_final = Y0 + delta

Radar-only inference: no render, no DINO, no clean-target-derived input, and E0
is never modified or fine-tuned. The refiner cannot be expected to reliably
recover information absent from the measurement, and no such claim will be made.

## Existing work checked

No refiner, second-stage or residual-refinement experiment exists in code,
configs, `experiments/` or results (searched 2026-09-14).

## Frozen base

`Holo_E0_fixed128_baseline`, resolved from its recorded validation selection
(`results/Holo_E0_fixed128_baseline/metadata/best_checkpoint.json`): iteration
**268,000**, `net_g_268000.pth`, md5 `a7d094b5e56bc13fc9660034a7bc0775`, matching
`KEPT_CHECKPOINTS.json`. Not the progressive baseline, not a render-guided model.
Eval mode, `requires_grad=False` on every parameter, never in an optimizer. Its
recorded full256 references: val **22.0767**, test **21.8725** (uint16).

## Data and cache

Established splits (train 6,101 / val 339 / test 338), normalisation uint16 /
65535. **Full 256x256 frames throughout**: E0 runs on whole frames (its full256
evaluation path, batch 1), the refiner trains and evaluates on whole frames. No
mixing with independently restored 128 crops. No augmentation.

E0's raw float32 outputs, **before clamping and quantisation**, are cached for
train and val (test only at evaluation) in
`experiments/<EXP>/e0_cache/`, with a provenance JSON per split: checkpoint and
md5, ids and their order, preprocessing, inference context (device, versions,
TF32 flags), dtype, out-of-range fractions, and whether quantising the cache
reproduces E0's recorded predictions. Run on **v100**, where the reference was
made, so reproduction can be bit-exact. Noisy frames and targets are not
duplicated. The target is used only for the loss and for evaluation.

## Architecture (fixed, not tuned)

U-Net: E1 2→16→16, avg-pool, E2 16→32→32, avg-pool, bottleneck 32→64→64,
bilinear up (align_corners=False) + concat E2 (96) → D2 96→32→32, bilinear up +
concat E1 (48) → D1 48→16→16, 1x1 16→1 signed residual. Every block
conv3x3(bias)–ReLU–conv3x3(bias)–ReLU. No BN, attention or extra branch.
Hidden convs Kaiming-normal (fan_in, ReLU), zero bias; **output conv zero weight
and bias**. **118,129 trainable parameters** (counted by hand per block and
verified in code: 2,624 / 13,888 / 55,424 / 36,928 / 9,248 / 17). For scale:
E0 has 26,124,052, so the refiner is **0.45% of E0's parameter count**
(denominator: E0's total trainable parameters).

## Training (fixed)

Only the refiner trains. Seed 100; AdamW lr 1e-4, betas (0.9, 0.999), weight
decay 1e-4; batch 16 (physical, no accumulation unless it does not fit); cosine
decay 1e-4 → 1e-6 over 10,000 updates, no warmup; grad-norm clip 1.0; float32
with TF32 **explicitly disabled**; loss `mean |Y0 + delta − Y_gt|` on the
UNBOUNDED output. No other loss terms. GPU: rtx3080 (a 118k-parameter model
does not need an a100).

## Validation, selection and stopping

Every 500 updates on all 339 validation frames. **Selection metric: mean full256
PSNR under the uint16 evaluation convention** (clamp to [0,1], ×65535, round,
skimage PSNR with data_range 1.0) — the convention every reported number in this
project uses. (E0's own checkpoint was selected on the 8-bit in-loop PSNR; that
path is a training-loop artefact and is not used here.)

A step-0 checkpoint (identity, delta = 0) is kept and validated; it must
reproduce E0's reference val mean. Best starts at step 0. Each check either
strictly improves the best or extends a non-improving streak; **training stops
once the update count is ≥ 3,000 and the streak has reached 5**, and in any
case at 10,000. (Streak counting includes checks before 3,000; the minimum only
delays the stop.) Best and final checkpoints are saved; no prediction
directories during training. If the identity checkpoint stays best, that is the
result.

## Evaluation (after selection is recorded)

A = the same frozen E0 · B = E0 + validation-selected refiner, val and test,
identical clamp + uint16 quantisation for both, scored by the unchanged
`Deraining_Holo/masked_metrics.py`.

* whole-image and foreground (gt > 0.01) PSNR and SSIM;
* per-image differences, paired 95% bootstrap CI (5,000, seed 0) and Wilcoxon,
  number improved / worsened — **sample-level uncertainty for this single run,
  not training-seed robustness**;
* background (gt ≤ 0.01): MAE, RMSE, mean prediction, fraction of pixels above
  0.05 and 0.10, mean positive change, and the correlation of delta with
  (X − Y0) in the background — **a positive correlation means the correction
  moves the background towards the noisy frame**, the sidelobe-reintroduction
  signature;
* parameter count, updates, selected checkpoint, measured training time.

**Practical target: +0.10 dB on test full256** — declared, not guaranteed, and
not the sole criterion. Reported separately: observed mean; reliable (CI
excludes zero AND validation has the same sign); exceeds target.

**Trade-off rule.** A foreground gain with worse background indicators is
reported as a trade-off, never as a plain improvement.

**Protocol note.** Only full256 is evaluated, by design of this brief (the
refiner is defined on whole frames). The project's standing both-protocols rule
concerns the DINO arms; its absence here is a stated decision, not an omission.

## Qualitative question — fixed before training

`qualitative_cases.json` (tracked, written by `select_cases.py` before
training, refuses to regenerate). Validation only, from E0's recorded
predictions. A 48x48 window is a weak-structure window if 0.03 ≤ mean(gt) ≤ 0.25
with ≥ 30% foreground. Inside it: E0 retention r_E = Σ(E0·m)/Σ(gt·m) and noisy
support c_N = corr(blur_σ2(X), gt).

* **recovery**: r_E ≤ 0.6 and c_N ≥ 0.5 — E0 weakened a weak structure the
  measurement still shows. Windows on 228 of 339 images.
* **control**: r_E ≤ 0.6 and c_N ≤ 0.2 — E0 weakened a structure the
  measurement does not clearly show. Windows on 191 images.

Panel: top 4 recovery (6689, 1918, 2708, 2886), top 2 control (2547, 4158), 2
typical (2906, 1076, E0 PSNR nearest the val median). Columns: noisy | target
| E0 | refined | signed correction | both error maps, identical scales and crops.
Plus post-hoc best / median / worst frames, labelled post hoc. These are
illustrations, not the evaluation population.

**Descriptive population comparison, fixed now:** window retention and window
PSNR, E0 vs refined, over ALL recovery windows and ALL control windows. These
windows were selected where E0 did badly, so any correction towards the target
looks good there; **the informative quantity is the contrast** — measurement-
supported recovery predicts a clearly larger gain in recovery than in control
windows. A similar gain in both would not support it.

## Expectation, stated as a hypothesis

A positive full256 change is expected, possibly above +0.10 dB, **for a reason
unrelated to the question**: E0-Fixed never trained on 256 frames (it scores
0.53 dB below the progressive baseline for that reason), and a second stage
trained on whole frames can absorb part of that full-frame deficit. **A
full-frame PSNR gain alone therefore does not show recovery of measurement-
supported structure.** The recovery-vs-control contrast, the background
indicators and the panels are what bear on the actual question. Guess, to be
scored: test full256 +0.1 to +0.5 dB; recovery windows gain more than control
windows; no sidelobe signature (background correlation with X − Y0 ≤ 0).

## Limitations, recorded before results

* The refiner is trained on E0 outputs for images E0 already saw in training
  (as 128 crops), so training residuals may be smaller or differently shaped
  than held-out residuals. Validation and test generalisation are essential.
* The pipeline receives additional supervised training. This does not show the
  same gain could not be obtained by training E0 further, or at 256.
* This one run does not isolate the value of access to the noisy input from the
  refiner architecture itself (no Y0-only control).
* Single seed; bilinear-upsampling backward is non-deterministic on CUDA, so the
  run is not bit-reproducible.

None of those controls is added automatically.
