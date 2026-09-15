# Foreground-balanced residual refiner — ONE follow-up to refiner_e0

`Holo_E0_frozen_noisy_output_fgbalanced_refiner`

**Pre-registration, written 2026-09-15, before training.** An **exploratory
follow-up informed by previously observed results** (refiner_e0, DEVLOG Step
51). Requested by the author as one run, no sweep; the brief was drafted in a
ChatGPT-assisted discussion. Outside the DINO study; the two final chains are
not touched.

**The test split is not a new untouched holdout.** It was already read for
refiner_e0, and the idea for this run comes from that reading.

## Question

Did refiner_e0 fail to recover weak object structure **partly because** its
training loss (whole-image L1) and its checkpoint selection (whole-image PSNR)
favoured background improvement? Not assumed. This run changes the loss AND the
selection rule together, so it tests an **objective-alignment package**; a
difference is not attributed to the loss alone.

## What is identical to refiner_e0 (checked mechanically)

`fgbal_common.load_configs` refuses to train if any shared config key differs
except `name`, `loss`, `selection_metric`. Identical: frozen E0 checkpoint
268,000 (md5 `a7d094b5…`, eval mode, never built during training, not in the
optimizer); the verified float32 caches of refiner_e0 (reused, sha256 checked;
same 6,101 / 339 ids, same preprocessing); full 256x256 frames; inputs [X, Y0];
Yhat = Y0 + refiner(X, Y0); the same U-Net, **118,129** parameters, and a fresh
init by the same procedure with seed 100 (not refiner_e0's trained weights). On
the login node it matches refiner_e0's step-0 weights to float32 rounding (the
10 hidden conv weights differ by at most 3.3e-7; biases and the zero output
layer are bit-identical); refiner_e0 drew its init on a v100 node's CPU. The
training job re-measures this on its own node, records it, and refuses to train
if any difference exceeds 1e-5; AdamW lr 1e-4, betas (0.9, 0.999), weight decay
1e-4; cosine to 1e-6 over 10,000 updates, no warmup; batch 16; clip 1.0;
float32, TF32 off; no augmentation; drop_last; validation every 500 on all 339
frames; minimum 3,000 updates, patience 5, maximum 10,000; unclamped training
prediction; GPU v100 (as refiner_e0's training job 1813217).

## What changes

**1. Loss.**

    L = 0.5 * mean_i[ mean_{p in F_i} |Yhat - Y| ] + 0.5 * mean_i[ mean_{p in B_i} |Yhat - Y| ]
    F_i = { p : Y_i(p) > 0.01 },  B_i = complement,  Y = uint16 target / 65535

Region means are per image over that image's own region, then averaged over the
images whose region is non-empty; an image with an empty region contributes
nothing to that term; if a region is empty in every image of a batch, its term
is omitted without renormalising. No NaN, no dilution by the other region's
pixel count. The mask is computed from the target inside the loss only.

**Mask choice, checked on TRAINING targets only (6,101 frames), before
training:** gt > 0.01 is the study's existing foreground (masked_metrics, as
run_evaluate.sh runs it), kept for comparability. Coverage: 32.0% of pixels on
average (min 3.5%, max 74.8%); no training image has an empty foreground or an
empty background. 67.4% of target pixels are exactly zero. The faint band
0.01 < gt ≤ 0.25 (12.0% of pixels) is inside the mask. Pixels with
0 < gt ≤ 0.01 are 0.62% of pixels and carry 0.022% of the target intensity, and
97.4% of them lie within 3 px of the mask (fade tails); no target value more
than 3 px from the mask reaches 0.01. The pre-declared weak-structure windows
require ≥ 30% of pixels above 0.01. **The mask does not exclude the targeted
structures, so it is used unchanged.** Not tuned on validation or test, not
derived from any model output.

**2. Checkpoint selection (validation only).** Candidates: the step-0 identity
(= E0 exactly) and every validation checkpoint. A checkpoint is **eligible**
only if its validation background MAE **and** background RMSE (gt ≤ 0.01,
uint16 convention, per-image then averaged) are no worse than E0's, computed by
the identical code at step 0. Among eligible checkpoints, the highest
validation **foreground PSNR** is selected if it exceeds E0's; otherwise **E0 is
retained and the experiment did not produce a useful refiner**.

**3. Derived: early stopping** keeps its mechanics but counts checks without a
new eligible foreground-PSNR best, because in both runs the stopping rule
follows the selection rule. Stated as the one derived difference.

**4. Logging only:** a checkpoint at every validation, so the original rule
(max validation whole-image PSNR) can be applied to this same run and reported
next to the new rule. refiner_e0 saved only three checkpoints; its validation
log records whole-image PSNR, foreground PSNR and background MAE at every check,
but not background RMSE.

**The new rule applied to refiner_e0, from its log (recorded before training
this run):** no check exceeds E0's validation foreground PSNR 17.978 (best
17.887 at 3,500), so the rule selects **E0**. The missing RMSE does not matter,
because the foreground condition fails at every check. The original rule
applied to this run is computed from this run's log.

## Evaluation (once, after selection)

Three methods on val and test, full256, identical clamp + uint16 + unchanged
`masked_metrics.py`: **A** frozen E0, **B** refiner_e0 (its recorded outputs),
**C** this run's selected checkpoint. Whole-image PSNR/SSIM, foreground
PSNR/SSIM, background MAE/RMSE and fractions above 0.05/0.10; paired per-image
differences C−A and C−B with bootstrap 95% CIs (5,000, seed 0) and Wilcoxon,
improved/worsened counts. **The intervals reflect image sampling only; one
training seed per refiner measures no training variability.**

**Primary recovery measures:**
1. test foreground PSNR, C − A (practical target **+0.10 dB**, reported
   separately as observed / reliable — CI excludes zero AND validation same
   sign / exceeds);
2. in the **228 recovery and 191 control windows fixed before refiner_e0 was
   trained** (validation), per-window **target agreement** (Pearson correlation
   with the target — gain- and offset-invariant, so brightening alone cannot
   raise it), retention, window PSNR, and false-addition fraction (pixels where
   prediction − target > 0.05);
3. the **recovery-minus-control contrast** of those changes relative to E0.
   Measurement-supported recovery requires recovery windows to improve in target
   agreement AND PSNR clearly more than control windows.

**Background constraints:** eligibility on validation (above); on test, background
MAE/RMSE and fractions reported for all three, plus the correlation of the
correction with (X − Y0) in the background (> 0 = moving towards the noisy frame,
the sidelobe signature).

**Useful outcome (declared):** test foreground PSNR improves over E0 reliably,
the recovery-window contrast is positive in target agreement and PSNR, and test
background MAE/RMSE are not worse than E0's. Otherwise the wording is: *"This
foreground-balanced refiner did not improve structure recovery under the tested
setup."* No conclusion that the information is absent, that refinement is
impossible, or that a larger network would solve it.

## Expected failure modes

* **Simple brightening.** E0 under-predicts the foreground on average (mean E0
  − target −0.013 on test, −0.017 val), so a near-uniform foreground brightening
  could raise foreground PSNR with no structural gain. Detected by mean signed
  foreground change, fraction brighter, unchanged window target correlation, and
  no recovery-vs-control contrast.
* **Background spill.** Brightening at object edges leaks into the background →
  background MAE/RMSE above E0 → ineligible → E0 fallback.
* **Sidelobe / false structure.** Brightening follows noisy-frame texture:
  positive background correlation with (X − Y0), false additions in control
  windows.
* **Nothing eligible improves** → E0 retained.

**Guess, to be scored:** the most likely outcome is either the E0 fallback or a
small foreground gain carried mainly by brightening, without a recovery-over-
control contrast in target agreement.

## Limitations

As refiner_e0: training outputs come from images E0 saw; extra supervised
training; no Y0-only control; one seed, non-deterministic bilinear backward.
Plus: loss and selection change together; the test split was already inspected.

Only full256 is evaluated, as in refiner_e0 and by the brief. The fixed cases
and populations are reused; no example is chosen after seeing this run, except a
post-hoc figure labelled as such.
