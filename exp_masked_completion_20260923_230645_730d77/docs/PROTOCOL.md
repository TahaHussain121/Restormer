# Oracle-assisted masked completion on the frozen E0 — protocol

`Holo_E0_oracle_masked_completion_diagnostic`

**Written 2026-09-23, before any training run of this experiment.** One
isolated diagnostic, fully contained in this experiment root. Nothing outside
this root is written, edited or retrained; existing data, splits, the frozen E0
checkpoint, its output caches and the two existing refiners' saved outputs are
used read-only.

This is **not** a repetition of `refiner_e0` (DEVLOG Step 51) or
`refiner_fgbal` (Step 52). Both of those take `[noisy X, E0 output Y0]` and
must discover *where* to correct. This run tests a different question by
supplying the location.

## 1. Hypothesis

A small network trained **explicitly on masked completion** — given the noisy
measurement, an image with a region removed, and a binary mask of that region —
can, when handed an **oracle mask** of E0's actual failures, fill in structure
that E0 removed.

Falsifiable prediction: on the pre-existing validation recovery/control windows,
the completion network raises in-mask agreement with the target **beyond what a
non-learned fill of the same mask achieves**, and it does so more in windows
where the measurement supports the structure than where it does not.

### This is an optimistic diagnostic, not a method

* The oracle mask is computed **from the ground truth**. It is not available at
  inference in any deployable pipeline. Every number produced under it is
  labelled **oracle-assisted**.
* Comparisons with E0, `refiner_e0` and `refiner_fgbal` are **descriptive**:
  this model receives privileged information those models never had. They are
  not a performance ranking.
* The oracle mask marks exactly where target structure exists, so **apparent
  recovery is not evidence that measurement information was used**. That is what
  the non-learned fill control, the shuffled-noisy ablation and the
  recovery-vs-control split are for.
* A successful completion under an oracle mask would **not** establish a
  practical inpainting pipeline; it would only justify asking whether a mask
  could be estimated without the target.

## 2. Architecture and parameter count

`code/completion_arch.py`. The same U-Net topology as the existing refiners
(2→16→32→64, bilinear decoder, zero-initialised signed 1×1 output), with a
**3-channel input** `concat(X, Y_in, M)`:

    delta = Net([X, Y_in, M]);   Y_out = Y_in + M * delta

**118,273 trainable parameters** (the existing refiners have 118,129; the extra
144 are the third input channel of the first convolution). Because the residual
is multiplied by `M`, `Y_out == Y_in` **bit-exactly wherever `M == 0`**, by
construction rather than by a penalty — in the tensors and, after the shared
clamp + ×65535 + round convention, in the saved uint16 PNGs.

Ground truth is used for supervision and to build the synthetic damage. **It is
never an input channel** (checked in the smoke run by comparing every input
channel against the target tensor).

## 3. Synthetic corruption recipe (fixed)

`code/corruption.py`, version `fixed-2026-09-23`. Chosen once from **training
data only** (`results/recipe_calibration.json`: the first 400 training images
and the existing frozen-E0 train cache), never adjusted afterwards. The
calibration measured the regions E0 actually erases under the documented rule
`target > 0.05 and E0 < 0.02`: median component area 26 px (p75 68, p90 179),
median bounding box 6×8 px, elongation median 1.8 / p90 5.4, about 29
components per image covering a median 2.2% of pixels.

Per clean training target `G`:

* 1–5 regions, each an **ellipse** (p 0.35), an axis-aligned **rectangle**
  (p 0.20) or an oriented **band** (p 0.45 — a contiguous missing section across
  a structure); area log-uniform on [25, 900] px, elongation U(1, 5), angle
  U(0, π);
* placement: p 0.75 centred on a **foreground** pixel (`G > 0.01`), p 0.25 on
  **deep background** (`G ≤ 0.01`, ≥ 6 px from any foreground pixel);
* damage, for foreground-placed regions only, with one factor per region:
  p 0.60 **erase** `Y_in = G·u`, `u ~ U(0, 0.05)` (the near-zero suppression
  signature), p 0.40 **attenuate** `u ~ U(0.05, 0.40)`;
* **background-placed regions are not damaged at all**: the mask is supplied
  where nothing is missing, so a mask never implies that an object belongs
  there;
* `M` is the union of all regions, damaged or not; outside `M` the input is
  exactly `G`.

The fixed validation task (`cache/val_synth_masks.npz`, seed 4242, built before
training) applies the same recipe once to all 339 validation targets: mean mask
coverage 1.25% of pixels (min 0.06%, max 4.0%), 3.0 regions per image, 185 of
339 images carry at least one background-only region, no image has an empty
mask.

## 4. Loss and checkpoint selection

**Loss** (per batch):

    L = 0.5 · mean_i[ mean_{p ∈ M_i} |Y_out − G| ]
      + 0.5 · mean_i[ mean_{p ∈ M_i ∩ (G > 0.01)} |Y_out − G| ]

Region means are per image over that image's own region, averaged over the
images whose region is non-empty; an empty region contributes nothing and the
other term is not renormalised. Outside `M` the error is exactly zero by
construction, so **unchanged pixels cannot dominate**; the second term prevents
the **black background inside a mask** from dominating the structure the mask
hides.

**Training budget (fixed, one seed, no sweep):** seed 100; AdamW lr 1e-4, betas
(0.9, 0.999), weight decay 1e-4; batch 16, drop_last; cosine 1e-4 → 1e-6 over
10,000 updates, no warmup; grad-norm clip 1.0; float32 with TF32 explicitly
disabled; no augmentation. Validation every 500 updates on the fixed synthetic
task; minimum 3,000 updates, patience 5 checks, hard maximum 10,000. One GPU
node (v100 or rtx3080), expected well under one hour.

**Checkpoint selection rule, fixed now:** among the step-0 identity checkpoint
and every 500-update checkpoint, the highest **mean per-image in-mask PSNR
(uint16 convention) on the fixed synthetic validation task**. The
actual-E0-failure diagnostic is **never** used for selection. Every checkpoint is
saved under a unique step-specific name `checkpoints/ckpt_update_XXXXXX.pth`
with exclusive creation; there is no `best.pth` or `latest.pth`, and the
selection is recorded in `results/checkpoint_selection.json`.

## 5. Oracle mask definition (fixed before any new prediction is inspected)

The **documented missing-structure rule** of the existing analysis
(`refiner_fgbal/analyze_windows_local.py`, constants `MISS_GT = 0.05`,
`MISS_E0 = 0.02`) is verified in the code and reused unchanged:

    oracle mask  =  target > 0.05  AND  E0 < 0.02

computed from the ground truth and E0's quantised prediction, and **restricted
to the fixed 48×48 window** of each case. Three mask variants, all fixed here:

1. **pure oracle** — the rule above, inside the window;
2. **expanded** — the pure oracle mask dilated by 3 px (3×3 kernel, 3
   iterations), clipped to the window: an imperfect mask that includes pixels
   where nothing is missing, to probe false additions;
3. **background control** — a mask of the same pixel count as that window's
   oracle mask, placed elsewhere in the same frame on deep background
   (`target ≤ 0.01`, ≥ 6 px from any foreground pixel), drawn with a fixed seed
   (7) from a deterministic location: it probes whether the model invents
   structure wherever a mask is supplied.

At diagnostic time the model receives `[noisy X, E0 output, mask]`. **It never
receives clean target pixel values as an image input.** Windows with an empty
mask are excluded and counted.

## 6. Controls and metrics (defined before any output is viewed)

Methods scored inside the same mask region:

| tag | method | note |
|---|---|---|
| E0 | the frozen baseline, unchanged | read-only |
| REF | `refiner_e0`'s saved validation output | read-only, descriptive |
| FGB | `refiner_fgbal`'s saved validation output | read-only, descriptive |
| FILL-RING | non-learned: fill the mask with the median of E0 in a 3-px ring around the mask | same mask, no learning |
| FILL-TELEA | non-learned: `cv2.inpaint` (Telea, radius 3) of E0 over the mask | same mask, no learning |
| COMP | the selected completion checkpoint, oracle mask | **oracle-assisted** |
| COMP-SHUF | the same checkpoint with another image's noisy frame | inference-only sensitivity diagnostic |

**Metrics, per window** (all on quantised uint16 outputs, the study's
convention, restricted to the mask unless stated):

* *error*: in-mask MAE and in-mask PSNR;
* *intensity retention*: fill ratio `Σ pred / Σ target`, and the fraction of
  mask pixels above 0.02 (E0's suppression threshold) — **intensity measures, a
  brightening raises them**;
* *structure / shape*, defined so that a uniform brightening cannot raise them:
  **target correlation** (Pearson `corr(pred, target)` over mask pixels;
  undefined when either is constant — such windows are counted and excluded,
  never scored as 0), **detail correlation** (Pearson correlation of the
  high-pass residuals `img − box5(img)` over mask pixels), and **gradient NCC**
  (normalised cross-correlation of Sobel gradient magnitudes over mask pixels).
  Brightening is *not* treated as geometric recovery anywhere in the report;
* *overshoot*: fraction of mask pixels with `pred − target > 0.05` and the mean
  positive excess;
* *outside the mask*: max absolute change relative to the supplied image over
  the whole frame outside `M` (expected exactly 0 for COMP, reported as
  measured);
* *background-mask control*: mean added intensity and fraction of pixels above
  0.05 inside a mask placed on true background;
* *expanded-mask control*: the same metrics plus, in the added ring only, the
  false-addition fraction (`pred > 0.05` where `target ≤ 0.05`).

**Reporting rules:** recovery (228 windows) and control (191 windows) groups are
reported **separately and never pooled**; eligible counts are stated for every
number; paired differences carry a 95% bootstrap CI resampled **at image level**
(5,000 resamples, seed 0), which is sample uncertainty for **one training seed**
and measures no training variability. The existing recovery/control distinction
is an **operational grouping from the existing proxy** (blurred-noisy
correlation), not proof that measurement information is present or absent.

**Qualitative:** the eight fixed cases of
`refiner_e0/qualitative_cases.json` (4 recovery, 2 control, 2 typical), reused
without selection, shown as noisy | target | E0 | mask | completion |
correction with identical display scales.

## 7. Split usage

Train 6,101 / validation 339 / test 338, unchanged membership and
preprocessing (uint16 / 65535). Training uses **training targets only**.
Selection uses the fixed synthetic validation task. The actual-failure
diagnostic uses **validation** windows only and is **exploratory**.
**The test split is not read in this experiment.**

## 8. Distribution mismatch, stated in advance

Training damages a **clean target**: outside the mask the context is noiseless
ground truth, the damage is a sharp region with a constant factor, and the
missing content is statistically whatever the recipe removed. At the diagnostic
the context outside the mask is an **E0 output** — blurrier, hazy in the
background, with its own residual errors — and the missing content is whatever
E0 actually failed to reconstruct, in regions shaped by E0's own behaviour
rather than by the recipe. A failure to transfer is therefore **consistent with
either** an inability to infer the structure **or** with this mismatch alone;
the report will not attribute it to one cause.

## 9. Limitations recorded before results

* **Privileged information.** The mask comes from the ground truth. No number
  here is achievable at inference today.
* **Oracle masks exclude true background by construction**, so false additions
  cannot appear inside a pure oracle mask; the expanded and background-mask
  controls exist for that reason and are the only evidence about imperfect
  masks.
* **One seed, one configuration.** No training variability is measured.
* The training set is the one E0 itself was trained on (as 128 crops).
* The recovery/control grouping is an existing proxy, and the two groups do not
  start from the same place (documented in Step 52).
* Bilinear-upsampling backward is non-deterministic on CUDA: the run is not
  bit-reproducible.

## 10. Criteria for continuing or stopping this direction

Decided before the results, and reported against:

* **Continue-worthy** only if *all three* hold: (a) the model solves the
  synthetic task clearly (in-mask PSNR well above the damaged input and above
  both non-learned fills); (b) under the oracle mask on **actual** E0 failures
  it beats **FILL-RING and FILL-TELEA on the structure metrics**, not only on
  intensity; (c) it does not invent comparable structure in the background-mask
  control and does not fill the expanded-mask ring with false additions.
* **Stop** if the transfer fails, or if the gain over the non-learned fills is
  confined to intensity, or if the background/expanded controls show the model
  fills any mask it is given. In that case the direction is recorded as not
  justified, and no mask-estimation follow-up is proposed.
* No follow-up training is launched automatically in either case.
