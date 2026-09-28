# Oracle-assisted masked completion on the frozen E0 — result

`Holo_E0_oracle_masked_completion_diagnostic`, run 2026-09-23. Protocol fixed
before training in `docs/PROTOCOL.md`; isolation and audit coverage in
`docs/ISOLATION.md`. One bounded run, one seed, validation split only — **the
test split was not read**.

**Every number below that involves an actual E0 failure is ORACLE-ASSISTED**:
the mask comes from the ground truth (`target > 0.05 and E0 < 0.02`, the
documented missing-structure rule) and is not available at inference in any
deployable pipeline. Comparisons with E0, `refiner_e0` (REF) and
`refiner_fgbal` (FGB) are **descriptive**, not a performance ranking: this model
is told where to fill and those models were not. The diagnostic on actual
failures is **exploratory**.

## What was run

A 118,273-parameter completion U-Net (the existing refiners' topology with a
third input channel) takes `[noisy X, damaged image Y_in, binary mask M]` and
predicts `Y_out = Y_in + M · delta`, so **outside the mask the output is
bit-identical to the input**, in the tensors and in the saved uint16 PNGs.
It was trained only on **synthetically damaged clean training targets** (fixed
recipe, calibrated on training data only), never on E0 outputs; the frozen E0
was never instantiated during training and its outputs were read from the
existing validated cache.

| | |
|---|---|
| jobs | smoke + training **1820611** (a100, 3.4 min, 1.4 GiB), evaluation **1820618** (a100, 2 × 233 s) |
| smoke checks | **28 / 28 passed** on real data (`results/smoke_checks.json`) |
| training | early stop at 5,000 updates (5 checks without improvement), 13.1 epochs |
| selected checkpoint | `checkpoints/ckpt_update_002500.pth`, by the pre-registered rule |
| windows | 228 recovery / 188 control (3 of 191 control windows have an empty oracle mask and are excluded and counted) |

## 1. Can the model solve synthetic completion? **Yes — but so can a parameter-free interpolant**

Fixed synthetic validation task, 339 images, metrics inside the mask:

| method | in-mask PSNR | fill ratio | target corr | grad NCC | detail corr |
|---|---|---|---|---|---|
| damaged input | 10.06 | 0.12 | 0.760 | 0.408 | −0.087 |
| FILL-RING (constant) | 12.95 | 0.59 | 0.000 | 0.337 | −0.170 |
| FILL-TELEA | 24.03 | 1.08 | 0.920 | 0.808 | 0.232 |
| FILL-DIFF (Laplace) | 25.44 | 1.17 | 0.935 | 0.863 | 0.503 |
| **COMP (learned)** | **26.45** | 1.03 | **0.944** | **0.875** | 0.225 |

Paired, per image: COMP − damaged input **+14.65 dB** [+14.04, +15.26], better
in 320/320 images where PSNR is finite; grad NCC +0.467 [+0.451, +0.483].
The model clearly performs the task it was trained on.

But against the **non-learned Laplace fill** on the same masks, COMP gains only
**+1.08 dB** [+0.30, +1.99] (better in 179/330), target correlation **+0.009**
[+0.0002, +0.020], grad NCC +0.012 [+0.004, +0.022] — and it is **worse on
detail correlation**, −0.279 [−0.312, −0.245], worse in 277 of 326 images.
Most of what the learned network achieves on its own training task is
achievable by interpolating from the mask boundary.

*Reported honestly:* PSNR is undefined (infinite) for 7–19 images per method
whose in-mask error is exactly zero; those images are excluded from the PSNR
means and the counts above (`n_defined` is recorded for every metric).

## 2. Does it transfer to actual E0 failures under an oracle mask? **Partly, and not as structure**

Validation windows fixed before `refiner_e0` was trained, pure oracle mask:

**Recovery windows (n = 228)** — COMP versus the frozen E0:

| metric | E0 | COMP | paired delta [95% CI] | better/worse |
|---|---|---|---|---|
| in-mask PSNR | 14.116 | 14.972 | **+0.856** [+0.769, +0.950] | 227 / 1 |
| fill ratio (Σpred/Σtarget) | 0.013 | 0.118 | +0.105 [+0.094, +0.117] | 227 / 1 |
| fraction above 0.02 | 0.000 | 0.328 | +0.328 [+0.305, +0.352] | 226 / 0 |
| target correlation | 0.230 | 0.341 | +0.111 [+0.081, +0.142] | 153 / 75 |
| gradient NCC | 0.478 | 0.650 | +0.172 [+0.158, +0.187] | 215 / 13 |
| **detail correlation** | 0.140 | −0.083 | **−0.223** [−0.272, −0.177] | 56 / 172 |
| overshoot (pred − target > 0.05) | 0.000 | 0.0024 | +0.0024 [+0.0009, +0.0045] | — |
| max change outside the mask | — | **0.0000** | exact preservation | 228 / 228 |

So the model does put intensity back: it lifts a third of the erased pixels
above E0's 0.02 suppression level and restores **12% of the missing intensity**
(88% is still missing). The two coarse agreement measures rise; the fine-detail
measure **falls below E0's own value**.

**Control windows (n = 188)** — the same comparison: PSNR **+0.825**
[+0.695, +0.969], fill ratio +0.106, fraction above 0.02 +0.327, target
correlation +0.059, gradient NCC +0.116. The **recovery-minus-control contrast**
is therefore +0.03 dB in PSNR, −0.001 in fill ratio and +0.056 in gradient NCC:
**no contrast worth the name**, the same null the two previous refiners
produced. Windows where the blurred noisy frame still shows the structure are
filled no better than windows where it does not.

*(The recovery/control split is an operational grouping from the existing
proxy, and the two groups do not start from the same place — see Step 52 — so
this is weak evidence about measurement information on its own. It agrees with
the shuffled-noisy ablation below, which is direct.)*

## 3. Does it beat simple filling in structure? **No — it beats it in brightness**

COMP versus **FILL-DIFF** (parameter-free Laplace fill, same mask), recovery
windows, n = 228:

| metric | FILL-DIFF | COMP | paired delta [95% CI] | better/worse |
|---|---|---|---|---|
| in-mask PSNR | 14.619 | 14.972 | +0.352 [+0.273, +0.436] | 162 / 66 |
| fill ratio | 0.069 | 0.118 | +0.049 [+0.039, +0.060] | 169 / 59 |
| fraction above 0.02 | 0.222 | 0.328 | +0.106 [+0.082, +0.130] | 170 / 54 |
| **target correlation** | 0.543 | 0.341 | **−0.201** [−0.229, −0.174] | 30 / 198 |
| **detail correlation** | 0.238 | −0.083 | **−0.321** [−0.365, −0.279] | 27 / 201 |
| **gradient NCC** | 0.746 | 0.650 | **−0.096** [−0.112, −0.079] | 54 / 174 |

The control group gives the same picture (PSNR +0.324, target correlation
−0.283, gradient NCC −0.146). Against the crude constant **FILL-RING** the
learned model wins on everything except detail correlation.

**This is the central result.** The learned completion is *brighter and closer
in PSNR* than a fill that knows nothing but the mask boundary, and *less
faithful to the target's structure* on all three structure measures that a
uniform brightening cannot influence. Every structure gain COMP shows against
E0 is smaller than what the interpolant gets for free from the mask geometry.
The qualitative panel shows the mechanism: inside the mask the completion draws
a dim version of **the mask's own shape**.

## 4. Sensitivity to the noisy conditioning and to mask errors

**Shuffled noisy frames** (inference-only, each image paired with the next
image's noisy frame — a distribution-shift diagnostic, not a clean ablation):
recovery windows PSNR **−0.030 dB** [−0.101, +0.036], target correlation
−0.027 [−0.052, −0.001]; control windows PSNR **+0.113** [+0.004, +0.233],
correlation +0.019 [−0.007, +0.046]. **Replacing the measurement with an
unrelated image changes essentially nothing**, and on control windows it is
marginally better. Whatever the model fills with, it is not the measurement.

**Background-mask control** (a mask of the same size on deep background,
fixed seed): the fraction of masked pixels above 0.02 rises from E0's 1.45% to
**3.14%** [+0.83, +2.72 points] and the mean positive excess by +0.0027, on
recovery-window images (control-window images: 1.46% → 3.51%). Small, but not
zero: the model adds a little wherever a mask is supplied, even on true
background. Against the interpolants it is well behaved there (FILL-DIFF lifts
32% of those pixels above 0.02).

**Expanded mask** (oracle dilated by 3 px). In the added ring alone, where by
construction little is missing: overshoot rises from E0's 1.35% of pixels to
**13.3%** [+10.7, +13.2 points], and false additions (`pred > 0.05` where
`target ≤ 0.05`) from 1.4% to **10.1%** (control ring 1.7% → 13.1%). **A mask
that is three pixels too generous makes the model paint structure that is not
there.** Any realistic mask estimator would be at least this imprecise.

## 5. Should this direction continue? **No**

The protocol's continue criteria required all three of: solving the synthetic
task, beating the non-learned fills **on structure** under the oracle mask, and
not filling background or expanded masks. Scored:

| criterion | verdict |
|---|---|
| (a) solves the synthetic task | **met** (+14.65 dB over the damaged input) — but only +1.08 dB over a parameter-free fill, and worse on detail correlation |
| (b) beats FILL-RING and FILL-DIFF on structure at actual failures | **not met** — worse than FILL-DIFF on all three structure metrics in both groups |
| (c) does not invent structure under imperfect masks | **not met** — 10–13% false additions in a 3-px ring; a small but consistent addition on pure background masks |

**Recommendation: stop.** An oracle mask — ground-truth information no
inference pipeline has — buys +0.86 dB inside the erased regions, recovers 12%
of the missing intensity, and produces *less* structural agreement than
Laplace interpolation from the mask boundary. The gain is not driven by the
measurement (shuffling the noisy frame changes nothing) and degrades quickly
when the mask is imperfect. Investigating a mask that could be estimated at
inference is therefore not justified by this evidence: even a perfect mask does
not deliver structure. No follow-up training was launched.

This does **not** show that the information is absent from the measurement,
that a larger network or a different objective could not do better, or that
inpainting is impossible for this data. It shows that this completion model,
under the most favourable mask it could ever receive, does not recover E0's
missing structures.

## Distribution mismatch (stated in the protocol, before results)

Training damaged **clean targets**: outside the mask the context was noiseless
ground truth and the damage was a region multiplied by a constant factor. At
the diagnostic the context is an **E0 output** — blurrier, hazy in the
background, with its own errors — and the missing content is whatever E0 failed
to reconstruct. The failure to transfer is therefore consistent with **either**
an inability to infer the structure **or** with this mismatch alone, and no
attribution to one cause is made. What the mismatch does **not** explain is the
comparison in section 3: FILL-DIFF faces no training distribution at all and
still agrees better with the target.

## Qualitative cases

`results/figures/fixed_cases_oracle.png` — the eight cases fixed in
`refiner_e0/qualitative_cases.json`, reused without selection: noisy | target |
E0 | oracle mask | FILL-RING | FILL-DIFF | completion | correction, identical
scales (gamma 0.5 on [0,1]; correction ±0.15).

* **Recovery 6689, 1918, 2886:** the completion does add faint stripe-shaped
  structure — and the stripes it draws are the stripes **of the mask**, at
  roughly a tenth of the target intensity (fill ratios 0.046, 0.033, 0.216).
* **Recovery 2708:** the mask is one large blob and the completion fills a dim
  uniform blob (fill 0.053).
* **Control 2547 and 4158:** the two windows where the measurement gives no
  support are filled **more strongly** than any recovery case (fill 0.454 and
  0.162, PSNR +3.4 and +1.4 dB). This is the clearest single illustration that
  the filling follows the mask, not the measurement.
* **Typical 2906 and 1076:** the oracle mask is **empty** (E0 erased nothing
  there), and the completion output is identical to E0 — the empty-mask path,
  also checked in the smoke run.

## Checkpoint selection — a flaw found after the fact

The pre-registered selection metric (mean per-image in-mask PSNR on the
synthetic task) is **dominated by images whose mask lies on background**: 15 of
339 validation masks contain no target intensity at all, and at any given
checkpoint a handful of images have exactly zero in-mask error, which the
training loop scored as ~300 dB. The metric therefore oscillated between 27.9
and 32.1 dB while in-mask MAE (0.0463 → 0.0381) and in-mask **foreground** PSNR
(23.2 → 24.7) improved monotonically to the last check. The rule as registered
selected **update 2,500**, and that is the checkpoint of record.

To check that the conclusions are not an artefact of that noisy rule, the
**final checkpoint (update 5,000)** was evaluated as a clearly labelled post-hoc
sensitivity run, chosen on validation *synthetic* metrics only and never on the
diagnostic (`results/*_posthoc5000.*`). That checkpoint is **better on the synthetic task** (in-mask PSNR 28.70 vs
26.45, detail correlation 0.413 vs 0.225, target correlation 0.956 vs 0.944)
and **slightly worse on actual E0 failures**: versus E0 on recovery windows
+0.721 dB [+0.651, +0.798] (against +0.856), target correlation +0.083
[+0.054, +0.113] (against +0.111), fill ratio +0.088 (against +0.105). Against
FILL-DIFF it remains worse on every structure metric (target correlation −0.230
[−0.263, −0.197], detail correlation −0.250, gradient NCC −0.100). Shuffling
the noisy frame still changes nothing (+0.038 dB [−0.023, +0.100]); the ring
false-addition rate is 8.4% (against E0's 1.4%) and the background-mask
fraction above 0.02 is 2.4% (against E0's 1.45%).

**Training further on the synthetic task therefore improved synthetic
completion and did not improve transfer.** Every conclusion in sections 2–5
holds for both checkpoints, so none of them depends on the flawed selection
metric.

One difference worth recording: at update 5,000 the target-correlation gain over
E0 is +0.083 [+0.054, +0.113] on recovery windows but +0.009 [−0.023, +0.042]
on control windows — a hint of the contrast that the pre-registered checkpoint
does not show, and which the PSNR gain contradicts (recovery +0.721 versus
control +0.782). It is one metric out of several at a post-hoc checkpoint, and
it is reported as such, not as evidence of measurement-supported recovery.

## Limitations

* **Privileged information.** The mask is derived from the ground truth. No
  number in section 2–4 is achievable at inference today.
* Oracle masks exclude true background by construction, so false additions
  cannot show up inside them; the expanded and background-mask controls are the
  only evidence about imperfect masks, and both are unfavourable.
* **One seed, one configuration, one training run.** The intervals resample
  windows/images; they measure no training variability.
* The shuffled-noisy ablation changes the input distribution as well as the
  content, so it bounds rather than isolates the use of the measurement.
* The recovery/control grouping is the existing proxy, and the groups do not
  start from the same place.
* Training used the split E0 itself was trained on (as 128 crops); the
  diagnostic is on held-out validation frames.
* Bilinear-upsampling backward is non-deterministic on CUDA: the run is not
  bit-reproducible.
* The job ran on an a100 because v100 and rtx3080 were fully allocated. The GPU
  type affects no reproduction claim here: E0 was never executed, only its
  existing cache was read, and float32 was enforced with TF32 disabled.
