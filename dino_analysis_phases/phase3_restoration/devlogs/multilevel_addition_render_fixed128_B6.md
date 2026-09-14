# multilevel-addition-render — FINAL EXPERIMENT A

`Holo_multilevel_addition_render_fixed128_B6`

**Pre-registration, written 2026-09-14, before any training.** This is an
EXPLORATORY follow-up, informed by earlier results in this study (post-latent
addition beat pre-latent addition by +0.306 dB on test). It was designed in a
ChatGPT-assisted discussion and authorised by the author as one of the two final
training runs; it is not a supervisor instruction. Together with its matched
partner (`multilevel_aca_render_fixed128_B6.md`) it CLOSES the architecture
training study.

## What changes

The B6 render prior, extracted ONCE per input and reused, added at three sites
through separate zero-initialised 1x1 projections:

| site | where | width | projection |
|---|---|---|---|
| pl | after the eight latent blocks | 384 | 768->384 (the reference arm's `P`) |
| d3 | decoder level 3 input, after skip concatenation and channel reduction | 192 | 768->192 |
| d2 | decoder level 2 input, after skip concatenation and channel reduction | 96 | 768->96 |

No pre-latent injection, no decoder-level-1 injection.

**Spatial mapping (new, documented).** Each projection is applied at the native
prior grid and expanded by NEAREST-NEIGHBOUR upsampling, 2x at d3 and 4x at d2.
The project's original "no feature-grid resizing" condition does NOT hold at the
decoder sites. The factors are identical at train128 (16x16 grid) and eval256
(32x32 grid), but matching ratios does not prove crop/full feature invariance.

**Capacity.** Added trainable parameters over E0: **516,768**; total trainable
**26,640,820**; frozen DINO counted separately. Reference `postlatent-render`:
+295,296.

Everything else is `postlatent-render`'s recipe: fixed 128 crops, batch 8, seed
100, 300k iterations, L1, AdamW, the schedule, clipping, validation frequency,
checkpoint policy, trained from scratch, a100.

## Pre-registered outcome and comparisons

**Primary outcome:** mean full256 PSNR on the test split, as a paired
per-image difference against the relevant comparator.

**Practical improvement threshold: +0.10 dB**, declared here and distinct from
statistical evidence. Every comparison reports separately: the observed paired
mean; the 95% paired bootstrap interval and Wilcoxon p; replication in sign on
validation; and whether the mean exceeds +0.10.

Also reported: whole-image and foreground PSNR/SSIM, matched crop128, parameter
counts, and representative improvements and failures. **A crop128 decline larger
than 0.10 dB is a practical trade-off, reported explicitly and never hidden by a
full256 gain.**

Comparisons this arm enters (numbering shared with experiment B):

  * **A. multi-level addition − post-latent addition** — does repeated delivery
    help the additive operator?
  * **C. multi-level ACA − multi-level addition** — operator at matched sites.
  * **D. (multi-level ACA − post-latent ACA) − (multi-level addition −
    post-latent addition)** — does ACA gain MORE than addition from the extra
    sites? Per-image difference-in-differences on the same images.

**Single seed.** Sample-level intervals measure image-to-image variation, NOT
training-seed variability, and are reported as such.

**Expectation, stated as a hypothesis:** limited incremental benefit, because the
decoder sites receive the same prior again. Reusing the same features does NOT
imply repeated delivery cannot help — it may still change how easily the decoder
stages use the information, which is what this arm tests. A non-significant
result is not evidence of equivalence.

## Interpretation rules, fixed now

  * Both operators improving similarly supports repeated guidance and does not
    establish an ACA advantage.
  * Little improvement supports the adequacy of the single-location design among
    the tested alternatives — not a universal ceiling.
  * Protocol disagreement is a trade-off, reported as such.
  * This tests the multi-level PACKAGE, not the contribution of each decoder
    stage.

## Monitoring

The stability gate keeps its three keys at the post-latent site, as the
reference arm measures them. All three sites are recorded separately (feature
norm, update norm, ratio, finiteness) and ride in any stability record; no new
threshold is defined on the decoder sites.
