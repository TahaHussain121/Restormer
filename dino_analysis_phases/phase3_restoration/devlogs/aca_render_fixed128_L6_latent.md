# aca-L6 — the FUSION-OPERATOR ablation of addition-render

`Holo_aca_render_fixed128_L6_latent`

**RUN ORDER: FIRST. This is the arm that matters most.**

Scope: **this arm only**. Every other Phase-3 arm is read-only from here;
affm-render and dinolight-render were TRAINING while this was built.
Written **before the run**, so the prediction below is a pre-registration.

## THE DECISION

**addition-render with ONLY the fusion operator changed.**

```
addition-render   D = centered B6   guided = F + P(D)
aca-L6            D = centered B6   guided = ACA(F, P(D))
```

Same layer (B6), same source (render), same centering mean, same injection
point. **Nothing else in Phase 3 gives a clean ACA-vs-addition comparison** —
dinolight-render changes the operator *and* the layer count, and
crossattn-render changed the operator, the direction and the attention type at
once. `dino_prior` is **inherited unchanged** from `RestormerDinoSpatialRender`,
so the prior this arm sees is byte-for-byte the prior addition-render sees.

**NO AFFM, DELIBERATELY.** With a single layer the AFFM softmax runs over an
axis of length 1 and is **identically 1.0** — a no-op. Including a degenerate
AFFM would add 769 dead parameters and one meaningless observation series.
The centered B6 grid goes straight to `P`. **Do not read this arm as
inconsistent with aca-L36 / aca-L6912 / dinolight-render**: those fuse 2, 3 and
4 layers and need AFFM; this one has nothing to fuse. The smoke test asserts
`self.affm` does not exist.

---

## TWO COMPARISONS, TWO DIFFERENT CAVEATS — DO NOT CONFLATE THEM

**vs addition-render — ONE factor, but CAPACITY CONFOUNDED.** The operator is
the only thing that changes, which is what makes this arm worth running. But it
carries **+1,349,773 parameters against addition-render's +295,296 — 4.57x**.
A win **cannot** be attributed to the operator alone.

**vs the ACA ladder — capacity is NOT a confound.** aca-L6 / aca-L36 /
aca-L6912 / dinolight-render differ only by AFFM scoring convs, **769 per
layer**. Measured spread across the whole ladder: **0.227%**. A difference
*across this ladder* is therefore **not** attributable to capacity.

Every report of this arm must say which of the two comparisons it is making.

---

## PARAMETERS

```
  dinolight-render measured delta over E0        = 1,352,849
  minus 4 absent AFFM scoring convs  4 x 769     =    -3,076
  -----------------------------------------------------------
  aca-L6 delta over E0                           = 1,349,773
```
Measured: E0 26,124,052 -> **27,473,825**. Matches exactly.

---

## PRE-REGISTERED PREDICTION — written before the run

**aca-L6 does not beat addition-render's 24.081 dB by more than 0.30 dB.**

Rationale: the fusion axis has produced no operator advantage on this problem
so far — concat vs addition was +0.023 val / -0.016 test with bootstrap CIs
spanning zero on both, and the two live multi-layer arms are tracking addition
rather than separating from it.

**CONTRARY EVIDENCE ON RECORD:** DINOLight's own ablation credits their ACA with
**+2.80 dB** over a no-DINO baseline. If this arm wins, **report it as a
surprise**, and remember the capacity caveat above before crediting the
operator.

**SECONDARY OUTCOME, INDEPENDENT OF PSNR.** `alpha` is a gate the network can
**close**. Decay toward 0 = "the prior does not help, falling back to plain
self-attention", a clean interpretable negative that crossattn could not
produce. Growth = the path is being used. Report the trajectory either way.

---

## WHAT IS HELD IDENTICAL

Splits, fixed 128 crop, augmentation, seed 100, Restormer hyper-parameters,
AdamW (lr 3e-4, wd 1e-4), `CosineAnnealingRestartCyclicLR` periods
[92000, 208000] summing to exactly 300,000, total_iter 300000, batch 8, L1,
val_freq 4000, checkpoint every 2000, the ACA block itself, and the stability
gate (cap 10, ratio rules from iteration 5000, NaN/Inf from 1).

**MEANS ARE SHARED, NOT RECOMPUTED.** This arm loads the same per-layer mean
files `dinolight-render` and `affm-render` load. Identical centering is what
keeps the ladder comparable on the feature side. Verified in the smoke test:
the B6 mean is **byte-identical (max abs diff 0.000e+00)** to the file
`addition-render` trains with, in both regimes.

**300k is not negotiable.** The periods sum to exactly 300,000, so a truncated
run is a different LR trajectory and would not be comparable to the finished
arms. No compression, no shortened schedule, no concurrent arms in one job.

---

## INITIALISATION AND THE THREE-STEP STAIRCASE

* `project_out` zero -> `guided == F` -> **step-0 output bit-identical to E0**
  (`max |arm - E0| = 0.000e+00`, measured).
* `alpha_logit = -2.0` -> **alpha = 0.119203**, read from dinolight-render's own
  module rather than typed in.
* Temperatures **1.0 per head, multiplicative** — Restormer's MDTA convention,
  compared in the smoke test against a real `restormer_arch.Attention` block.
* RNG: trunk built first by the parent, ViT and `P` inside the parent's fence,
  the new modules inside a fence added in this arm's arch. **494/494 trunk
  tensors byte-identical to E0.**

**THE STAIRCASE IS THREE STEPS.** Two zero-inits sit in series — `project_out`
at the ACA output and `P` at the prior's entrance:

| step | live | dead |
|---|---|---|
| 1 | `project_out` only | everything else, exactly 0 |
| 2 | + the feature path (`to_q`) and `P` | the DINO cross path still 0 |
| 3 | + `to_v_cross`, alpha | — |

`P` has zero gradient at step 1, so AdamW leaves it at exactly 0, so `d_proj` is
still 0 at step 2 and the cross-path inputs are 0. Arithmetic, not a defect;
asserted at all three steps.

---

## MONITORING

`dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`, same tags,
same `dino_stability.csv`, same gate, unchanged thresholds.

**`projected_norm` logs `||alpha * F_ca||`**, the DINO contribution, measured
BEFORE the zero-init output conv — the convention dinolight-render set.

> **THIS MAKES THIS ARM'S `injection_ratio` ROUGHLY AN ORDER OF MAGNITUDE
> SMALLER THAN THE ADDITION ARMS' ~1.03, AND NOT DIRECTLY COMPARABLE TO THEM.**
> dinolight-render runs at ~0.05 where affm-render runs at ~0.85. Compare ACA
> arms to ACA arms. The true injected delta into F is logged separately as
> `aca_injected_norm`, and it starts at exactly 0 like every other arm.

Every 5000 forwards, through the wrapper's existing generic hook (arriving with
a `dino/` prefix): `aca_alpha`, `aca_ca_to_sa_ratio`, `aca_injected_norm`, the
per-head temperatures of both attentions, the per-head channel-attention
entropy of both. **None is a gate rule.** Like every arm's
observations they are a **step function** — measured every N forwards, reprinted
every `print_freq` — so de-duplicate before plotting.

---

## THE SCALE CHECK

The ACA attention matrix is `C/heads x C/heads = 64x64` at **both** the
16x16-token train regime and the 32x32-token eval regime, verified on real
forwards while the `d_proj` grid demonstrably changes 16x16 -> 32x32. It is
never 256x256 or 1024x1024. This is the property crossattn-render lacked: that
arm scored **+2.00 dB at crop128 and -7.56 dB at full256 from identical
weights**, because a spatial softmax must renormalise over four times as many
competitors between train and eval.

It does **not** guarantee this arm will score well. It removes one specific,
measured failure mode.

---

## SMOKE TEST

**31/31 PASS** (`scripts/smoke_tests_aca_arms.py --config
configs/aca_render_fixed128_L6_latent.yml` --scale-check).

---

## STATUS

Built, smoke-passed, **submitted as STAGE 1**. Entries below this line are appended by the
run itself.

---
