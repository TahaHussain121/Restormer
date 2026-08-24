# aca-L36 — ACA fusion over the two SHALLOWEST measured depths

`Holo_aca_render_fixed128_L36_latent`

**RUN ORDER: SECOND.** Stage 2, launched only after aca-L6 is confirmed healthy.

Scope: **this arm only**. Every other Phase-3 arm is read-only from here;
affm-render and dinolight-render were TRAINING while this was built.
Written **before the run**, so the prediction below is a pre-registration.

## THE DECISION

**ACA fusion over the DINO layer set {3, 6}** — an interior point of the ACA
layer ladder, pairing B6 with the shallowest measured depth.

```
  D_l = centered DINOv2 grid, l in {3,6}
  W   = softmax([s_3, s_6], dim=layers)    -> [B,2,g,g], sums to 1 per position
  D_fus = W_3*D_3 + W_6*D_6                -> still 768 channels
  guided = project_out(F_sa + alpha*F_ca) + F
```

The fusion operator is **identical to every other ACA arm** — the same `DinoAca`
block imported from `dino_aca.py`. **Only the layer set changes.** The layer set
is **hard-coded in this arm's own arch file**, not a config flag, so a change to
one arm can never silently alter another.

---

## WHERE IT SITS IN THE LADDER, AND WHAT THAT BUYS

| arm | layers | AFFM | delta over E0 |
|---|---|---|---|
| aca-L6 | {6} | none (1-layer softmax is a no-op) | 1,349,773 |
| **aca-L36** | **{3,6}** | **2 convs** | **1,351,311** |
| aca-L6912 | {6,9,12} | 3 convs | 1,352,080 |
| dinolight-render | {3,6,9,12} | 4 convs | 1,352,849 |

**Measured spread across the ladder: 0.114% between this arm and dinolight.**
A difference *across this ladder* is **not** attributable to capacity — that is
the entire reason the ladder is built this way.

**Against addition-render the story is different and must not be conflated:**
4.58x the parameters, so *that* comparison IS capacity confounded.

---

## PRE-REGISTERED PREDICTION — written before the run

**aca-L36 does not beat aca-L6 by more than 0.10 dB.**

Rationale: B3 wins the same-vs-different scene advantage (+0.1966 vs B6's
+0.1453) but loses on absolute correspondence (0.6091 vs 0.6694), and the
layer axis has so far produced no clear winner — affm-render's AFFM weights
have wandered non-stationarily over 245k iterations without settling on B6 or
anything else.

**SECONDARY OUTCOME:** the `affm_w_b3` / `affm_w_b6` split. With only two
layers competing, this is the cleanest read we will get of whether the network
prefers the shallow or the mid depth, uncontaminated by B9/B12. Report it
whichever way it goes, and **average over a window** — the four-layer arms have
already shown these weights are non-stationary and a single measurement is not
a result.

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
| 3 | + `to_v_cross`, alpha, AFFM | — |

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
entropy of both, and `affm_w_b3` / `affm_w_b6` / `affm_w_sum` (asserted to sum to 1.0). **None is a gate rule.** Like every arm's
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
configs/aca_render_fixed128_L36_latent.yml`).

---

## STATUS

Built, smoke-passed, **built and smoke-passed, NOT submitted**. Entries below this line are appended by the
run itself.

---
