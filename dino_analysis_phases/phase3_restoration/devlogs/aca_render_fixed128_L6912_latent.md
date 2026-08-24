# aca-L6912 — ACA fusion over B6 plus the two DEEPEST measured depths

`Holo_aca_render_fixed128_L6912_latent`

**RUN ORDER: THIRD.** Stage 2, launched only after aca-L6 is confirmed healthy.

Scope: **this arm only**. Every other Phase-3 arm is read-only from here;
affm-render and dinolight-render were TRAINING while this was built.
Written **before the run**, so the prediction below is a pre-registration.

## THE DECISION

**ACA fusion over the DINO layer set {6, 9, 12}** — the mirror of aca-L36:
B6 plus the two *deepest* measured depths instead of the shallowest.

```
  D_l = centered DINOv2 grid, l in {6,9,12}
  W   = softmax([s_6, s_9, s_12], dim=layers)  -> [B,3,g,g], sums to 1 per position
  D_fus = sum_l W_l * D_l                      -> still 768 channels
  guided = project_out(F_sa + alpha*F_ca) + F
```

Fusion operator identical to every other ACA arm; **only the layer set changes**,
and it is hard-coded in this arm's own arch file rather than being a config flag.

---

## WHERE IT SITS IN THE LADDER

| arm | layers | AFFM | delta over E0 |
|---|---|---|---|
| aca-L6 | {6} | none | 1,349,773 |
| aca-L36 | {3,6} | 2 convs | 1,351,311 |
| **aca-L6912** | **{6,9,12}** | **3 convs** | **1,352,080** |
| dinolight-render | {3,6,9,12} | 4 convs | 1,352,849 |

**Measured spread: 0.057% below dinolight.** Together with aca-L6 and
dinolight-render this gives both endpoints of the layer axis plus two interior
points, all on an identical operator and all within 0.23% on parameters — so
**the layer axis can be read without a capacity confound.**

Against addition-render: 4.58x parameters, capacity confounded. Different
comparison, different caveat, do not conflate.

---

## PRE-REGISTERED PREDICTION — written before the run

**aca-L6912 does not beat aca-L6 by more than 0.10 dB, and does not beat
aca-L36 by more than 0.10 dB either.**

Rationale: B9 (0.5482) and B12 (0.2337) both sit **below** B6 (0.6694) on
centered same-scene correspondence at the fixed-128 training scale, and B12 is
close to its 0.1175 different-scene floor. Adding them to B6 should dilute
rather than add.

**CONTRARY EVIDENCE ON RECORD, AND IT IS REAL:** the live four-layer arms have
NOT down-weighted B12 the way this rationale assumes. At ~185k dinolight-render
rated B12 its **highest** layer (0.309) before reversing to lowest (0.173) by
239k. If B12 turns out to carry usable signal on this data, that contradicts the
feature-space ranking the B6 lock rests on, and **that is the more interesting
outcome** — report it as such rather than as an anomaly.

**SECONDARY OUTCOME:** the `affm_w_b9` / `affm_w_b12` share against aca-L36's
`affm_w_b3`. Between the two interior arms this is a direct shallow-vs-deep
comparison at matched capacity.

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
entropy of both, and `affm_w_b6` / `affm_w_b9` / `affm_w_b12` / `affm_w_sum` (asserted to sum to 1.0). **None is a gate rule.** Like every arm's
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
configs/aca_render_fixed128_L6912_latent.yml`).

---

## STATUS

Built, smoke-passed, **built and smoke-passed, NOT submitted**. Entries below this line are appended by the
run itself.

---
