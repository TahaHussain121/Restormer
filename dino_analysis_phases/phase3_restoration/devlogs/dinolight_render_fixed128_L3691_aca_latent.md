# dinolight-render — DINOLight's method adapted to radar

`Holo_dinolight_render_fixed128_L3691_aca_latent`

Scope: **this arm only**. Every other Phase-3 arm is read-only from here, and
the affm arm was training while this was built. Written **before the run**, so
the prediction below is a pre-registration.

---

## THE DECISION

**AFFM multi-layer fusion over {3,6,9,12} plus ACA gated channel cross-attention
at the latent** — a faithful adaptation of DINOLight (arXiv 2603.12579).

```
addition-render   D = centered B6              guided = F + P(D)
affm-render       D = sum_l W_l * centered D_l guided = F + P(D)
dinolight-render  D = sum_l W_l * centered D_l guided = ACA(F, P(D))

  ACA:  X = LN(F)                     X' = LN(P(D))
        Q, K, V, Q' <- X               K', V' <- X'
        each projection = Conv2d(384,384,1) then 3x3 DEPTHWISE  (Restormer MDTA style)
        F_sa = TransposedAttn(Q, K, V)         F_ca = TransposedAttn(Q', K', V')
        alpha = sigmoid(alpha_logit)
        guided = project_out(F_sa + alpha * F_ca) + F
```

**AFFM and ACA are both adopted from DINOLight and cited. Neither is claimed as
a contribution.** What is ours is the question: does the published method
transfer to radar heatmaps.

**DINOLight's frequency-domain branch is OMITTED.** Their own ablation puts
spatial-only at 22.289 and dual-domain at 22.600 — about **0.31 dB for double
the implementation surface**, which does not fit before the 2026-08-28 shutdown.
Recorded as a deliberate scope cut, not an oversight.

---

## THIS IS NOT A ONE-FACTOR ABLATION. SAY SO WHEN REPORTING IT.

Against addition-render this changes **two** things at once — the **layer count**
and the **fusion operator** — and it is the largest arm in the ladder. **A win
here cannot be separated from added capacity.** It answers "does the published
method transfer", not "which factor matters". The single-factor arms are
`affm-render` (layer count alone, addition unchanged, running now) and a planned
`aca-L6` (operator alone, B6 unchanged) — for which `dino_aca.py` was written as
a standalone module so it can be reused rather than reimplemented.

---

## RELATIONSHIP TO OUR FAILED crossattn-render, AND THE PREDICTION IT BUYS

crossattn-render used a **spatial 256x256 softmax**, **replaced** rather than
augmented, and had **no gate**. Its diagnosis (phase4_crossattn_diagnosis)
found:

* it did not fail the way the first devlog said. Its published −3.150 dB is a
  **4,000-iteration model**, selected by a rule that measures full-256;
* the decisive failure is **scale transfer**: **+2.00 dB at crop128 and
  −7.56 dB at full256 from identical weights**, because the DINO grid goes
  16x16 (256 tokens) at train to 32x32 (1024 tokens) at eval and a spatial
  softmax must renormalise over four times as many competitors;
* re-softening the collapsed attention at inference recovered **nothing**
  (T=2/4 within ±0.00 dB), and forcing the attention to the identity cost only
  0.112 dB — so entropy collapse was never the cause.

**Channel-transposed attention has no such failure mode**: the matrix is
`C/heads x C/heads`, independent of token count. **That is a prediction, and the
smoke test is where it gets tested rather than asserted** — see below, where it
passed on real forwards at both scales.

---

## DEVIATIONS FROM DINOLight, ALL DELIBERATE

| | DINOLight | this arm | why |
|---|---|---|---|
| layers | **{1, 6, 12}** | **{3, 6, 9, 12}** | ours contains B6 and all four have a measured fixed-128 correspondence on radar (WO1, n=339). B1/B4/B8 have never been measured here. |
| DINO input | the **degraded image** | the aligned **clay render** | our equivalent arm, addition-1e5, **LOST 0.58 dB**. The input-only route does not transfer to radar heatmaps. |
| centering | none stated | per-layer train-only [768] means, before the scoring conv | our pre-E1 analysis found the usable signal lives entirely in the centered residual. |
| encoder | ViT-B/14 **with registers** | ViT-B/14, **0 registers** | ours is the checkpoint every previous phase used; changing it would invalidate Phase 0/1/2 and every finished arm. |
| injection points | every scale | **one** (`inp_enc_level4`) | to stay comparable with the rest of our ladder. |
| frequency branch | present | **omitted** | ~0.31 dB by their own ablation, for double the surface. Time. |
| AFFM activation | **SiLU** | **GELU** | the work order's pseudocode said SiLU, but it also said stage 1 must be *identical to the affm arm already built*. Identity won: `DinoAffm` is **imported** from that arch, so stage 1 is the same code object and the two arms stay comparable on the feature side. The repo also standardises on GELU (`restormer_arch.py:91`). Not load-bearing. |

---

## PARAMETERS — MEASURED, AND 4,986 BELOW THE WORK ORDER'S ESTIMATE

```
  aca.to_*  six MDTA projections   6 x (147,456 + 3,456)  =   905,472
  P         768->384, 1x1 (+bias)  768*384 + 384          =   295,296
  aca.project_out  384->384, 1x1   147,456                =   147,456
  affm      four scoring convs     4 x 769                =     3,076
  aca.norm  two LayerNorms(384)    2 x 768                =     1,536
  aca.temperature  2 x 6 heads                            =        12
  aca.alpha_logit                                         =         1
  ---------------------------------------------------------------------
  measured delta over E0                                  = 1,352,849
```

Measured: E0 26,124,052 -> dinolight 27,476,901. **The work order estimated
1,357,835; the difference of 4,986 is fully itemised** and is not a mistake in
either direction:

* **−4,608**: the six projections take `bias` from the **trunk's own config key**
  (`bias: False`), as Restormer's MDTA does. The estimate assumed biased convs.
* **−384**: `project_out` likewise unbiased.
* **+6**: the self- and cross-attention get **separate** per-head temperatures
  (2 x 6), because they are two different attention operations over two
  different key spaces. The estimate assumed one shared set of 6.

**4.58x addition-render's 295,296 — the largest arm in the ladder.** Recorded as
a hard limit on interpretation.

---

## PRE-REGISTERED PREDICTION — written before the run

**This arm does not beat addition-render's 24.081 dB by more than 0.30 dB.**

Rationale: our fusion axis has shown no operator advantage so far (concat vs
addition +0.023 val / −0.016 test, CIs spanning zero), and B12 measures near the
different-scene floor on our data (0.2337 same-scene against a 0.1175 floor).

**CONTRARY EVIDENCE ON RECORD:** DINOLight's own ablation found multi-layer beat
every single layer, and their ACA added 2.80 dB over a no-DINO baseline. **If
this arm wins, report it as a surprise**, not as the expected outcome.

**SECONDARY OUTCOME, INDEPENDENT OF PSNR — and this one is the interesting
diagnostic.** `alpha` is a gate the network can *close*. If it decays toward 0
the model is saying the DINO path does not help and is falling back to plain
self-attention: a clean, interpretable negative result of a kind crossattn could
not produce, because that arm had no escape hatch. If it grows, the path is
being used. **Report the trajectory either way.**

---

## WHAT IS HELD IDENTICAL

Splits, fixed 128 crop, augmentation, seed 100, Restormer hyper-parameters,
AdamW (lr 3e-4, wd 1e-4), `CosineAnnealingRestartCyclicLR` periods
[92000, 208000], total_iter 300000, batch 8, L1, val_freq 4000, checkpoint every
2000, the stability gate (cap 10, ratio rules from 5000, NaN/Inf from 1), **and
the four centering mean FILES, reused verbatim from the affm arm — not
recomputed**, so the two arms are comparable on the feature side. Verified
bit-identical in the smoke test.

**The config differs from the affm arm's in 9 of 112 keys**: name, type, devlog
and tb_logger paths, `dino_fusion: affm -> aca`, and the three `dino_aca_*` keys
replacing `dino_affm_stats_freq`. **103 keys identical.**

**300k is not negotiable** — the periods sum to exactly 300,000.

---

## INITIALISATION, AND THE THREE-STEP STAIRCASE

* AFFM scoring convs zero -> the layer softmax starts **exactly uniform at
  0.25**, verified on a real batch.
* `project_out` zero -> `guided == F` -> **step-0 output bit-identical to E0**
  (`max |dinolight − E0| = 0.000e+00`).
* `alpha_logit = -2.0` -> **alpha = 0.119203** at init.
* temperatures **1.0 per head, multiplicative** — compared in the smoke test
  against a real `restormer_arch.Attention` block's own parameter, not against a
  number typed into the test.
* RNG: trunk built first by the stock `__init__`; ViT and `P` inside the
  parent's fence; AFFM and ACA inside a fence added here. **494/494 trunk
  tensors byte-identical to E0.**

**THE STAIRCASE IS THREE STEPS, NOT TWO, and the work order expected two.** Two
zero-inits sit in series — `project_out` at the ACA output and `P` at the
prior's entrance:

| step | live | dead |
|---|---|---|
| 1 | `project_out` only (2.60e-05) | everything else, exactly 0 |
| 2 | + feature path `to_q` (6.90e-07), `P` (1.67e-03) | **cross path still 0** |
| 3 | + `to_v_cross` (5.32e-07), AFFM (5.67e-06), `alpha` (1.80e-05) | — |

Step 2 is dead on the cross path for a precise reason: `P` had **zero** gradient
at step 1, so AdamW left it at exactly 0, so `d_proj` is still 0 and the
cross-path inputs are 0, so `dL/dW = 0` there regardless of what flows back.
Once `P` moves at step 2, everything is live at step 3.

This is arithmetic, not a defect, and 3 of 300,000 steps is not worth
redesigning for. **Dropping `P`'s zero-init would shorten it to two steps
without affecting step-0 equality** (which `project_out` alone already
guarantees) — deliberately **not** done, because `P` is inherited from the parent
under the config's `dino_init: zero` and keeping it identical to the other arms
is worth more than one step.

---

## MONITORING

`dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`, same tags,
same `dino_stability.csv`, same gate, unchanged thresholds.

**`projected_norm` logs `||alpha * F_ca||`** — the actual DINO contribution —
as the work order specifies. It **starts at exactly 0** like every other arm's
`||P(D)||`, because `P` is zero-init too, so `injection_ratio` rises from 0 and
the gate reads the same shape of quantity everywhere. (An earlier draft of this
devlog claimed it would start non-zero; the smoke test showed otherwise and the
claim was corrected rather than left standing.)

Every 5000 forwards, through the wrapper's existing generic hook, arriving with
a `dino/` prefix: `affm_w_b3/b6/b9/b12` and `affm_w_sum` (asserted to 1.0),
`aca_alpha`, `aca_ca_to_sa_ratio`, `aca_injected_norm` (the TRUE delta into F),
`aca_temp_{sa,ca}_h*`, and `aca_entropy_{sa,ca}_h*` (channel-attention entropy
per head, over the C/heads axis). **32 observation keys.** None is a gate rule.

Like every arm's observations these are a **step function** — measured every
`dino_aca_stats_freq` forwards, reprinted every `print_freq` — so de-duplicate
before plotting.

---

## THE SCALE CHECK — THE POINT OF THE DESIGN, AND IT PASSED

Run on real forwards at both regimes, not asserted:

| regime | radar | DINO tokens | fused grid | **attention matrix** |
|---|---|---|---|---|
| train128 | 128 | 16x16 = 256 | [1, 768, 16, 16] | **[1, 6, 64, 64]** |
| eval256 | 256 | 32x32 = 1024 | [1, 768, 32, 32] | **[1, 6, 64, 64]** |

**The attention matrix is identical at both scales while the DINO grid demonstrably
changes** — so the invariance is real, not an artefact of nothing changing. It is
`C/heads x C/heads = 64x64`, never 256x256 or 1024x1024. eval256 produces a
correct [1,1,256,256] output and uses the `mu_eval256` buffers.

This is the property crossattn-render lacked. **It does not guarantee this arm
will score well** — it only removes one specific, measured failure mode.

**Architectural smoke test: 44/44 PASS** (`smoke_tests_dinolight_render.py`).

---

## THE 6000-ITERATION TRAINING SMOKE (job 1788921, v100, gate ON) — PASSED

`rc=0`, reached 6000, **no STABILITY_FAILURE**, ratio rules genuinely ENFORCED
over 5000-6000. Six criteria were fixed in advance, before any result was seen.

**Gate telemetry.** `injection_ratio` 0.23-0.74 across the run (0.32 at 6000) —
finite, stable, and nowhere near the cap of 10. `latent_norm` 303 -> 924,
`projected_norm` (= ||alpha*F_ca||) 224 -> 297. Validation ran three times
through the eval256 switch: **20.38 / 20.96 / 21.64 dB** at 2k/4k/6k.

**The diagnostics this arm exists to produce**, 59 distinct measurements:

| iter | alpha | ca/sa | injected_norm | w_b3 | w_b6 | w_b9 | w_b12 | sum |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.11920 | 0.000 | 0.00 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 1500 | 0.12341 | 2.563 | 378.70 | 0.4808 | 0.2291 | 0.1215 | 0.1686 | 1.0000 |
| 3600 | 0.12600 | 3.094 | 632.70 | 0.4383 | 0.1952 | 0.1342 | 0.2322 | 1.0000 |
| 6000 | **0.12846** | 3.516 | 779.56 | 0.3524 | 0.3616 | 0.1127 | 0.1732 | 1.0000 |

**alpha is RISING**, 0.11920 -> 0.12846 (+0.00926) — the network is opening the
DINO gate, not closing it. It is small movement over 2% of the run and must not
be over-read, but the direction is the one that says the path is being used.
The AFFM weights sum to 1.0000 at all 59 measurements. As in the affm arm's
smoke, **B3 is the layer gaining most** early — the same direction, on a
different fusion operator.

### THE SCALE CHECK ON A TRAINED CHECKPOINT — THE RESULT THAT MATTERS

`net_g_6000`, val n=339, both protocols, same weights:

| | crop128 | full256 |
|---|---|---|
| **dinolight-render @6k** | 19.562 | **21.645** |
| crossattn-render @178k (for contrast) | 21.812 | **14.517** |

**full256 is HIGHER than crop128, not 7.5 dB lower.** crossattn's signature
failure -- the same weights collapsing when the DINO grid goes 16x16 -> 32x32 --
does not appear. Combined with the architectural check (attention matrix
[1,6,64,64] at BOTH scales while the grid demonstrably changes), the
scale-robustness prediction made in this devlog is **confirmed** at 6k. Both
numbers are from a 2%-trained model and neither is a result.

### SPEED AND MEMORY

**0.6837 s/iter on v100**, against the affm arm's 0.6800 on the same card:
**the ACA costs +0.5%** in step time. Peak VRAM **31,450 MiB on a 32 GB v100**
(the affm/addition arms measured ~25,117 MiB), so it fits a100 40 GB but would
NOT fit a 10 GB card.

Projected to a100 using the v100->a100 ratio measured on affm *while it trains*
(0.6800 -> 0.4474 s/iter, x1.52): **0.4498 s/iter -> 37.5 h for 300k**, i.e.
**two chained 24 h jobs**, the same shape as every other arm.

---

## STATUS

Built, architectural smoke **44/44**, 6000-iteration training smoke **passed on
all six pre-registered criteria**, and **LAUNCHED** for the full 300k. Entries
below this line are appended by the run itself.

---
