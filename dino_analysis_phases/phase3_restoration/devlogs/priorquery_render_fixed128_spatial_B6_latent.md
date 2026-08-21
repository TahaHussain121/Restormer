# priorquery-render — Holo_priorquery_render_fixed128_spatial_B6_latent

DINO reads the RENDER, spatial B6 patch tokens, injected at the latent. The
fusion is cross-attention with the PRIOR AS QUERY — the direction counterpart of
crossattn-render.

---

## 2026-08-20 — BUILD, BEFORE ANY TRAINING

Written before launching. Nothing here has been trained.

### VERIFIED FIRST, FROM THE REPO

Read from the crossattn-render config, the model code and that arm's own smoke
artifact before a line of this arm was written. All CONFIRMED, no mismatches:

| | value | source |
|---|---|---|
| latent at a 128 crop | `[B, 384, 16, 16]` | crossattn-render smoke artifact |
| DINO B6 at a 128 crop | `[B, 768, 16, 16]`, 256 tokens | same |
| grids equal | 256 queries == 256 keys/values | same |
| batch / iters / crop / seed | 8 / 300,000 / 128 / 100 | config |
| scheduler | CosineAnnealingRestartCyclicLR [92000, 208000], restart_weights [1,1], eta_mins [0.0003, 0.000001] | config |
| optimizer / loss | AdamW 3e-4, wd 1e-4, betas [0.9, 0.999]; L1Loss weight 1, mean | config |
| means | `render_B6_train128_dino224_mean.pt`, `render_B6_eval256_dino448_mean.pt` | config |
| E0 params | 26,124,052 | constructed and counted |
| gate + tags | `image_restoration_dino_model.py`; `dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`; CSV `experiments/<name>/dino_stability.csv` | code |

The equal token counts matter more here than in any previous arm — see MECHANISM.

### DECISION

**A second cross-attention arm with the direction reversed: the DINO prior is
the QUERY, the Restormer latent is both KEY and VALUE.**

```
crossattn-render   Q = radar F,  K = V = DINO D    (our direction)
priorquery-render  Q = DINO D,   K = V = radar F   (this arm)
```

This is the direction used by **Perceive-IR's PGCA** and inherited unchanged by
**DSGIR**. crossattn-render tests the opposite direction, so the pair isolates
**direction** as a factor. The two are not to be harmonised, and share no code.

Perceive-IR's use of it was verified from the paper rather than from memory:
its features are `𝐅l ∈ ℝ^{1×768}` — a **global pooled vector**, four layers
(1, 4, 8, 12) — and `we take 𝐘l as query to perform the cross-attention`. So the
published direction pairs a *pooled* prior with prior-as-query. This arm keeps
our **spatial** 16×16 prior and changes only the direction, which is a
combination neither paper tests.

### MECHANISM NOTE — do not smooth this over

**In this direction DINO content does not reach the output directly.** The
attention output is a re-mix of RADAR values; DINO influences it only through
the softmax weights. **DINO is a router over radar positions, not a content
source.**

Two consequences, both enforced in code rather than asserted in prose:

- **The output token count follows the QUERY**, i.e. the number of DINO tokens.
  The residual add back onto `F` is well defined only because the DINO grid is
  16×16. `DinoPriorQueryAttention.forward` raises if the two grids ever differ,
  explicitly refusing to interpolate, pad or crop to make a mismatch fit.
- **There is no second path injecting D.** No concat, no residual `P(D)` — `P`
  is deleted in `__init__`. Routing-only is the hypothesis under test, and a
  content path would silently turn this into a different experiment. The smoke
  test proves the point structurally: `W_v` is `Linear(384 → 384)`, so it is
  *dimensionally incapable* of accepting the 768-wide prior.

### PRE-REGISTERED PREDICTION — written before the run

**This arm lands closer to E0-Fixed than to addition-render.**

Rationale: the shuffled-render control cost addition-render **9.297 dB** with
**339/339 images worse** and per-image correlation to the correct-render run of
only r=0.251. The render's **content** is what carries the gain. A design that
reduces the prior to routing weights should forfeit most of it.

**If the prediction fails, that is the more interesting outcome and must be
reported as such, not quietly dropped.** A routing-only arm that matches
addition-render would mean the prior's value lies in *where to look* rather than
*what to add* — which would reframe the whole fusion axis.

### THE MODULE

```
D : [B,768,16,16] -> 256 QUERY tokens, dim 768
F : [B,384,16,16] -> 256 key/value tokens, dim 384

LayerNorm(768) on queries, LayerNorm(384) on keys/values  (pre-norm, affine)
W_q 768->384   W_k 384->384   W_v 384->384   W_o 384->384 (ZERO-INIT)
heads = 6, head_dim = 64, attn = softmax(Q K^T / sqrt(64)) V
guided = F + reshape(W_o(attn))
```

`DinoPriorQueryAttention` is a **separate class**, deliberately duplicating
structure rather than sharing a module with crossattn-render or parametrising
one with a direction flag. A shared module would mean a later edit for one
direction could silently change the other, and the two arms only mean anything
if each is frozen independently.

### THE GRADIENT STAIRCASE

`W_o` is zero at the output of the fusion, so on the first backward
`dL/dW_q = dL/dW_k = dL/dW_v = 0` — the gradient must travel through `W_o`.
`W_o` learns at step 1, q/k/v from step 2. A step-1 assertion on q/k/v would
fail, correctly.

### RNG ORDER — approach used: EXPLICIT SAVE/RESTORE FENCE

The attention module is built inside `torch.get_rng_state()` /
`cuda.get_rng_state_all()` … restore-in-`finally`, the same device the parent
uses for the ViT and `P`, so every trunk weight and every later draw stays
byte-identical to E0 at the same seed.

### PARAMETER COUNT

```
W_q   768*384 + 384 = 295,296
W_k   384*384 + 384 = 147,840
W_v   384*384 + 384 = 147,840
W_o   384*384 + 384 = 147,840
LN_q       768*2    =   1,536
LN_kv      384*2    =     768
------------------------------
              total   741,120     E0 26,124,052 -> 26,865,172
```

**KNOWN ASYMMETRY — recorded, not corrected.** The two direction arms are **not
parameter-matched to each other**: 741,120 here versus 888,576 for
crossattn-render. The asymmetry is structural rather than a choice — there both
K and V came from the wide 768 side, here only Q does. The ladder is now

```
addition 295,296 < concat 442,752 < prior-Q 741,120 < radar-Q 888,576
```

### MONITORING — AND ONE NUMBER THAT DOES NOT MEAN WHAT IT MEANS ELSEWHERE

```
latent_norm     = ||F||
projected_norm  = ||W_o(attn)||
injection_ratio = projected_norm / (latent_norm + eps)
```

logged under the same tags and into the same `dino_stability.csv` for
continuity.

> **⚠ THIS ARM'S `injection_ratio` IS NOT COMPARABLE TO THE OTHER ARMS'.** In
> every other arm the numerator is built from **DINO features**. Here
> `W_o(attn)` is built from **RADAR values** — it measures how much the network
> re-mixes its own latent, not how much prior it injects. **It must not be put
> in the same table column as the other arms' ratio without this caveat
> attached.**

Observations every 5000 forwards:

- **`attn_entropy`** — mean entropy of the 256-way softmax rows (uniform =
  log 256 = 5.545 nats).
- **`attn_diag_mass`** — **the key metric for this arm.** Near 1.0 means each
  DINO token routes to the radar token at its own position, i.e. the module has
  collapsed to a per-position reweighting that a far simpler operator could
  perform. That is a RESULT to report, not a failure. At initialisation it sits
  at 0.003902 ≈ 1/256, as it should.

Gate unchanged: ratio cap 10.0, ratio rules from iteration 5000,
`growth_factor_max` 10.0, NaN/Inf hard-stop from iteration 1. **No rule is
defined on entropy or diagonal mass**, deliberately.

### CONTEXT — why direction is worth a run at all

The fusion axis has so far produced **no operator advantage**. concat-render vs
addition-render was **+0.023 dB on val and −0.016 dB on test**, bootstrap 95% CI
spanning zero on both (val [−0.071, +0.119], test [−0.116, +0.085]), for
+147,456 parameters; only crop128/test separated from zero (+0.214, CI
[+0.099, +0.334]) and even there the CI does not clear the +0.30 dB threshold.
**Direction is one of the few remaining factors that could make this axis
informative.**

### VERIFICATION — 52/52 smoke checks passed (v100, job 1784410)

- **STEP-0 OUTPUT BIT-IDENTICAL TO E0**: max deviation **0.000e+00**.
- **Gradient staircase**: step 1 `W_o` 3.520e-04 with q/k/v at exactly
  0.000e+00 as predicted; step 2 all three alive (W_q 6.021e-07, W_k 6.122e-08,
  W_v 2.163e-06); still alive at step 10.
- **Parameter delta 741,120**, exactly as derived.
- **RNG fence held**: 494/494 shared tensors byte-identical to E0; the only
  extra keys are `pqattn.*`.
- **Direction, structurally**: `W_q` is `Linear(768→384)`, `W_k` and `W_v` are
  `Linear(384→384)`, LN over the query side is 768-wide and over keys/values
  384-wide — the mirror image of the radar-query arm.
- **ROUTING-ONLY, causally**: with `W_o` temporarily set to the identity so the
  injection is observable, changing the radar key/value changed the injection
  (max Δ 2.280e-01) and changing the prior also changed it (max Δ 1.027e-01) —
  but only through the weights, since `W_v` cannot accept a 768-wide input.
- **No direct D path**: no `P` module, no `P.*` parameter.
- **Spatial**: 16×16 grid preserved end to end; queries are not a pooled
  broadcast vector; 256 queries and 256 keys, so the residual add is defined.
- DINO reads the render channel, frozen, eval, gradients None; both mean files
  are addition-render's, no new mean.
- 256 eval: 1024 tokens → 32×32 grid == 32×32 latent, no interpolation.

### PEAK VRAM

Measured on **v100**, one fresh process per arm, batch 8 / crop 128 (job
1784420):

| arm | peak allocated | vs addition |
|---|---|---|
| addition-render | 25,117 MiB | — |
| **priorquery-render** | **25,147 MiB** | **+0.12%** |

Measured on v100 rather than a100 because crossattn-render was training and
holding the a100 GRES association (`AssocGrpGRES`), and addition-render was
re-measured on the *same* card so the comparison is valid — a v100 number is not
comparable to the earlier a100 figures (25,170 / 25,176 / 25,203 MiB for
addition / concat / crossattn). The relative cost is negligible either way; the
`[B,6,256,256]` attention matrix is ~12 MiB and is not even materialised except
on measurement steps.

### ISOLATION

No existing experiment directory, chain-state directory, `tb_logger/` tree,
results directory, config or devlog for this name — all confirmed clear before
the build. E0-Fixed, addition-1e5, addition-render, global-addition-render,
concat-render and crossattn-render were **not** edited, renamed, moved or
deleted; they were opened read-only as reference. crossattn-render was
**training throughout this build** (job 1784331) and was not touched; the chain
lock in `chain_core.sh` is keyed by experiment name, so this arm cannot start a
trainer against its directory.

**Files created by this arm** (all new):
```
basicsr/models/archs/restormer_dino_priorquery_render_arch.py
dino_analysis_phases/phase3_restoration/configs/priorquery_render_fixed128_spatial_B6_latent.yml
dino_analysis_phases/phase3_restoration/scripts/smoke_tests_priorquery_render.py
dino_analysis_phases/phase3_restoration/scripts/chain_priorquery_render.sh
dino_analysis_phases/phase3_restoration/scripts/run_smoke_priorquery_render.sh
dino_analysis_phases/phase3_restoration/scripts/run_peak_vram_priorquery.sh
dino_analysis_phases/phase3_restoration/devlogs/priorquery_render_fixed128_spatial_B6_latent.md
```

**Files modified belonging to another arm: NONE.**

### CONFIG DIFF versus crossattn-render — machine-generated

**96 keys identical**, 0 added, 0 removed, **5 changed**: `name`,
`network_g.type`, `network_g.dino_fusion`, `dino_stability.devlog`,
`logger.tb_logger_dir`. Only the fusion module and the per-arm output paths.

### OPEN / NOT YET RESOLVED

- Co-inflation still has no gate rule.
- B3 vs B6 was a criterion choice, not a measurement.
- Token-shuffle control not run. It is informative for both attention arms,
  which could in principle recover from shuffling by attending elsewhere — but
  note this arm could only recover its *routing*, never its content, since it
  has no content path at all.
- The two direction arms are not parameter-matched, as recorded above.
