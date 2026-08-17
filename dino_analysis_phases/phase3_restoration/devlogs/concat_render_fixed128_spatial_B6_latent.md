# concat-render — Holo_concat_render_fixed128_spatial_B6_latent

DINO reads the RENDER, spatial B6 patch tokens, injected at the latent — the
same as addition-render. The fusion operator is the one thing that differs.

---

## 2026-08-17 — BUILD, BEFORE ANY TRAINING

Written before launching. Nothing in this entry has been trained; it records
the decision, the implementation and the verification.

### DECISION

**The fusion operator changes from residual addition to concat-then-project.**

```
addition-render   guided = F + P(D_centered)                 P:    1x1,  768 -> 384
concat-render     guided = fuse(cat([F, D_centered], 1))     fuse: 1x1, 1152 -> 384
```

The residual addition is **removed, not kept alongside** the concatenation:
`RestormerDinoConcatRender.__init__` deletes `self.P`, so no checkpoint of this
arm carries a `P.*` key and no unused parameter joins the optimizer.

**Motivation.** In addition, the radar path is fixed at identity — F enters the
sum with an implicit weight of 1 that no gradient can move, so the only way the
network can rebalance the two streams is by changing ‖P(D)‖. In concat the
radar-path weight `W_F` is a real learnable parameter, and the balance can move
from either side. That is the theoretical answer to the injection-ratio
saturation the addition arms show.

**Citations.**
- ACT (arXiv 2203.07682) — ablates direct concatenation against summation for
  feature fusion; concatenation wins.
- Feature-Fused SSD (arXiv 1709.05054) — learned versus fixed combination
  weights for fused features.

### WHAT IS DELIBERATELY UNCHANGED

Layer B6 (0-indexed 5). DINO source = the render. Injection at
`inp_enc_level4`, before the 8 latent blocks. Frozen ViT in eval(). Both render
centering means, the **same files** addition-render uses — no new mean was
computed. Fixed 128 crop, batch 8, 300k iterations,
CosineAnnealingRestartCyclicLR periods [92000, 208000], AdamW 3e-4, L1, seed
100, val every 4k, and every gate threshold.

**The features stay SPATIAL.** The `[B,768,g,g]` patch-token grid is
concatenated along channels position by position — g=16 at the 128 training
crop, g=32 at 256 eval. There is no pooling and no broadcast anywhere in this
arm. It extends `RestormerDinoSpatialRender` and inherits nothing from
`RestormerDinoSpatialGlobalRender`, which is a control for a different question.

Config diff versus `E1_addition_render_fixed128_spatial_B6_latent.yml`, key by
key, machine-generated — **94 keys identical**, 1 added, 4 changed:

| key | addition-render | concat-render |
|---|---|---|
| `name` | Holo_E1_addition_render_fixed128_spatial_B6_latent | Holo_concat_render_fixed128_spatial_B6_latent |
| `network_g.type` | RestormerDinoSpatialRender | **RestormerDinoConcatRender** |
| `network_g.dino_fusion` | *(absent)* | **concat** |
| `dino_stability.devlog` | …/E1_addition_render_….md | …/concat_render_….md |
| `logger.tb_logger_dir` | tb_logger/Holo_E1_addition_render_… | tb_logger/Holo_concat_render_… |

Only the fusion block and the two per-arm output paths differ. No mean, crop,
schedule, threshold or seed moved.

### INITIALIZATION — the arm starts functionally equal to E0

```
W[:, :384]  identity   (torch.eye reshaped to [384,384,1,1])
W[:, 384:]  zero
bias        zero
```

so `guided == F` exactly at step 0, which is what E0 feeds its latent blocks.
This is the concat analogue of the zero-initialised `P` that made
addition-render's step 0 equal to E0's.

**The zero DINO half does not stay at zero.** `dL/dW_D = dL/d(guided) ⊗ D` with
D non-zero, so `W[:, 384:]` receives a real gradient on the very first backward.
Both facts are asserted, not assumed — see the verification below.

### RNG ORDER — approach used: EXPLICIT SAVE/RESTORE FENCE

`nn.Conv2d.__init__` draws from the generator before `eye`/`zeros_` overwrite
the result, which would shift every later draw (data order, crop offsets,
augmentation flags) away from E0's for the same seed. `fuse` is therefore built
inside a save/restore fence — `torch.get_rng_state()` / `cuda.get_rng_state_all()`
before, restore in a `finally` — the same device `RestormerDinoSpatial.__init__`
already uses for the ViT and P. The parent fences its own construction, so RNG
state on entry to this fence is already E0's, and restoring it on exit leaves
every trunk weight and every later draw byte-identical to E0.

Verified rather than argued: **494/494 shared state_dict tensors are
byte-identical to E0's** at the same seed, and the only extra keys are
`fuse.weight` and `fuse.bias`.

### PARAMETER COUNT

| | trainable params | delta over E0 |
|---|---|---|
| E0-Fixed | 26,124,052 | — |
| addition-render | 26,419,348 | +295,296 |
| **concat-render** | **26,566,804** | **+442,752** |

442,752 = 1152 × 384 + 384. **Confirmed equal to the expected value.**

**KNOWN ASYMMETRY — recorded, not corrected.** concat-render is **not**
parameter-matched to addition-render (+442,752 vs +295,296, a difference of
147,456), and its radar-path weight `W_F` is learnable where addition fixes the
radar path at identity. So the two arms differ in **capacity as well as in
fusion operator**. This is inherent to a fusion-operator comparison and is not
something to correct for; it is a limit on what a concat-vs-addition difference
can be attributed to, and any result from this arm must be read with it in view.

### MONITORING — the existing gate, reused unchanged

A 1×1 conv on a concatenation is exactly the sum of two 1×1 convs:

```
fuse(cat([F, D])) = conv(F, W_F) + conv(D, W_D) + b
W_F = W[:, :384]      W_D = W[:, 384:]

latent_norm     = ‖ conv(F, W_F) ‖
projected_norm  = ‖ conv(D_centered, W_D) ‖
injection_ratio = projected_norm / (latent_norm + eps)
```

Written into the same `last_dino_stats` keys the addition arms use, so the
model wrapper emits the same TensorBoard tags (`dino/latent_norm`,
`dino/projected_norm`, `dino/injection_ratio`) and the same
`dino_stability.csv` with no change to the wrapper or the gate. At
initialisation `W_F` is the identity, so `latent_norm` starts as literally ‖F‖
— the exact quantity addition-render logs at step 0 — which is what makes the
two arms' ratios directly comparable. The bias is excluded from the split; it
belongs to neither stream.

Gate rules and thresholds are **unchanged as amended**: `injection_ratio_max`
10.0, ratio rules start at iteration 5000, `growth_factor_max` 10.0, NaN/Inf
hard-stop active from iteration 1.

### VERIFICATION — 52/52 smoke checks passed (CPU, pre-launch)

`scripts/smoke_tests_concat_render.py`, inference / one-step only, no
experiment identity and no checkpoint written. The load-bearing ones:

- **STEP-0 OUTPUT EQUALS E0**: max |concat − E0| = **0.000e+00** on the same
  radar tensor, E0 built from its own config at the same seed.
- **DINO half has non-zero gradient on the FIRST backward**:
  max |dL/dW_D| = 2.504e-02. (Radar half: 5.802e-05, also non-zero and finite.)
- parameter delta == 442,752, exactly.
- `P` is gone: no `P` module, no `P.*` parameter.
- `W_F` == identity, `W_D` == 0, bias == 0 at init.
- RNG fence held: 494/494 shared tensors identical to E0.
- **SPATIAL, not pooled**: 16×16 tokens → [B,1152,16,16] concat →
  [B,384,16,16] guided; and D ≠ pool(D) broadcast (max difference 64.36), so
  no global vector is being fed.
- DINO reads the render channel (`torch.equal`), never the radar; frozen, eval,
  gradients None.
- Both render mean files are the addition-render ones; no new mean.
- At init `projected_norm` == 0 and `latent_norm` == ‖F‖ = 46.4322, matching
  the direct computation to 4 decimals.
- 256 eval: 1024 tokens → 32×32 grid == 32×32 latent, no interpolation,
  `mu_eval256` in use.

Run on GPU before launching with
`sbatch dino_analysis_phases/phase3_restoration/scripts/run_smoke_concat_render.sh`.

### ISOLATION

No existing `experiments/Holo_concat_render_fixed128_spatial_B6_latent`, no
chain-state directory and no `tb_logger/` tree for this name — confirmed before
writing this entry. E0-Fixed, addition-1e5, addition-render and
global-addition-render directories, configs, devlogs and checkpoints were not
touched, renamed or interrupted by this build.

### OPEN / NOT YET RESOLVED

- **Co-inflation still has no gate rule.** The gate watches the injection ratio;
  it cannot see both streams growing together.
- **B3 vs B6 was a criterion choice, not a measurement.** B3 has the stronger
  same-vs-different-scene advantage and B6 the stronger centered same-scene
  correspondence at the fixed-128 scale. The conflict is documented and still
  unresolved; this arm does not revisit it.
- **The token-shuffle control has not been run** for this arm. (The
  mismatched-render control was run for addition-render only, post-hoc, on val.)
- concat-render is not parameter-matched to addition-render, as recorded above.
