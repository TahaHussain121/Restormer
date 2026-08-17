# global-render — `Holo_global_addition_render_fixed128_B6_latent`

Append-only. One entry per event. Never rewritten.

---

## 2026-08-13 — arm created (implementation, not launched)

**What this arm asks.** E1-render is well ahead of E0 at matched iterations
(+2.43 dB val PSNR at 60k). This arm asks *why*: does that gain come from
DINO's **spatial layout** of the render, or would a **single global descriptor**
of the render do just as well?

**Reference arm is E1-render**, not E1-addition-noisy. Exactly one thing
changes: the centered DINO grid is pooled to one vector and broadcast back
before the projection.

```
E1-render       D_centered [B,768,16,16] ------------------> P -> + at latent
global-render   D_centered [B,768,16,16] -> mean -> [B,768,1,1]
                                         -> broadcast ----> P -> + at latent
```

Every spatial position receives the SAME vector. The only difference from
E1-render is the presence or absence of spatial variation.

**DINO source is still the render.** `dino_source: render`, the render means and
the stacked render dataset are all unchanged. Restormer still receives the 1e5
crop; the target is still 1e7; the render is DINO input only.

### Config delta vs E1-render — 5 keys of 99, verified key-by-key

| key | E1-render | global-render |
|---|---|---|
| `network_g.type` | `RestormerDinoSpatialRender` | `RestormerDinoSpatialGlobalRender` |
| `network_g.dino_pooling` | *(absent)* | `global_mean` |
| `name` | `Holo_E1_addition_render_...` | `Holo_global_addition_render_...` |
| `logger.tb_logger_dir` | `tb_logger/Holo_E1_addition_render_...` | `tb_logger/Holo_global_addition_render_...` |
| `dino_stability.devlog` | the E1-render devlog | this file |

Two are the scientific change; three are the identity/bookkeeping paths a
separate experiment must have. Everything else — `dino_source: render`, both
render mean paths, block, init, injection, projection, dataset type and roots,
crop, batch, seed, LR, scheduler, loss, iterations and **every gate threshold** —
is byte-identical, checked programmatically rather than by eye.

### Implementation

`basicsr/models/archs/restormer_dino_global_render_arch.py`,
`RestormerDinoSpatialGlobalRender(RestormerDinoSpatialRender)`. It overrides
**`dino_prior` and nothing else**:

```python
grid      = super().dino_prior(inp_img, mode, latent_hw)  # [B,768,g,g], centered + asserted
pooled    = grid.mean(dim=(2, 3), keepdim=True)           # [B,768,1,1]
broadcast = pooled.expand_as(grid).contiguous()           # [B,768,g,g]
```

`forward` is **inherited unmodified**, which is what keeps the radar/render
split, the global residual against the radar and the stacked-input validation
identical to E1-render.

**Crop alignment is inherited, not re-implemented.** The render still rides as
channel 1 of the LQ tensor, so `train.py`'s sub-crop is one slice hitting both
channels. Pooling happens *after* the crop, inside the model — so a desynced
crop would still be fatal; it simply cannot occur, because there is only one
tensor to crop. Re-proved from disk in this arm's own smoke test rather than
inherited on trust.

**Means reused, not recomputed.** The mean is one position-independent [768]
vector, so centering and pooling commute:
`mean_p(D_p - mu) == mean_p(D_p) - mu`. The arch centers first, then pools.

**Patch tokens, not CLS.** Extraction runs with `return_class_token=False`, so
CLS is never available — substituting it would change two things at once.

**The no-interpolation assertion stays active** on the pre-pool grid. A pooled
vector would broadcast happily to any grid size, so without this, pooling could
silently paper over a scale mismatch.

### Verification — **61/61 checks passed** (job 1775532, v100)

Machine-readable record: `results/wo2_implementation/smoke_results_global_render.json`.
(An earlier submission, job 1775515, was cancelled while still PENDING on a
saturated a100 and never ran; it produced nothing.)

| check | result |
|---|---|
| parameter count vs E1-render | **26,419,348 == 26,419,348**, identical |
| parameter delta vs E0 | E0 26,124,052 + **295,296** — pooling adds none |
| zero-init identity vs E0 | **max diff 0.0** |
| zero-init identity vs E1-render | **max diff 0.0** |
| `pooled == mean(pre-pool grid)` | max diff 0.000e+00 |
| broadcast spatially constant | max spatial spread 0.000e+00 |
| spatial variance destroyed | per-channel 7.627 → **0.000000** |
| `P(broadcast)` constant, random 1×1 W | max spatial spread 0.000e+00 |
| commutation `mean(D-mu) == mean(D)-mu` | max diff **2.384e-06** on scale 9.663 (relative 2.5e-07) |
| arch computes center-then-pool | max diff 7.749e-07 |
| crop+aug identity, re-derived from disk | `torch.equal` true on radar AND render |

The zero-init identity ran on CPU, where it is bitwise exact; on CUDA two
separately-constructed instances differ by ~1e-4 through float
non-determinism, which is hardware, not this arm.

Note `injection_ratio` is exactly **0.0** at init (latent_norm 61.78,
projected_norm 0.0) — the branch really is an exact no-op before training, and
the ratio has to climb from zero, which is precisely why the gate's ratio rules
do not start until iteration 5000.

### Expected trajectory, registered before the run

`injection_ratio` should sit **below E1-render's ~1.03**, because pooling
cancels token-to-token variation and therefore shrinks `‖P(D)‖`. A ratio much
*higher* than E1-render's would be a surprise worth stopping for.

**Read both raw norms, not only the ratio.** Co-inflation — `latent_norm` and
`projected_norm` growing together with a flat ratio — remains an **ungated**
failure mode across all DINO arms (see HANDOVER §5).

### Status

**NOT LAUNCHED.** No experiment directory, no checkpoint, no training state.
E0, E1-addition-noisy and E1-addition-render were untouched throughout; the new
`*_arch.py` file was import-checked immediately after creation, because
`basicsr/models/archs/__init__.py` imports **every** `*_arch.py` in the folder,
so a broken file would have taken down all three running arms at their next
resume.

---

## 2026-08-13 — 6000-iteration smoke run, GATE ON (job 1775633, v100, 1:12:13)

Throwaway identity `SMOKE6K_global_addition_render_B6_latent`. Completed all
6000 iterations, exit 0. **No abort, no trigger, no NaN.** No
`STABILITY_FAILURE*` or `STABILITY_TRIGGER*` file was written.

### injection_ratio — the registered prediction held

| iter | loss | latent_norm | projected_norm | ratio |
|---|---|---|---|---|
| 1 | 1.725e-01 | 9.094e+01 | **0.000e+00** | **0.0000** |
| 10 | 1.627e-01 | 1.401e+02 | 6.383e+01 | 0.4556 |
| 50 | 1.714e-01 | 1.215e+02 | 1.616e+02 | 1.3299 |
| 1000 | 1.208e-01 | 7.879e+02 | 3.727e+02 | 0.4730 |
| 4000 | 9.006e-02 | 1.292e+03 | 5.618e+02 | 0.4349 |
| 5000 | 5.058e-02 | 1.240e+03 | 7.898e+02 | 0.6371 |
| 6000 | 5.136e-02 | 1.358e+03 | 8.765e+02 | 0.6454 |

Steady state over the enforcement window (5000-6000): mean **0.529**, range
0.366-0.849. Max over the entire run **1.987** (an early transient), against a
cap of 10.0. 5k reference recorded at 0.6371.

**PREDICTION CONFIRMED.** The registered expectation was that the ratio would sit
BELOW E1-render's ~1.03 because pooling cancels token-to-token variation. It sits
at roughly **half** of it. `‖P(D)‖` is 8.76e+02 here against E1-render's 5.06e+03
at 61k — different training stage, but the direction is unambiguous.

**Co-inflation, checked explicitly rather than inferred from the flat ratio:**
over 5000→6000, latent grew 1.10x and projected 1.11x — in lockstep, no runaway.
Caveat: a 1000-iteration window is a weak test, and this failure mode stays
ungated. Read both norms in `dino_stability.csv`, not only the ratio.

**The gate amendment is load-bearing for this arm.** Steady-state ratio is
0.43-0.65 with an early excursion to 1.99, so the ORIGINAL 0.5 cap active from
iteration 1 would have aborted this arm within ~10 iterations and kept tripping
in steady state. The amendment (cap 10, ratio rules from 5000) is what makes it
runnable at all — a second, independent instance of the same startup-transient
diagnosis that produced the amendment.

### Early validation — a scientific yellow flag, not a technical one

Smoke val_freq is 2000, so the only iteration matched against the real arms is
**4000**. LR confirmed identical at that point (3.000e-04, first cosine period is
flat), same seed, same data order, so the comparison is fair.

| arm | val PSNR @4000 |
|---|---|
| E1-addition-render | **21.2728** |
| E1-addition-noisy | 19.6836 |
| E0-Fixed | 19.4023 |
| **global-render** | **17.3166** |

global-render at 2000/4000/6000: 18.076 / 17.317 / 18.235.

So the pooled arm is running **below E1-render by ~3.9 dB and below even the E0
baseline** at matched iterations. If that holds, it is exactly the answer this
ablation was built to get: E1-render's advantage would come from DINO's SPATIAL
LAYOUT, not from a global descriptor of the render — and a single broadcast
vector would be actively worse than no prior at all.

**It must not be reported as a result yet.** Three validation points, 6000 of
300,000 iterations, single seed, and early val PSNR is demonstrably unstable on
this data (E0 itself went 19.40 @4k → 19.22 @20k before climbing). This is a
hint about where the full run will land, nothing more.

### Verdict

Technically clear to launch: the gate question the smoke existed to answer is
answered, and answered safely. The early val signal is the experiment working,
not a reason to withhold it.

---

## 2026-08-14 — LAUNCHED, and in flight

Submitted by the user after the 6k smoke passed. Job **1776459** (a100, chain
job #1 of max 6), successor 1776464 queued `afterany`. START FRESH — no prior
experiment directory, checkpoint or state.

**Progress: 88,000 / 300,000.** Best val so far **20.5911 @ 60,000**, last
20.1651 @ 84,000.

Stability at 88k: `latent_norm` 7.528e+03, `projected_norm` 3.146e+03,
`injection_ratio` **0.418**. Still roughly half of E1-render's ~1.03 and far
inside the cap of 10. No gate rule has fired.

**The early signal from the smoke run is holding.** At 88k this arm sits
**below the E0 baseline** (22.075) by ~1.5 dB and roughly 3.5 dB below
E1-render (24.117). If that persists to 300k it answers the question the arm was
built for:

> E1-render's +2.043 dB comes from DINO's SPATIAL LAYOUT of the render, not from
> a global descriptor of it. Pooling to one broadcast vector does not merely
> lose the advantage — it lands worse than injecting no prior at all.

That is a strong mechanistic result, and it makes the render arm's gain
specifically about *where* structure is, not merely *what object* is present.

**Not a conclusion yet.** 88k of 300k, single seed, and E1-noisy is a live
reminder that an arm can peak early and decline (it peaked at 128k) — the
reverse is equally possible. Wait for 300k.

**Co-inflation watch** (still ungated): latent 7.53e+03 and projected 3.15e+03
at 88k. Both are large in absolute terms — E1-render sat at 5.21e+03 / 5.06e+03
at 61k — so read both columns of `dino_stability.csv`, not just the flat ratio.
