# affm-render — the LAYER-COUNT ablation of addition-render

`Holo_affm_render_fixed128_spatial_L3691_latent`

Scope: **this arm only**. Every other Phase-3 arm is read-only from here.
Written **before the run**, so the prediction below is a pre-registration and
not a description of a result that already exists.

---

## THE DECISION

**A multi-layer DINO arm over blocks {3, 6, 9, 12}, fused with DINOLight's
Adaptive Feature Fusion Module (arXiv 2603.12579), injected by the SAME
mechanism as addition-render.**

```
addition-render   D = centered B6 grid                              -> P -> F + P(D)
affm-render       D = sum_l W_l * centered D_l,  l in {3,6,9,12}    -> P -> F + P(D)

  s_l   = Conv2d(768, 1, 1)( GELU(D_l) )              [B, 1, 16, 16]
  W     = softmax([s_3, s_6, s_9, s_12], dim=1)       [B, 4, 16, 16]   sums to 1
  D_fus = sum_l W[:, l:l+1] * D_l                     [B, 768, 16, 16]
```

The softmax runs **across layers at each spatial position**, so the four weights
sum to 1 at every one of the 256 positions. The output stays **768 channels**
because it is a weighted SUM, not a concatenation — which is the entire reason
this fusion was chosen (see PARAMETERS).

**AFFM is adopted from DINOLight and cited. It is not claimed as a contribution
of this thesis.** What is ours is the question it is being used to answer:
does reading DINO at four depths beat reading it at one, on radar data, under
an otherwise identical recipe.

**The injection operator is UNCHANGED**: the same zero-initialized
`Conv2d(768, 384, 1)` residual added at `inp_enc_level4`. No attention, no
second injection point, no gate change. One factor at a time.

---

## DEVIATIONS FROM DINOLight, ALL DELIBERATE

Verified against the paper (arXiv HTML, 2026-08-21), not from memory:

| | DINOLight | this arm | why |
|---|---|---|---|
| layers | **{1, 6, 12}** ("layers 1,6,12 of a distilled ViT-B/14 DINOv2 with registers") | **{3, 6, 9, 12}** | ours CONTAINS B6, so this arm differs from addition-render by layer COUNT alone; and all four have a measured fixed-128 correspondence on radar (WO1 Task 1.3, n=339). B1, B4, B8 have never been measured here. |
| encoder | ViT-B/14 **with registers** | ViT-B/14, **0 register tokens** | ours is the checkpoint every previous phase used; changing it would invalidate Phase 0/1/2 and every finished arm. Asserted at build time (`build_dino` raises on a non-zero register count). |
| DINO input | the **degraded image itself** ("given a degraded image I_D ... we first extract DINOv2 features") | the aligned **clay render** | our own source ablation settled this: DINO on the 1e5 radar loses 0.58 dB, DINO on the render gains 2.21 dB. |
| centering | none stated ("processed by a SiLU activation followed by a 1×1 convolution layer") | per-layer, train-only [768] means, applied **before** the scoring conv | our pre-E1 analysis found the usable signal on this data lives in the centered residual; raw 1e5↔render similarity is almost entirely a shared common component. |
| activation | **SiLU** | **GELU** | this repo standardises on GELU (`restormer_arch.py:91`). Not load-bearing — it only shapes the scoring conv's input. Recorded, not hidden. |
| injection | auxiliary **cross-attention** | **addition**, unchanged from our reference | our cross-attention arm is a separate, already-run experiment. Changing fusion AND layer count at once would confound them. |

---

## PARAMETERS — THE POINT OF USING AFFM

```
  P              768*384 + 384 = 295,296     unchanged from addition-render
  AFFM score x4  4 * (768 + 1) =   3,076
  ------------------------------------------------------------------
  delta over E0                 = 298,372
```

Measured: E0 26,124,052 -> affm-render 26,422,424 trainable (excluding the
frozen ViT, which no checkpoint holds).

Against addition-render's **295,296** that is **+3,076, i.e. +1.04%**. This arm
is very nearly **parameter-matched to its own reference**, and that is the main
thing the design buys: **a decline across the layer ladder cannot be explained
by capacity.** A naive four-layer concatenation would have needed
`Conv2d(3072, 384, 1)` = **1,180,032** parameters and would have reintroduced
exactly the confound that made concat-render's +50% hard to read.

---

## PRE-REGISTERED PREDICTION — written before the run

**Multi-layer does NOT beat single-layer B6 by more than 0.10 dB.**

Rationale, from our own measurements at the actual fixed-128 training scale
(WO1 Task 1.3, centered, n=339):

| block | same-scene | different-scene | advantage |
|---|---|---|---|
| B3 | 0.6091 | 0.4124 | +0.1966 |
| **B6** | **0.6694** | 0.5241 | +0.1453 |
| B9 | 0.5482 | 0.4235 | +0.1247 |
| B12 | **0.2337** | 0.1175 | +0.1162 |

B12's same-scene correspondence is 0.2337 against a 0.1175 floor, and B9 is also
below B6. Adding them to B6 should **dilute** rather than add.

**CONTRARY EVIDENCE, RECORDED IN ADVANCE.** DINOLight's own ablation (Table 2)
found multi-layer beat single-layer: **22.207** (Model A, shallow only) →
**22.414** (all three layers, no AFFM) → **22.600** (with AFFM). If our result
matches theirs instead of our prediction, **that is a surprise and must be
reported as one**, not retrofitted into an expectation. (Caveat on that quote:
Model A is described as *shallow only*; whether it is the best of the three
single-layer variants is not something I could confirm from the text I read.)

**SECONDARY OUTCOME, INDEPENDENT OF PSNR.** The logged AFFM weights are a result
in their own right. If `w_b6` grows while `w_b12` decays toward zero, the
network has independently reproduced our feature-space ranking from a completely
different objective. Report the trajectory **either way** — a flat trajectory,
or one that prefers B12, is equally informative and equally publishable.

**KNOWN LIMIT.** AFFM is a soft average, so a useless layer is down-weighted but
**never removed**. Nothing forces `w_b12` to zero, and a residual weight on a
weak layer is not evidence that the layer helps.

---

## WHAT IS HELD IDENTICAL TO addition-render

Splits, crop protocol (fixed 128), augmentation, seed (100), Restormer
hyper-parameters, optimizer (AdamW, lr 3e-4, wd 1e-4), scheduler
(`CosineAnnealingRestartCyclicLR`, periods [92000, 208000], eta_mins
[3e-4, 1e-6]), total_iter 300000, batch 8, L1 loss, val_freq 4000, checkpoint
every 2000, the injection operator, and the stability gate (ratio cap 10, ratio
rules from iteration 5000, NaN/Inf from iteration 1).

**The config differs from `E1_addition_render_fixed128_spatial_B6_latent.yml` in
exactly four places**: `name`, `network_g.type`, the layer set
(`dino_layers` + `dino_fusion` + `dino_affm_stats_freq`, replacing the
single-block reading of `dino_block`), and the mean paths (one pair -> one pair
per layer). Nothing else, key for key.

**300k is not negotiable.** The periods sum to exactly 300,000, so a truncated
run is a different LR trajectory, not this recipe stopped early, and would not
be comparable to the finished arms.

---

## INITIALISATION AND MONITORING

- **Scoring convs zero-initialised** (weight and bias) -> every score is 0 ->
  the softmax starts **exactly uniform at 0.25 per layer**, deterministically
  and seed-independently. The arm's starting prior is the unweighted mean of the
  four centered grids. Asserted on a real batch in the smoke test.
- **`P` zero-initialised** -> `P(D_fused) == 0` -> the step-0 output is E0's
  exactly. Asserted with `torch.allclose(atol=1e-6)` against a stock E0 built
  from E0's own config with the same seed.
- **THERE IS A ONE-STEP GRADIENT STAIRCASE, and the work order predicted there
  would not be.** The prediction is incompatible with the zero-init of `P` that
  the same work order requires for step-0 equality with E0, so the two could
  never both have held. `D_fused` *is* non-zero from step 0 (a zero score still
  gives a weight of 0.25), which is why `P` takes gradient immediately. But
  `D_fused` reaches the loss ONLY through `P`, and `d(P(D))/dD == P.weight ==
  0`, so the AFFM scoring convs receive **exactly zero** gradient on the first
  backward. Measured on a real batch:

  | | step 1 | step 2 | step 3 |
  |---|---|---|---|
  | `max abs dL/dP` | 1.634e-02 | 2.117e-02 | 8.183e-03 |
  | `max abs dL/dAFFM` (B3) | **0** | 1.828e-03 | 5.284e-03 |
  | `max abs dL/dAFFM` (B6/B9/B12) | **0** | 6.8e-05 / 2.5e-04 / 1.0e-04 | ... |
  | `max abs P.weight` | 0 | 3.000e-04 | 6.004e-04 |

  The staircase is **one step deep and it clears**; this is the same structure
  the attention arms have, one level shallower. The smoke test now asserts the
  true invariant — zero at step 1, non-zero at step 2 — instead of the one the
  work order expected. **Nothing was changed in the design to make this go
  away**: removing it would mean giving up either the zero-init of `P` or
  step-0 equality with E0, and both are worth more than a one-step delay.
- **RNG order.** The parent builds the whole Restormer trunk first; the ViT and
  `P` are built inside the parent's RNG save/restore fence; the AFFM is built
  inside **its own** fence in this class. Every trunk weight is byte-identical
  to E0's for the same seed, and every later draw (data order, crop offsets,
  augmentation flags) is unshifted. Asserted tensor by tensor.
- **Gate telemetry unchanged**: `dino/latent_norm`, `dino/projected_norm`,
  `dino/injection_ratio`, same tags, same `dino_stability.csv`, same rules.
- **AFFM weights**, every 5000 forwards: the position-averaged softmax weights
  are published through the model wrapper's existing generic observation hook,
  arriving as **`dino/affm_w_b3`, `dino/affm_w_b6`, `dino/affm_w_b9`,
  `dino/affm_w_b12`** plus `dino/affm_w_sum`.

  **TWO THINGS A READER MUST KNOW.** (1) The prefix is `dino/`, not `affm/`:
  the model wrapper prefixes every observation with `dino/`, and an `affm/`
  prefix would have required editing a file five finished arms also run. That
  edit was refused on isolation grounds; the names are unambiguous as they
  stand. (2) The value is a **step function** — measured every 5000 forwards,
  reprinted at every `print_freq` of 1000 — so it must be de-duplicated before
  plotting. This is the same trap the crossattn attention statistics set.
- **No gate rule is defined on the AFFM weights**, deliberately. They are
  observations. A stopping criterion on them would turn a result into a
  constraint.
- **Weight MAPS**, not just their means, are dumped for a fixed val batch at
  5k / 100k / 300k by `scripts/dump_affm_weight_maps.py`, run **after**
  training. Flat maps mean the layer choice is global; structured maps mean the
  network picks different depths in different places. Those are different
  findings and the position-averaged log cannot distinguish them.

---

## CENTERING — FOUR MEANS, AND ONE OF THEM IS A CORRECTNESS CHECK

Each layer is centered with its **own** position-independent [768] train-only
mean, computed **before** AFFM sees it, because the four blocks sit in genuinely
different places in feature space. Centering all four against B6's mean would
bias three of them by a constant and make the scoring convs read that bias
rather than the content.

Computed by `scripts/compute_affm_means.py`, which **imports
`compute_production_means` and rebinds only its block constants**, so the
sampling, crop draw, preprocessing, accumulation dtype and metadata are the same
code objects that produced the existing B6 mean. 1000 train images each, seed 0,
float64 accumulation, float32 storage, train split asserted.

| regime | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| train128 (224, 256 tokens/img) | 70.9559 | 56.8906 | 45.1594 | 24.1637 |
| eval256 (448, 1024 tokens/img) | 77.4982 | 61.6951 | 51.3676 | 27.5146 |

(‖μ‖; all shape [768], all finite, all four distinct.)

**THE CHECK.** B6 was recomputed through the new path into a scratch directory
and compared element-wise against the file addition-render already trains with.
The production file was **never** overwritten.

```
B6 train128   max abs diff = 1.192e-06   (relative to max|mu| = 3.369e-08)  PASS
B6 eval256    max abs diff = 1.907e-06   (relative to max|mu| = 5.832e-08)  PASS
```

That is GPU float32 non-determinism, not a path difference. The extraction and
preprocessing this arm uses are the ones addition-render trains with. The config
additionally points the B6 entry at the *same file*, and the arch refuses to
build unless `mu_b6_* == mu_*` under `torch.equal`.

---

## ONE IMPLEMENTATION NOTE WORTH KEEPING

The first smoke run **failed on purpose**, and correctly: the parent's
`_load_mean` validates a mean's metadata `block_1indexed` against
`self.dino_block_1indexed`, which is 6, so loading the B3 mean through it was
refused with `block 3 != B6`. That guard is right for a single-layer arm and was
**left alone**. This class got its own `_load_layer_mean`, which performs every
one of the same checks — width, finiteness, DINO input size for the regime,
train-only provenance — with the block made an explicit argument, plus one the
parent does not have (`domain == render`). The guard was parameterised, not
weakened.

---

## THE SMOKE RUNS, AND THE TWO BUGS THEY CAUGHT

Three levels of verification were run before launch. The architectural suite
(`smoke_tests_affm_render.py`, **72/72 PASS** on both CPU and GPU) covers the
module in isolation. It did NOT catch either of the following, because both
live in the basicsr integration path — which is exactly why a training smoke
run exists.

**BUG 1 — the real run would have crashed at startup, before iteration 1.**
`basicsr/utils/options.py:109` builds its startup option dump with
`k + ': '`. The per-layer mean maps were written with **bare integer** YAML
keys (`3:`, `6:`, ...), so that line raised
`TypeError: can only concatenate str (not "int") to str` during
`init_loggers`, before the model was even constructed. Fixed in **this arm's
config** by quoting the keys (`'3':`); `basicsr/utils/options.py` was not
touched, and the arch accepts either form (`_key`). A 300k a100 job would have
died in its first seconds.

**BUG 2 — the AFFM observations would have been silently absent at freq 1.**
The publish condition was `count % freq == 1`, copied from the attention arms'
idiom. That is correct for every `freq > 1` — including the real config's 5000,
which fires at forwards 1, 5001, 10001 — but at `freq == 1` it **never fires**,
because `count % 1` is always 0. The CPU integration run sets freq 1 and
produced six log lines with no `dino/affm_w_*` tags at all. Changed to
`(count - 1) % freq == 0`, which is identical for freq > 1 and correct at 1.
**The real config was never affected**, but the idiom was one config value away
from producing a 300k run with no observations. Noted, not fixed elsewhere: the
crossattn arm carries the same idiom at freq 5000, where it behaves correctly;
that arm is finished and was not touched.

**CPU integration run** (`--iters 6 --batch 2 --cpu --no-val --affm-freq 1`,
throwaway identity `SMOKE_INT_affm_render_L3691`) then completed end to end:

| iter | latent_norm | projected_norm | injection_ratio | w_b3 | w_b6 | w_b9 | w_b12 | sum |
|---|---|---|---|---|---|---|---|---|
| 1 | 43.62 | **0.0000** | **0.0000** | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 3 | 65.28 | 21.66 | 0.3318 | 0.2568 | 0.2427 | 0.2371 | 0.2634 | 1.0000 |
| 6 | 70.44 | 42.73 | 0.6066 | 0.2604 | 0.2320 | 0.2186 | 0.2891 | 1.0000 |

`projected_norm == 0` at iteration 1 confirms step-0 equality with E0 **in the
real pipeline**, not only in the unit test. `injection_ratio` rises from 0
exactly as the gate amendment documents. The weights leave 0.2500 and the sum
stays 1.0000 to four decimals throughout. Checkpointing, `dino_stability.csv`
and training-state saving all work; no `STABILITY_FAILURE` was written; the
checkpoint reloads with all 8 AFFM tensors and all 10 mean buffers.

**DO NOT READ A TREND INTO THAT TABLE.** Six iterations at batch 2 on CPU is an
integration check, not evidence. It is worth one line only because the early
direction — `w_b12` **rising** and `w_b9` falling — runs opposite to the
pre-registered expectation. At this sample size that is noise, and it is
recorded here so that nobody later mistakes it for an early confirmation of
anything.

**Two limitations of the CPU run, both artefacts of having no GPU:** batch was
2 rather than 8 and validation was skipped, because basicsr's final
`model.validation()` call is unconditional and its path reaches CUDA even at
`num_gpu: 0`. The eval256 path is covered by the architectural suite instead
(32x32 tokens, correct mean buffers, no interpolation), and the 6000-iteration
GPU smoke run covers validation and the enforced ratio rules.

### The 6000-iteration GPU smoke run (job 1787032, v100, gate ON)

`rc=0`, ran to 6000, **no STABILITY_FAILURE written**, and the ratio rules were
genuinely enforced over iterations 5000-6000 rather than only measured.

| iter | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|---|
| latent_norm | 298 | 552 | 732 | 781 | 976 | 849 | 1040 |
| projected_norm | 404 | 539 | 765 | 933 | 922 | 937 | 1029 |
| **injection_ratio** | 1.354 | 0.977 | 1.045 | 1.193 | 0.945 | 1.103 | **0.990** |

**`injection_ratio` settles at ~0.94-1.19 against addition-render's ~1.03** —
the same regime, nowhere near the cap of 10, and without the 2.3-6.6 startup
transient the 1e5 arm showed. Validation (the eval256 switch, 32x32 tokens) ran
three times: **21.18 / 21.41 / 22.06 dB** at 2k / 4k / 6k.

**AFFM weight trajectory**, 60 distinct measurements at `affm_stats_freq: 100`:

| iter | w_b3 | w_b6 | w_b9 | w_b12 | sum |
|---|---|---|---|---|---|
| 100 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 600 | 0.2983 | 0.3902 | 0.1034 | 0.2082 | 1.0000 |
| 2100 | 0.3987 | 0.1635 | 0.2818 | 0.1559 | 1.0000 |
| 4100 | 0.3191 | 0.2992 | 0.1466 | 0.2352 | 1.0000 |
| 6000 | **0.3336** | 0.2323 | 0.2112 | 0.2228 | 1.0000 |

Net movement from uniform: **B3 +0.084**, B6 −0.018, B9 −0.039, B12 −0.027. The
sum is 1.0000 at every one of the 60 measurements.

**READ THIS CAREFULLY AND DO NOT OVERSTATE IT.** 6000 of 300000 iterations, each
weight measured on a SINGLE training batch, and the trajectory is visibly noisy
(w_b6 goes 0.39 at 600, 0.16 at 2100, 0.30 at 4100, 0.23 at 6000). It is not
evidence of a converged preference. It is worth recording for exactly one
reason: the only layer gaining weight so far is **B3**, and B3 is the layer that
wins the *scene-advantage* criterion (+0.1966 vs B6's +0.1453) — the documented
tension behind the B6 lock. If that holds to 300k it is a second, independent
line of evidence on a conflict this project has so far only been able to argue
about. If it does not hold, this paragraph stands as a record that the early
signal was noise.

### Peak VRAM (job 1787031, v100 32GB, batch 8, crop 128)

| arm | peak allocated | peak reserved |
|---|---|---|
| addition-render (reference, same card, same job) | 25,117.4 MiB | 25,914.0 MiB |
| **affm-render** | **25,166.4 MiB** | 25,962.0 MiB |

**Ratio 1.0020** — the four-layer extraction costs 0.2% more memory than the
single-layer arm, because the four token grids are extracted under `no_grad` and
only the fused [B,768,16,16] result carries gradient. Comfortable on a100 40GB.
(A first attempt on rtx3080 OOM'd for BOTH arms, reference included — a
batch-8 128-crop step of this model does not fit in 10 GB. That is a fact about
the card, not about this arm.)

---

## STATUS

Built, verified, **not launched**. Nothing has been trained. Entries below this
line are appended by the run itself.

---
