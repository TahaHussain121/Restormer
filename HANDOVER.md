# HANDOVER — Phase 3 restoration, as of 2026-08-24

Read this + `CONTEXT.md` + `DEVLOG.md` at the start of a new session. This file
covers the **Phase-3 restoration experiments** specifically; `CONTEXT.md` is the
project-wide primer and `DEVLOG.md` is the append-only step log.

> **This file replaces the old HANDOVER.md**, which documented the abandoned
> DINOv2-**FiLM** experiment and was deleted in commit `6216542` (DEVLOG Step
> 28). That experiment is dead and is not being resumed — its code lives on
> branch `dino_prior`, its post-mortem in DEVLOG "Steps 20-26". Nothing in this
> file is FiLM. If you are reading advice about `gamma`/`beta` runaway, you are
> reading the old file out of git history.

---

## One-line status

**Two arms are training right now and will finish today; everything else is
done.** Reading the **render** gains **+2.208 dB** on test/full256 and reading
the **noisy radar** *loses* **0.577 dB** — that result is settled and unchanged.
Eight arms now exist. The two in flight are **affm-render** (layer count alone)
and **dinolight-render** (DINOLight's published method: layer count AND operator).
Both are at ~241k/300k, both gates clean, both projected to finish **2026-08-24
~18:00**, well inside the **2026-08-28** cluster shutdown.

**Nothing in the fusion ladder has beaten plain addition yet**, and the two live
arms are not separating from it either — see §1a before quoting any number.

---

## §1a. THE TWO LIVE ARMS — read this before quoting their numbers

At 240k on the **8-bit training-time validation** curve: affm **24.364**,
dinolight **24.239**, against addition-render's best-ever **24.117** on the same
curve. That looks like a win. **It is not yet a result, for four reasons:**

1. **Wrong metric for comparison.** This is the 8-bit training-time val path.
   The published 24.081 test / 24.120 val figures come from the **uint16
   evaluation**. The two are not comparable. Neither live arm has been evaluated
   on the real path yet.
2. **Point noise is ~0.4 dB.** E0 moves 21.738 -> 21.328 between 180k and 188k
   on this same curve. Differences under ~0.3 dB at a single iteration mean
   nothing.
3. **No checkpoint selection has happened.** addition-render peaked at 204k,
   concat at 292k. Best-val selection is pre-registered and has not been run.
4. **Precedent.** concat-render looked +0.025 ahead on this curve and the full
   evaluation showed a statistical tie.

**The honest reading today: both live arms sit in the render-arm cluster (~24 dB)
alongside addition and concat, far above E0 (~21.5), E1-noisy (~21) and
global-render (~19.7). Nothing separates them from addition-render.** Both
pre-registered predictions (affm "not >0.10 dB", dinolight "not >0.30 dB") are
so far holding.

---

## READ FIRST — the two things most likely to be misquoted

**1. crossattn-render is not a clean operator failure.** Same weights, val n=339:

| checkpoint | crop128 | full256 |
|---|---|---|
| E0-Fixed 268k | 19.816 | 22.077 |
| crossattn **4k** (the selected one) | 18.284 | 18.563 |
| crossattn **178k** (final) | **21.812 (+2.00)** | **14.517 (−7.56)** |

Checkpoint selection uses the training-time **full-256** validation — the one
regime this arm cannot do — so it selected iteration 4,000 out of 179,000, and
every published crossattn number comes from that model. This is exactly the
**pre-registered matched-128 escape clause**, and it was never invoked. Using
the last checkpoint instead is *not* a legitimate selection rule either, so
nobody should quote +2.00 dB as a result — it is a diagnostic fact that the
selection metric and in-distribution performance are **anti-correlated** for
this arm.

**2. No paper in our reference set does what our attention arms do.**
Perceive-IR's prior is `F_l ∈ ℝ^{1×768}` — a global vector — and PGM's
FiLM-style affine runs *before* PGCA, so the attention query is the
prior-modulated feature, not a prior token grid. Whether PGCA's softmax is
spatial or Restormer-MDTA channel attention is **not stated in the paper**; it
does not matter, because a 1×768 prior has no spatial tokens to attend over
either way. Both our attention arms are novel constructions. DSGIR is paywalled
and was **not** verified — what the repo says about it is second-hand.

## THE RESULT — FINAL, on the locked TEST split (n=338)

Checkpoints selected on **validation** alone (E0 268k, E1-noisy 128k, E1-render
204k); the test split was read once, after all three arms completed 300k.
Jobs 1776745-1776750.

| arm | DINO reads | **test full256** | vs E0 | **test crop128** | vs E0 | improved (full256) |
|---|---|---|---|---|---|---|
| **E0-Fixed** | — | 21.873 | baseline | 19.546 | baseline | — |
| **E1-addition-noisy** | 1e5 radar | **21.296** | **−0.577** | 19.062 | −0.484 | 111/338 (32.8%) |
| **E1-addition-render** | render | **24.081** | **+2.208** | **22.259** | **+2.713** | **298/338 (88.2%)** |
| global-render | render, pooled | *in flight* | — | — | — | 90k/300k |

| metric (test, full256) | E0 | E1-noisy | E1-render |
|---|---|---|---|
| PSNR object mask | 17.599 | 17.210 | **19.673** |
| SSIM whole / mask | 0.783 / 0.560 | 0.762 / 0.540 | **0.822 / 0.643** |
| HF energy ratio | 0.218 | 0.275 | **0.328** |
| Laplacian / Sobel ratio | 0.286 / 0.764 | 0.370 / 0.779 | **0.438 / 0.871** |

**Verdict against the pre-registered thresholds** (>+0.30 dB = meaningful):
E1-render is **MEANINGFUL on both protocols**; E1-noisy is negative on both.
Test confirmed validation (val was +2.043 / −0.608) rather than overturning it.

**The matched-128 escape clause does not apply.** It was registered as "if E1
improves on matched-128 but not full-256, the prior is useful in-distribution and
scale transfer is the limiter". E1-render improves on **both**, and by *more* at
crop128. There is no scale-transfer limitation to invoke.

**E1-render is the best model this project has produced** — 24.081 dB test beats
the old progressive baseline (22.405) by **+1.676 dB**. E0-Fixed itself scores
21.873, i.e. 0.53 dB *below* that old baseline, exactly as registered in advance
(it never trains at the evaluation resolution). That does not weaken the Phase-3
comparison, which is internal — all arms share E0-Fixed's recipe exactly.

⚠ **HF-ratio framing, corrected.** The old baseline's motivating 0.216 was
measured on **test**. Test-to-test, E0-Fixed is **0.218 — indistinguishable**,
not "marginally worse" as an earlier val-vs-test comparison suggested. Quote
0.218 (test) or 0.200 (val), matched to the split of whatever sits beside it.
Also do not read the crop128 HF ratios (~0.9 for every arm): on a 128 crop the
HF band is narrow and the statistic is unstable (std ~0.47).

Validation, for reference: E0 22.077 · E1-noisy 21.469 (−0.608, 100/339) ·
E1-render 24.120 (+2.043, 290/339); best val at 268k / 128k / 204k.

**What the pairing licenses.** E1-noisy is the *same code* as E1-render with one
tensor swapped — identical parameter count (+295,296), seed, schedule, crop,
fusion and gate. They land on opposite sides of the baseline. So the gain is
attributable to the DINO **input**, not to added capacity and not to "DINO
features" generically.

**What it does NOT license.** The render is a clean view of the same object and
carries the target's geometry almost directly. Write the claim narrowly: *a
clean geometric view of the object, delivered through frozen DINO features,
substantially improves restoration — while the identical mechanism fed the noisy
radar makes it worse.* The render **is** normally available in this pipeline, so
this is a usable method, not an oracle — but do not overstate the source.

**Three findings that are easy to miss:**
1. **E1-noisy is sharper but less accurate** — HF ratio 0.280 vs the baseline's
   0.200 while losing PSNR. It *hallucinates* structure (a spurious bright cross
   on val image 4467), a different failure from over-smoothing.
2. **E1-noisy helps where the baseline is desperate** — on E0's hardest decile it
   is **+0.478 dB, 20/34 wins**, while negative overall.
3. **E1-render helps most exactly there too** — hardest decile **+3.07 dB,
   31/34 wins**, versus its own +2.043 average.

Figures: `results/comparisons/three_arm_{harsh,median}_full256_val.png`.
**The test split has now been read**, once, for these three arms. global-render
will need its own test pass when it finishes; because selection is per-arm on
validation, adding it later contaminates nothing.

---

## 1. Run status (verified 2026-08-24 11:00)

| arm | iters | checkpoint | test full256 | vs E0 | state |
|---|---|---|---|---|---|
| **E0-Fixed** | 300k | 268k | 21.873 | baseline | done |
| **E1-addition-noisy** | 300k | 128k | 21.296 | −0.577 | done |
| **E1-addition-render** | 300k | 204k | **24.081** | **+2.208** | done |
| **global-render** | 300k | 60k | 20.589 | −1.284 | done — pooling HURTS |
| **concat-render** | 300k | 292k | 24.065 | +2.193 | done — ties addition |
| **crossattn-render** | 179k | 4k | 18.723 | −3.150 | stopped; see READ FIRST |
| **priorquery-render** | 90k | — | — | — | **DROPPED** (cancelled 2026-08-23) |
| **affm-render** | **243k** | — | — | — | **RUNNING** job 1788585, tg097 |
| **dinolight-render** | **241k** | — | — | — | **RUNNING** job 1789321, tg091 |

**The live jobs.** Each has ~17 h of walltime left and needs ~7 h, so both
should finish inside their current job. Each has a self-queued successor
(1791652 affm, 1791655 dinolight) sitting on `Dependency` — that is the HEALTHY
state, not a problem; `chain_core` cancels the successor when `net_g_300000.pth`
appears. Both are on `CHAIN_COUNT=2` (job 1 hit its 24 h walltime at 188k, as
designed, and resumed from `188000.state`).

**Two stale locks, both to be LEFT ALONE**: crossattn-render
(`RUNNING_JOB=1784331`) and priorquery-render. Both jobs were cancelled by hand;
deleting the lock would let a dead arm auto-resume.

### WHAT TO DO WHEN THE TWO ARMS FINISH

1. Confirm `TRAINING_DONE` exists in each `Phase3_chain_state_*` dir and that
   the successors were cancelled.
2. Select checkpoints on **validation** only (pre-registered), then run
   `scripts/run_evaluate.sh` for both protocols on **val**.
3. **Do not touch the test split** until you are ready to read it once.
4. Then and only then compare against addition-render's 24.081.

---

## 2. What each arm is

Exactly one thing differs between arms. Everything else — splits, fixed 128
crop, augmentation, seed 100, Restormer hyper-parameters, AdamW, LR 3e-4,
scheduler `[92000, 208000]`, 300k iters, batch 8, L1, val frequency, metrics —
is held identical.

```
E0:  1e5 → Restormer → predicted 1e7

E1-addition-noisy:
     1e5 ─┬─────────────────────────────→ encoder → inp_enc_level4 ─┐
          └→ frozen DINOv2 B6 → centering → P (1×1, zero-init) ─────┤ +
                                                                     ↓  latent → decoder → 1e7
E1-addition-render:
     1e5 ──────────────────────────────→ encoder → inp_enc_level4 ─┐
     render → frozen DINOv2 B6 → centering → P (1×1, zero-init) ───┤ +
                                                                     ↓  latent → decoder → 1e7
```

- **The injection is a zero-initialized residual projection**, not FiLM. No
  learnable alpha, no gate, no multiplicative path, no cross-attention, no
  concatenation, no decoder injection.
- `P = nn.Conv2d(768, 384, 1)` with zero weight **and** zero bias, so `P(D) = 0`
  at step 0 and E1's step-0 output is identical to E0's — while
  `dL/dW = dL/d(guided) ⊗ D` is still non-zero, so the branch can grow.
- **Parameter delta = 295,296** exactly (`768×384 + 384`).
- **1e7 is target only.** Never conditioning, guidance, centering statistics or
  auxiliary input.
- **The render is normally available** in this pipeline, so E1-render is a
  usable method, not merely an oracle. Say so when reporting it.

### The four later arms, in one line each

| arm | the one thing that differs | params vs E0 |
|---|---|---|
| **global-render** | the centered grid is POOLED to `[B,768,1,1]` and broadcast — tests whether the render's value is spatial | +295,296 |
| **concat-render** | `fuse(cat([F, D]))`, a 1x1 1152→384, replacing the addition; `P` deleted | +442,752 |
| **crossattn-render** | `F + W_o(MHCA(Q=F, K=V=D))` — radar queries, DINO keys/values. **Direction reversed from the papers on purpose** | +888,576 |
| **priorquery-render** | `Q=D, K=V=F` — the papers' direction. DINO content never reaches the output; it is a ROUTER over radar positions, not a content source | +741,120 |
| **affm-render** | LAYER COUNT: {3,6,9,12} instead of {6}, per-layer centering, AFFM softmax across layers. Injection UNCHANGED | +298,372 |
| **dinolight-render** | layer count AND operator: the same AFFM, then `guided = project_out(F_sa + alpha*F_ca) + F` — gated CHANNEL cross-attention. DINOLight's published method | +1,352,849 |

### The two live arms, in more detail

**affm-render** is the clean one-factor arm. Four DINO depths, each centred with
its own train-only mean, combined by a per-position softmax ACROSS LAYERS (the
four weights sum to 1 at each of the 256 positions). The output stays 768
channels because it is a weighted SUM, so `P` is unchanged and the arm is within
**1.04%** of addition-render's parameter count — *a loss here cannot be blamed on
capacity*, which is the entire point of using AFFM instead of a concat.

**dinolight-render** changes TWO factors and is the largest arm in the ladder at
**4.58x** addition-render. It is a "does the published method transfer to our
data" arm, **not an ablation** — say so whenever it is reported. Stage 1 is the
SAME code object (it imports `DinoAffm` from the affm arch), then:

```
D_proj = P(D_fused)                     [B,384,16,16]
X = LN(F)   X' = LN(D_proj)
Q,K,V,Q' <- X      K',V' <- X'          each = 1x1 conv + 3x3 depthwise (MDTA style)
F_sa = TransposedAttn(Q,K,V)   F_ca = TransposedAttn(Q',K',V')
guided = project_out(F_sa + sigmoid(alpha_logit)*F_ca) + F
```

**THE ATTENTION IS CHANNEL-TRANSPOSED, NOT SPATIAL, AND THAT IS THE POINT.** The
matrix is `C/heads x C/heads = 64x64` at BOTH scales — verified on real forwards
at 16x16 and 32x32 tokens. crossattn-render used a spatial 256x256 softmax and
scored +2.00 dB at crop128 vs −7.56 dB at full256 from identical weights. This
operator structurally cannot have that failure mode.

**`alpha` is the headline non-PSNR diagnostic.** It is a gate the network can
CLOSE — decay toward 0 means "the prior does not help, falling back to plain
self-attention", a clean interpretable negative that crossattn could not produce.
Observed: 0.1192 -> ~0.139, risen then plateaued, so the path is being used
modestly.

### THE AFFM WEIGHTS ARE A RESULT IN THEMSELVES, AND THEY SURPRISED US

Both arms log the per-layer softmax weights every 5k iterations. At ~185k:

| | w_b3 | w_b6 | w_b9 | w_b12 |
|---|---|---|---|---|
| affm-render | **0.172** (lowest) | 0.229 | **0.333** (highest) | 0.266 |
| dinolight-render | 0.249 | **0.211** (lowest) | 0.231 | **0.309** (highest) |

**B6 is not winning in either arm**, and **B12 — the layer WO1 put nearest the
different-scene floor (0.2337 same-scene against a 0.1175 floor) — is doing
fine, and is the highest-weighted layer in dinolight.** That contradicts the
feature-space ranking the whole B6 lock rests on. The two arms also disagree
with each other about which layer wins, which argues the weights are weakly
determined rather than reading a strong signal. Report it either way; it is
pre-registered in both devlogs as a secondary outcome.

**A CORRECTION ON RECORD:** the 6k smoke runs showed B3 gaining and that was
read as an early signal. Over 185k iterations it did not hold — it was noise.
Do not repeat that reading.

**The fusion-operator verdict, and its caveat.** concat ties addition within
bootstrap CIs on three of four cells for +50% parameters; crossattn and (predicted)
priorquery fail. So the operator does not matter — **but** the crossattn scale
reversal in READ FIRST means that conclusion rests, for that arm, on a
4,000-iteration model. State the caveat when you state the verdict.

### Layer lock — B6, with a documented tension

B6 (0-indexed **5**) locked on the pre-registered primary criterion: centered
same-scene 1e5↔1e7 correspondence at the actual fixed-128 training scale
(Work Order 1, val n=339): B3 0.6091, **B6 0.6694**, B9 0.5482, B12 0.2337.

**B3 wins the same-vs-different scene advantage** (+0.1966 vs +0.1453) and the
gap is *wider* at 128 than at 256. That is a criterion conflict, not a
measurement error. A **B3 run under an identical recipe is pre-registered** as
the follow-up so it is settled empirically, not by argument.

**New, weak, and worth watching:** in affm-render's 6000-iteration smoke run the
only layer whose AFFM weight GAINS is **B3** (0.250 → 0.334; B6 −0.018, B9
−0.039, B12 −0.027). That is the scene-advantage layer winning on an entirely
different objective. It is 6k of 300k with single-batch measurements and a
visibly noisy trajectory — **not** evidence yet, but it is the second
independent hint pointing the same way.

---

## 3. Key files (all committed as of 2026-08-21 — see §6)

**Architecture / model / data**
- `basicsr/models/archs/restormer_dino_spatial_arch.py` — `RestormerDinoSpatial`.
  Trunk built first by the stock `__init__` (so RNG consumption matches E0),
  then an **RNG fence** around ViT + `P` construction so every later draw (data
  order, crop positions, augmentation flags) is identical to E0's.
  `state_dict` strips the frozen ViT; `load_state_dict` re-injects it so
  `strict=True` stays meaningful. Checkpoints are 105.8 MB / 498 entries / 0
  DINO entries.
- `basicsr/models/archs/restormer_dino_render_arch.py` — `RestormerDinoSpatialRender`,
  a subclass that re-routes **one tensor**: DINO sees `inp_img[:, 1:2]` (render),
  Restormer sees `inp_img[:, 0:1]` (radar), global residual against the radar.
- `basicsr/models/image_restoration_dino_model.py` — `ImageCleanModelDinoSpatial`.
  Explicit `train128` / `eval256` regime switching (never inferred from tensor
  size), the stability logging, and the abort gate.
- `basicsr/data/paired_radar_render_stacked_dataset.py` —
  `Dataset_PairedImage_uint16_RenderStacked`. Packs the render as **channel 1 of
  the LQ tensor**.

**Phase-3 tree** — `dino_analysis_phases/phase3_restoration/`
- `README.md` — the scientific spec: scale consistency, the two evaluation
  protocols, pre-registered thresholds, config immutability. **Read it.**
- `configs/` — one YAML per arm; `devlogs/` — one append-only devlog per arm.
- `means/` — four centering means (see §4).
- `scripts/dino_shared.py` — **the single sanctioned DINO extraction path.**
  Every Phase-3 consumer calls this and nothing else.
- `scripts/chain_core.sh` + one `chain_<arm>.sh` per arm — shared logic,
  separate identity per arm. **Never override `--partition`**: `ARM_PART` is read
  by every successor, so an override moves only the first job and silently splits
  a run across TF32 (a100) and non-TF32 (v100) regimes.
- `results/wo1_verification/`, `results/wo2_implementation/` — the verification
  evidence. **Do not modify.**

**The affm-render arm** (built 2026-08-21, not launched)
- `basicsr/models/archs/restormer_dino_affm_render_arch.py` — `RestormerDinoAffmRender`
  + `DinoAffm`. Overrides `dino_prior` only; the inherited forward does the rest.
  Own `_load_layer_mean` (the parent's guard validates against B6 and correctly
  refuses a B3 mean — it was parameterised, not weakened).
- `configs/affm_render_fixed128_spatial_L3691_latent.yml` — differs from
  addition-render's in **15 of 109 keys**, all identity / type / layer set / mean
  paths. **The per-layer mean keys must stay QUOTED strings**; bare ints crash
  `basicsr/utils/options.py:109` at startup.
- `scripts/compute_affm_means.py` — imports `compute_production_means` and rebinds
  only its block constants, so the means are produced by the same code objects.
- **To launch it** (submit ONCE; it self-chains and auto-resumes):
  ```bash
  sbatch dino_analysis_phases/phase3_restoration/scripts/chain_affm_render.sh
  ```
- `scripts/smoke_tests_affm_render.py` (72 checks), `make_smoke6k_affm_render.py`
  (+ `--iters/--name/--batch/--cpu/--no-val/--affm-freq`), `run_smoke6k_affm_render.sh`,
  `run_peak_vram_affm.sh`, `dump_affm_weight_maps.py` (run AFTER training, at
  5k/100k/300k).

**The dinolight-render arm** (built 2026-08-22)
- `basicsr/models/archs/dino_aca.py` — `DinoAca`, the gated channel
  cross-attention block. **Deliberately NOT a `*_arch.py` file** so the registry
  ignores it; it is a building block, not a network. The planned `aca-L6` arm
  (operator alone, B6 only) should IMPORT this, not reimplement it.
- `basicsr/models/archs/restormer_dinolight_render_arch.py` —
  `RestormerDinoLightRender`. Imports `DinoAffm` from the affm arch so stage 1 is
  the same code object.
- `configs/dinolight_render_fixed128_L3691_aca_latent.yml` — differs from the
  affm config in **9 of 112 keys**, all identity / type / fusion. **Reuses the
  affm arm's four mean files verbatim** (verified bit-identical), so the two arms
  are comparable on the feature side.
- `scripts/smoke_tests_dinolight_render.py` (44 checks, includes the scale
  check), `make_smoke6k_dinolight.py`, `run_smoke6k_dinolight.sh`,
  `chain_dinolight_render.sh`.
- **`chain_dinolight_render.sh` carries a HARD DEADLINE GUARD** for 2026-08-28:
  it writes `DEADLINE_REACHED` and exits BEFORE sourcing `chain_core.sh`, so no
  successor is queued. The guard lives in the arm's own wrapper and NOT in
  `chain_core.sh`, because that file is sourced by every arm and an edit would be
  picked up by a running arm's next resume, silently, hours later.
  **affm-render has NO such guard** — asymmetry worth knowing.

**The crossattn diagnosis** — `dino_analysis_phases/phase4_crossattn_diagnosis/`
- `scripts/forensics_and_series.py` — SLURM forensics + val curves + the logged
  attention statistics.
- `scripts/attention_probe.py` — dumps the real 256x256 attention matrices and
  per-head statistics from a checkpoint. **This is the script to point at eval256.**
- `scripts/inference_interventions.py` — the four interventions; self-checks
  against the published numbers before reporting anything.

---

## 4. Things that will bite you if you don't know them

**The `train.py` sub-crop only touches `lq` and `gt`.** The progressive block at
`basicsr/train.py:241-270` sub-crops and subsamples **only** `lq`/`gt` and hands
the model only `{'lq','gt'}` — a third aligned tensor would be silently
misaligned (`REPO_INVESTIGATION_REPORT.md` §N.1, "highest-probability silent
bug"). Both E1 arms dodge it *by construction*, in two different ways:
- **noisy arm**: the DINO input is derived from `inp_img` inside `forward` —
  one stream, nothing to misalign.
- **render arm**: the render rides as **channel 1 of the LQ tensor**, so the
  crop is one slice hitting both channels. **This needed zero edits to
  `train.py`** — which matters, because E0 and E1-noisy depend on that file at
  every resume.

**BasicSR auto-resume overrides your YAML.** `train.py:138-149` scans
`experiments/<name>/training_states/` and resumes from the highest `.state`.
**Reusing an experiment name silently resumes it.** Conversely `make_exp_dirs`
/ `mkdir_and_rename` **renames** an existing `experiments/<name>/` to
`_archived_<timestamp>` on a fresh start — which is why the chain lock lives in
`CHAIN_DIR`, never in `EXP_DIR`.

**Never interpolate the DINO feature grid.** 128 → DINO 224 → 16×16 tokens =
latent 16×16; 256 → DINO 448 → 32×32. ≈8 radar px/token in both regimes.
`dino_shared.assert_no_interpolation_needed` is a hard gate, not a comment.

**`get_intermediate_layers` returns ASCENDING block order** regardless of the
order you pass indices in. That was the Phase-2 bug (`4240521`). This is why
`extract_blocks` returns a **dict keyed by block index**, never a list.

**Four centering means, and they are not interchangeable.** One
position-independent [768] vector per (domain, regime), train split only, ≥1000
images:

| file | norm | tokens |
|---|---|---|
| `1e5_B6_train128_dino224_mean.pt` | 43.5031 | 256k |
| `1e5_B6_eval256_dino448_mean.pt` | 41.2580 | 1024k |
| `render_B6_train128_dino224_mean.pt` | 56.8906 | 256k |
| `render_B6_eval256_dino448_mean.pt` | 61.6951 | 1024k |

cosine(1e5, render) = **0.1799** at train128 and **0.3288** at eval256 — the two
domains genuinely sit in different places, so reusing the 1e5 mean for the
render arm would centre it against the wrong distribution. The arch validates
block / input size / split in the mean's metadata and refuses a mismatch.

**Hardware.** Restormer at 128²×batch 8 **OOMs on a 10 GB RTX 3080** (measured,
job 1773194 — that was the *stock E0 trunk alone*, not a Phase-3 defect). Use
a100 or v100 (32 GB). **Login-node compute is not an option** — the watchdog
SIGTERMs substantial CPU work (exit 143). Run everything through SLURM.
`squeue` shows only your own jobs (`PrivateData=jobs`), so it is not evidence
about how busy a partition is.

---

## 5. The stability gate (and its one amendment)

Logged every step, written to `experiments/<name>/dino_stability.csv`:
`dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`.

| rule | threshold | active from |
|---|---|---|
| NaN/Inf in loss, latent, projection or ratio | any | **iteration 1** |
| rule 1: `injection_ratio` cap | **10.0** | iteration **5000** |
| rule 2: growth vs the 5k reference | **10×** | iteration 5000 |

**Gate amendment (2026-08-12).** The original cap was 0.5 from iteration 1, and
it aborted E1 at **iteration 4**. That was *not* a pathology: `P` is zero-init,
so `injection_ratio` **must** rise from 0, and AdamW's first steps move ~lr per
weight regardless of gradient magnitude. A diagnostic run (abort disabled,
throwaway identity) showed the ratio **plateaus at 2.3-6.6 and drifts down after
iter 1000**. So the amendment changed *when the gate is valid*, not how much
injection is tolerated in steady state. The NaN/Inf stop was never windowed.

Current references: noisy **4.0398** @5k, render **1.1656** @5k. Both arms are
sitting comfortably inside the cap (noisy ~2.6, render ~1.03 at last check), and
the reference persists across resume so a walltime kill cannot silently reset it.

**A stability failure is investigated and reported, never restarted into.** The
chain scripts stop on `STABILITY_FAILURE.json` and set `CHAIN_ABORTED`. Records
are archived per (rule, iteration, job) and never overwritten.

### ⚠ Known gap: co-inflation is not gated

Both norms grow together while the *ratio* stays flat — E1-noisy went from
latent 215 / projected 872 at 1k to latent 2310 / projected 5237 at 41k. Rule 2
watches the **ratio**, so it cannot see this. Proposed rule (**not
implemented**, needs a decision): abort if `‖F_latent‖` or `‖P(D)‖` exceeds 5×
its own 5k reference. Adding it changes an experiment definition, so it needs a
new identity for any run it would apply to.

---

## 6. Git state

Branch **`dino_e2`**, tracking `origin/dino_e2` (pushed 2026-08-21). Remotes:
`origin` = TahaHussain121/Restormer (ours), `upstream` = swz30/Restormer (never
push there).

Last commit `1f343a0`. **Committed through the crossattn diagnosis and the affm
arm**, in one commit per piece of work.

**NOT YET COMMITTED (9 items):**
```
 M dino_analysis_phases/phase3_restoration/scripts/chain_affm_render.sh   (walltime 23h -> 24h)
?? basicsr/models/archs/dino_aca.py
?? basicsr/models/archs/restormer_dinolight_render_arch.py
?? dino_analysis_phases/phase3_restoration/configs/dinolight_render_fixed128_L3691_aca_latent.yml
?? dino_analysis_phases/phase3_restoration/devlogs/dinolight_render_fixed128_L3691_aca_latent.md
?? dino_analysis_phases/phase3_restoration/scripts/chain_dinolight_render.sh
?? dino_analysis_phases/phase3_restoration/scripts/make_smoke6k_dinolight.py
?? dino_analysis_phases/phase3_restoration/scripts/run_smoke6k_dinolight.sh
?? dino_analysis_phases/phase3_restoration/scripts/smoke_tests_dinolight_render.py
```
Committing is safe **while the arms train** — the files on disk are already what
they are running. **Editing is not**: a change to a shared arch or to
`chain_core.sh` is read by the NEXT RESUME, not the next iteration, so it lands
silently hours later. New arms get NEW files with NEW class names.

`.gitignore` excludes `experiments/`, `tb_logger/`, `**/results/`, `*.pth`,
`*.pt`, `*.png`, `*.log`, `*.state`. **Checkpoints, curves, prediction images,
figures and the centering-mean tensors exist on disk ONLY.** The mean
`*_meta.json` files ARE tracked, so a lost `.pt` can be recomputed with a known
recipe.

Commits on this repo take **no `Co-Authored-By` trailer**.

## 7. Open items

| item | status | cost |
|---|---|---|
| **Evaluate affm + dinolight when they hit 300k** — best-val selection, then `run_evaluate.sh` on val for both protocols. See §1's "WHAT TO DO WHEN..." | **the immediate next task** | ~30 min |
| **Commit the dinolight arm** (9 files, §6) | ready | 2 min |
| **crossattn at eval256** — point `phase4_crossattn_diagnosis/scripts/attention_probe.py` at `net_g_178000` in the eval256 regime and read entropy/diag over the 1024x1024 matrix. Separates "the softmax denominator changed" from "the routing is tied to 16x16 geometry" | **still the cheapest open item** | ~2 min |
| **Checkpoint-selection rule** — is best-val-on-full256 right for arms trained at 128? It is PRE-REGISTERED, so changing it after seeing crossattn's reversal needs a written decision, not drift | decision needed | — |
| **aca-L6** — operator alone, B6 only, importing `DinoAca`. Completes the 2x2 with affm (layer alone) and dinolight (both) | designed, not built | 300k — will NOT fit before Aug 28 |
| **Crop-size feature drift study** | specified, never started; blocked on 3 decisions incl. that 0.6694/+0.1453 is the **1e5<->1e7** pair, not render<->1e5 | 1 sbatch |
| **B3 run** under an identical recipe | pre-registered, never run | 300k |
| **Co-inflation gate rule** | proposed, not implemented; hole confirmed quantitatively | small |
| **Token-shuffle control**, **global arm on 1e5**, **E1-noisy's early peak** | not run / not investigated | — |
| ~~priorquery-render~~ | **DROPPED** 2026-08-23 at 90k, cancelled by hand | — |

### THE AUG 28 DEADLINE

The cluster goes down **2026-08-28** for maintenance. As of 2026-08-24 11:00
there are **~85 h** left. The two live arms need ~7 h each and finish today.
**Nothing else that needs 300k can be started** — a 300k run is ~39 h of compute
plus queue wait, and the queue has already cost 15 h on one occasion. Plan the
remaining time around evaluation and writing, not new training.

## 8. Standing rules for this work

- **No silent fixes.** If a check fails, stop and report. Do not repair it by
  changing the experiment design, the layer, the crop, the scheduler or the
  fusion.
- **Config immutability.** Once a run starts its YAML is frozen. Changing the
  block, crop, a mean, the LR, the scheduler, the projection, the seed or the
  loss requires a **new experiment identity** — never an edit, never a deletion,
  never a silent restart.
- **All non-trivial compute through SLURM.** Never on the login node.
- **Never initialize E0 from the old 292k checkpoint.** Both arms train from
  scratch.
- **Logs are never overwritten.** Every arm has its own job name, log directory,
  chain-state directory and TensorBoard tree; stability records are archived per
  (rule, iteration, job).
- **Do not touch the test split** until final evaluation. Checkpoint selection is
  highest **validation** PSNR, both arms.
- **Do not modify** `phase1/`, `phase2/`, `dino_spatial_layer_means.pt`, the old
  pooled means, the Work Order 1 verification outputs, the old progressive
  baseline, old checkpoints, old predictions, or previous devlogs/reports.
