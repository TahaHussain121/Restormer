# HANDOVER — Phase 3 restoration (E0 / E1-addition), as of 2026-08-14

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

**Phase 3 is decided, on the locked test split.** All three original arms
finished 300k and were evaluated on test (n=338): reading the **render** gains
**+2.208 dB**, reading the **noisy radar** *loses* **0.577 dB** against no prior
at all. Both clear/miss the pre-registered thresholds on *both* protocols. A
fourth arm (global-render) is at 90k/300k testing whether the render's value is
spatial. Nothing in Phase 3 is committed to git yet.

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

## 1. Run status (verified 2026-08-14)

| arm | experiment name | state |
|---|---|---|
| **E0** baseline | `Holo_E0_fixed128_baseline` | **300k DONE**, evaluated |
| **E1-addition-noisy** | `Holo_E1_addition_noisy_fixed128_spatial_B6_latent` | **300k DONE**, evaluated |
| **E1-addition-render** | `Holo_E1_addition_render_fixed128_spatial_B6_latent` | **300k DONE**, evaluated |
| **global-render** | `Holo_global_addition_render_fixed128_B6_latent` | **90k / 300k**, job 1776459 on a100, successor 1776464 queued |

All chains behaved: each completed arm wrote `TRAINING_DONE` and cancelled its
own successor. Chain counts: 2 jobs each for the completed arms, 1 so far for
global-render.

**Caveats that stay attached to every number above.** Single seed; one layer
(B6); one fusion (zero-init 1×1 residual at the latent); one dataset. The 16-bit
evaluation PSNR is **not** comparable to the 8-bit training-time val PSNR,
despite similar values. Checkpoint selection used validation only, which is what
makes the test read legitimate.

### A log-parsing trap that has already caused one wrong claim

`grep 'iter:'` over the training logs matches **`total_iter: 300000`** in the
config dump at the head of every log, and `train_*.log` files do not sort
chronologically by shell glob. Both together made three finished arms look like
they were stuck at 182k. **Use the checkpoint files as ground truth for
progress** (`ls experiments/<name>/models/`), and anchor any log regex as
`(?<!total_)iter:\s*([\d,]+),`.

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

### Layer lock — B6, with a documented tension

B6 (0-indexed **5**) locked on the pre-registered primary criterion: centered
same-scene 1e5↔1e7 correspondence at the actual fixed-128 training scale
(Work Order 1, val n=339): B3 0.6091, **B6 0.6694**, B9 0.5482, B12 0.2337.

**B3 wins the same-vs-different scene advantage** (+0.1966 vs +0.1453) and the
gap is *wider* at 128 than at 256. That is a criterion conflict, not a
measurement error. A **B3 run under an identical recipe is pre-registered** as
the follow-up so it is settled empirically, not by argument.

---

## 3. Key files (none of this is committed yet — see §6)

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
- `scripts/chain_core.sh` + `chain_E0.sh` / `chain_E1_addition_noisy.sh` /
  `chain_E1_addition_render.sh` — shared logic, separate identity per arm.
- `results/wo1_verification/`, `results/wo2_implementation/` — the verification
  evidence. **Do not modify.**

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

## 6. Git state — read this before committing anything

Branch **`dino_e2`**. Last commit `4e37beb` (DEVLOG Step 28). **The entire
Phase-3 body of work is untracked:**

```
?? REPO_INVESTIGATION_REPORT.md
?? basicsr/data/paired_radar_render_stacked_dataset.py
?? basicsr/models/archs/restormer_dino_render_arch.py
?? basicsr/models/archs/restormer_dino_spatial_arch.py
?? basicsr/models/image_restoration_dino_model.py
?? dino_analysis_phases/phase3_restoration/
 M .claude/settings.json
```

Three live runs depend on these files on disk. **Committing is safe; editing is
not** — a change to `restormer_dino_spatial_arch.py` is picked up by the *next
resume* of both E1 arms, not at the next iteration, so it lands silently hours
later. Note also that `experiments/` and `tb_logger/` are **gitignored**, so
checkpoints and TensorBoard curves exist on disk only — exactly how the E1-FiLM
runs became unrecoverable (Step 28).

Commits on this repo take **no `Co-Authored-By` trailer**.

---

## 7. Open items — none started, all need a decision

| item | status |
|---|---|
| ~~**FINAL TEST EVALUATION**~~ — three completed arms, both protocols | **DONE 2026-08-14** (jobs 1776745-1776750) |
| **global-render** — launched 2026-08-14, at 88k/300k. Early signal: below baseline, which would say the render's value is *spatial* | in flight |
| **global arm on the 1e5 (noisy) source** — the same pooling against E1-addition-noisy | requested earlier, not implemented |
| **E1-noisy's early peak** — best at 128k then declining, vs 268k/204k for the others | observed, not investigated |
| **Co-inflation gate rule** (§5) | proposed, not implemented |
| **B3 run** under an identical recipe | pre-registered |
| **Concat-then-project** fusion variant | pre-registered |
| **Token-shuffle control** for the 0.524 different-scene floor | not run |
| ~~**Baseline re-characterisation** on E0-Fixed~~ | **DONE 2026-08-13** (jobs 1775533/1775534, val n=339) |

**The re-characterisation is now done, and the premise held.** E0-Fixed keeps
**0.200** of the target's HF energy against **0.216** for the old progressive
baseline — marginally *worse*, not better. Over-smoothing is the headline
weakness of this baseline too, so motivation and results now describe the same
model. **From here quote 0.200 (full256, val, n=339), never 0.216.**

Measured on `net_g_268000.pth` (best val): PSNR 22.077 / SSIM 0.783 whole-image,
17.978 / 0.569 on the object mask, Laplacian ratio 0.284, Sobel ratio 0.756.
The Sobel-vs-Laplacian gap is the finding — first-order edges survive, second-order
detail does not. Two things not to misquote: the crop128 HF ratio of 0.910 has a
std of 0.475 (the HF band is nearly empty on a 128 crop — full256 is the figure of
record), and this evaluation PSNR is the 16-bit path, **not** comparable to the
8-bit training-time val PSNR. Test split still untouched, deliberately, so all
arms can be evaluated together under one protocol.

---

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
