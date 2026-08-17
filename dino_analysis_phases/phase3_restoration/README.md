# Phase 3 — Restoration: does a spatial DINOv2 prior help?

**Scientific question.** Does spatial DINOv2 guidance extracted from the *noisy
1e5 radar observation* improve Restormer reconstruction of the clean 1e7 target?
And — the source ablation — does it matter whether DINO reads the noisy radar or
the clean render?

> This README is the **frozen spec**: what the arms are, what is held identical,
> and the rules fixed before any result existed. For **live run status, current
> numbers and open items**, see `HANDOVER.md` at the repo root.

> **Outcome (2026-08-14, val n=339, full256).** E0 22.077 dB · E1-noisy **21.469
> (−0.608)** · E1-render **24.120 (+2.043)**. The DINO *source* decides the sign:
> the render helps well past the pre-registered +0.30 dB threshold, the noisy
> radar hurts. A fourth arm, **global-render**, pools the grid to one broadcast
> vector to test whether the render's value is spatial. Full numbers and the
> claim's exact scope: `HANDOVER.md`.

Phases 0–2 measured the prior in representation space (patch-wise cosine).
Nothing before Phase 3 tested whether that signal converts into restoration
quality. This phase runs **three** arms and nothing else.

| arm | what it is |
|---|---|
| **E0-Fixed** `Holo_E0_fixed128_baseline` | stock Restormer, fixed 128×128 crops, no DINO, from scratch, 300k iters |
| **E1-addition-noisy** `Holo_E1_addition_noisy_fixed128_spatial_B6_latent` | identical in every respect, plus centered spatial DINOv2 **B6** features from the **same** 1e5 crop, added at the latent through a zero-initialized 1×1 residual projection |
| **E1-addition-render** `Holo_E1_addition_render_fixed128_spatial_B6_latent` | the **source ablation**: identical to the noisy arm, except DINO reads the aligned **render** instead of the 1e5 crop |
| **global-render** `Holo_global_addition_render_fixed128_B6_latent` | the **spatial ablation**, added 2026-08-13: identical to E1-render, except the centered DINO grid is pooled to `[B,768,1,1]` and broadcast back, so every position receives the same vector. Same parameter count; pooling adds none |

```
E0:  1e5 → Restormer → predicted 1e7

E1-addition-noisy:
     1e5 ─┬──────────────────────────────→ encoder → inp_enc_level4 ─┐
          │                                                          │
          └→ DINO preproc → frozen DINOv2 B6 → centering → P(1×1) ────┤  +
                                                                      ↓
                                              latent blocks → decoder → predicted 1e7

E1-addition-render:
     1e5 ───────────────────────────────→ encoder → inp_enc_level4 ─┐
                                                                     │
  render → DINO preproc → frozen DINOv2 B6 → centering → P(1×1) ─────┤  +
                                                                      ↓
                                              latent blocks → decoder → predicted 1e7
```

Between the two E1 arms the **only** difference is which tensor DINO reads.
Restormer always receives the 1e5 crop and the target is always 1e7; the render
is DINO input and nothing else — never a Restormer input, never a target, never
a centering statistic for the radar arm. **The render is normally available in
this pipeline, so the render arm is a usable method, not merely an oracle** —
state that whenever the arm is reported.

**1e7 is TARGET ONLY** — never used for conditioning, guidance, centering
statistics or auxiliary input.

---

## The one permitted difference

**All three arms** share splits, crop protocol, augmentation, seed, Restormer
hyper-parameters, optimizer, LR schedule, iterations, batch size, loss,
validation frequency, checkpoint-selection rule and evaluation procedure.
E0 → E1 adds the DINO branch; E1-noisy → E1-render changes **only** which tensor
DINO reads. Verified mechanically, not asserted in prose:

- parameter delta vs E0 = `768×384 + 384` = **295,296** exactly, for **both** E1
  arms — the render arm adds no parameters, it re-routes one tensor
- trunk initialisation is **bit-identical** to E0 (an RNG fence around the ViT
  and projection construction keeps every later RNG draw aligned)
- step-0 outputs are identical, because `P(D) = 0` at initialisation
- the render is cropped and augmented **identically** to the LQ/GT pair by
  construction, not by two coordinate calculations that have to agree: it rides
  as **channel 1 of the LQ tensor**, so `train.py`'s sub-crop is one slice
  hitting both channels. This needed **zero edits to `basicsr/train.py`**, which
  matters because every arm re-reads that file at each resume.

## The layer lock, and the tension behind it

**B6 (0-indexed 5)** is locked on the *pre-registered primary criterion* —
centered same-scene 1e5↔1e7 correspondence measured at the actual fixed-128
training scale (Work Order 1, val n=339):

| block | same-scene (centered) | scene advantage |
|---|---|---|
| B3 | +0.6091 | **+0.1966** |
| **B6** | **+0.6694** | +0.1453 |
| B9 | +0.5482 | +0.1247 |
| B12 | +0.2337 | +0.1162 |

**Documented tension.** B3 wins the *same-vs-different scene advantage*, the
criterion Phase 2's own interpretation called "the one to trust", and the gap is
**wider at 128 than it was at 256** (B3−B6 advantage: +0.026 at 256 → +0.051 at
128). This is a criterion conflict, not a measurement error. It does not change
E1. It is why a **B3 run under an identical recipe is pre-registered** as the
immediate follow-up (see the E1 devlog) — so the conflict is settled empirically
rather than by argument, and not by fishing after the fact.

## Scale consistency — never interpolate the feature grid

| mode | radar | DINO input | tokens | grid | latent | mean |
|---|---|---|---|---|---|---|
| training | 128 | 224 | `[B,256,768]` | 16×16 | 16×16 | `train128` |
| matched-128 eval | 128 | 224 | `[B,256,768]` | 16×16 | 16×16 | `train128` |
| full-256 eval | 256 | 448 | `[B,1024,768]` | 32×32 | 32×32 | `eval256` |

224/14 = 16 and 448/14 = 32, so radar-pixels-per-token stays ≈8 in both regimes
and the DINO grid *already equals* the Restormer latent grid.
`dino_shared.assert_no_interpolation_needed` is a hard gate, not a comment.
The regime is set by an **explicit flag** (`set_dino_mode`), never inferred from
tensor size.

## Two evaluation protocols, never mixed

- **full-256** — scale-consistent full-image evaluation, `eval256` mean.
- **matched-128** — one 128×128 crop per image, coordinates read from a
  **manifest on disk** (`results/crop_manifests/matched128_{val,test}.csv`), the
  same crop applied to the 1e5 input, the 1e7 target and both models'
  predictions, `train128` mean.

**Pre-registered interpretation rule:** if E1 improves on matched-128 but not on
full-256, that indicates the prior is useful in-distribution and that full-image
scale/context transfer is the limiter — **not** that the prior is useless.

## Pre-registered result thresholds (frozen before any result existed)

```
Delta PSNR = E1 test PSNR − E0 test PSNR
  <  +0.10 dB   → no meaningful PSNR improvement
  +0.10..+0.30  → marginal / promising
  >  +0.30 dB   → meaningful improvement
```

PSNR is not the sole criterion: SSIM, object-only (masked) PSNR/SSIM, HF-energy
ratio, Laplacian variance, Sobel gradient and the radial power spectrum are all
reported. A modest PSNR gain with clear high-frequency or object-structure
recovery may still be scientifically important.

**Checkpoint selection: highest validation PSNR, for both arms. The test split
never drives selection.**

## ⚠ E0-Fixed is a NEW baseline — the old numbers do not describe it

E0-Fixed uses a constant batch of 8 and **never trains at 256**, unlike the old
progressive `Holo_Baseline_Restormer_verynoisy` (8/5/4/2 over 128→160→192→256).
Consequences that must be stated, not hidden:

- The recorded **22.446 dB val / 22.405 dB test / 18.313 dB masked** figures
  belong to the OLD baseline and are **not** E0-Fixed's numbers.
- The **HF-energy ratio 0.216** — the over-smoothing figure the whole project
  motivation rests on — was measured on the OLD baseline too.
- **Expect E0-Fixed to score lower in absolute PSNR** than the old baseline,
  because it never trains at the evaluation resolution. That is expected and
  must be reported, not explained away.
- Therefore the full over-smoothing characterisation (HF-energy, Laplacian,
  Sobel, radial power) **must be re-measured on E0-Fixed** so the motivation and
  the results describe the same model. `scripts/run_evaluate.sh` runs exactly
  that chain; it is wired in and must be run once E0-Fixed exists.
- The val split is now 339 images (not the 677 the old val curve used), so even
  val-to-val comparison with the old run is not like-for-like.

## Centering means — four files, not interchangeable

One position-independent `[768]` vector per (domain, regime), computed on the
**train split only**, ≥1000 images, accumulated float64 and stored float32.
This is feature centering, **not** ImageNet pixel normalisation.

| file | ‖μ‖ | tokens |
|---|---|---|
| `1e5_B6_train128_dino224_mean.pt` | 43.5031 | 256k |
| `1e5_B6_eval256_dino448_mean.pt` | 41.2580 | 1024k |
| `render_B6_train128_dino224_mean.pt` | 56.8906 | 256k |
| `render_B6_eval256_dino448_mean.pt` | 61.6951 | 1024k |

cosine(1e5, render) = **0.1799** at train128 and **0.3288** at eval256 — the two
domains sit in genuinely different places, so the render arm needs its own
means; centering it against the 1e5 mean would use the wrong distribution. The
architecture validates block / input size / split from each mean's metadata and
refuses a mismatch rather than silently biasing every token by a constant.

## The stability gate

Evaluated every optimizer step; `dino/latent_norm`, `dino/projected_norm` and
`dino/injection_ratio` are logged to `experiments/<name>/dino_stability.csv`.

| rule | threshold | active from |
|---|---|---|
| NaN/Inf in loss, latent, projection or ratio | any | **iteration 1** |
| rule 1: `injection_ratio` cap | **10.0** | iteration **5000** |
| rule 2: growth vs the 5k reference | **10×** | iteration 5000 |

**Amendment (2026-08-12).** The original cap of 0.5 from iteration 1 aborted E1
at iteration 4. A diagnostic run (abort disabled, throwaway identity, every
measurement still taken) showed the ratio **plateaus at 2.3–6.6 and drifts down
after iteration 1000**. `P` is zero-initialized so the ratio *must* rise from 0,
and AdamW's first steps move ≈`lr` per weight regardless of gradient magnitude —
the old cap was firing on a startup transient, not a pathology. The amendment
changed **when the gate is valid**, not how much injection is tolerated in
steady state. The NaN/Inf stop was never windowed.

A stability failure is **investigated and reported, never restarted into**: the
chain stops and sets `CHAIN_ABORTED`, and each failure is archived per (rule,
iteration, job) so a later event cannot destroy an earlier record.

> ⚠ **Known gap: co-inflation is not gated.** Rule 2 watches the *ratio*, so it
> cannot see both norms growing together (E1-noisy: latent 215 → 2310, projected
> 872 → 5237 between 1k and 41k, ratio flat). A proposed rule — abort if either
> norm exceeds 5× its own 5k reference — is **not implemented**, because adding
> it changes an experiment definition and would require a new identity.

## Layout

```
phase3_restoration/
├── README.md                    this file
├── configs/                     E0_fixed128_baseline.yml
│                                E1_addition_noisy_fixed128_spatial_B6_latent.yml
│                                E1_addition_render_fixed128_spatial_B6_latent.yml
├── means/                       1e5_B6_train128_dino224_mean.pt     (+ _meta.json)
│                                1e5_B6_eval256_dino448_mean.pt      (+ _meta.json)
│                                render_B6_train128_dino224_mean.pt  (+ _meta.json)
│                                render_B6_eval256_dino448_mean.pt   (+ _meta.json)
├── devlogs/                     one append-only devlog per arm
├── scripts/                     dino_shared.py (WO1) + all Phase-3 helpers
│                                chain_core.sh + chain_<ARM>.sh, one per arm
└── results/
    ├── wo1_verification/        Work Order 1 evidence (do not modify)
    ├── wo2_implementation/      smoke results, smoke configs, job logs
    ├── crop_manifests/          matched-128 coordinate manifests
    ├── <experiment>/{metrics,predictions,visuals,metadata}/
    └── comparisons/E0_vs_E1/
```

Live BasicSR output stays under `experiments/<name>/`, TensorBoard under
`tb_logger/<name>/`. Checkpoints are never duplicated into this tree; paths are
recorded in metadata instead.

## Hardware note

Restormer at 128²×batch 8 **OOMs on a 10 GB RTX 3080** (measured, job 1773194:
the stock E0 trunk alone exceeded 9.6 GB — not a Phase-3 defect). Jobs target
**a100** or v100 (32 GB). Login-node compute is not an option either — the
watchdog SIGTERMs substantial CPU work (exit 143).

**All three arms now run on a100**, so per-iteration timings are comparable
across the whole Phase-3 table. Note that `squeue` shows only your own jobs
(`PrivateData=jobs`) — it is not evidence about how busy a partition is.

## Config immutability

Once a run starts, its YAML is frozen. Changing the DINO block, the crop, a
centering mean, the LR, the scheduler, the projection/fusion, the seed or the
loss requires a **new experiment identity** — never an edit, never a deletion,
never a silent restart.
