# Phase 2 Devlog — DINO Prior Source: 1e5 Radar vs Render

Scope: **Phase 2 only** — at the DINO depth Phase 1 selected, does DINO(1e5) or
DINO(Render) give a spatial representation better aligned with the clean
DINO(1e7) target?

Judged on two quantities, never one alone:
1. **absolute** same-scene corresponding-patch cosine to DINO(1e7)
2. **scene-specific advantage** = same-scene − different-scene cosine

Phase 1 showed different-scene pairs can score highly, so absolute similarity on
its own can reflect a generic shared component rather than scene content.

Out of scope: Restormer, training, and any PSNR/SSIM claim — that is Phase 3,
which gets its own `dino_analysis/phase3/PHASE3_DEVLOG.md`.

Predecessors: `dino_analysis/DINO_ANALYSIS_DEVLOG.md` (single sample) and
`dino_analysis/phase1/PHASE1_DEVLOG.md` (validation-set layer sweep).

Entries below are **appended automatically by**
`dino_analysis/phase2/analyze_dino_prior_source.py` at the end of a full run,
using the real local execution timestamp and the real numbers from that run.
Smoke tests (`--max-samples`) never write here. An empty log below means Phase 2
has not been run yet.

---

## 2026-08-10 11:47 CEST — DINO Prior Source: 1e5 Radar vs Render

### Why
Phase 1 established that Block 6 (2/4 depth) gives the strongest
cross-noise spatial consistency between the 1e5 and 1e7 radar representations
(centered mean +0.6856). The open question is which source we could actually
feed a restoration model: the noisy 1e5 radar itself, or the corresponding
render. Phase 2 asks which of the two produces a spatial DINO representation
better aligned with the clean 1e7 target, judged both in absolute terms and
against a different-scene control.

### What we did
- read the primary block from the Phase 1 summary CSV (not hardcoded)
- reused the exact Phase 1 validation triplet set (339 samples)
- compared DINO(1e5) with DINO(1e7), and DINO(Render) with DINO(1e7)
- evaluated raw and centered features
- built a deterministic seeded derangement as the different-scene control, using
  the same mapping for both candidate sources
- measured scene-specific advantage (same-scene − different-scene)
- ran paired Wilcoxon signed-rank tests on both quantities
- visualised representative best/worst/tied samples

### How
- selected block: **B6** (2/4 depth), chosen as the highest mean
  centered same-scene 1e5↔1e7 in `dino_analysis/phase1/outputs/dino_spatial_similarity_val_summary.csv`
- secondary block (context only): B3
- model `dinov2_vitb14`, checkpoint `/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- 339 valid triplets (Phase 1 had 339); all Phase 1 samples reproduced
- patch grid 16×16 = 256 tokens, 768-d, patch 14, input 224²
- centering: per-domain spatial means over 150 TRAIN images (unchanged from Phase 1)
- metric: mean corresponding-patch cosine on full 768-d features (never PCA)
- different-scene control: seeded derangement, seed 0, no fixed points
- test: paired Wilcoxon signed-rank; magnitude reported as mean/median paired difference and Cohen's dz

### Results

**CENTERED features, Block 6 (primary):**

| Quantity | 1e5 → clean | Render → clean |
|---|---|---|
| same-scene mean | +0.6045 | +0.7807 |
| same-scene median | +0.6057 | +0.7856 |
| different-scene mean | +0.5377 | +0.7141 |
| **scene advantage (mean)** | **+0.0668** | **+0.0666** |
| scene advantage (median) | +0.0618 | +0.0582 |

- direct delta (Render − 1e5), absolute: mean +0.1762, median +0.1683
  → Render wins on 98.2%, 1e5 wins on 1.5%, tied 0.3%
  → Wilcoxon p = 3.538e-57, Cohen's dz = +1.906
- delta scene advantage (Render − 1e5): mean -0.0002, median -0.0043
  → Render larger on 49.3%, 1e5 larger on 50.1%
  → Wilcoxon p = 9.175e-01, Cohen's dz = -0.002

**RAW features, Block 6:**

| Quantity | 1e5 → clean | Render → clean |
|---|---|---|
| same-scene mean | +0.5218 | +0.7843 |
| different-scene mean | +0.4546 | +0.7211 |
| **scene advantage (mean)** | **+0.0672** | **+0.0633** |

- direct delta mean +0.2625 (Render wins 98.2%)
- delta scene advantage mean -0.0039 (Render larger 48.1%)

**Complementarity (same scene, centered, B6):** mean 1e5↔Render +0.4907,
median +0.4794; Phase 1 different-scene control for the same pair/block
was +0.6263.

**Representative samples:** render_wins = 5120, 1e5_wins = 1910, near_tie = 0731, render_scene_advantage = 0592, 1e5_scene_advantage = 5465

### Interpretation
- On ABSOLUTE alignment to the clean DINO representation, **Render** is ahead
  (centered mean delta +0.1762).
- On SCENE-SPECIFIC advantage over the different-scene control, **1e5** is ahead
  (centered mean delta -0.0002, dz -0.002).
- The two criteria DISAGREE. Absolute similarity is therefore partly generic domain similarity rather than scene content, exactly the failure mode Phase 1 warned about — the scene-advantage column is the one to trust.
- **The absolute gap is 915.6x the scene-advantage gap.** Almost all of Render's absolute lead also shows up against WRONG-scene targets, so it is generic domain similarity, not scene-specific content. The scene-advantage column is the one that reflects usable structure (delta -0.0002, dz -0.002).
- 1e5 and Render are not interchangeable representations (same-scene 1e5↔Render
  mean +0.4907), so complementarity is not ruled out; testing fusion is
  out of scope here.
- No claim is made about restoration PSNR/SSIM. This measures alignment to the
  clean DINO spatial representation only; whether that translates into
  restoration quality is Phase 3.

### Outputs
- `dino_analysis/phase2/outputs/dino_prior_source_per_sample.csv`
- `dino_analysis/phase2/outputs/dino_prior_source_summary.csv`
- `dino_analysis/phase2/outputs/dino_prior_source_paired_comparison.png`
- `dino_analysis/phase2/outputs/dino_prior_source_scene_advantage.png`
- `dino_analysis/phase2/outputs/dino_prior_source_delta_histogram.png`
- `dino_analysis/phase2/outputs/representative_samples/`
- `dino_analysis/phase2/outputs/phase2_metadata.json`

### Next Step
Phase 3 — controlled restoration experiments informed by Phase 1 and Phase 2.
Not implemented.

---

## 2026-08-10 11:53 CEST — DINO Prior Source: 1e5 Radar vs Render

### Why
Phase 1 established that Block 6 (2/4 depth) gives the strongest
cross-noise spatial consistency between the 1e5 and 1e7 radar representations
(centered mean +0.6856). The open question is which source we could actually
feed a restoration model: the noisy 1e5 radar itself, or the corresponding
render. Phase 2 asks which of the two produces a spatial DINO representation
better aligned with the clean 1e7 target, judged both in absolute terms and
against a different-scene control.

### What we did
- read the primary block from the Phase 1 summary CSV (not hardcoded)
- reused the exact Phase 1 validation triplet set (339 samples)
- compared DINO(1e5) with DINO(1e7), and DINO(Render) with DINO(1e7)
- evaluated raw and centered features
- built a deterministic seeded derangement as the different-scene control, using
  the same mapping for both candidate sources
- measured scene-specific advantage (same-scene − different-scene)
- ran paired Wilcoxon signed-rank tests on both quantities
- visualised representative best/worst/tied samples

### How
- selected block: **B6** (2/4 depth), chosen as the highest mean
  centered same-scene 1e5↔1e7 in `dino_analysis/phase1/outputs/dino_spatial_similarity_val_summary.csv`
- secondary block (context only): B3
- model `dinov2_vitb14`, checkpoint `/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- 339 valid triplets (Phase 1 had 339); all Phase 1 samples reproduced
- patch grid 16×16 = 256 tokens, 768-d, patch 14, input 224²
- centering: per-domain spatial means over 150 TRAIN images (unchanged from Phase 1)
- metric: mean corresponding-patch cosine on full 768-d features (never PCA)
- different-scene control: seeded derangement, seed 0, no fixed points
- test: paired Wilcoxon signed-rank; magnitude reported as mean/median paired difference and Cohen's dz

### Results

**CENTERED features, Block 6 (primary):**

| Quantity | 1e5 → clean | Render → clean |
|---|---|---|
| same-scene mean | +0.6856 | +0.7827 |
| same-scene median | +0.6839 | +0.7869 |
| different-scene mean | +0.6225 | +0.7106 |
| **scene advantage (mean)** | **+0.0632** | **+0.0721** |
| scene advantage (median) | +0.0574 | +0.0661 |

- direct delta (Render − 1e5), absolute: mean +0.0971, median +0.0937
  → Render wins on 98.2%, 1e5 wins on 1.8%, tied 0.0%
  → Wilcoxon p = 4.909e-57, Cohen's dz = +2.052
- delta scene advantage (Render − 1e5): mean +0.0089, median +0.0062
  → Render larger on 53.1%, 1e5 larger on 46.0%
  → Wilcoxon p = 1.178e-02, Cohen's dz = +0.163

**RAW features, Block 6:**

| Quantity | 1e5 → clean | Render → clean |
|---|---|---|
| same-scene mean | +0.6402 | +0.7791 |
| different-scene mean | +0.5859 | +0.7237 |
| **scene advantage (mean)** | **+0.0542** | **+0.0554** |

- direct delta mean +0.1389 (Render wins 95.3%)
- delta scene advantage mean +0.0012 (Render larger 48.1%)

**Complementarity (same scene, centered, B6):** mean 1e5↔Render +0.6570,
median +0.6601; Phase 1 different-scene control for the same pair/block
was +0.6263.

**Representative samples:** render_wins = 5120, 1e5_wins = 5465, near_tie = 4725, render_scene_advantage = 0592, 1e5_scene_advantage = 5465

### Interpretation
- On ABSOLUTE alignment to the clean DINO representation, **Render** is ahead
  (centered mean delta +0.0971).
- On SCENE-SPECIFIC advantage over the different-scene control, **Render** is ahead
  (centered mean delta +0.0089, dz +0.163).
- The two criteria point the same way, but that is not the whole story.
- **The absolute gap is 10.9x the scene-advantage gap.** Almost all of Render's absolute lead also shows up against WRONG-scene targets, so it is generic domain similarity, not scene-specific content. The scene-advantage column is the one that reflects usable structure (delta +0.0089, dz +0.163).
- 1e5 and Render are not interchangeable representations (same-scene 1e5↔Render
  mean +0.6570), so complementarity is not ruled out; testing fusion is
  out of scope here.
- No claim is made about restoration PSNR/SSIM. This measures alignment to the
  clean DINO spatial representation only; whether that translates into
  restoration quality is Phase 3.

### Outputs
- `dino_analysis/phase2/outputs/dino_prior_source_per_sample.csv`
- `dino_analysis/phase2/outputs/dino_prior_source_summary.csv`
- `dino_analysis/phase2/outputs/dino_prior_source_paired_comparison.png`
- `dino_analysis/phase2/outputs/dino_prior_source_scene_advantage.png`
- `dino_analysis/phase2/outputs/dino_prior_source_delta_histogram.png`
- `dino_analysis/phase2/outputs/representative_samples/`
- `dino_analysis/phase2/outputs/phase2_metadata.json`

### Next Step
Phase 3 — controlled restoration experiments informed by Phase 1 and Phase 2.
Not implemented.

---
