# Phase 1 Devlog — Dataset-Level Spatial DINO Consistency

Scope: **Phase 1 only** — establishing, across the validation split, which DINO
depth most consistently preserves spatial structure across Very Noisy Radar
(1e5 rays), Clean Radar (1e7 rays) and Render.

Phase 2 (comparing DINO(1e5) and DINO(Render) as candidate structural priors
relative to DINO(1e7)) gets its own `dino_analysis/phase2/PHASE2_DEVLOG.md` and
must not be logged here.

Predecessor: the single-sample run in
`dino_analysis/DINO_ANALYSIS_DEVLOG.md` (sample 0196, 2026-08-09).

Entries below are **appended automatically by**
`dino_analysis/phase1/analyze_dino_spatial_consistency.py` at the end of a full
run, using the real local execution timestamp and the real numbers from that
run. Smoke tests (`--max-samples`) never write here. Nothing is written before
the analysis actually executes, so an empty log below means Phase 1 has not been
run yet.

---

## 2026-08-10 11:05 CEST — Dataset-Level Spatial DINO Consistency

### Why
The Phase-0 single-sample run (sample 0196) suggested that intermediate DINO
blocks might retain usable spatial structure even from the extremely noisy
1e5-ray radar image. One example cannot support that, so we measured the whole
validation split to see whether the behaviour generalises. Block 9 was not
assumed to be the answer.

### What we did
- iterated the complete val split and verified every 1e5/1e7/render triplet
- extracted spatial DINO patch tokens (CLS/register removed, nothing pooled)
- evaluated blocks [3, 6, 9, 12] (quarter depths, 1-indexed)
- measured mean corresponding-patch cosine similarity on the full 768-d features
- repeated for RAW and CENTERED features (fixed training means)
- added a different-scene control (previous valid sample) so the absolute
  cosines are interpretable
- selected and visualised the 3 best and 3 worst samples by 1e5↔1e7

### How
- model: `dinov2_vitb14`, frozen, offline local load, strict=True
- checkpoint: `/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- split: val — 339 considered, **339 valid triplets**, 0 skipped
- blocks 1-indexed [3, 6, 9, 12] → 0-indexed [2, 5, 8, 11]
- DINO input 224×224, patch 14 → **16×16 = 256 patch tokens**, 768-d
- centering: per-domain spatial means over 150 TRAIN images
  (`dino_analysis/dino_spatial_layer_means.pt`); validation never enters the statistic
- primary metric: mean corresponding-patch cosine on full feature vectors (not PCA)
- uncertainty in plots: ± 1 standard error of the mean over the 339 triplets
- seed 0, device cuda

### Results

**1e5 ↔ 1e7 (the primary comparison), mean ± SEM:**

| Features | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| RAW | +0.5218 ± 0.0074 | +0.6402 ± 0.0043 | +0.6064 ± 0.0036 | +0.4498 ± 0.0043 |
| CENTERED | +0.5932 ± 0.0043 | +0.6856 ± 0.0024 | +0.5763 ± 0.0030 | +0.2703 ± 0.0043 |

**1e5 ↔ Render:**

| Features | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| RAW | +0.3500 ± 0.0080 | +0.4977 ± 0.0046 | +0.5240 ± 0.0029 | +0.2679 ± 0.0034 |
| CENTERED | +0.5593 ± 0.0021 | +0.6570 ± 0.0015 | +0.5466 ± 0.0020 | +0.1968 ± 0.0025 |

**1e7 ↔ Render:**

| Features | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| RAW | +0.7843 ± 0.0033 | +0.7791 ± 0.0027 | +0.7168 ± 0.0025 | +0.4614 ± 0.0040 |
| CENTERED | +0.7573 ± 0.0029 | +0.7827 ± 0.0021 | +0.6982 ± 0.0023 | +0.3731 ± 0.0037 |

**Different-scene control (wrong object), 1e5↔1e7:**

| Features | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| RAW | +0.4553 | +0.5864 | +0.5650 | +0.4114 |
| CENTERED | +0.5048 | +0.6230 | +0.5266 | +0.2119 |

**Direct answers to the Phase-1 questions:**
1. Highest mean RAW 1e5↔1e7: **Block 6** (+0.6402)
2. Highest mean CENTERED 1e5↔1e7: **Block 6** (+0.6856)
3. Highest 1e5↔Render (centered): **Block 6** (+0.6570)
4. Highest 1e7↔Render (centered): **Block 6** (+0.7827)
5. Does centering change the layer ranking? **YES** — raw order [6, 9, 3, 12], centered order [6, 3, 9, 12]
6. Are all intermediate blocks above the final block (12) for centered 1e5↔1e7? **YES**
7. Block 9 across the dataset: rank **3 of 4** on centered 1e5↔1e7 (+0.5763) — it is NOT the top block; the single-sample impression did not generalise

**Best / worst samples** (block 6, centered features, 1e5↔1e7):
- best: 6115 (+0.7928), 3017 (+0.7874), 3819 (+0.7868)
- worst: 4467 (+0.5671), 1995 (+0.5802), 2547 (+0.5872)

**Skipped samples:** none

### Interpretation
- The layer ordering above is what the validation set actually supports; the
  Phase-0 single-sample reading is superseded by it.
- Read the same-scene numbers against the different-scene control on the same
  row. Where the two are close, DINO is reporting a component shared by all
  images rather than this scene's structure.
- Centering does not change which block ranks HIGHEST for 1e5↔1e7
  (raw best B6, centered best B6), and does change the full ordering.
- Every intermediate block beats the final block on centered 1e5↔1e7, which is the
  direct evidence on whether noise-stability is an intermediate-layer property.

### Outputs
- `dino_analysis/phase1/outputs/dino_spatial_similarity_val_per_sample.csv`
- `dino_analysis/phase1/outputs/dino_spatial_similarity_val_summary.csv`
- `dino_analysis/phase1/outputs/dino_spatial_similarity_by_layer_raw.png`
- `dino_analysis/phase1/outputs/dino_spatial_similarity_by_layer_centered.png`
- `dino_analysis/phase1/outputs/best_samples/`, `.../worst_samples/`
- `dino_analysis/phase1/outputs/phase1_metadata.json`

### Next Step
Phase 2 — compare DINO(1e5) and DINO(Render) as candidate structural priors
relative to DINO(1e7). Not implemented yet.

---
