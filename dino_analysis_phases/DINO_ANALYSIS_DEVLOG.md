# DINO Analysis Devlog

Chronological, append-only log of the DINO investigation for the radar
restoration project. Newest entries at the bottom. Every entry carries the real
local execution timestamp of the run it reports.

Scope: analysis/visualisation only — nothing here touches the training pipeline.

---

## 2026-08-09 14:16 CEST — Spatial DINO PCA: Very Noisy 1e5 vs Clean 1e7 vs Render

### Why
All previous DINO measurements in this project used the **pooled** feature (one
vector per image, patch tokens averaged), and they came out weak: the pooled
vector is ~85–90% a constant offset and raw pooled features were object-blind.
Pooling throws away the spatial layout, which is exactly what a restoration
network needs. So the question here is the spatial one: does DINO keep a similar
**spatial/structural** representation when the same scene arrives as an extremely
noisy 1e5-ray radar heatmap, a clean 1e7-ray radar heatmap, and its render — and
how does that change from shallow to deep blocks?

### What we did
- selected the same scene across 1e5 / 1e7 / render through the project's own
  dataset class, and verified the three basenames are identical
- extracted **spatial patch tokens** from DINOv2 blocks 3, 6, 9, 12 (quarter
  depths); CLS + register tokens removed; nothing pooled anywhere
- reshaped 256 tokens to a 16×16 spatial grid (asserted N == H·W)
- fitted **one joint PCA per layer** on the concatenated 1e5 + 1e7 + render patch
  tokens, and applied that same transform to all three rows
- displayed PCA-1 with **one shared range per layer** across the three rows
  (robust p1–p99 of that layer's joint values); also made PCA-RGB (PC1/2/3)
- repeated everything on **centered** features, using new per-layer spatial means
  computed from training data
- measured mean corresponding-patch cosine similarity for all three pairings
- added a **different-scene control** (12 other val objects): the same
  measurement against the wrong object. Without it an absolute cosine of +0.52
  is uninterpretable, because DINO patch tokens on this dataset share a large
  common component.

### How
- model: `dinov2_vitb14`, frozen, loaded offline (`hub_source: local`,
  `strict=True`, missing/unexpected keys both empty) — the project's own
  `DINOv2Extractor`, config read from `Holo_DINOv2_renderDINO_Restormer.yml`
- checkpoint: `torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- sample: **0196**, val split (index 8) — 1e5 `val_verynoisy/0196.png`,
  1e7 `val_clean/0196.png`, render `val_renders_blackbg/0196.png`
- DINO input: 224×224, patch 14 → **16×16 = 256 patch tokens**, 768-d,
  12 blocks, 0 register tokens
- selected layers: 1-indexed [3, 6, 9, 12] → 0-indexed [2, 5, 8, 11].
  **Note:** the training recipe uses [1, 4, 8, 12]; this study uses quarter
  depths, so the two layer sets are not identical.
- preprocessing: unchanged project code (`dino_preprocess`). Radar uint16/65535
  → 1ch [0,1] → replicated to 3ch → bilinear 224 → ImageNet norm. Render 8-bit →
  [0,1] 3ch → same. No colormap applied before DINO.
- centering: the repo's existing `dino_feat_mean_*.pt` are **3072-d pooled**
  vectors and are **not** usable for spatial patch-token centering — the script
  says so and refuses them. New statistic: mean over 150 TRAIN-split images
  **and** over all patch positions, one 768-d vector per block, cached to
  `dino_analysis/dino_spatial_layer_means.pt`. Figures use the per-domain mean
  (one mean per domain per layer, matching the convention already used in
  `Deraining_Holo/render_radar_similarity.py`); a global mean is also cached and
  reported in the CSV.
- seed 0, CPU, ~5 min for the mean pass + ~1 min per analysis run.

### Results

**Mean corresponding-patch cosine similarity.** Bracketed value = the 12-scene
different-scene control; `gap` = same scene − different scene, and the gap is
what actually carries information.

RAW spatial features:

| Layer | Block | 1e5 vs 1e7 | 1e5 vs Render | 1e7 vs Render |
|---|---|---|---|---|
| 1/4 | 3 | +0.520 [+0.351] **gap +0.170** | +0.225 [+0.200] **gap +0.026** | +0.664 [+0.593] **gap +0.071** |
| 2/4 | 6 | +0.632 [+0.533] **gap +0.099** | +0.422 [+0.413] **gap +0.009** | +0.680 [+0.620] **gap +0.060** |
| 3/4 | 9 | +0.578 [+0.525] **gap +0.053** | +0.488 [+0.472] **gap +0.016** | +0.627 [+0.582] **gap +0.045** |
| 4/4 | 12 | +0.417 [+0.367] **gap +0.050** | +0.240 [+0.238] **gap +0.002** | +0.358 [+0.314] **gap +0.044** |

CENTERED spatial features (per-domain training mean, n=150):

| Layer | Block | 1e5 vs 1e7 | 1e5 vs Render | 1e7 vs Render |
|---|---|---|---|---|
| 1/4 | 3 | +0.582 [+0.444] **gap +0.139** | +0.517 [+0.487] **gap +0.030** | +0.656 [+0.545] **gap +0.111** |
| 2/4 | 6 | +0.671 [+0.595] **gap +0.076** | +0.620 [+0.609] **gap +0.011** | +0.707 [+0.627] **gap +0.080** |
| 3/4 | 9 | +0.546 [+0.495] **gap +0.051** | +0.525 [+0.503] **gap +0.023** | +0.609 [+0.546] **gap +0.062** |
| 4/4 | 12 | +0.249 [+0.180] **gap +0.069** | +0.184 [+0.163] **gap +0.021** | +0.270 [+0.184] **gap +0.086** |

(The global-mean centering variant is in the CSV; it gives the same ordering with
larger gaps for 1e5-vs-1e7, because it leaves the domain offset in.)

**Numerical observations (single sample, so these are indicative, not estimates
with error bars):**
- The absolute cosines are misleading on their own. 1e5-vs-Render reaches +0.62
  at block 6 centered, but the different-scene control is +0.61 — i.e. **almost
  the entire similarity is the shared common component, not this scene**. The
  1e5↔Render gap never exceeds +0.030 at any depth, raw or centered.
- 1e5-vs-1e7 is the only clearly non-trivial pairing, and it is **strongest at
  the shallowest layer** (gap +0.170 raw / +0.139 centered at block 3) and decays
  monotonically with depth to ~+0.05–0.07 at block 12.
- 1e7-vs-Render holds a modest but consistent gap at every depth
  (+0.044…+0.071 raw, +0.062…+0.111 centered), so the render does correspond to
  the clean radar somewhat — it is the *noise*, not the render/radar domain gap,
  that destroys the correspondence.
- Centering does not change the ranking, but it roughly doubles the 1e7↔Render
  gap at block 12 (+0.044 → +0.086) and lifts the 1e5↔Render gap out of the
  floor at deep layers (+0.002 → +0.021).

**Visual observations (from the generated figures):**
- In the RAW PCA-1 figure, PC1 mostly encodes **which input modality a patch came
  from**, not scene structure: the 1e5 row sits near the top of the shared range
  and the render row near the bottom at blocks 3, 6 and 12. PC1 explains 30.5% of
  joint variance at block 3, dropping to ~16% deeper. Block 9 is the exception —
  there all three rows show a bright object-shaped blob, so it is the only raw
  depth where a figure/ground reading survives the domain offset.
- After centering, PC1 turns into a **figure/ground map** (object dark,
  background bright) for 1e7 and Render at blocks 6, 9 and 12. The 1e5 row shows
  a visibly weaker but correctly-placed dark object region at blocks 6 and 9, and
  by block 12 it has essentially washed out into texture.
- Centered block 3 is the most modality-specific layer visually: 1e7 shows a
  bright filled silhouette, the render shows only its edges/horizontal bars, and
  1e5 shows very little — consistent with block 3 being dominated by low-level
  appearance rather than shape.
- The PCA-RGB figures tell the same story more sharply: at block 12 the render's
  object is a single flat region (a clean object/background segmentation), the
  1e7 object is a coherent darker region, and the 1e5 panel is dominated by
  noise-driven texture with only a faint object outline.
- So, answering the figure's question by eye: **1e5 and 1e7 do highlight the same
  spatial region at mid depth (blocks 6 and 9), but the 1e5 map is markedly
  degraded, and by the final block the noisy input no longer produces a
  recognisable object representation.**

### Output
- `dino_analysis/outputs/dino_spatial_pca_1e5_1e7_render_raw.png`
- `dino_analysis/outputs/dino_spatial_pca_1e5_1e7_render_centered.png`
- `dino_analysis/outputs/dino_spatial_pca_rgb_1e5_1e7_render_raw.png`
- `dino_analysis/outputs/dino_spatial_pca_rgb_1e5_1e7_render_centered.png`
- `dino_analysis/outputs/dino_spatial_similarity_1e5_1e7_render.csv`
- `dino_analysis/outputs/dino_spatial_analysis_metadata.json`
- `dino_analysis/dino_spatial_layer_means.pt` (cached spatial per-layer means)
- script: `dino_analysis/visualize_dino_spatial_pca.py`

Run command:
```bash
PY=/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
$PY dino_analysis/visualize_dino_spatial_pca.py \
    --sample-id 0196 --device cpu --compute-centered \
    --compute-layer-means --num-mean-samples 150   # means cached after 1st run
```

### Interpretation / Next Step
- **The render is not a usable spatial prior for the noisy radar.** The
  1e5↔Render corresponding-patch gap is ≤ +0.03 at every depth — near the
  different-scene floor. That is an independent, spatial-level replication of
  the weak pooled render↔radar signal, and it is a real argument against the
  renderDINO arm's premise, not just against its FiLM implementation.
- **If DINO is used spatially, use a mid-shallow block, not the last one.** The
  1e5↔1e7 correspondence is largest at block 3 and decays with depth; block 12
  is where the noisy input's representation collapses. The training recipe's
  [1,4,8,12] pooled set is not obviously the right choice for a spatial prior.
- **Next test:** repeat this over ~50 val samples instead of one, and report
  gap ± SE per block. One sample cannot distinguish +0.170 from +0.099; a
  small sweep would say whether the shallow-block advantage is real and whether
  the 1e5↔Render null holds dataset-wide.
