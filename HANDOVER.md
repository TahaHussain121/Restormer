# HANDOVER — DINOv2 FiLM guidance (E1), as of 2026-08-05

Read this + `CONTEXT.md` + `DEVLOG.md` at the start of a new session. This file
covers the DINO/E1 work specifically; CONTEXT.md is the project-wide primer and
DEVLOG.md is the append-only step log.

## One-line status
E1 (DINOv2 FiLM guidance) is **built, wired, and verified but NOT trained**.
Two arms are ready to launch. Several decisions are pending — see "Open decisions".

---

## What E1 is
Add semantic guidance from a **frozen DINOv2 ViT-B/14** to the verynoisy
Restormer baseline (Exp 2), as FiLM `(1+γ)·F+β` at the bottleneck + 3 decoder
stages, γ/β predicted by a small MLP from pooled DINO features. Zero-init final
projection ⇒ identical to baseline at init. Restormer trained from scratch; DINO
is the only pretrained part. **One variable studied: what image DINO looks at.**

Full pre-registration: `Deraining_Holo/experiment_results/exp3_dino_film/design.md`.

### Two arms
| Arm | DINO sees | Config | model / dataset |
|---|---|---|---|
| **B lqDINO** | the noisy LQ crop (same as Restormer) | `Holo_DINOv2_lqDINO_Restormer.yml` | `ImageCleanModel` / `Dataset_PairedImage_uint16` |
| **A renderDINO** | the black-bg render, cropped+flipped like the LQ | `Holo_DINOv2_renderDINO_Restormer.yml` | `ImageCleanModelRender` / `Dataset_PairedImage_uint16_Render` |

Both differ from the Exp 2 baseline ONLY in the DINO/FiLM block (+ arm A's render
data path). Everything else (optimizer, LR schedule, progressive crops
[128,160,192,256]@batch[8,5,4,2], 300k iters, L1, split, seed 100, mixup off) is
held identical.

---

## Key files
- arch: `basicsr/models/archs/restormer_dino_arch.py`
  (`RestormerDINO`, `DINOv2Extractor`, `FiLMHead`, `dino_preprocess/denormalize`)
- render dataset: `basicsr/data/paired_image_uint16_render_dataset.py`
- render model: `basicsr/models/image_clean_render_model.py`
- configs: `Deraining_Holo/Options/Holo_DINOv2_{lqDINO,renderDINO}_Restormer.yml`
- sanity check (identity at init): `Deraining_Holo/sanity_check_dino_film.py`
- weights-loaded verification: `Deraining_Holo/verify_dino_weights.py`
- input debug dumps: `Deraining_Holo/debug_dino_inputs.py`
- feature-separation analysis: `Deraining_Holo/analyze_dino_features.py`
- render↔radar similarity (corrected, no synthetic noise): `Deraining_Holo/render_radar_similarity.py`

## DINOv2 offline cache (compute nodes have NO internet — do not rely on it)
Cached on shared `/home/woody` so all nodes read from disk:
- repo dir: `/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub/hub/facebookresearch_dinov2_main`
- weights:  `/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth`
Both configs point `dino_hub_source: local` at these. `DINOv2Extractor` loads with
`strict=True` and **fails loudly** (FileNotFoundError) if a path is missing.
xformers/timm NOT needed (DINO falls back to vanilla attention).

## Model / build facts
- ViT-B/14 = 86.6M params, embed 768; layers {1,4,8,12} → 0-indexed {0,3,7,11};
  4 layers × 768 pooled+concat → **3072-d** feature.
- FiLM target channels (from dim=48): bottleneck 384, decoder 192/96/96 (sum 768).
- Trainable params 28,485,396 = 26,124,052 backbone + 2,361,344 FiLM head. DINO frozen.
- DINO input pipeline: resize→224 (bilinear, align_corners=False; 224=16×14),
  ImageNet-normalized (range ~[-2.1,2.6]). Same for both arms; only the source image differs.

---

## Verification done (all before launch)
1. **Identity at init**: FiLM-on == FiLM-off == baseline, bit-identical (max diff 0.0).
2. **Alignment (arm A)**: render crop shares crop+flip RNG with the LQ — 200/200 marker
   trials identical + visual overlay. `debug/alignment_overlay.png`.
3. **DINO pipeline**: resize/ImageNet-norm confirmed applied; arm B reuses Restormer's
   [0,1] input (correct pre-norm range, not a double-norm bug). Content is grayscale = OOD.
4. **Weights genuinely loaded** (not silent-random): strict keys empty; params == .pth
   file AND ≠ random init; cross-model cos(loaded,random)≈0.02. `verify_dino_weights.py`.
5. **Feature-separation (pre-E1)**: the pooled DINO feature is offset-dominated —
   ~95.6% of each feature's energy is a shared mean vector (‖mean‖≈106 vs ‖residual‖≈22).
   Raw cosine ~0.9 for everything; centering reveals a WEAK object signal (see design.md
   "Feature-separation analysis"). CORRECTED render↔radar test (`render_radar_similarity.py`,
   replaces the wrong synthetic-Gaussian arm-A test): does DINO link a render to the SAME
   object's 1e7 radar heatmap more than a different object's? RAW is object-blind (d≈0.03@128,
   null@256); CENTERED shows a real, significant signal (d=+0.25/+8.6σ @128, +0.17/+6σ @256;
   same-object residual cos 0.27 vs different 0.02). So the render carries modest object
   identity that transfers to radar — but ONLY in the centered residual. Strengthens the case
   that centering the FiLM input is needed for renderDINO to help.

---

## Open decisions (nothing launches until resolved)
1. **Centering the FiLM input?** The pooled feature is 95%+ shared offset. Option: subtract
   a per-arm precomputed training mean (register a buffer, subtract before the MLP; preserves
   zero-init identity). It's a design change to E1 — needs a pre-registration note. Not applied.
2. **Which arm(s) to run** — both, or start with renderDINO (stronger in the analysis)?
3. **Single seed** confirmed (user: no multi-seed for now).
4. **[SOURCE NEEDED]**: the "layers {1,4,8,12}" recipe citation — user's paper to fill in.
5. **Test-time eval for arm A**: `test_holo.py` doesn't yet load/pass the render; needed at
   eval time for the renderDINO arm.

## Pre-registered E1 prediction (do not move it post hoc)
Gain small (≤ ~0.3 dB masked PSNR, plausibly within noise / leaning null),
renderDINO ≥ lqDINO if any effect; FiLM head likely needs centered features.
Rationale: object signal in the pooled feature is weak and offset-buried.

---

## Commands (run from repo root, venv python)
```bash
PY=/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

$PY Deraining_Holo/sanity_check_dino_film.py        # identity-at-init (offline, CPU)
$PY Deraining_Holo/verify_dino_weights.py           # weights really loaded
$PY Deraining_Holo/debug_dino_inputs.py             # dump input grids + bg numbers
# launch (only when decisions above are settled) — build chain drivers first:
#   sbatch Deraining_Holo/train_holo_chain_lqDINO.sh      (NOT YET CREATED)
#   sbatch Deraining_Holo/train_holo_chain_renderDINO.sh  (NOT YET CREATED)
```
NOTE: launch scripts for the two arms do NOT exist yet — mirror
`train_holo_chain_verynoisy.sh` (self-chaining resume driver), one per arm, with
its own experiment dir + `Holo_chain_state_*` bookkeeping.

## Dataset (holographic_image_dataset), split seed 42
Per object, aligned by filename across: `*_clean` (1e6 GT), `*_verynoisy` (1e7 LQ),
`*_renders_blackbg` (black-bg render, DINO ref for arm A). Splits: train 6101 / val 339 /
test 338. `splits/{train,val,test}.txt`. renders_blackbg made from renders/ at threshold 250.

## Results archive
`Deraining_Holo/experiment_results/` — README + running `results.md` + per-exp
`report.md`/figures/metrics. Exp 1 (noisy) and Exp 2 (verynoisy) done; Exp 2 test
22.405 dB full / 18.313 dB masked, over-smoothing (HF ratio 0.216) is the weakness E1 targets.

## Working style (user preferences)
- User runs on FAU HPC (tinygpu/SLURM); Claude executes commands directly.
- Report honestly: a predicted null is useful, a rationalized one is not. Do not soften flat results.
- Do not silently fix — flag what's wrong and where, let the user decide.
- One change at a time; extra ideas go at the end as separate suggestions.
- Commit + push meaningful work to branch `dino_prior`.
