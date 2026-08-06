# HANDOVER — DINOv2 FiLM guidance (E1), as of 2026-08-06

Read this + `CONTEXT.md` + `DEVLOG.md` at the start of a new session. This file
covers the DINO/E1 work specifically; CONTEXT.md is the project-wide primer and
DEVLOG.md is the append-only step log.

## One-line status
E1 (DINOv2 FiLM guidance) is **TRAINING** as of 2026-08-06 — both arms launched:
job **1771016 renderDINO on a100**, job **1771017 lqDINO on v100**, self-chaining
to 300k (23 h walltime, MAX_CHAIN=8). Submitted ONCE per arm — **do not resubmit**;
each job queues its own successor and basicsr auto-resumes. See DEVLOG Step 21 for
the isolation audit and the pre-launch smoke results.

---

## What E1 is
Add semantic guidance from a **frozen DINOv2 ViT-B/14** to the verynoisy
Restormer baseline (Exp 2), as FiLM `(1+γ)·F+β` at the bottleneck + 3 decoder
stages, γ/β predicted by a small MLP from **centered** pooled DINO features. Zero-init final
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
- sanity check (identity at init, stub + real DINO + real means): `Deraining_Holo/sanity_check_dino_film.py`
- centering vectors: `Deraining_Holo/compute_dino_feat_mean.py` ->
  `Deraining_Holo/experiment_results/exp3_dino_film/dino_feat_mean_{lqDINO,renderDINO}.pt`
- launch drivers (self-chaining, one per arm): `Deraining_Holo/train_holo_chain_{lqDINO,renderDINO}.sh`
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
   **Re-run 2026-08-06 after the centering change — PASSED.** The check now has a second
   part using the REAL frozen DINOv2 + each arm's REAL mean vector read from its yml:
   both arms `max|film_on − film_off| = 0.000e+00`, `max|film_model − baseline| = 0.000e+00`;
   extractor output verified == `raw_pooled − feat_mean` (max residual 0.00e+00) and the
   buffer byte-equal to the `.pt` the yml names. (Identity could not break — γ=β=0 whatever
   the feature — but it was confirmed, not assumed.)
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

## Decisions taken 2026-08-06 (all implemented; full text in design.md "Amendments")
1. **Centering: ADOPTED.** Fixed per-arm mean pooled feature over 300 training crops
   (crop sizes drawn in proportion to the schedule's iters: 92/64/48/96 at 128/160/192/256,
   train split only, seed 0), registered as a buffer in `DINOv2Extractor` and subtracted at
   the end of `forward`. New config field `dino_feat_mean`, set in both ymls.
   Fixed mean, **not** BatchNorm — batch drops to 2 at 256px and a two-sample mean is noise.
   Measured: renderDINO ‖mean‖ 102.23 / ‖resid‖ 32.63 (offset = 90.0% of ‖feat‖²);
   lqDINO ‖mean‖ 85.63 / ‖resid‖ 35.40 (84.9%). Buffer ⇒ travels with the checkpoint, so a
   chained resume cannot silently use a different mean.
2. **No raw-feature baseline arm.** Raw pooled features were measured object-blind
   (render↔radar d ≈ 0.03, n.s.); paying ~3 GPU-days to confirm a predicted null is a bad
   trade against the deadline. Written up as a measured design choice, not an ablation.
   Stated limitation: E1 cannot attribute a gain to centering specifically.
   Optional later row if GPU time frees up.
3. **Both arms run.** lqDINO stays: it matches the published recipes, needs no render at
   inference, and a flat lqDINO alongside a non-flat renderDINO is itself a result.
4. **Identity at init re-verified after centering — PASSED** (see below).
5. **Registered in advance:** crop-size decay of the DINO signal as a *candidate
   explanation if E1 underperforms* (A4 in design.md). Schedule NOT changed.
6. **Single seed** confirmed (user: no multi-seed for now).

## Still open
- **[SOURCE NEEDED]**: the "layers {1,4,8,12}" recipe citation — user's paper to fill in.
- **Test-time eval for arm A**: `test_holo.py` doesn't yet load/pass the render; needed at
  eval time for the renderDINO arm (not needed to launch, needed before results).
- **Partition choice**: both chain scripts default to `a100` (CONTEXT.md's stated default);
  the Exp 2 baseline ran on `v100`. Switch if a100 is congested and record which was used.

## Pre-registered E1 prediction (do not move it post hoc)
Gain small (≤ ~0.3 dB masked PSNR, plausibly within noise / leaning null),
renderDINO ≥ lqDINO if any effect; FiLM head fed centered features (now applied).
Rationale: object signal in the pooled feature is weak and offset-buried.
Note the two arms' d values measure **different things** — lqDINO's d is noise
robustness inside the radar domain, renderDINO's is cross-domain correspondence —
so they are not comparable scores and "+0.249 > +0.183" ranks nothing.

---

## Commands (run from repo root, venv python)
```bash
PY=/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

$PY Deraining_Holo/sanity_check_dino_film.py        # identity-at-init (offline, CPU, ~2 min)
$PY Deraining_Holo/verify_dino_weights.py           # weights really loaded
$PY Deraining_Holo/debug_dino_inputs.py             # dump input grids + bg numbers

# regenerate the centering vectors (only if the data/schedule/layers change):
$PY Deraining_Holo/compute_dino_feat_mean.py \
    --opt Deraining_Holo/Options/Holo_DINOv2_renderDINO_Restormer.yml \
    --out Deraining_Holo/experiment_results/exp3_dino_film/dino_feat_mean_renderDINO.pt
$PY Deraining_Holo/compute_dino_feat_mean.py \
    --opt Deraining_Holo/Options/Holo_DINOv2_lqDINO_Restormer.yml \
    --out Deraining_Holo/experiment_results/exp3_dino_film/dino_feat_mean_lqDINO.pt

# LAUNCH — ONLY on the user's explicit go-ahead. Submit ONCE per arm; each job
# queues its own successor and auto-resumes (23 h walltime, ~3 jobs for 300k).
#   sbatch Deraining_Holo/train_holo_chain_lqDINO.sh
#   sbatch Deraining_Holo/train_holo_chain_renderDINO.sh
```
The two chain drivers exist (created 2026-08-06, mirrored from
`train_holo_chain_verynoisy.sh`), each with its own experiment dir and
`experiments/Holo_chain_state_{lqDINO,renderDINO}/` bookkeeping, so the arms can
run concurrently without touching each other or the Exp 2 baseline. Neither has
ever been submitted — they are untested against the scheduler.

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
