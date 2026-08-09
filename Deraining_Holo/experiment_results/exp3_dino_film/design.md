# Exp 3 (pre-registration) — DINOv2 FiLM guidance on the verynoisy baseline

**Status:** designed, not yet trained. This is a pre-registration — written
before the run so the prediction can't be moved to fit the result.
**Baseline:** Exp 2, `Holo_Baseline_Restormer_verynoisy` (see `../results.md`).
**Date:** 2026-08-05.

---

## The question this run answers

> Does adding semantic guidance from a frozen DINOv2, injected as FiLM at the
> bottleneck and decoder, improve verynoisy holographic denoising over the pure
> Restormer baseline — when *nothing else* changes? And does the SOURCE of the
> semantic signal matter: a clean canonical render vs. the actual noisy input?

The baseline's headline weakness (Exp 2) is over-smoothing: the prediction keeps
only ~22 % of the ground-truth high-frequency energy. The hypothesis is that a
semantic prior tells the network *what object* it is reconstructing, so it can
restore structure the pixel loss alone has no incentive to keep.

## Two arms (this is the one variable being studied)

Both arms are identical to each other and to the Exp 2 baseline in every way
except **what image DINO looks at**. Restormer always processes the noisy LQ.

| Arm | DINO sees | Config | Dataset / model |
|---|---|---|---|
| **B — lqDINO** | the **noisy LQ crop** (same tensor Restormer gets) | `Holo_DINOv2_lqDINO_Restormer.yml` | `Dataset_PairedImage_uint16` / `ImageCleanModel` |
| **A — renderDINO** | the **black-bg render**, cropped+augmented identically to the LQ | `Holo_DINOv2_renderDINO_Restormer.yml` | `Dataset_PairedImage_uint16_Render` / `ImageCleanModelRender` |

Arm A tests "clean semantic reference"; arm B tests "self-conditioning on the
degraded input". The render in arm A goes through the **same random crop and
flip** as the LQ (via `paired_random_crop`/`random_augmentation`), so DINO sees
the exact same region — the only difference from arm B is clean-render pixels vs
noisy pixels. Both are compared against the Exp 2 no-DINO baseline.

The FiLM wiring, injection points, zero-init identity, and everything in the
"held identical" list are shared by both arms.

---

## Design → source mapping

Each row names the paper it comes from, or is labelled as **[your extension]**
(a choice not taken from a specific paper) or **[SOURCE NEEDED]** (you said
"following the published recipe" — fill in the exact paper + section, I will not
invent a citation).

| Design element | Source |
|---|---|
| Restormer backbone (encoder/bottleneck/decoder, MDTA, GDFN) | Zamir et al., *Restormer*, CVPR 2022 |
| Frozen DINOv2 ViT-B/14 as the visual prior | Oquab et al., *DINOv2*, TMLR 2024 (arXiv 2304.07193) |
| FiLM mechanism: modulate features by per-channel (γ, β) | Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018 — FiLM layer definition (their Method section) |
| Zero-init the final γ/β projection → identity at init | ControlNet, Zhang et al., ICCV 2023 — "zero convolution" (their §3.1). (1+γ)·F+β form so zero γ = identity |
| **Which** DINO layers to read: {1,4,8,12} | **[SOURCE NEEDED]** — your "published recipe". Wired as 0-indexed {0,3,7,11}; confirm that matches your paper's layer numbering |
| Inject at bottleneck + decoder, **not** encoder | **[your extension]** — you specified this. Rationale (yours): the encoder should stay a faithful low-level feature extractor; guidance belongs where the image is reconstructed. No paper cited yet |
| Pool DINO patch tokens to one global vector per layer | **[your extension]** — "pooled to a single global vector". Mean-pool over patches |
| MLP hidden width 512 | **[your extension]** — arbitrary small head; not from a paper |
| Feed DINO the same (cropped) LQ tensor Restormer sees | **[your extension]** — keeps the baseline data path untouched (see "held identical"). A recipe that feeds full images would change the data path |

---

## What is held identical to the baseline

Optimizer (AdamW, lr 3e-4, wd 1e-4), LR schedule (CosineAnnealingRestartCyclicLR,
periods [92000, 208000]), progressive crop policy ([128,160,192,256] over
[92000,64000,48000,96000], batch [8,5,4,2]), total_iter 300000, L1 loss,
train/val/test split, mixup off, seed 100, uint16 loader, geometric augs.
Verified: the config diff touches only `name`, `network_g.type` + the DINO block,
and `tb_logger_dir`.

Two things do change as a *necessary consequence* of the one idea, not as extra
knobs: (a) the optimizer now also sees the FiLM head's ~2.36 M trainable params
(DINO stays frozen, auto-excluded because `requires_grad=False`); (b) the network
has the DINO+FiLM modules. At init these produce γ=β=0, so the run starts
numerically identical to the baseline (proved by the sanity check).

---

## Config values (the diff vs Exp 2 baseline)

Shared by both arms — the FiLM/DINO block on `network_g` (type `RestormerDINO`):
```yaml
network_g:
  type: RestormerDINO                       # was Restormer
  # ... all Restormer kwargs unchanged ...
  dino_layers: [0, 3, 7, 11]                # 1-indexed {1,4,8,12}
  dino_img_size: 224
  dino_model_name: dinov2_vitb14
  dino_hub_source: github ; dino_hub_dir: ~ ; dino_weights: ~   # set for offline
  dino_feat_mean: .../dino_feat_mean_{lqDINO,renderDINO}.pt     # centering (see below)
  dino_stub: false                          # true ONLY for the sanity check
  film_hidden: 512
```
Arm B (lqDINO) changes nothing else — `model_type: ImageCleanModel`,
`Dataset_PairedImage_uint16`. Arm A (renderDINO) additionally sets:
```yaml
model_type: ImageCleanModelRender
datasets: {train,val}:
  type: Dataset_PairedImage_uint16_Render
  dataroot_render: .../{train,val}_renders_blackbg   # the only new data path
```

Files:
- arch `basicsr/models/archs/restormer_dino_arch.py`
- render dataset `basicsr/data/paired_image_uint16_render_dataset.py`
- render model `basicsr/models/image_clean_render_model.py`
- configs `Deraining_Holo/Options/Holo_DINOv2_lqDINO_Restormer.yml` (B),
  `Holo_DINOv2_renderDINO_Restormer.yml` (A)
- sanity check `Deraining_Holo/sanity_check_dino_film.py`

---

## Where DINO features are computed, and caching

- **Computed inside `RestormerDINO.forward`**, once per forward pass.
  - Arm B: `net_g(self.lq)` — DINO sees the LQ (dino_img defaults to None).
  - Arm A: `ImageCleanModelRender` passes the render as `dino_img`, so
    `net_g(self.lq, dino_img=render)` — DINO sees the render crop.
- **Recompute per step; do NOT cache — in both arms.** DINO is frozen, so caching
  is only valid if the *exact same pixels* recur. They do not: training uses
  random crops + geometric augmentation, so every step sees a different tensor —
  and in arm A the render is cropped/flipped identically to the LQ, so it changes
  every step too. A cache keyed by image id would be stale the moment the crop or
  flip changes. (Caching *would* be possible if arm A fed DINO the whole,
  uncropped render — but then it would see a different spatial extent than the LQ
  and the A/B comparison would confound "clean vs noisy" with "whole vs crop", so
  that option is rejected.)
- **Cost:** one frozen ViT-B forward per step at 224×224, under `no_grad`, in
  both arms — so they are equal in compute and differ only in DINO's input.

---

## Pre-registered prediction and falsification

**Prediction.** On the 338-image test set, measured on the **masked (foreground)**
metric (the honest measure), vs the Exp 2 baseline (18.313 dB, HF ratio 0.216):

1. At least one DINO arm improves masked PSNR by **≥ 0.3 dB**, **and** raises the
   Pred/GT high-frequency energy ratio above **0.22** (measurably less
   over-smoothing).
2. Arm A (renderDINO, clean semantic reference) **≥** arm B (lqDINO) on masked
   PSNR — a clean object descriptor should help at least as much as conditioning
   on the degraded input.

**What would falsify it.** Any of:
- both arms within ±0.3 dB of baseline → guidance does nothing useful;
- either arm *drops* masked PSNR → guidance hurts;
- PSNR rises but the HF-energy ratio does **not** move → improved by some other
  route, not by fixing the over-smoothing the hypothesis targets;
- arm B beats arm A by > 0.3 dB → the clean render is *not* the better signal,
  which would itself be an interesting (hypothesis-2-falsifying) result.

Single seed per arm (as decided) — so treat a sub-0.3 dB gap between A and B as a
tie, not a ranking. The ±0.3 dB band is a stand-in for a proper significance
test; a stricter version would train ≥2 seeds per arm. Checkpoint selection and
the test-split caveat from Exp 2 (`../results.md`) apply here too — keep them
consistent across both arms so they cancel.

---

## Before spending GPU hours

1. `python Deraining_Holo/sanity_check_dino_film.py` — proves identity at init
   (offline, CPU). **PASSED** on 2026-08-05: max abs diff 0.000e+00.
   **Re-run and PASSED again on 2026-08-06, after the centering change (A1)**,
   now including a real-DINOv2 part that loads each arm's actual mean vector:
   for both arms `max|film_on − film_off| = 0.000e+00` and
   `max|film_model − baseline| = 0.000e+00`, and the extractor output was
   verified equal to `raw_pooled − feat_mean` (max residual 0.00e+00) with the
   buffer byte-equal to the `.pt` named in the yml.
2. Make DINOv2 available (see the arch header): the compute nodes are offline and
   `xformers`/`timm` are absent, so `torch.hub.load(..., source='github')` will
   fail on a compute node. Pre-download `dinov2_vitb14` weights + clone the repo
   on a node with internet, then set `dino_hub_source: local`, `dino_hub_dir`,
   and `dino_weights` in the config. Do one real forward pass to confirm the
   feature shape [B, 3072] before launching the full run.

---

## Feature-separation analysis (pre-E1, written 2026-08-05, before training)

Question: is the pooled DINO feature the FiLM head consumes actually
object-discriminative, or a thin residual on a large shared offset?
Script: `Deraining_Holo/analyze_dino_features.py` (frozen DINOv2, verified weights).

**Item 1 — 3-probe pairwise cosine (renders), raw vs centered**

| pair | raw | centered |
|---|---|---|
| object_A vs object_B | +0.935 | **-0.237** |
| object_A vs empty | +0.881 | -0.161 |
| object_B vs empty | +0.901 | -0.075 |

Raw ~0.9 for everything; after subtracting the training-set mean feature it
collapses, and the two different objects become the *most* dissimilar pair. The
raw similarity was almost entirely a shared offset.

**Item 2/3 — discrimination d = sim(A, A_noisy) - sim(A, B), N=100, mean±std**
(arm A A_noisy = render+N(0,0.1) synthetic; arm B A_noisy = real clean/noisy pair)

| arm @ crop | raw d | centered d | centered significance |
|---|---|---|---|
| A (render) @128 | -0.261 ± 0.068 | **+0.211 ± 0.383** | +5.5σ |
| B (lq) @128 | -0.156 ± 0.126 | **+0.183 ± 0.495** | +3.7σ |
| A (render) @256 | -0.252 ± 0.031 | **+0.182 ± 0.354** | +5.1σ |
| B (lq) @256 | -0.312 ± 0.195 | -0.123 ± 0.762 | -1.6σ (n.s.) |

**Reading (not softened):** In *raw* pooled space the feature is object-blind —
d is negative everywhere, i.e. a different object is *more* similar to the anchor
than the anchor's own noised version, because a large shared offset dominates.
*Centering removes that offset and a real but WEAK object signal appears*: mean d
flips positive and is aggregate-significant for 3 of 4 configs (renderDINO both
sizes, lqDINO@128), but the per-pair std is ~2× the mean, so it is reliable only
in aggregate, not per image. lqDINO@256 stays null. So the signal exists and lives
entirely in the residual; the raw vector buries it.

**Pre-registered E1 prediction (follows from the above):** because the object
signal is weak and offset-buried, E1's masked-PSNR gain over the Exp 2 baseline
will be small (predict ≤ ~0.3 dB, plausibly within noise / leaning null), with
renderDINO ≥ lqDINO if any effect appears; the FiLM head will likely need
centered (offset-removed) input features to extract even that.

### CORRECTED render↔radar test (2026-08-05, supersedes the synthetic-noise arm-A row)

The arm-A discrimination above used render + synthetic Gaussian noise for
"same object under noise" -- wrong. The real question is whether DINO links a
render to the SAME object's actual 1e5 radar heatmap more than to a DIFFERENT
object's. No synthetic noise; the "other view" is the real radar image.
Script: `Deraining_Holo/render_radar_similarity.py`.

d = sim(render_i, radar_i) − sim(render_i, radar_j), N=100, mean±std:

| crop | raw d | centered d | centered same / diff | centered σ |
|---|---|---|---|---|
| 128 | +0.027 ± 0.145 (SE .015) | **+0.249 ± 0.295 (SE .029)** | +0.272 / +0.023 | +8.6σ |
| 256 | −0.008 ± 0.270 (SE .027) | **+0.167 ± 0.280 (SE .028)** | +0.159 / −0.008 | +6.0σ |

**Reading (not softened):** raw cross-modal is object-blind (d≈0.03 at 128, null
at 256 — the modality offset dominates). *Centered*, a real and clearly
significant object-linking signal appears: a render's residual points ~0.27
cosine toward the SAME object's radar residual vs ~0.02 for a different object
(d=+0.25, +8.6σ at 128; +0.17, +6σ at 256). So the render DOES carry object
identity that transfers to the radar domain — but the absolute alignment is
MODEST (0.27) and exists ONLY in the residual; the raw pooled feature buries it.

**Updated E1 implication:** renderDINO has a genuine (if modest) object signal to
work with, but essentially all of it is in the centered residual — so centering
the FiLM input is now well-justified, not optional. Without centering, raw
render features are near object-blind and renderDINO would likely come back
near-null. Prediction stays: small gain, renderDINO ≥ lqDINO, conditional on
centered features.

---

## Amendments made before training (2026-08-06)

All five items below were written **before either arm was launched**. Nothing
here is a post-hoc reading. The training schedule is unchanged.

### A1. Centering is now part of E1 (design change, applied)

The FiLM head is fed `pooled − mean` instead of `pooled`, where `mean` is a
**fixed, per-arm vector precomputed over 300 training crops** and subtracted at
the end of the DINO extractor's forward.

*Why this and not BatchNorm:* the progressive schedule drops the batch to **2**
at 256px, and a two-sample batch mean is noise, not a mean. A fixed vector also
keeps the guidance signal stationary over the run, so a change in FiLM output
can only come from the image, never from batch composition.

*Why it counts as measured, not guessed:* in raw pooled space the cross-domain
signal does not exist — the corrected render↔radar test gives **d ≈ 0.03 (n.s.)
at crop 128 and null at 256**. Centered, the same test gives **d = +0.25 (+8.6σ)
and +0.17 (+6.0σ)**. The signal the renderDINO arm is supposed to exploit is a
property of the centered residual only. Feeding raw features would be testing a
representation already measured to be object-blind.

How the mean is built (`Deraining_Holo/compute_dino_feat_mean.py`, seed 0):

- crops drawn through the arm's **own dataset class**, so the exact training data
  path (same random crop, same geometric augs, same loader and value range);
- **train split only** — no val/test images touch the mean;
- crop sizes drawn **in proportion to the progressive schedule's iteration
  counts** (92/64/48/96 crops at 128/160/192/256, matching 92k/64k/48k/96k), so
  the mean matches the crop-size mix the run actually sees rather than one stage;
- arm-specific by construction: renderDINO's mean is over **render** crops,
  lqDINO's over **noisy-LQ** crops. The two arms still differ only in what image
  DINO looks at; the mean follows the arm's domain because that is what an offset
  is.

Measured on those 300 crops (schedule-mixed, so not directly comparable to the
95.6 % figure quoted earlier for full-frame renders at one size — mixing crop
sizes adds real variance and lowers the offset share):

| arm | ‖mean‖ | mean ‖residual‖ | offset share of ‖feature‖² |
|---|---|---|---|
| renderDINO | 102.23 | 32.63 | 90.0 % |
| lqDINO | 85.63 | 35.40 | 84.9 % |

Implementation: `dino_feat_mean` config field → registered **buffer** in
`DINOv2Extractor`. Being a buffer, it is saved into the checkpoint, so a chained
resume cannot silently pick up a different mean. Vectors are committed at
`experiment_results/exp3_dino_film/dino_feat_mean_{renderDINO,lqDINO}.pt` with
full provenance metadata.

**Zero-init identity re-verified after this change** — see "Before spending GPU
hours" below. Subtracting a constant cannot break identity (γ=β=0 whatever the
feature), but it was confirmed rather than assumed.

### A2. Raw (uncentered) features are NOT run as a separate baseline arm

A third arm would cost ~3 GPU-days to confirm a **predicted null**: raw pooled
features were already measured object-blind (render↔radar d ≈ 0.03, n.s.). With
the thesis deadline that is a bad trade. Reported as: *raw pooled features were
measured object-blind (d ≈ 0.03, n.s.); centering was therefore adopted before
training rather than ablated.* If GPU time frees up later it can be added as an
optional row — it is not part of E1's claim set now.

Consequence, stated plainly: E1 cannot attribute any observed gain to centering
specifically, because the uncentered variant is not trained. The evidence for
centering is the pre-training feature measurement, not a training ablation.

### A3. Both arms are run

lqDINO is not dropped. It matches the published recipes, it works without a
render at inference, and if it comes back flat while renderDINO does not, that
contrast is itself a result.

### A4. Named candidate explanation, registered in advance, if E1 underperforms

**The DINO object signal weakens as crop size grows, in both arms:**

| arm | crop 128 | crop 256 | source test |
|---|---|---|---|
| renderDINO | +0.249 (+8.6σ) | +0.167 (+6.0σ) | render↔radar, `render_radar_similarity.py` |
| lqDINO | +0.183 (+3.7σ) | −0.123 (n.s.) | noisy↔clean, `analyze_dino_features.py` |

The progressive schedule spends its **final 96k iterations at crop 256** — the
phase doing the finest reconstruction, and the phase whose weights are kept. So
the guidance signal is **strongest early and weakest exactly where it matters
most**; for lqDINO it is measurably gone by then.

Registered now as a **candidate explanation to be invoked only if E1
underperforms**, so it cannot be mistaken for a post-hoc rescue. It is not a
prediction that E1 will fail, and it is **not** a reason to change the schedule:
the schedule is held identical to Exp 2 so the arms remain comparable to the
baseline. Testing this explanation would need a separate experiment (e.g. tiled
DINO inputs at 256, or holding the crop at 128), which is a different row.

### A5. The two arms measure different things — their d values are not scores

Restated so the numbers in the table above are not misread as a ranking:

- **lqDINO's d** comes from *noisy heatmap vs its own clean pair* — it measures
  **noise robustness** of the pooled feature within the radar domain.
- **renderDINO's d** comes from *render vs the same object's radar heatmap* — it
  measures **cross-domain correspondence** between two different modalities.

Different anchors, different comparison domains, different tests. "+0.249 >
+0.183" does **not** mean renderDINO carries more signal than lqDINO. Each d is
only interpretable against its own null (0) and its own σ. The pre-registered
prediction that renderDINO ≥ lqDINO rests on the argument that a clean object
descriptor should be at least as useful as one extracted from the degraded
input, **not** on comparing these two numbers.
