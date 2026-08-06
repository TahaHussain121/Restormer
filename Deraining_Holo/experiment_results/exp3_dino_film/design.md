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
   (offline, CPU, seconds). **PASSED** on 2026-08-05: max abs diff 0.000e+00.
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
