# Exp 3 (pre-registration) — DINOv2 FiLM guidance on the verynoisy baseline

**Status:** designed, not yet trained. This is a pre-registration — written
before the run so the prediction can't be moved to fit the result.
**Baseline:** Exp 2, `Holo_Baseline_Restormer_verynoisy` (see `../results.md`).
**Date:** 2026-08-05.

---

## The question this run answers

> Does adding semantic guidance from a frozen DINOv2, injected as FiLM at the
> bottleneck and decoder, improve verynoisy holographic denoising over the pure
> Restormer baseline — when *nothing else* changes?

The baseline's headline weakness (Exp 2) is over-smoothing: the prediction keeps
only ~22 % of the ground-truth high-frequency energy. The hypothesis is that a
semantic prior tells the network *what object* it is reconstructing, so it can
restore structure the pixel loss alone has no incentive to keep.

This is a **single-variable** experiment. The only difference from Exp 2 is the
semantic guidance. Everything in the "held identical" list below is unchanged.

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

## Config values (the diff)

```yaml
name: Holo_DINOv2_Restormer_verynoisy      # was Holo_Baseline_Restormer_verynoisy
network_g:
  type: RestormerDINO                       # was Restormer
  # ... all Restormer kwargs unchanged ...
  dino_layers: [0, 3, 7, 11]                # 1-indexed {1,4,8,12}
  dino_img_size: 224
  dino_model_name: dinov2_vitb14
  dino_hub_source: github
  dino_hub_dir: ~
  dino_weights: ~
  dino_stub: false
  film_hidden: 512
logger:
  tb_logger_dir: tb_logger/Holo_DINOv2_Restormer_verynoisy
```

Files: arch `basicsr/models/archs/restormer_dino_arch.py`; config
`Deraining_Holo/Options/Holo_DINOv2_Restormer.yml`; sanity check
`Deraining_Holo/sanity_check_dino_film.py`.

---

## Where DINO features are computed, and caching

- **Computed inside `RestormerDINO.forward`**, from the LQ input, once per
  forward pass. basicsr calls `net_g(self.lq)` unchanged — no training-loop edit.
- **Recompute per step; do NOT cache.** DINO is frozen, so caching is only valid
  if the *exact same pixels* recur. They do not: training uses random crops +
  geometric augmentation, so every step sees a different tensor. A cache keyed by
  image id would be stale the moment the crop or flip changes. Precomputing
  per-image (whole-image, no aug) would be a *different* design — DINO would see
  something the Restormer never sees — so it is rejected here.
- **Cost:** one frozen ViT-B forward per step at 224×224, under `no_grad`. This
  is the price of the guidance and is inherent to the design.

---

## Pre-registered prediction and falsification

**Prediction.** On the 338-image test set, DINO-FiLM beats the Exp 2 baseline on
the **masked (foreground)** metric — the honest measure — by a margin that clears
run-to-run noise:

- masked PSNR improves by **≥ 0.3 dB** over 18.313 dB, **and**
- Pred/GT high-frequency energy ratio rises meaningfully above **0.22**
  (i.e. measurably less over-smoothing).

**What would falsify it.** Any of:
- masked PSNR change within ±0.3 dB of baseline → guidance did nothing useful;
- masked PSNR *drops* → guidance hurts;
- PSNR rises but the HF-energy ratio does **not** move → it improved by some
  other route, not by fixing the over-smoothing the hypothesis targets.

The ±0.3 dB band is a stand-in for a proper significance test; a stricter version
would train ≥2 seeds per arm and compare distributions. Checkpoint selection and
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
