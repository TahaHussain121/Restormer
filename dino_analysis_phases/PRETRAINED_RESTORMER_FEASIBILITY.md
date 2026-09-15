# Could E0 start from the official pretrained Restormer? — read-only feasibility note (2026-09-15)

Nothing was downloaded, no official checkpoint was loaded, no code was changed
and no run was started. Sources: the official `Denoising/README.md`,
`Denoising/Options/GaussianGrayDenoising_Restormer.yml` and
`Denoising/test_gaussian_gray_denoising.py` (github.com/swz30/Restormer, read
2026-09-15), and the local code. Local parameter counts were measured by building
the architecture on CPU.

## Architecture: local E0 against the official grayscale denoiser

| | local E0 (`E0_fixed128_baseline.yml`) | official gray Gaussian (`GaussianGrayDenoising_Restormer.yml`) |
|---|---|---|
| class | `Restormer` (`basicsr/models/archs/restormer_arch.py`) | `Restormer` |
| input / output channels | 1 / 1 | 1 / 1 |
| width `dim` | 48 | 48 |
| `num_blocks` | [4, 6, 6, 8] | [4, 6, 6, 8] |
| refinement blocks | 4 | 4 |
| heads | [1, 2, 4, 8] | [1, 2, 4, 8] |
| FFN expansion | 2.66 | 2.66 |
| conv `bias` | False | False |
| **LayerNorm** | **`WithBias`** — (x − mean) / std · w + b | **`BiasFree`** — x / std · w, no mean subtraction, no bias |
| global residual | output + input | output + input |
| trainable parameters | **26,124,052** | **26,109,076** |

**The only difference is the LayerNorm variant.** Measured on CPU: the local
state dict has **88 extra keys**, all `*.norm{1,2}.body.bias` (14,976
parameters); no official key is missing locally and no shared tensor differs in
shape.

Consequences:

* A strict load of official weights into the E0 configuration **fails** (88
  missing keys).
* A non-strict load is **not** equivalent even with zero biases, because
  `WithBias` also subtracts the per-token mean, which `BiasFree` does not: a
  different function.
* **What initialising from official weights requires:** a new configuration with
  `LayerNorm_type: BiasFree` (the local class already supports it) and a new
  experiment identity. The official weights should then load strictly
  (`checkpoint['params']`, as the official test script does). This is not yet
  verified against the actual file, because it was not downloaded.

## Preprocessing and data: official against ours

| | official gray denoiser | this project |
|---|---|---|
| images | natural images (DIV2K, Flickr2K, WED, BSD — "DFWB") | radar holography reconstructions |
| bit depth / scaling | 8-bit, `/255` | uint16, `/65535` |
| degradation | synthetic additive white Gaussian noise, σ ∈ [0, 50] (blind model) or fixed σ 15/25/50 (non-blind) | 1e5 ray-budget reconstruction against a 1e7 target: structured, non-Gaussian, with sidelobes |
| training crops | progressive 128 → 384, 300k iterations, AdamW 3e-4 | E0: fixed 128, batch 8, 300k, AdamW 3e-4 |
| test padding | reflect-pad to a multiple of 8, clamp to [0, 1] | 256x256 needs no padding; clamp to [0, 1] |

Four checkpoints exist: blind (one model for σ 15/25/50) and non-blind σ15,
σ25, σ50. Weight files use the suffixes `_blind.pth` / `_sigma{σ}.pth` in the
official test script, and the README points to a Google Drive folder.

## What a comparison would and would not show

* **Transfer is an open question.** Natural-image Gaussian denoising priors
  (8-bit, texture statistics, white noise) may or may not transfer to radar
  reconstructions, whose background is mostly exactly zero and whose
  degradation is structured.
* **Fine-tuning updates Restormer itself.** That is unlike the frozen-backbone
  refiners (refiner_e0, the foreground-balanced follow-up), where E0 never
  changes. It would be a different kind of experiment.
* **Practical performance, not the effect of pretraining.** A short pretrained
  fine-tune compared with the existing long-trained E0 (300k iterations from
  scratch) measures how well that pipeline works in practice. It does not
  isolate the effect of pretraining: iterations, schedule and the LayerNorm
  variant would all differ.
* **A causal initialisation comparison needs matched training.** Pretrained
  versus random initialisation must share the downstream recipe (same
  `BiasFree` architecture, crops, iterations, schedule, seed policy and
  selection rule). The existing E0 (`WithBias`) is therefore **not** the
  matched random-init control; a `BiasFree` from-scratch run would be.
* **Comparability with the DINO arms.** Every DINO arm is built on E0's
  `WithBias` trunk, so a `BiasFree` pretrained trunk would not be comparable
  with them unless they were retrained on it.

No fine-tuning or SwinIR experiment is proposed or started here.
