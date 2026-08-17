# Repository Investigation Report

**Date:** 2026-08-10
**Branch:** `dino_e2` (HEAD = `4e37beb`)
**Scope:** ANALYSIS ONLY. Nothing in the repository or the dataset was created, modified
or deleted to produce this report. No training was launched, no job was submitted, no
checkpoint or mean was recomputed. The only executions were two read-only probe scripts
in a scratch directory (see §19).

**Purpose:** establish, before designing the next DINO + Restormer experiment, exactly
what the Restormer baseline is, what its real tensor shapes are, what Phase 0/1/2
actually did, what DINO preprocessing was actually used, how centering was actually
done, whether the existing centering statistics match the training-crop pipeline, and
what infrastructure already exists.

---

## Table of contents

- [A. Important repository structure](#a-important-repository-structure)
- [B. Existing Restormer baseline](#b-existing-restormer-baseline)
- [C. Restormer tensor shapes](#c-restormer-tensor-shapes)
- [C-2. The exact latent / bottleneck injection point](#c-2-the-exact-latent--bottleneck-injection-point)
- [D. Dataset and pairing](#d-dataset-and-pairing)
- [D-2. Train / validation / test splits](#d-2-train--validation--test-splits)
- [E. Restormer preprocessing](#e-restormer-preprocessing-exact-pipeline)
- [F. Previous DINO preprocessing](#f-previous-dino-preprocessing)
- [G. Feature centering](#g-feature-centering)
- [H. Potential preprocessing / centering mismatch](#h-potential-preprocessing--centering-mismatch)
- [I. Phase 1 findings](#i-phase-1-findings--why-b6-was-selected-and-what-b3-actually-shows)
- [J. Phase 2 findings](#j-phase-2-findings--dino1e5-vs-dinorender-relative-to-dino1e7)
- [K. Existing metrics](#k-existing-metrics)
- [L. Existing visualization / result infrastructure](#l-existing-visualization--result-infrastructure)
- [M. Existing configuration system](#m-existing-configuration-system)
- [N. Important risks / unknowns](#n-important-risks--unknowns-to-resolve-before-implementing-e1-n)
- [§19. What was actually run](#19--what-was-actually-run)

---

## A. Important repository structure

Only files relevant to the model, training, data, metrics, results and DINO work.

```
Restormer/
├── CONTEXT.md                  project state, decisions, env, "what NOT to do"
├── DEVLOG.md                   append-only history, Steps 0-28 (E1 post-mortem in "Steps 20-26")
├── COMMANDS.md                 recurring commands (submit/resume/tensorboard/eval)
├── train_holo.sh               SLURM job -> basicsr/train.py -opt .../Holo_Baseline_Restormer.yml
│
├── basicsr/                    stock Restormer/BasicSR + 4 project-owned files
│   ├── train.py                TRAINING ENTRY POINT (auto-resume + progressive crop loop)
│   ├── test.py                 basicsr inference entry (unused for holo)
│   ├── models/
│   │   ├── image_restoration_model.py    ImageCleanModel: train step, val loop, mixup
│   │   ├── base_model.py                 save/load/resume, schedulers
│   │   ├── losses/losses.py              L1Loss, MSELoss, PSNRLoss, CharbonnierLoss
│   │   └── archs/
│   │       ├── restormer_arch.py         *** the Restormer model ***
│   │       └── dinov2_feature_extractor.py   [PROJECT] frozen DINOv2 loader + dino_preprocess
│   ├── data/
│   │   ├── paired_image_uint16_dataset.py    [PROJECT] Dataset_PairedImage_uint16 (LQ/GT)
│   │   ├── radar_render_triplet_dataset.py   [PROJECT] ..._uint16_Render (LQ/GT/render)
│   │   ├── transforms.py                 paired_random_crop, augment, random_augmentation
│   │   └── data_util.py                  paired_paths_from_folder (pairing by GT basename)
│   ├── utils/img_util.py       [PROJECT] imfrombytes_uint16, padding, tensor2img
│   ├── utils/options.py        YAML->dict config parser, path derivation
│   └── metrics/psnr_ssim.py    calculate_psnr / calculate_ssim (used in val)
│
├── Deraining_Holo/             the project's task directory
│   ├── Options/
│   │   ├── Holo_Baseline_Restormer.yml       *** the baseline training config ***
│   │   ├── Holo_Baseline_Restormer_test.yml  inference-only config
│   │   └── DINO_analysis_data.yml            DATA-ONLY spec read by dino_analysis_phases/
│   ├── train_holo_chain.sh / train_holo_chain_verynoisy.sh   self-chaining SLURM drivers
│   ├── eval_test.sh / eval_test_verynoisy.sh                 test-set eval jobs
│   ├── test_holo.py            uint16 inference + PSNR/SSIM + sharpness + spectrum + viz
│   ├── masked_metrics.py       foreground-masked PSNR/SSIM + per-image CSV + mask viz
│   ├── analyze_sharpness.py    Laplacian/Sobel/HF-energy/radial power spectrum
│   ├── plot_curves.py plot_overfit.py plot_compare.py plot_val_table.py plot_loss_vs_loss.py
│   ├── create_split.py create_val_test_split.py create_symlinks.py verify_splits.py
│   ├── experiment_results/     COMMITTED results (git-ignored dirs excepted)
│   │   ├── results.md, README.md
│   │   ├── exp1_noisy/{report.md,figures/,metrics/}
│   │   ├── exp2_verynoisy/{report.md,figures/,metrics/}     <- the current baseline
│   │   ├── dataset_splits/holographic/{train,val,test}.txt
│   │   └── dino_pooled_means_reference/dino_feat_mean_{lq,render}DINO.pt  (E1 relics, DO NOT DELETE)
│   ├── results/Holo_verynoisy_test_292k/{raw/,viz/,*.csv,*.png}   338 test predictions
│   ├── results/Holo_test_224k/{raw/,viz/,for_professor/,...}      Exp 1 test predictions
│   └── debug/                  4 leftover E1 crop-alignment PNGs
│
├── dino_analysis_phases/       *** all previous DINO work (do not touch) ***
│   ├── DINO_ANALYSIS_DEVLOG.md         Phase 0 (single sample 0196)
│   ├── visualize_dino_spatial_pca.py   Phase 0 + the shared library everything imports
│   ├── dino_spatial_layer_means.pt     *** the spatial centering means ***
│   ├── outputs/                        Phase 0 figures + CSV + metadata
│   ├── phase1/analyze_dino_spatial_consistency.py, PHASE1_DEVLOG.md, outputs/
│   └── phase2/analyze_dino_prior_source.py,        PHASE2_DEVLOG.md, outputs/
│
├── experiments/                (gitignored) checkpoints + training logs
│   ├── Holo_Baseline_Restormer/models/net_g_{128000,224000,latest}.pth     Exp 1
│   └── Holo_Baseline_Restormer_verynoisy/models/net_g_{128000,292000}.pth  Exp 2 <- current baseline
└── tb_logger/                  (gitignored) TensorBoard events for both baselines
```

---

## B. Existing Restormer baseline

The baseline to treat as "the" baseline is **Exp 2 (verynoisy)**: 1e5 -> 1e7 on
`holographic_image_dataset`. It is already trained. Do not retrain it.

| Item | Finding |
|---|---|
| Training entry point | `basicsr/train.py` (`main()`, progressive loop at lines 241-270) |
| Baseline config | `Deraining_Holo/Options/Holo_Baseline_Restormer.yml` — note `name: Holo_Baseline_Restormer_verynoisy` |
| Command actually used | `sbatch Deraining_Holo/train_holo_chain_verynoisy.sh` -> `python basicsr/train.py -opt Deraining_Holo/Options/Holo_Baseline_Restormer.yml --launcher none` (jobs 1752760 -> 1752761 -> 1753869) |
| Existing checkpoint | `experiments/Holo_Baseline_Restormer_verynoisy/models/net_g_292000.pth` (best val, the reported one) and `net_g_128000.pth`. **No training states left** — this run cannot be resumed |
| Exp 1 checkpoints (older dataset) | `experiments/Holo_Baseline_Restormer/models/net_g_{128000,224000,latest}.pth` |
| Existing validation results | In-log: `experiments/Holo_Baseline_Restormer_verynoisy/train_*.log` (3 logs) + `tb_logger/Holo_Baseline_Restormer_verynoisy/`. Best **22.446 dB / 0.8156 @ 292k**, final 22.437 / 0.8158 @ 300k |
| Existing test results | `Deraining_Holo/experiment_results/exp2_verynoisy/{report.md,metrics/masked_metrics_per_image.csv}` and `results/Holo_verynoisy_test_292k/masked_metrics_per_image.csv`. Full **22.405 +/- 3.052 dB / 0.7999**, masked **18.313 +/- 2.917 dB / 0.5438**, input 12.354 dB -> **+10.05 dB**. HF-energy ratio pred/GT **0.216** |
| Existing prediction images | `Deraining_Holo/results/Holo_verynoisy_test_292k/raw/` (338 uint16 PNGs) and `.../viz/` (338 3-panel heatmap PNGs). Exp 1: `results/Holo_test_224k/{raw,viz,for_professor}` |
| Crop size | Progressive **[128, 160, 192, 256]** over iters [92k, 64k, 48k, 96k]; `gt_size: 256` |
| Input range | float32 **[0, 1]** (uint16 PNG / 65535). No mean/std normalization (`mean`/`std` absent from the yml) |
| Loss | `L1Loss`, weight 1, reduction mean (`basicsr/models/losses/losses.py`, `self.cri_pix`) |
| Optimizer | AdamW, lr 3e-4, weight_decay 1e-4, betas [0.9, 0.999]; grad-clip norm 0.01; `CosineAnnealingRestartCyclicLR` periods [92k, 208k], eta_mins [3e-4, 1e-6] |
| Training iterations | 300,000 (mini-batch 8/5/4/2 per stage, effective batch = mini-batch, 1 GPU) |
| mixup | **off** for this run |
| Hardware / wall time | 1x V100-PCIE-32GB, 97% util, 29.4 GB peak, 16 h 47 m, 394 epochs, ~0.81 s/iter |

### Architecture facts (read from `basicsr/models/archs/restormer_arch.py`)

| Property | Value |
|---|---|
| Model file / class | `basicsr/models/archs/restormer_arch.py` -> `class Restormer` |
| inp_channels / out_channels | 1 / 1 |
| dim | 48 |
| Encoder levels | 3 (`encoder_level1/2/3`) + 1 latent level (level 4) |
| Decoder levels | 3 (`decoder_level3/2/1`) |
| num_blocks per level | [4, 6, 6, 8] -> enc1=4, enc2=6, enc3=6, latent=8; decoders mirror (dec3=6, dec2=6, dec1=4) |
| heads | [1, 2, 4, 8] (dec1 and refinement use heads[0]=1) |
| num_refinement_blocks | 4 |
| ffn_expansion_factor / bias / LayerNorm | 2.66 / False / `WithBias` |
| dual_pixel_task | False |
| Parameter count | **26,124,052** (verified by running the model) |
| **Residual behaviour** | **Global residual**: `out = self.output(x) + inp_img` (restormer_arch.py:281). The network predicts a **residual**; the clean image only exists after adding the input |
| Skip connections | `torch.cat` of decoder-upsampled features with the matching encoder output at levels 3, 2, 1 (lines 260, 265, 270). Levels 3 and 2 then get a 1x1 `reduce_chan_level{3,2}`; **level 1 has no channel reduction** (stays at 2*dim=96 through decoder1, refinement, and the output conv) |
| Down/Up sampling | `Downsample` = 3x3 conv (C -> C/2) + `PixelUnshuffle(2)`; `Upsample` = 3x3 conv (C -> 2C) + `PixelShuffle(2)` |

---

## C. Restormer tensor shapes

Measured by registering forward hooks on every submodule and running one batch (B=2)
through the real model on CPU under `torch.no_grad`. Not inferred from documentation.

**Training crop 128x128 (stage 1, 92k of 300k iters):**

| Stage | Tensor variable / module | Shape [B, C, H, W] | C | H | W |
|---|---|---|---|---|---|
| input | `inp_img` | [B, 1, 128, 128] | 1 | 128 | 128 |
| shallow / patch embed | `inp_enc_level1 = patch_embed(inp_img)` | [B, 48, 128, 128] | 48 | 128 | 128 |
| encoder 1 | `out_enc_level1 = encoder_level1(...)` | [B, 48, 128, 128] | 48 | 128 | 128 |
| after down1_2 | `inp_enc_level2` | [B, 96, 64, 64] | 96 | 64 | 64 |
| encoder 2 | `out_enc_level2` | [B, 96, 64, 64] | 96 | 64 | 64 |
| after down2_3 | `inp_enc_level3` | [B, 192, 32, 32] | 192 | 32 | 32 |
| encoder 3 | `out_enc_level3` | [B, 192, 32, 32] | 192 | 32 | 32 |
| after down3_4 | `inp_enc_level4`  **<- LATENT INPUT / injection point** | [B, 384, 16, 16] | 384 | 16 | 16 |
| latent output | `latent = self.latent(inp_enc_level4)` | [B, 384, 16, 16] | 384 | 16 | 16 |
| after up4_3 (+cat, +reduce) | `inp_dec_level3` (cat is [B,384,32,32] -> reduce -> 192) | [B, 192, 32, 32] | 192 | 32 | 32 |
| decoder 3 | `out_dec_level3` | [B, 192, 32, 32] | 192 | 32 | 32 |
| decoder 2 | `out_dec_level2` (cat [B,192,64,64] -> reduce -> 96) | [B, 96, 64, 64] | 96 | 64 | 64 |
| decoder 1 | `out_dec_level1` (cat [B,96,128,128], **no reduce**) | [B, 96, 128, 128] | 96 | 128 | 128 |
| refinement | `refinement(out_dec_level1)` | [B, 96, 128, 128] | 96 | 128 | 128 |
| output | `self.output(...) + inp_img` | [B, 1, 128, 128] | 1 | 128 | 128 |

**All four progressive crop sizes (everything scales by the same /2 per level):**

| Input | enc1 HxW | enc2 | enc3 | **latent** | decoder1 | output |
|---|---|---|---|---|---|---|
| 128 (92k iters) | 128^2 | 64^2 | 32^2 | **[B, 384, 16, 16]** | 128^2 | [B,1,128,128] |
| 160 (64k iters) | 160^2 | 80^2 | 40^2 | **[B, 384, 20, 20]** | 160^2 | [B,1,160,160] |
| 192 (48k iters) | 192^2 | 96^2 | 48^2 | **[B, 384, 24, 24]** | 192^2 | [B,1,192,192] |
| 256 (96k iters + ALL validation/test) | 256^2 | 128^2 | 64^2 | **[B, 384, 32, 32]** | 256^2 | [B,1,256,256] |

Channel widths are fixed regardless of crop size: 48 / 96 / 192 / 384 / 192 / 96 / 96 / 1.

> **Likely future LATENT injection point:** `inp_enc_level4` — the output of `down3_4`,
> immediately before `self.latent`. Channels **384 = dim * 2^3**, spatial **H/8 x W/8**,
> i.e. **16x16 at the 128 crop and 32x32 at 256**.
>
> The DINO B6 patch grid is a fixed **16x16** — it coincides with the latent grid *only*
> at the 128 crop. At 160/192/256 the latent is 20/24/32 and a resample is required.

---

## C-2. The exact latent / bottleneck injection point

| Question | Answer |
|---|---|
| File | `basicsr/models/archs/restormer_arch.py` |
| Class | `Restormer` |
| Function | `forward(self, inp_img)` |
| Module construction | line 219-220: `self.down3_4 = Downsample(int(dim*2**2))`, `self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ...) for i in range(num_blocks[3])])` |
| Forward code | lines 256-257: `inp_enc_level4 = self.down3_4(out_enc_level3)` then `latent = self.latent(inp_enc_level4)` |
| Tensor variable | `inp_enc_level4` (input) / `latent` (output) |
| Tensor shape | [B, 384, H/8, W/8] — 16x16 @ crop 128, 32x32 @ crop 256 |

The four points you asked to separate:

| Point | Tensor | Is it distinct? |
|---|---|---|
| **A.** feature immediately AFTER the final encoder downsampling | `inp_enc_level4 = self.down3_4(out_enc_level3)` | — |
| **B.** feature immediately BEFORE latent Transformer processing | `inp_enc_level4` | **Same tensor as A.** There is nothing between `down3_4` and `self.latent` |
| **C.** feature immediately AFTER latent Transformer processing | `latent = self.latent(inp_enc_level4)` | Distinct from A/B |
| **D.** feature immediately BEFORE decoder upsampling begins | `latent` (consumed by `self.up4_3(latent)`) | **Same tensor as C.** There is nothing between `self.latent` and `up4_3` |

So this implementation has only **two** distinct bottleneck tensors, not four:
**A == B** (`inp_enc_level4`) and **C == D** (`latent`).

**Which point allows a spatial prior to be introduced BEFORE the latent Transformer
blocks, so those blocks process radar + prior jointly:** point **A/B**, i.e.
`inp_enc_level4`, between line 256 and line 257. Anything injected at C/D is only
seen by the decoder, never by the 8 latent Transformer blocks.

Not implemented. No model file was modified.

---

## D. Dataset and pairing

**Root:** `/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset/`

| Domain | Directory (master) | Split dirs | Format |
|---|---|---|---|
| **1e5 — very noisy radar (LQ input)** | `verynoisy/` (6778) | `train_verynoisy` 6101, `val_verynoisy` 339, `test_verynoisy` 338 | uint16 grayscale PNG, 256x256, 1 channel |
| **1e7 — clean radar (GT target)** | `clean/` (6778) | `train_clean` / `val_clean` / `test_clean` | uint16 grayscale PNG, 256x256, 1 channel |
| 1e6 — "noisy" (Exp 1 input, count UNCONFIRMED) | `noisy/` (6778) | `train_/val_/test_noisy` | uint16 grayscale PNG |
| **render** | `renders_blackbg/` (6778) | `train_/val_/test_renders_blackbg` (6101/339/338) | **uint8 RGB PNG**, 256x256x3, all three channels identical |
| render (white bg, UNUSED) | `renders/` (6778) | — | uint8 RGB, mean ~247 (white background) |

- **Naming convention / sample IDs:** 4-digit zero-padded stem, `NNNN.png`, e.g. `0196.png`,
  `3752.png`, `6115.png`. **The same filename identifies the same scene in all four
  domains** — that is the entire pairing mechanism.
- **How 1e5 and 1e7 are paired:** `paired_paths_from_folder([lq_folder, gt_folder], ...)`
  (`basicsr/data/data_util.py:208`) enumerates the GT folder and constructs the LQ path from
  the **GT basename**, asserting the name exists in the LQ folder. Pairing is by filename,
  not by list order — safe. It does **not** sort, so index order is filesystem order.
- **How the render corresponds:** `Dataset_PairedImage_uint16_Render._render_path()` =
  `os.path.join(render_folder, basename(gt_path))`
  (`basicsr/data/radar_render_triplet_dataset.py:46-48`). Phase 0/1/2 additionally assert
  all three basenames are identical per sample.
- **All three accessible from one class?** Yes — `Dataset_PairedImage_uint16_Render` returns
  `{'lq','gt','dino','lq_path','gt_path'}`, where `dino` is the render. It is *not*
  registered in any training config.
- **Is the render loaded during Restormer training?** **No.** The baseline uses
  `Dataset_PairedImage_uint16` (LQ/GT only). The triplet dataset is used exclusively by
  `dino_analysis_phases/`.
- **Numeric ranges when loaded:** radar -> `imfrombytes_uint16` = `cv2.IMREAD_UNCHANGED` ->
  `(H,W,1)` -> `float32 / 65535` -> **[0, 1]**. Render -> `imfrombytes(flag='color',
  float32=True)` -> `(H,W,3)` `float32 / 255` -> **[0, 1]** (verified live: render tensor
  max 0.9765). No clipping, no percentile stretch, no colormap.
- **Raw dtypes on disk (verified on `0196.png`):** clean uint16 min 0 max 65535 mean 17300;
  verynoisy uint16 mean 25612; noisy uint16 mean 17532; renders_blackbg uint8 min 0 max 249
  mean 74.4; renders uint8 min 130 max 255 mean 247.0.
- **Example matched IDs:** `0196.png`, `0060.png`, `0072.png`, `0073.png`, `6115.png`,
  `3017.png`, `4467.png` — each exists in `verynoisy/`, `clean/`, `noisy/`,
  `renders_blackbg/`.
- Split dirs are **symlinks** into the master dirs (verified:
  `val_clean/0060.png -> clean/0060.png`, `train_verynoisy/... -> verynoisy/...`).

**Ray-count labelling note (DEVLOG Step 27):** clean = 1e7, verynoisy = 1e5. The docs had
this backwards until 2026-08-09. Verified empirically (PSNR vs clean over 30 val images:
noisy 29.55 dB, verynoisy 13.11 dB). Only labels were wrong; every experiment trained
degraded -> clean and is unaffected. `noisy`'s exact count remains UNCONFIRMED (1e6 assumed
from the ladder).

---

## D-2. Train / validation / test splits

| Item | Finding |
|---|---|
| Training samples | **6101** (`splits/train.txt`, and `train_*` dirs each hold 6101) |
| Validation samples | **339** (`splits/val.txt`, `val_*` dirs = 339) |
| Test samples | **338** (`splits/test.txt`, `test_*` dirs = 338) |
| Where defined | `holographic_image_dataset/splits/{train,val,test}.txt`, **materialized as symlink dirs**; the dataset classes read the *directories*, never the txt files |
| Committed copy | `Deraining_Holo/experiment_results/dataset_splits/holographic/{train,val,test}.txt` — **byte-identical** to the live splits (verified by `diff`) |
| Fixed or dynamic | **Fixed files on disk.** Generated once by `create_split.py` (seed 42) and `create_val_test_split.py --seed 42`, then frozen |
| Seed effect on the split | None at runtime. The seeds only affected generation. `manual_seed: 100` in the yml affects sampling/augmentation, **not** the split |
| Sample IDs saved | Yes — the three txt files, plus the committed copies above |

**The 339 validation triplets — confirmed, not assumed.** `phase1_metadata.json` records
`split: val`, `total_in_split: 339`, `valid_triplets_analyzed: 339`, `skipped_count: 0`,
with dataroots `val_verynoisy` / `val_clean` / `val_renders_blackbg`. Verified live that
`val_clean/` contains exactly the 339 entries of `splits/val.txt` (sorted diff, empty), and
that all four `val_*` variants start with the same IDs `0060, 0072, 0073`. So
**339 = the entire holographic val split, every one of which had a complete
1e5/1e7/render triplet.**

**CAVEAT (DEVLOG Step 19a):** the 339/338 val/test split was carved **after** Exp 2 finished
training, so the 338 test images influenced *checkpoint selection* (never the weights —
those images are not in `train.txt`). Top-5 checkpoints span 0.016 dB, so the effect is
negligible, but Exp 2's test number is not a fully clean held-out estimate and must be
described that way. Exp 1's split predated its training and is clean.

**Side effect:** the training config's val dataroots now resolve to 339 images instead of
677. Any FUTURE run validates on the smaller set — correct behaviour, but its val PSNR is
not directly comparable to Exp 2's logged val curve.

---

## E. Restormer preprocessing (exact pipeline)

**Training** — `basicsr/data/paired_image_uint16_dataset.py:82-133`, in this exact order:

```
GT (clean/1e7) PNG on disk, uint16 256x256
LQ (verynoisy/1e5) PNG on disk, uint16 256x256
  | FileClient.get()                      -> raw bytes
  | cv2.imdecode(IMREAD_UNCHANGED)        -> (256,256) uint16
  | img[:, :, np.newaxis]                 -> (256,256,1)
  | .astype(np.float32) / 65535.          -> [0,1] float32       <- the ONLY scaling
  | padding(img_lq, img_gt, gt_size=256)  -> NO-OP (256 >= 256; would BORDER_REFLECT otherwise)
  | paired_random_crop(gt, lq, 256, scale=1, gt_path)   -> 256x256 (no-op crop at gt_size 256)
  | random_augmentation(img_gt, img_lq)   -> one of 8 dihedral ops (see below)
  | img2tensor(bgr2rgb=False, float32=True)  -> CHW float32 [1,256,256]
  | (no normalize: opt has no 'mean'/'std')
  | collate                               -> batch [8,1,256,256]
  ===== IN basicsr/train.py, AFTER the dataloader =====
  | mini-batch subsample: lq = lq[indices]; gt = gt[indices]   (random.sample, k = 8/5/4/2)
  | progressive sub-crop: x0,y0 random; lq = lq[:,:,x0:x1,y0:y1]; gt likewise
  |                                       -> [mb, 1, 128|160|192|256, ...]
  | model.feed_train_data({'lq','gt'})    -> .to(device)
  | (mixup: DISABLED for this run)
  -> Restormer, values in [0,1]
```

**No resize anywhere.** Images are native 256x256 and only ever cropped, never rescaled.

**Geometric augmentation code** — `basicsr/data/transforms.py:270-275`:

```python
def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)        # ONE flag for ALL arguments
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
```

`data_augmentation(image, mode)` covers the full dihedral group:
0 = identity, 1 = flipud, 2 = rot90, 3 = rot90+flipud, 4 = rot180, 5 = rot180+flip,
6 = rot270, 7 = rot270+flip. So there is **no independent horizontal-flip probability** —
one integer chooses one of 8 rigid transforms. (`augment()` in the same file, with its
separate hflip/vflip/rot90 coins, is **not** used by this dataset.)

**Are paired images guaranteed identical crops and flips? YES, at the dataset level:**

- `paired_random_crop` draws `top`/`left` **once** and applies the same offsets to every
  image in both lists (`transforms.py:63-78`); with `scale=1` the GT offsets equal the LQ
  offsets exactly.
- `random_augmentation` draws `flag_aug` **once** and applies it to every `*args` entry.
- The triplet dataset explicitly passes the render through both:
  `paired_random_crop(img_gt, [img_lq, img_render], ...)` and
  `random_augmentation(img_gt, img_lq, img_render)`
  (`radar_render_triplet_dataset.py:74-81`).

**BUT there is a second crop that is NOT dataset-level.** The progressive sub-crop and the
mini-batch subsample happen in `basicsr/train.py:255-269` and touch **only
`train_data['lq']` and `train_data['gt']`**:

```python
lq = train_data['lq']
gt = train_data['gt']
if mini_batch_size < batch_size:
    indices = random.sample(range(0, batch_size), k=mini_batch_size)
    lq = lq[indices]; gt = gt[indices]
if mini_gt_size < gt_size:
    x0 = int((gt_size - mini_gt_size) * random.random())
    y0 = int((gt_size - mini_gt_size) * random.random())
    lq = lq[:,:,x0:x0+mini_gt_size, y0:y0+mini_gt_size]
    gt = gt[:,:,x0*scale:..., y0*scale:...]
```

A third tensor (e.g. `'dino'`) coming out of the dataset would be handed back untouched at
256x256 and at full batch size — **silently misaligned** with the crop the network actually
sees, for 204k of 300k iterations. This is the single most important pipeline fact for a
future spatial-DINO experiment.

**Validation** (`basicsr/models/image_restoration_model.py:213-297`): `phase='val'` ->
**no padding, no crop, no augmentation** — full 256x256, batch size 1. `window_size: 8`
triggers `pad_test`, but 256 % 8 == 0 so no padding occurs. Metrics are computed with
`use_image: true`, i.e. on `tensor2img` output -> **clamped to [0,1] and quantized to uint8
0-255**, then `calculate_psnr`/`calculate_ssim` with `crop_border=0`,
`test_y_channel=false`. So logged val PSNR is an **8-bit** number, not 16-bit.

**Test** (`Deraining_Holo/test_holo.py`): reads uint16 directly (no dataset class),
`/65535` -> `[0,1]`, reflect-pads to a multiple of 8 (no-op at 256), forward,
`clamp(0,1)`, `x65535` -> uint16 PNG. PSNR is computed **on the uint16 arrays** with peak
65535; SSIM on floats via skimage. So test PSNR (16-bit) and val PSNR (8-bit) are computed
on different quantizations and are not the same scale.

**Value range entering Restormer: [0, 1] float32, single channel, no normalization.**

---

## F. Previous DINO preprocessing

| Item | Actual Phase 1/2 behavior |
|---|---|
| Model | **DINOv2** (not DINO v1), `dinov2_vitb14`, ViT-B/14, 12 blocks, embed dim 768, patch 14, `num_register_tokens = 0` (verified at runtime) |
| Loading mechanism | `torch.hub.load(hub_dir, 'dinov2_vitb14', source='local', pretrained=False)` then `load_state_dict(sd, strict=True)` — `dinov2_feature_extractor.py:100-131`. Fails loudly if `hubconf.py` or the `.pth` is missing. Verified live: missing = [], unexpected = [] |
| Local cache | repo `/home/woody/.../code/torch_hub/hub/facebookresearch_dinov2_main`, weights `/home/woody/.../torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth` (346 MB) |
| Offline behaviour | `hub_source: local` in the yml. An online GitHub fallback exists but is not used |
| Frozen / eval | `p.requires_grad_(False)` for all params; `self.dino.eval()`; `train()` is overridden so the parent going to train mode cannot un-eval DINO. All extraction under `@torch.no_grad()` |
| Strict weight checks | `strict=True`, and the `load_result` (missing/unexpected key lists) is stored and printed by every phase script — a silently-random ViT cannot slip through |
| Input size | **224 x 224** (`dino_img_size: 224` in `DINO_analysis_data.yml`) |
| Image range before DINO | **[0, 1]** — radar uint16/65535, render uint8/255. No clipping, no min-max stretch, no dataset normalization, no percentile normalization |
| Grayscale handling | 1-channel radar is **replicated** to 3 channels: `img.repeat(1,3,1,1)`. No channel mapping, no colormap (`colormap_before_dino: false` recorded in metadata) |
| RGB conversion (render) | None needed — loaded as 3-channel by `imfrombytes(flag='color')`; all three channels are identical anyway (verified) |
| Resize | `F.interpolate(img, size=(224,224), mode='bilinear', align_corners=False)` — **from the full 256x256 image**, not from a crop |
| Interpolation | bilinear, `align_corners=False` |
| Crop behavior | **None.** `build_dataset(cfg, split)` forces `opt['phase'] = 'val'`, so the dataset skips padding, cropping and augmentation entirely |
| ImageNet / DINO normalization | `(img - mean)/std` with mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), as registered buffers. Verified live: the 224^2 tensor spans ~[-2.118, +2.638] |
| torchvision transforms / AutoImageProcessor | **Neither.** No `torchvision.transforms`, no HuggingFace processor. One hand-written function: `dino_preprocess(img, img_size, mean, std)` |
| Transform code path | `basicsr/models/archs/dinov2_feature_extractor.py:50-60` (`dino_preprocess`), called via `batch_to_dino_inputs()` in `dino_analysis_phases/visualize_dino_spatial_pca.py:174-188` |
| Tensor layout | `[B, 3, 224, 224]` float32, NCHW |
| B6 extraction | `ext.dino.get_intermediate_layers(x, n=(2,5,8,11), reshape=False, return_class_token=False, norm=True)` — `visualize_dino_spatial_pca.py:204-212` |
| CLS handling | DINOv2 strips it internally: `outputs = [out[:, 1 + num_register_tokens:] for out in outputs]` (vision_transformer.py:321). CLS is never seen by the analysis |
| Register tokens | **0** for this checkpoint. Phase 0 and Phase 1 both **abort** if it is not 0 |
| Patch grid | **16 x 16 = 256 tokens** (224 / 14). Asserted at runtime by both Phase 0 and Phase 1 |
| Embedding dimension | **768** |

### Exact preprocessing sequence

Identical for 1e5, 1e7 and render except for the loader:

```
1e5 / 1e7:  uint16 PNG -> cv2.IMREAD_UNCHANGED -> (256,256,1) -> float32/65535 -> [0,1]
render:     uint8  PNG -> cv2 color            -> (256,256,3) -> float32/255   -> [0,1]
   | img2tensor(bgr2rgb=False) -> [C,256,256] -> unsqueeze -> [1,C,256,256]
   | if C==1: repeat(1,3,1,1)                 -> [1,3,256,256]
   | F.interpolate(size=(224,224), bilinear, align_corners=False) -> [1,3,224,224]
   | (x - ImageNet_mean) / ImageNet_std       -> [1,3,224,224], ~[-2.12, +2.64]
   -> DINOv2 ViT-B/14
   | get_intermediate_layers(n=(2,5,8,11), norm=True, return_class_token=False)
   -> per block: [1, 256, 768]     (final LayerNorm applied; CLS already dropped)
```

### Block numbering convention — verified

`get_intermediate_layers` collects `x` *after* `blk(x)` for `i in n`
(`vision_transformer.py:280-283`), so 0-indexed `5` = the output of the **6th** block =
"B6". Phase 0/1/2 all compute `blocks1 = [round(12*k/4) for k in 1..4] = [3,6,9,12]` and
`blocks0 = [2,5,8,11]`, and record both in metadata. **"B6" means the same block
everywhere in the analysis.** Two footnotes:

1. The **E1 training** recipe used a different set — 1-indexed [1,4,8,12] / 0-indexed
   [0,3,7,11] — which is why the surviving pooled means are labelled
   `dino_layers: [0,3,7,11]`. **The analysis and E1 do not share a block set.** Both files
   say so explicitly.
2. `get_intermediate_layers` returns outputs in **ascending block order regardless of the
   order `n` is passed in**. Phase 2 originally passed `(5,2)` and indexed positionally,
   which swapped B6 and B3 for one run. Found and fixed (commit `4240521`); the committed
   Phase 2 outputs are from the corrected 11:53 run.
   **The first Phase 2 devlog entry (11:47 CEST) holds the buggy numbers — do not read it.**

### Runtime-verified shapes (probe, §19)

```
B6 token tensor            : [1, 256, 768]
reshape to grid            : [1, 16, 16, 768]
channel-first (theoretical): [1, 768, 16, 16]
```
The manual reshape+permute is **bit-identical** to DINOv2's own `reshape=True` output
(`torch.allclose` -> True), confirming row-major token -> grid ordering.
A 128x128 crop through the same `dino_preprocess` also yields `[1,3,224,224]` ->
`[1,256,768]` — same grid, different magnification (see §H).

---

## G. Feature centering

**Keep these two apart. They are different operations at different stages:**

- **(A) IMAGE PREPROCESSING BEFORE DINO** = §F. `[0,1]` -> 3-channel replicate ->
  resize 224 -> **ImageNet mean/std**. That ImageNet normalization is an *input* transform
  applied to pixels. Radar images were **NOT** "centered" in any dataset-statistics sense —
  no radar mean image, no radar mean/std, no per-image standardization.
- **(B) FEATURE CENTERING AFTER DINO** = subtracting a fixed 768-d vector from the *patch
  tokens*, below. It happens on features, after the ViT, and is completely independent
  of (A).

```
IMAGE
  |
DINO input preprocessing        (A) -- ImageNet mean/std, pixels
  |
DINO
  |
B6 feature  [1, 256, 768]
  |
feature centering               (B) -- subtract mu_{domain, block}, features
```

| Item | Actual behavior |
|---|---|
| Centering used? | **Yes, but as one of two reported variants.** Every Phase 0/1/2 table exists in both `raw` and `centered` form. The extractor itself is built with `feat_mean=None` (`build_extractor`, `visualize_dino_spatial_pca.py:194-201`) — DINO returns raw tokens and the analysis subtracts its own mean afterwards |
| Which experiments used raw | Both variants are always computed and written to CSV. Phase 1 reports raw + centered for all 4 blocks; Phase 2 reports raw + centered for B6 and B3 |
| Which used centered | The **primary/headline** numbers and the block selection are the **centered** ones (`best_worst.feature_type: "centered"`; block choice read from the centered summary) |
| Formula | `f_centered[p] = f[p] - mu_{domain, block}` for every patch position p — `analyze_dino_spatial_consistency.py:144-150`. Applied **before** PCA and before cosine |
| Per DINO block? | **Yes** — a separate mean per block |
| Per domain? | **Yes** (default `--centering-mode per-domain`) |
| Mean tensor shape | Per domain: **[4, 768]** = (4 selected blocks) x 768. One **768-d vector per (domain, block)**. A `global` [4, 768] variant (pooled over the three domains) is also cached |
| Averaged dimensions | Over **training images** AND over **all 256 patch positions**: `sums[d][li] += f[0].sum(0)`, `counts[d] += n_tokens`, then `sums/counts` (`visualize_dino_spatial_pca.py:244-275`) |
| Spatial positions averaged? | **Yes** — the mean is position-agnostic. **No per-position mean exists.** Every spatial location gets the *same* 768-d vector subtracted |
| Source split | **train** only. Phase 1 hard-aborts if `meta['split'] != 'train'` (`refusing to proceed (validation leakage)`) |
| Number of samples | **150** train images, `random.Random(seed=0).sample(range(6101), 150)`, sorted. Example files recorded: `0012, 0135, 0156, 0313, 0364` |
| Separate 1e5 mean? | **Yes** — `per_domain['1e5']`, per-block norms [50.10, 42.72, 38.69, 29.54] |
| Separate 1e7 mean? | **Yes** — `per_domain['1e7']`, norms [59.23, 47.02, 44.43, 28.25] |
| Separate render mean? | **Yes** — `per_domain['render']`, norms [72.55, 58.18, 47.54, 26.37] |
| Full-image or crop based? | **Full 256x256 images, no crop, no augmentation** — `build_dataset` forces `phase='val'` even for the train split, explicitly "the mean must be over full images too, not over random crops". Recorded in meta as `phase: 'val (full 256x256, no crop)'` |
| Validation/test contribution | **None.** Train split only, asserted |
| dtype | Accumulated in **float64**, stored as **float32** |
| Saved file format | `torch.save({'per_domain': {d: [4,768]}, 'global': [4,768], 'meta': {...}})` |
| Saved path | **`dino_analysis_phases/dino_spatial_layer_means.pt`** (created 2026-08-09T14:09:34+02:00) |
| L2 normalization BEFORE centering? | **No** |
| L2 normalization AFTER centering? | **No** |
| LayerNorm? | **Yes, but DINOv2's own.** `get_intermediate_layers(..., norm=True)` applies the ViT's final `self.norm` LayerNorm to *every* extracted intermediate layer before returning. No additional LayerNorm is applied by the project |
| Cosine normalization only at metric time? | **Yes.** `patchwise_cosine` = `F.cosine_similarity(a, b, dim=1).mean()` — the L2 normalization lives inside the metric only; stored/centered features are never renormalized |

### The other means on disk — the E1 pooled vectors

`Deraining_Holo/experiment_results/dino_pooled_means_reference/dino_feat_mean_{lq,render}DINO.pt`

- **3072-d = 4 layers x 768**, **mean-POOLED over patch tokens** (one vector per image)
- computed from **300 random training crops** with the progressive size distribution
  `sizes [128,160,192,256]`, `counts [92,64,48,96]`, `geometric_augs: True`
- layers `[0,3,7,11]` (the E1 set, NOT the analysis set)
- lqDINO: `mean_norm 85.63`, `resid_norm 35.40`, `offset_energy_frac 0.849`
- renderDINO: `mean_norm 102.23`, `resid_norm 32.63`, `offset_energy_frac 0.900`
- `report_pooled_mean_incompatibility()` reads them purely to *record* that they cannot
  centre spatial patch tokens; metadata carries
  `existing_pooled_means_compatible: false`
- **Not used for anything numerical.** CONTEXT.md and DEVLOG Step 28 both say: do not
  delete them — a rerun would then record `[]` where the committed runs record two entries

---

## H. Potential preprocessing / centering mismatch

**Answer: the existing spatial means are NOT directly compatible with Restormer training
crops.**

| | (A) How the existing spatial means were produced | (B) How Restormer training crops are produced |
|---|---|---|
| Source image | full 256x256 | full 256x256 |
| Crop | **none** | `paired_random_crop` at gt_size 256 (no-op) **then** `train.py` progressive sub-crop to **128 / 160 / 192 / 256** |
| Augmentation | **none** (`phase='val'`) | one of 8 dihedral transforms, per sample |
| Batch composition | 150 fixed train images, seed 0 | random, 8->5->4->2 per stage, 300k iters |
| What DINO would see | 256^2 -> bilinear -> 224^2 | (hypothetically) 128^2 -> bilinear -> 224^2 for 92k of 300k iters |
| Effective magnification | 1.0x of the scene | **1.75x / 1.40x / 1.17x / 0.875x** relative to (A) |
| Field of view | always the whole scene | a 128^2 window that may contain the whole object, part of it, or **only background** |

**What the mismatch actually is:**

1. **Scale / field-of-view shift.** A 128x128 crop upsampled to 224 shows the scene at
   ~1.75x the magnification the mean was measured at. DINOv2 patch statistics are strongly
   scale-dependent, so the 768-d offset for a 128-crop token distribution is not the offset
   measured on full frames. The mismatch is worst at exactly the stage that gets the most
   iterations (92k of 300k at crop 128).
2. **Content-distribution shift.** Objects occupy ~39.5% of a frame on average (from the
   masked-metric coverage), and a random 128^2 window can be pure background. The
   full-image mean averages a fixed object/background mixture; the crop distribution does
   not.
3. **Augmentation is NOT a source of mismatch.** Because the mean averages over all patch
   positions, it is invariant to flips and 90-degree rotations — those permute positions
   without changing the position-averaged mean. So (1) and (2) are the real issues, not the
   dihedral group.
4. **The per-domain aspect is fine.** A `1e5` mean exists and is the correct one to use for
   a DINO(1e5) arm; no new domain statistic would be needed.
5. **The E1 pooled means are the wrong shape AND were built from the crop distribution**
   (300 crops at the progressive sizes). So the one statistic that matches the crop
   distribution is 3072-d and pooled, and the one that is spatially usable is 768-d and
   full-frame. **Neither is both.**

**Why it matters for a future spatial-DINO experiment:** the centered feature `f - mu` is
where Phase 1/2 measured the signal. If `mu` is estimated on a different input distribution
than the one seen at training time, the residual carries a systematic, crop-size-dependent
offset. That offset is *large* relative to the signal: the pooled-vector measurement
recorded `offset_energy_frac ~ 0.85-0.90`, i.e. the shared component dominates, and the
whole reason for centering is that the object signal lives entirely in the residual. A
biased `mu` therefore leaks a scale-correlated constant straight into whatever consumes
the prior — and a bias that correlates with the progressive stage would also correlate with
the training schedule.

**NOT fixed. NOT recomputed. Flagged only.**

---

## I. Phase 1 findings — why B6 was selected, and what B3 actually shows

Source: `dino_analysis_phases/phase1/outputs/dino_spatial_similarity_val_summary.csv`.
n = 339 same-scene / 338 different-scene (the first sample has no control partner).
Run: 2026-08-10 11:05 CEST, device cuda, seed 0, 339/339 valid triplets, 0 skipped.

### 1. Absolute same-scene 1e5 <-> 1e7 (the criterion actually used)

| Features | B3 | **B6** | B9 | B12 |
|---|---|---|---|---|
| RAW | +0.5218 +/- 0.0074 | **+0.6402 +/- 0.0043** | +0.6064 +/- 0.0036 | +0.4498 +/- 0.0043 |
| CENTERED | +0.5932 +/- 0.0043 | **+0.6856 +/- 0.0024** | +0.5763 +/- 0.0030 | +0.2703 +/- 0.0043 |

### 2. Different-scene control (wrong object), 1e5 <-> 1e7

| Features | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| RAW | +0.4553 | +0.5864 | +0.5650 | +0.4114 |
| CENTERED | +0.5048 | +0.6230 | +0.5266 | +0.2119 |

### 3. Same-scene advantage / discrimination (= same - different)

| Features | **B3** | B6 | B9 | B12 |
|---|---|---|---|---|
| RAW | **+0.0665** | +0.0538 | +0.0414 | +0.0383 |
| CENTERED | **+0.0885** | +0.0627 | +0.0497 | +0.0583 |

### Supporting pairings (centered)

| Pair | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| 1e5 <-> Render | +0.5593 | **+0.6570** | +0.5466 | +0.1968 |
| 1e7 <-> Render | +0.7573 | **+0.7827** | +0.6982 | +0.3731 |

### 4. Why B6 was selected as the primary restoration representation

The selection rule is in the code, not in prose:
`best_block('centered', ('1e5','1e7'))` takes `max` over the **same-scene mean**,
`scene_match='same'` (`analyze_dino_spatial_consistency.py:640-644`). B6 wins that on
**every** pairing and both feature types:

- highest raw 1e5<->1e7 (+0.6402)
- highest centered 1e5<->1e7 (+0.6856)
- highest centered 1e5<->render (+0.6570)
- highest centered 1e7<->render (+0.7827)

It also has the smallest dispersion (centered std 0.0445 vs B3's 0.0787), and it overturned
the Phase 0 single-sample impression that B9 was special — B9 ranks **3 of 4** ("the
single-sample impression did not generalise"). Every intermediate block beats the final
block B12 on centered 1e5<->1e7. Phase 2 then read the block out of the CSV rather than
hardcoding it.

### 5. Why B3 may still look better under a different discrimination statistic

Under the same-minus-different gap, the ranking **inverts**: B3 first (+0.0885 centered,
+0.0665 raw), B6 second, decaying with depth apart from B12. Phase 0 saw this more starkly
on one sample (gap +0.170 at B3 vs +0.099 at B6) and concluded "**if DINO is used
spatially, use a mid-shallow block, not the last one**". Also note the raw vs centered
ordering differs: raw [6, 9, 3, 12], centered [6, 3, 9, 12] — centering changes the full
ranking but not the top block.

### 6. Why these are not contradictory

They measure different things.

- The **absolute** cosine is *how much of the clean representation is reproduced at all*,
  and it includes a large component every image in this dataset shares. At B6 centered,
  the wrong-object baseline is already +0.6230 of the +0.6856.
- The **gap** is *how much of the similarity is specific to this scene*.

B6 has more total structure but a larger shared floor; B3 has less total structure but a
proportionally larger scene-specific fraction. Both statements are true of the same
features. Which one is "right" depends on what the downstream consumer can use: a network
that can learn to subtract a constant offset cares about the scene-specific part; one fed
the vector directly is affected by both.

### Best / worst samples (block 6, centered, 1e5<->1e7)

- best: 6115 (+0.7928), 3017 (+0.7874), 3819 (+0.7868)
- worst: 4467 (+0.5671), 1995 (+0.5802), 2547 (+0.5872)

**FLAG:** Phase 1 selected B6 on the *absolute* criterion, then Phase 2's own
interpretation states "**the scene-advantage column is the one to trust**". By that later
standard, **B3** is the block the data selects. The B6 choice has never been re-examined
against the criterion the analysis line itself ended up endorsing. See §N.4.

---

## J. Phase 2 findings — DINO(1e5) vs DINO(render), relative to DINO(1e7)

Source: `phase2/outputs/dino_prior_source_summary.csv` and the **11:53 CEST** devlog entry
(the corrected run; metadata confirms `execution_local_time: 11:53:51`). n = 339, block
**B6**, control = a seeded derangement (no fixed points, same mapping for both sources),
test = paired Wilcoxon signed-rank, effect size = Cohen's dz.

### CENTERED, Block 6 (primary)

| Quantity | 1e5 -> clean | Render -> clean |
|---|---|---|
| same-scene mean | +0.6856 | +0.7827 |
| same-scene median | +0.6839 | +0.7869 |
| different-scene mean | +0.6225 | +0.7106 |
| **scene advantage (mean)** | **+0.0632** | **+0.0721** |
| scene advantage (median) | +0.0574 | +0.0661 |

- Absolute delta (Render - 1e5): **mean +0.0971**, median +0.0937, render wins 98.2% of
  samples, Wilcoxon p = 4.909e-57, **dz = +2.052**
- Scene-advantage delta (Render - 1e5): **mean +0.0089**, median +0.0062, render larger on
  53.1% vs 46.0%, p = 1.178e-02, **dz = +0.163**

### RAW, Block 6

| Quantity | 1e5 -> clean | Render -> clean |
|---|---|---|
| same-scene mean | +0.6402 | +0.7791 |
| different-scene mean | +0.5859 | +0.7237 |
| **scene advantage (mean)** | **+0.0542** | **+0.0554** |

- Absolute delta mean +0.1389 (render wins 95.3%)
- Scene-advantage delta mean +0.0012 (render larger on only 48.1% — i.e. no lead at all)

### What Phase 2 established

1. On **absolute** alignment to DINO(1e7), the render is clearly ahead (+0.0971 centered,
   dz +2.05, essentially every sample).
2. On **scene-specific** advantage, the render's lead collapses to +0.0089 — statistically
   detectable at n=339 but with dz = +0.163, and the win rate is barely above chance
   (53.1 / 46.0). Raw features show no lead at all.
3. The devlog states the ratio explicitly: **"the absolute gap is 10.9x the scene-advantage
   gap"** — i.e. ~90% of the render's apparent superiority also appears against
   **wrong-scene** targets, so it is generic domain similarity (render and clean radar
   simply look more alike to DINO than noisy radar does), not scene content.
4. **Complementarity:** same-scene 1e5<->render at B6 centered is +0.6570, against a
   Phase 1 different-scene floor of +0.6263 for that same pair/block — a gap of only
   ~+0.031. The two sources are neither redundant nor strongly aligned; fusion was
   explicitly declared out of scope.
5. Phase 0 (single sample) reached the sharper version of the same conclusion:
   "**the render is not a usable spatial prior for the noisy radar**" — the 1e5<->render
   gap never exceeded +0.030 at any depth, raw or centered.
6. Phase 2 makes **no** claim about restoration PSNR/SSIM. That was designated Phase 3,
   which does not exist.

### Honest summary

Neither source shows a large scene-specific alignment to the clean DINO representation.
1e5 reaches +0.063 and render +0.072 scene advantage at B6 centered, on a scale where the
shared floor is +0.62-0.71. The measured signal is real and consistent (it survives 339
samples with tight SEMs), but it is a small fraction of the total similarity, and Phase 2's
own conclusion is that the render's advantage is almost entirely generic.

---

## K. Existing metrics

| Metric | Implementation | How computed | Used where | Baseline value exists? |
|---|---|---|---|---|
| PSNR (val) | `basicsr/metrics/psnr_ssim.py:9` `calculate_psnr` | `20*log10(max/sqrt(mse))`, max=255 because `use_image: true` quantizes to uint8; `crop_border=0`, `test_y_channel=false` | **training validation** every 4000 iters | Yes — 22.446 @ 292k |
| SSIM (val) | `psnr_ssim.py:225` `calculate_ssim` | BasicSR 11x11 Gaussian SSIM on uint8 | training validation | Yes — 0.8156 |
| PSNR (test, 16-bit) | `Deraining_Holo/test_holo.py:61` `psnr_uint16` | `20*log10(65535/sqrt(mse))` on uint16 arrays | analysis/eval only | Yes — 22.405 +/- 3.052 |
| SSIM (test) | `skimage.metrics.structural_similarity(data_range=1.0)` | on float [0,1] | analysis/eval only | Yes — 0.7999 +/- 0.077 |
| **object-only PSNR** | `masked_metrics.py:62` `psnr_masked` | mask = `GT > 0.01`, optional close, `dilate=3`; `10*log10(1/mean((gt-pred)^2[mask]))` | analysis only | Yes — 18.313 +/- 2.917 |
| **object-only SSIM** | `masked_metrics.py:66` `ssim_masked` | full SSIM map, then `smap[mask].mean()` | analysis only | Yes — 0.5438 +/- 0.136 |
| Foreground coverage | `masked_metrics.py` | `mask.mean()` per image | analysis only | Yes — 39.5% mean (13.9-80.3%) |
| **HF energy fraction** | `hf_energy` in `test_holo.py:81` and `analyze_sharpness.py:54` | FFT2 -> fftshift -> power; fraction of total power at radius > 0.25*r_max. analyze_sharpness adds an optional Hann window (`--window`) | analysis only | Yes — pred/GT **0.216** |
| Laplacian variance | `lap_var`, both files | `cv2.Laplacian(...).var()` | analysis only | Yes — pred/GT 0.283 |
| Sobel gradient magnitude | `grad_mag`, both files | mean `sqrt(gx^2+gy^2)`, ksize 3 | analysis only | Yes — pred/GT 0.730 |
| **Radial power spectrum** | `radial_power`, both files | azimuthally-averaged FFT power, 128 bins vs fraction of Nyquist | analysis only | Yes — `exp2_verynoisy/figures/radial_power_spectrum.png`; Hann-windowed variant for Exp 1 |
| Patch-wise cosine similarity | `visualize_dino_spatial_pca.py:310` `patchwise_cosine` | `F.cosine_similarity(a,b,dim=1).mean()` over 256 corresponding patches, 768-d | DINO analysis only | Yes — all Phase 1/2 tables |
| Wilcoxon signed-rank + Cohen's dz | `phase2/analyze_dino_prior_source.py` | `scipy.stats.wilcoxon`, dz on paired differences, tie tolerance 1e-3 | DINO analysis only | Yes |
| NIQE / FID | `basicsr/metrics/{niqe,fid}.py` | upstream | **never used** | No |

No over-smoothing metric exists beyond the three sharpness ratios + the spectrum. No
perceptual/LPIPS metric exists anywhere. No new metrics were invented for this report.

**Derived masked analysis already on record (DEVLOG Step 19c):**
full-minus-masked PSNR gap 4.09 +/- 1.24 dB (range 0.95-8.54);
`corr(mask_frac, psnr_mask) = +0.13` (masked score is not an artifact of object size);
worst cases by masked PSNR: `3752.png` (8.88 dB), `3796.png` (11.69), `0616.png` (11.71).

---

## L. Existing visualization / result infrastructure

### Already exists and is directly reusable

| Capability | Where | Notes |
|---|---|---|
| Predictions as uint16 PNGs | `test_holo.py` -> `<result_dir>/raw/` | 338 files per run currently |
| 3-panel comparison PNGs (noisy / prediction / GT, inferno heatmap, shared 0-1 scale, PSNR in titles) | `test_holo.py:save_heatmap_grid` -> `<result_dir>/viz/` | one per test image |
| Radial power spectrum figure | `test_holo.py` + `analyze_sharpness.py` | `radial_power_spectrum.png` |
| Mask visualization (GT / mask / overlay) | `masked_metrics.py --save_mask_viz` | `mask_visualization.png` |
| Per-image metrics CSV | `masked_metrics.py --csv`, `analyze_sharpness.py --csv` | `filename, mask_frac, psnr_full, psnr_mask, ssim_full, ssim_mask` |
| Training curves (loss + val PSNR + val SSIM) | `plot_curves.py --exp <name> --title` | parses `train_*.log` with regexes |
| Overfit check (train loss vs val PSNR, verdict derived from data) | `plot_overfit.py --exp` | |
| Two-run overlay | `plot_compare.py --exp_a --exp_b --label_a --label_b --out --note` | already generalized |
| Val metrics table image | `plot_val_table.py` | WARNING: still hardcodes `../experiments/Holo_Baseline_Restormer` |
| Loss-vs-loss scatter | `plot_loss_vs_loss.py` | WARNING: also hardcoded to Exp 1 |
| TensorBoard | `logger.use_tb_logger: true`, `tb_logger/<name>/` + `launch_tensorboard.sh` | scalars: `l_pix`, `lr`, `metrics/psnr`, `metrics/ssim` |
| Checkpoints + resume states | `experiments/<name>/models/net_g_<iter>.pth`, `training_states/<iter>.state`, every 2000 iters | auto-resume scans `training_states/` and overrides the yml |
| Self-chaining SLURM driver | `train_holo_chain*.sh` + `experiments/Holo_chain_state*/`{CHAIN_COUNT, TRAINING_DONE, CHAIN_ABORTED} | includes a crash guard; MAX_CHAIN=5 |
| Split tooling | `create_split.py`, `create_val_test_split.py` (refuses double-halving), `create_symlinks.py`, `verify_splits.py` (non-zero exit -> usable as a gate) | already `--root/--variants/--splits` parameterized |
| Committed results convention | `experiment_results/expN_<tag>/{report.md, figures/, metrics/}` + a section in `results.md` | documented in `experiment_results/README.md` |
| DINO figure machinery | `visualize_dino_spatial_pca.py`: `joint_pca`, `shared_range`, `to_map`, `upsample`, `make_figure`, `input_panel` (denormalized DINO input, so preprocessing errors are visible in the figure) | reused unchanged by Phase 1 and Phase 2 |
| Auto-devlog appending with real run numbers | Phase 1/2 `append_devlog()` | smoke runs never write |

### Existing qualitative baseline examples

- `Deraining_Holo/results/Holo_verynoisy_test_292k/viz/*.png` — all 338 Exp 2 test predictions
- `Deraining_Holo/experiment_results/exp2_verynoisy/figures/` —
  `data_example_verynoisy_vs_clean.png`, `mask_visualization.png`,
  `radial_power_spectrum.png`, `training_curves.png`, `overfit_check.png`
- `Deraining_Holo/results/Holo_test_224k/for_professor/` — `1_training_curves.png`,
  `2_example_typical.png`, `3_example_blurry.png` (Exp 1)
- Named Exp 2 failure cases: `3752.png` (8.88 dB masked), `3796.png` (11.69),
  `0616.png` (11.71)
- DINO qualitative: `phase1/outputs/{best_samples,worst_samples}/` (3 each),
  `phase2/outputs/representative_samples/` (5), `outputs/dino_spatial_pca_*`
  (4 Phase 0 figures)

### Not present

Error/difference maps, per-pixel residual visualizations, and any saved validation images
during training (`val.save_img: false`; both `experiments/*/visualization/` dirs are empty).

---

## M. Existing configuration system

- **Format:** YAML. **Parser:** PyYAML with an `OrderedDict`-preserving loader
  (`basicsr/utils/options.py`, `ordered_yaml()` / `parse()`), `CLoader` when available.
- **Loading:** `basicsr/train.py:parse_options()` takes exactly `-opt <path>`,
  `--launcher {none,pytorch,slurm}`, `--local_rank`. `parse()` then injects `is_train`,
  sets `dataset['phase']` from the datasets key name (`train`/`val`/`test_1`->`test`),
  copies `scale` into every dataset, expands `~`, and **derives all output paths from
  `opt['name']`**: `experiments/<name>/{models,training_states,visualization}` and the log
  there too.
- **Command-line overrides:** **there are none.** No `--set key=value`, no argparse merge.
  Everything is in the yml; the only CLI knobs are the three above. Changing an experiment
  means editing the yml or copying it.
- **Are model options passed cleanly through config?** Yes. `network_g` is a dict;
  `define_network(deepcopy(opt['network_g']))` pops `type`, looks the class up in the
  auto-scanned arch registry, and passes **every remaining key as a kwarg**. Adding an arch
  parameter = adding a yml key, no plumbing.
- **Are different model variants already supported?** The mechanism is generic —
  `basicsr/models/archs/__init__.py` scans `*_arch.py`, `basicsr/models/__init__.py` scans
  `*_model.py`, `basicsr/data/__init__.py` scans `*_dataset.py`, all keyed by `type`. But
  **right now only one arch is registered** (`Restormer`) — `dinov2_feature_extractor.py`
  deliberately does *not* end in `_arch.py` because it registers nothing. Two datasets are
  registered (`Dataset_PairedImage_uint16`, `Dataset_PairedImage_uint16_Render`) and one
  model type (`ImageCleanModel`).
- **Output paths:** entirely `opt['name']`-driven, plus `logger.tb_logger_dir`.
  WARNING: `init_tb_logger` actually uses `osp.join('tb_logger', opt['name'])` and
  **ignores** `tb_logger_dir` — that yml key is decorative.
- **Checkpoint / resume:** `path.pretrain_network_g` + `strict_load_g` for weight init;
  `path.resume_state` for full resume. But `train.py:138-149` **always** scans
  `experiments/<name>/training_states/`, takes the highest `<iter>.state` and **overrides
  the yml** — so resume needs no config edit, and *reusing an experiment name silently
  resumes that experiment*. `check_resume()` then rewires `pretrain_network_g` to the
  matching checkpoint.

### One existing config, structurally — `Holo_Baseline_Restormer.yml`

```
name / model_type / scale / num_gpu / manual_seed     <- name drives ALL output paths
datasets:
  train:  type + dataroot_gt + dataroot_lq + geometric_augs + io_backend
          loader: use_shuffle, num_worker_per_gpu, batch_size_per_gpu
          progressive: mini_batch_sizes [8,5,4,2], iters [92k,64k,48k,96k],
                       gt_size 256, gt_sizes [128,160,192,256]
  val:    type + dataroot_gt + dataroot_lq + io_backend   (no crop keys -> full images)
network_g:  type: Restormer + 10 arch kwargs
path:       pretrain_network_g / strict_load_g / resume_state
train:      total_iter, warmup_iter, use_grad_clip, scheduler, mixing_augs, optim_g, pixel_opt
val:        window_size, val_freq, save_img, rgb2bgr, use_image, max_minibatch, metrics{psnr,ssim}
logger:     print_freq, save_checkpoint_freq, use_tb_logger, tb_logger_dir, wandb
dist_params: backend, port
```

`DINO_analysis_data.yml` reuses this shape but is **not a training config**: only
`datasets.val` (three dataroots) and `network_g`'s five `dino_*` keys are read. Its `/val_`
path segments are load-bearing — Phase 1 rewrites `/val_` -> `/train_` to reach the training
split. Phase 2 asserts `dino_model_name` / `dino_weights` / `dino_hub_source` against
`phase1_metadata.json` and aborts on mismatch.

**No redesign proposed. Inspection only.**

---

## N. Important risks / unknowns to resolve before implementing E1-N

1. **The progressive crop is applied after the dataloader and only to `lq`/`gt`.**
   `train.py:255-269` sub-crops and mini-batch-subsamples `train_data['lq']` and
   `train_data['gt']` only. Any third stream from the dataset (a render, or a precomputed
   DINO tensor) arrives at 256x256 and full batch and would be **silently misaligned** with
   what the network sees, for 204k of 300k iterations. This must be handled explicitly, and
   it is the highest-probability silent bug in the whole design.

2. **The latent grid is not fixed.** DINO B6 is always 16x16; the latent is 16/20/24/32
   depending on the progressive stage, and **32x32 at every validation and test forward
   pass**. Train and eval would see different prior-to-latent scale relationships unless the
   resampling is defined carefully. The 16x16 match at crop 128 is a coincidence.

3. **Centering-distribution mismatch (§H).** The 768-d spatial means were measured on full
   256^2 frames; training crops are 128/160/192. No decision has been made about whether to
   accept, recompute, or sidestep this. Related: the means come from only 150 images.

4. **The B6 selection rests on the criterion Phase 2 later argued against.** Phase 1 chose
   B6 on *absolute* same-scene cosine. Under the scene-advantage criterion — which Phase 2's
   own interpretation calls "the one to trust" — **B3 ranks first** (+0.0885 vs B6's
   +0.0627 centered). Never reconciled. If a spatial prior is taken at B6, that choice
   should be stated as being made on absolute alignment, or B3 should be included as an arm.

5. **The absolute size of the signal is small.** At B6 centered, 1e5<->1e7 same-scene is
   +0.686 against a wrong-object floor of +0.623 — the scene-specific part is ~+0.063 on a
   scale where ~0.62 is shared by every image pair in the dataset. Phase 2 never tested
   whether that translates into restoration quality; that was designated Phase 3 and does
   not exist. The premise of the next experiment is still unverified in the only currency
   that matters (PSNR / SSIM / HF energy).

6. **E1 failed three times on FiLM runaway**, and its post-mortem is compressed (full detail
   only on the `dino_prior` branch, and **all E1 checkpoints/logs/TensorBoard were deleted
   from disk and exist in no branch**). CONTEXT.md: do not resurrect FiLM guidance without a
   reason the three recorded failures do not already cover. Whatever injection mechanism is
   chosen needs bounded modulation and logging of the modulation itself from iteration 1 —
   attempt 1 trained 73k iters with nothing logging it.

7. **No model wrapper accepts a third stream.** `ImageCleanModel.feed_train_data` /
   `feed_data` read only `lq`/`gt`, and `optimize_parameters` calls `net_g(self.lq)`. A new
   `*_model.py` (auto-registered) plus a new `*_arch.py` is required; the two E1 wrappers
   were deleted.

8. **The baseline cannot be resumed or warm-started from mid-run state.** All Exp 2 training
   states were deleted (Step 19b). Only `net_g_292000.pth` and `net_g_128000.pth` survive,
   as inference weights. Any "initialize from baseline" plan must use those.

9. **Val PSNR is computed on 8-bit quantized images** (`use_image: true` -> `tensor2img` ->
   uint8), while test PSNR is 16-bit. The two numbers are not on the same scale; a new
   experiment comparing against 22.446 (val) or 22.405 (test) must pick the right one.

10. **Exp 2's test number is not a clean held-out estimate** (split carved after training;
    test images influenced checkpoint selection). A new experiment validating on the
    339-image val half will produce val numbers **not directly comparable** to Exp 2's
    logged val curve, which used all 677.

11. **Renders exist in two flavours** — `renders/` (white background, mean 247) and
    `renders_blackbg/` (mean 74). Only `renders_blackbg` is wired anywhere. Easy to grab the
    wrong one.

12. **`paired_paths_from_folder` does not sort.** Pairing is safe (by GT basename), but
    dataset index -> sample-ID order is filesystem order. Anything relying on index order
    (e.g. Phase 1's "previous sample" control) inherits that.

13. **Small landmines:**
    - `logger.tb_logger_dir` is ignored; TensorBoard always writes to `tb_logger/<name>/`
    - reusing an `opt['name']` silently auto-resumes the old run
    - the two E1 pooled-mean `.pt` files must not be deleted (a rerun would no longer match
      the committed metadata)
    - `plot_val_table.py` and `plot_loss_vs_loss.py` still hardcode Exp 1
    - the first Phase 2 devlog entry (11:47 CEST) holds pre-bugfix numbers
    - `use_identity` under `mixing_augs` is dead config while `mixup: false`
    - `noisy`'s ray count is UNCONFIRMED (1e6 assumed)

---

## §19 — What was actually run

Two read-only probe scripts in the session scratchpad, both on CPU, both under
`torch.no_grad`, both write-free:

```bash
# 1. Restormer shape probe -- forward hooks on all 18 submodules, B=2 random input,
#    at 128/160/192/256. No weights loaded, no data read, nothing written.
/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python \
  <scratchpad>/shape_probe.py

# 2. DINO probe -- one real val sample (0196) through the project's own
#    build_dataset / build_extractor / batch_to_dino_inputs / spatial_tokens path.
TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub \
/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python \
  <scratchpad>/dino_probe.py
```

Plus read-only `cv2.imread` inspections of five dataset PNGs, `torch.load` on the three
saved mean files, and `diff` / `ls` / `git log` over the repo.

**No file in the repository or the dataset was created, modified or deleted** (other than
this report itself, written on request). No job was submitted. No checkpoint, mean or
config was touched.

### Runtime confirmations from probe 2 (nothing here was assumed)

- DINOv2: 12 blocks, 768-d, patch 14, **0 register tokens**
- `load_state_dict` strict with **missing=[] unexpected=[]**
- DINO input `[1,3,224,224]` in ~[-2.118, +2.638]
- Every selected block returns **`[1, 256, 768]`**
- `-> [1,16,16,768] -> permute -> [1,768,16,16]`, and that manual reshape is
  **bit-identical** to DINOv2's own `reshape=True` output (`allclose` True), confirming
  row-major token -> grid ordering
- A 128x128 crop through `dino_preprocess` also yields `[1,3,224,224] -> [1,256,768]`,
  i.e. the same grid at a different magnification — exactly the mismatch in §H
- Dataset batch tensors: `lq [1,256,256]` and `gt [1,256,256]` in [0,1];
  `dino (render) [3,256,256]` in [0, 0.9765]

### Runtime confirmations from probe 1

- Parameter count **26,124,052**
- All shapes in §C, at all four progressive crop sizes
- Output shape always equals input shape; residual add confirmed in code

---

**END OF REPORT. Nothing implemented. Awaiting review before the next experiment is
designed.**
