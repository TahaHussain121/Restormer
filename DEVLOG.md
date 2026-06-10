# Restormer Radar CFR — Dev Log
> Append-only. Never rewrite existing entries. Add new entries at the bottom when a step is completed.

## Step 0 — Thesis direction established (~April 2026)
Professor rejected U-Net direction. New direction: Restormer backbone + DINOv2 visual prior injected via cross-attention (Perceive-IR style). Dataset: ShapeNetCore.v2 chairs. DINOv2 frozen.

## Step 1 — Restormer architecture understood
MDTA: transposed attention over channels [CxC] not spatial [HWxHW]. O(C²) not O(N²).
GDFN: expand x4, split, DWConv(F_a) * GELU(F_b) — learned spatial gate.
Residual output: model predicts artifact to remove, final = model(X) + X.
Deraining chosen over denoising: side lobes are structured like rain streaks, not random noise.

## Step 2 — shapenet_radar package built (May-June 2026)
10 modules, 35/35 tests passing. Clean (1e6) + degraded (1e7) PNG pairs per chair view.
Key bugs fixed: RadarData unwrap, signed int32 seed mask.

## Step 3 — Restormer repo cloned and trimmed
Kept: Deraining/, Denoising/, basicsr/, train.py, test.py, requirements.txt
Deleted: Motion_Deblurring/, Defocus_Deblurring/

## Step 4 — Deraining_Holo/ created
Copied Deraining/ → Deraining_Holo/
yml changes: inp_channels=1, out_channels=1, experiments_root=experiments/Deraining_Holo
Dataset paths: not yet set.

## Step 5 — Repo trimmed, Deraining_Holo/ set up (2026-06-10)
Deleted Motion_Deblurring/ and Defocus_Deblurring/.
Copied Deraining/ → Deraining_Holo/.
Renamed yml: Deraining_Restormer.yml → Deraining_Holo_Restormer.yml.
yml changes made: name=Deraining_Holo (controls experiments_root → experiments/Deraining_Holo), inp_channels=1, out_channels=1.
Dataset paths left as TODO placeholders — awaiting real paths from Taha.
Note: experiments_root is NOT a yml key; basicsr computes it as <root>/experiments/<name> from options.py.

## Step 6 — Dataset paths filled, pre-training checks run (2026-06-10)
Filled dataroot_gt/lq in yml: clean/ and noisy/ under holographic_image_dataset/.
6778 paired files confirmed, filenames match (0001.png … both dirs).
Train and val both point to same folder — no val split exists yet.

CRITICAL issues found in BasicSR loader (imfrombytes, paired_image_dataset.py):
- Issue A (precision): cv2.IMREAD_COLOR + /255. silently truncates uint16→uint8.
  Full 16-bit dynamic range (65536 levels) → 256 levels. Must fix before training.
- Issue B (crash): IMREAD_COLOR on grayscale PNG returns (H,W,3). Network has inp_channels=1.
  Shape mismatch → crash at first forward pass. Must fix before smoke test.

Not fixed yet — awaiting decision from Taha on how to handle (custom loader vs flag change).

## Step 7 — uint16 loader fixed, dataloader test passed (2026-06-10)
Added imfrombytes_uint16 to basicsr/utils/img_util.py (below imfrombytes).
  - Uses cv2.IMREAD_UNCHANGED to preserve uint16 bit depth
  - Adds channel dim for grayscale: (H,W) → (H,W,1)
  - Normalizes by /65535. (not /255.)
Exported from basicsr/utils/__init__.py.
Created basicsr/data/paired_image_dataset_uint16.py:
  - Class: Dataset_PairedImage_uint16
  - imfrombytes → imfrombytes_uint16 for both gt and lq
  - bgr2rgb=False (grayscale, no channel reordering)
  - geometric_augs uses .get() with False default (safe for test opt)
Updated yml: type: Dataset_PairedImage_uint16 for both train and val.
Test (Deraining_Holo/test_dataloader.py) output:
  gt shape: [1, 256, 256], lq shape: [1, 256, 256], dtype: float32, range [0,1] — PASS

## Step 8 — Train/val split created (2026-06-10)
90/10 split, random.seed(42), 6778 total → 6101 train / 677 val.
Split lists written to holographic_image_dataset/splits/train.txt and val.txt.
Symlink dirs created (no data copied):
  train_clean/, train_noisy/ — 6101 symlinks each
  val_clean/,   val_noisy/   — 677 symlinks each
Overlap check: 0 — PASS.
yml updated: train paths → train_clean/train_noisy, val paths → val_clean/val_noisy.
Dataloader test re-run against new train paths — PASS.
Scripts: Deraining_Holo/create_split.py, Deraining_Holo/create_symlinks.py

## Step 9 — 1-batch smoke test passed (2026-06-10)
Config: Deraining_Holo_Restormer_smoketest.yml (total_iter=2, batch=1, num_gpu=1, seed=42)
Command: CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt ... --launcher none
GPU: RTX 4070 Ti SUPER (16 GB), PyTorch 2.6.0+cu124
Model: 26,124,052 parameters. patch_embed Conv2d(1,48), output Conv2d(96,1) — single-channel confirmed.
iter 1: l_pix = 7.2340e-02
iter 2: l_pix = 8.1395e-02
Validation PSNR (untrained baseline, 677 val images): 20.72 dB
No NaN, no shape mismatch, no OOM. Exit code 0.
Smoketest yml deleted after pass.

## Step 10 — Config renamed and DINOv2 placeholder created (2026-06-10)
Renamed Deraining_Holo_Restormer.yml → Holo_Baseline_Restormer.yml.
  - name: Deraining_Holo → Holo_Baseline_Restormer (experiments/ dir follows automatically)
  - All other settings unchanged
Created Holo_DINOv2_Restormer.yml as a comments-only placeholder for July.
Config naming convention going forward:
  Holo_Baseline_Restormer.yml   — pure Restormer, no DINOv2
  Holo_DINOv2_Restormer.yml     — DINOv2 cross-attention injection (Perceive-IR style)

## Step 11 — Full training run launched (2026-06-10)
Config: Holo_Baseline_Restormer.yml
  - num_gpu: 1 (RTX 4070 Ti SUPER, 16 GB)
  - batch_size_per_gpu: 4 (overridden by progressive schedule: mini_batch_sizes=[8,5,4,2,1,1])
  - Stage 1 active: patch=128×128, batch=8
  - total_iter: 300,000
Command: CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Deraining_Holo/Options/Holo_Baseline_Restormer.yml --launcher none
Session: tmux attach -t restormer_baseline | log: /tmp/restormer_baseline_train.log
iter 1,000: l_pix = 1.5066e-02
iter 2,000: l_pix = 1.3755e-02
Loss finite and decreasing — training healthy.
ETA displayed ~2 days (stage 1 speed only); actual total ~3–4 days (later stages use larger patches).
First val checkpoint at iter 4,000.

## Step 12 — Training config finalized for 50-epoch run (2026-06-10)
Reduced total_iter: 300000 → 75000 (~50 epochs × 1526 iters/epoch).
LR schedule periods: [92000, 208000] → [52000, 75000] (70/30 split).
  Note: cycle 2 period (75k) extends past training end → partial cosine anneal only.
  LR reaches ~2.8e-4 at iter 75k instead of 1e-6. Acceptable for baseline run.
Checkpoint freq: 4000 → 2000 iters (~20 min safety window on HPC).
TensorBoard: use_tb_logger=true, logs → experiments/Holo_Baseline_Restormer/tb_logger/
  Launch: tmux new-session -d -s tensorboard 'bash Deraining_Holo/launch_tensorboard.sh'
  Access: http://localhost:6006 (SSH tunnel: ssh -L 6006:localhost:6006 taha.hussain@ad.five-d.ai@fived08)
Resume workflow: edit resume_state in yml to point at latest .state file, rerun training command.
  State files at: experiments/Holo_Baseline_Restormer/training_states/ITER.state
Progressive training note: with total_iter=75000 and first stage threshold at 92000 iters,
  all 75k iters run at patch=128×128, batch=8. Higher stages never activate in this run.
Old tmux session (300k config) still running — kill before fresh start.

## TODO
- [ ] Step 13: kill old session, fresh start with 75k config, log PSNR at each 4k val
- [ ] Step 14: evaluate on val set at convergence, visualize results
- [ ] Step 15 (July): DINOv2 injection into bottleneck
- [ ] Step 16 (July): ablation with/without DINOv2
- [ ] Step 17 (Aug): write thesis chapter
