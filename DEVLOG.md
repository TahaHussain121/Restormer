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

## Step 13 — Moved to HPC (2026-06-11)
All development up to Step 12 was done on fived08 (RTX 4070 Ti SUPER, 16 GB, tmux-based).
A 300k-iter run was started on fived08 and cancelled — no usable checkpoint was produced from it.

Cluster: tinygpu (FAU HPC), SLURM v25.11.2.
Venv transferred: /home/woody/iwnt/iwnt174h/thesis_dino/code/venv/ — PyTorch 2.5.1+cu121.
Dataset is at the same absolute path (/home/woody/…/holo_image_dataset/) — symlinks survived intact.
Default partition going forward: a100 (A100 GPU, 24 h walltime). v100 was used for first run.
Job script: train_holo.sh (repo root), submitted via sbatch.

## Step 14 — 75k baseline training run complete (2026-06-11)
Config: Holo_Baseline_Restormer.yml — 75k iters, all at patch=128×128 batch=8
  (progressive stages never activate because total_iter=75k < first stage threshold=92k)
Hardware: SLURM job 1693579, node tg071, Tesla V100-PCIE-32GB (32 GB)
Runtime: 8 h 02 m 59 s (well within 15 h requested; 24 h available)
Peak VRAM: 26,126 MiB / 32,768 MiB (80%)
GPU util: 97%

PSNR progression at val checkpoints (every 4k iters):
  iter  4k: 31.30 dB
  iter 52k: ~32.1 dB (LR decay begins here)
  iter 75k: 32.2650 dB  ← final
  Total gain: +0.97 dB over training

One loss spike at iter 14k (1.4033e-01) — recovered immediately, no lasting effect.
Checkpoint: experiments/Holo_Baseline_Restormer/models/net_g_latest.pth
State files: experiments/Holo_Baseline_Restormer/training_states/ (every 2k iters up to 75k)

NOTE: This 75k run will be superseded. The progressive training schedule was not active
(all iters used 128×128 patches). A proper 300k run using the full progressive schedule
(patches 128→384, batches 16→2) is planned next to get the intended baseline number.

## Step 15 — Val set split into val + test (2026-06-11)
Motivation: upcoming 300k retrain will monitor val loss during training. Test set must be
held out and untouched from this point to avoid any leakage into model selection.

Procedure:
  - Backed up: splits/val.txt.bak
  - Script: Deraining_Holo/create_val_test_split.py  (seed=42, 50/50 split by count)
  - Old val.txt (~677 entries) shuffled with random.seed(42), split at midpoint
  - New splits/val.txt  — second half sorted → 339 entries
  - New splits/test.txt — first half sorted  → 338 entries
  - Symlink dirs rebuilt via updated create_symlinks.py (idempotent, all three splits)
  - test_clean/, test_noisy/ created (338 symlinks each)
  - Zero overlap confirmed: train∩val=0  train∩test=0  val∩test=0

test.txt committed to git for reproducibility. Test set is now locked.
All future val-time PSNR numbers (during and after 300k retrain) are on the ~338-sample val set.
Final model evaluation will be run once, on the test set.

## Step 16 — Retrain baseline with full original 300k recipe (2026-06-11)
The 75k run is OBSOLETE. Reason: total_iter=75k was below the first progressive
stage threshold (92k), so all 75k iters ran at stage 1 (128px, batch 8). The
progressive patch schedule (128->384) never activated — it was not a faithful
Restormer deraining baseline.

Action: restored Holo_Baseline_Restormer.yml to the upstream Restormer deraining
recipe:
  - total_iter: 75000 -> 300000
  - scheduler periods: [52000,75000] -> [92000,208000]
  - scheduler eta_mins: [0.0006,1e-6] -> [0.0003,1e-6]
  - optim_g lr: 6e-4 -> 3e-4
  - mini_batch_sizes: [16,10,8,4,2,2] -> [8,5,4,2,1,1]
  - batch_size_per_gpu: 4 -> 8
  - num_worker_per_gpu: 4 -> 2
  - (iters, gt_size, gt_sizes already matched original)
Intentional deltas from upstream kept: 1-channel I/O, uint16 loader, our paths,
num_gpu=1, save_checkpoint_freq=2000, TB logging, resume mechanism.
Header comment block added to the yml documenting all intentional differences.

CAVEAT (thesis): upstream recipe assumes 8 GPUs => effective batch 64. We run
1 GPU => effective batch 8. This difference must be reported in the thesis.

mixup decision (resolved): set mixup: true to match the upstream deraining recipe
(was false in the 75k run). batch_size_per_gpu also set 4 -> 8 to match upstream
(overridden by progressive mini_batch_sizes[0]=8 in stage 1, so effectively inert).

Execution model: a100 partition, 23h walltime (NOT 24h). Full 300k run will not
fit one job; expect 2-3 jobs with resume between them (checkpoints every 2k iters).
Obsolete run moved to experiments/Holo_Baseline_Restormer_75k_obsolete (not deleted).
Commands recorded in COMMANDS.md. Job not yet submitted (awaiting confirmation).

## Step 16b — 300k job submitted and RUNNING (2026-06-11)
Submitted via sbatch train_holo.sh (repo root). First job:
  - JOB ID: 1694322
  - Partition: v100 (NOT a100 — a100 was queue/GRES-blocked at submit time; Taha
    chose v100 to start sooner. train_holo.sh header notes how to switch back.)
  - Walltime: 23h. Started 2026-06-11 11:26, started fresh from iter 0.
  - Node: tg071, Tesla V100-PCIE-32GB.

Startup verified healthy:
  - Dataset_PairedImage_uint16, 6101 train / 339 val images, 300000 iters / 394 epochs
  - Progressive stage 1 active (patch 128, batch 8), lr 3.000e-04, mixup True
  - First loss print finite: iter 1,000 l_pix = 1.4868e-02

Early val PSNR/SSIM (already surpassing the obsolete 75k run's final 32.265):
  iter  4k: 31.03 / 0.913
  iter  8k: 31.47 / 0.925
  iter 12k: 31.74 / 0.925
  iter 16k: 32.02 / 0.930
  iter 20k: 32.11 / 0.931
ETA ~2 days 5h => will NOT fit one 23h job. Expect ~3 jobs with resume between them.

NOTE: a stray mkdir of the experiment dir before submit triggered basicsr's
auto-archive; the SLURM stdout (slurm_1694322.out) landed in
experiments/Holo_Baseline_Restormer_archived_20260611_112635/. Harmless — the
real progress log is experiments/Holo_Baseline_Restormer/train_*.log. On resume
basicsr continues in place (no archiving), so future jobs are unaffected.

## Step 16c — Self-chaining resume script (2026-06-11)
Goal: make the ~3 walltime-resume cycles autonomous (no human / no live Claude).

Key discovery: basicsr/train.py (lines ~138-149) has BUILT-IN auto-resume. At every
startup it scans experiments/<name>/training_states/, takes max([int(name)]).state,
and sets opt['path']['resume_state'] to it — OVERRIDING the yml. Verified the
dry-run detection (ls -t ... | head -1) agrees with this max(int) logic: both pick
22000.state currently. => resume needs NO yml edit and runs identical training code.
No --auto_resume CLI flag exists (argparse has only -opt/--launcher/--local_rank).

Created Deraining_Holo/train_holo_chain.sh (NEW file; train_holo.sh left as-is):
  - same SBATCH as current run: v100, gpu:v100:1, 23:00:00
  - exits if TRAINING_DONE marker exists (before incrementing counter)
  - increments CHAIN_COUNT; exits if > MAX_CHAIN=5 (infinite-loop guard)
  - queues successor: sbatch --dependency=afterany:$SLURM_JOB_ID <self> (abs path)
  - runs IDENTICAL command: python basicsr/train.py -opt <same yml> --launcher none
  - after training, if models/net_g_300000.pth exists -> touch TRAINING_DONE (stops chain)
  - crash guard: if python exits in < MIN_RUNTIME=1800s WITHOUT the 300k ckpt,
    assume persistent crash -> scancel the queued successor + touch CHAIN_ABORTED;
    successor also checks CHAIN_ABORTED at startup and exits. (A walltime kill runs
    ~23h and kills this script too, so the guard never falsely fires on a healthy run.)
  Markers (CHAIN_COUNT, TRAINING_DONE, CHAIN_ABORTED) live in the experiment dir root
  (NOT in training_states/, which must contain only *.state for the max(int) parser).
  To restart after an abort: rm experiments/Holo_Baseline_Restormer/CHAIN_ABORTED.

Safety: created only the new script + docs. Did NOT touch basicsr/, the yml,
the running job 1694322, or anything inside experiments/Holo_Baseline_Restormer/.
Chain is NOT submitted yet — Taha submits it once, manually, after the walltime kill.

## Step 17 — Reflect-padding contamination found; run restarted, schedule truncated at 256 (2026-06-11)
Finding (at iter ~28k of job 1694322): with gt_size: 384, the dataset pads every
256x256 image UP to 384x384 via cv2.BORDER_REFLECT (img_util.py padding(), bottom+right
borders) before cropping. The progressive sub-crop then samples from this 384 canvas, so
a large fraction of crops at ALL stages contain MIRRORED pixels that are physically fake
radar structure (mirrored side lobes). Quantified: ~75% of stage-1 (128px) crops touch
reflected pixels; stages 5 (320) and 6 (384) would be ~unavoidably/entirely reflected.
=> the baseline was training partly on artifacts that never occur at test time.

Decision: cancelled job 1694322 (scancel) at iter ~28k (~9% in) and restarted with the
progressive schedule TRUNCATED at native 256 resolution (4 stages), so padding() is a
no-op and every patch is 100% real data.

yml change (only the schedule; total_iter/scheduler/lr/mixup/batch_size unchanged):
  gt_size: 384 -> 256
  gt_sizes: [128,160,192,256,320,384] -> [128,160,192,256]
  iters: [92000,64000,48000,36000,36000,24000] -> [92000,64000,48000,96000]  (sum=300000)
  mini_batch_sizes: [8,5,4,2,1,1] -> [8,5,4,2]
Corrected 4-stage table (cumulative boundaries 92k/156k/204k/300k):
  Stage 1: 128px, batch 8, iters     0 - 92k
  Stage 2: 160px, batch 5, iters   92k - 156k
  Stage 3: 192px, batch 4, iters  156k - 204k
  Stage 4: 256px (full image), batch 2, iters 204k - 300k

Numbers recorded before deleting the two obsolete runs (folders deleted; key numbers kept here):
  - 75k run (Step 14): final val PSNR 32.2650 (all 128px, progressive never activated).
  - 26-28k reflect-pad run (job 1694322): val PSNR/SSIM progression
      4k 31.03/0.913 | 8k 31.47/0.925 | 12k 31.74/0.925 | 16k 32.02/0.930
      20k 32.11/0.931 | 24k 32.20/0.935 | 28k 32.2261/0.9354  (CONTAMINATED — discard).
Deleted: experiments/Holo_Baseline_Restormer (26-28k run),
  experiments/Holo_Baseline_Restormer_75k_obsolete, the two _archived_* dirs, and
  tb_logger/Holo_Baseline_Restormer* — so the fresh run finds NO training_states/ to
  auto-resume from.

NOTE: this same native-resolution cap (gt_size 256, patches <= 256) must also be applied
to the future denoising-schedule ablation config.

## Step 18 — Loss function verified: L1Loss (docs correction) (2026-06-15)
Verified the actual training loss from the live config + code (no code/config changed):
  - Deraining_Holo/Options/Holo_Baseline_Restormer.yml -> train.pixel_opt.type: L1Loss
    (loss_weight 1, reduction mean)
  - basicsr/models/image_restoration_model.py init_training_settings() builds
    self.cri_pix = L1Loss(...) from pixel_opt.type, applied as l_pix in
    optimize_parameters() (l_pix = self.cri_pix(pred, gt); l_pix.backward()).
So the optimized objective is L1Loss, which also matches the upstream Restormer
deraining recipe.
Correction: earlier CONTEXT.md "Architecture Decisions" said the loss was "Charbonnier".
That was an error — corrected to L1Loss. Docs-only change; training untouched.

## TODO
- [ ] Step 17-run: fresh 4-stage run submitted via chain script; drive to 300k, log final PSNR/SSIM
- [ ] Step 18: evaluate on val set, then run test_holo.py on the locked test set
- [ ] Step 19 (July): DINOv2 injection into bottleneck
- [ ] Step 20 (July): ablation with/without DINOv2
- [ ] Step 21 (Aug): write thesis chapter
