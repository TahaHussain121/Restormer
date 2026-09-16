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

## Step 19 — Verynoisy baseline: new dataset, mixup off, 300k complete (2026-07-18 → 07-21)

Switched the baseline onto the newer `holographic_image_dataset` using the
**verynoisy** variant as the degraded input, with mixup disabled.

Dataset work:
  - `holographic_image_dataset/` ships `clean/`, `noisy/`, `verynoisy/`, `renders/`
    (6778 each) plus `splits/{train,val}.txt`, but only `train_/val_` dirs for
    clean+noisy. Built `train_verynoisy/` (6101) and `val_verynoisy/` (677) as
    symlinks into `verynoisy/`, driven by the SAME splits/*.txt, so GT/LQ pairing
    matches the clean split exactly.
  - Gotcha: `splits/*.txt` have no trailing newline, so a naive `while read` loop
    silently drops the last image (6100 instead of 6101). Loop must use
    `while read -r f || [ -n "$f" ]`.

Config (`Options/Holo_Baseline_Restormer.yml`):
  - dataroots -> holographic_image_dataset/{train,val}_clean (GT)
    and /{train,val}_verynoisy (LQ); test yml updated to match
  - `mixing_augs.mixup: true -> false`
  - `name:` -> `Holo_Baseline_Restormer_verynoisy`, `tb_logger_dir` to match.
    This rename is load-bearing: basicsr keys both its output dir AND its
    auto-resume scan off the experiment name, so reusing the old name would have
    resumed the previous noisy run's 224k weights and written over its results.
  - Note `use_identity` is only read when mixup is true, so it is dead config
    here and was left at its upstream value.

Run: `train_holo_chain_verynoisy.sh` (clone of the chain driver pointed at the new
experiment dir, with its own `Holo_chain_state_verynoisy/` bookkeeping).
Jobs 1752760 -> 1752761 -> 1753869, 3 of max 5, each auto-resuming. Completed the
full 300k in 16 h 47 m (394 epochs) on a V100 at 97 % utilization; final job wrote
TRAINING_DONE and cancelled its successor.

Results:
  - best val PSNR/SSIM 22.4460 / 0.8156 @ 292k
  - final val PSNR/SSIM 22.4374 / 0.8158 @ 300k  (delta 0.009 dB -> use 300k)
  - no overfitting; train loss falls throughout, val plateaus without decline
  - the ~21.5 plateau seen mid-run at 174k resolved after the cosine LR restart
    at 208k, gaining ~0.9 dB. Heavy eval-to-eval oscillation before ~200k is
    LR-driven and damps out as LR anneals.

Comparison vs Step 17's noisy baseline: the interesting result is convergence
shape, not the absolute gap. The noisy run is essentially converged by ~25k and
its 300k budget was mostly wasted; the verynoisy run improves across the whole
schedule and needs the full budget. The raw 33.5 vs 22.4 dB gap is NOT a valid
comparison — dataset, noise level and mixup all changed at once.

Tooling: `plot_curves.py` and `plot_overfit.py` now take `--exp/--title` instead of
a hardcoded experiment (overfit verdict is derived from the data, not hardcoded);
added `plot_compare.py` for two-run overlays. Figures + metrics are archived under
`Deraining_Holo/experiment_results/` (tracked in git via a .gitignore exception,
since `experiments/`, `results/` and `*.png` are all ignored) with the running
write-up in `experiment_results/results.md`.

OPEN ISSUE: `holographic_image_dataset` has NO test split — only train.txt and
val.txt. The Step 15 locked test set exists only for the old `holo_image_dataset`.
So the verynoisy run currently has no held-out test number; evaluating on val is
possible but val was used for monitoring. To report a genuine test figure, carve
a test half out of the 677-image val set and re-evaluate.

UNRESOLVED: why `net_g_128000.pth` was kept from the Step 17 noisy run is not
recorded anywhere. It is not the peak (224k), not the knee (~156k per
early_stop_knee.png), and not a progressive-stage boundary (92k/156k/204k).
Most likely it was simply whatever survived a disk prune. Do not treat it as a
deliberately selected checkpoint without confirming.

## Step 19a — Held-out test split for holographic_image_dataset (2026-07-22)

Mirrored the Step 15 recipe onto holographic_image_dataset so the verynoisy run
can be reported on a held-out set instead of on validation.

Procedure (identical method and seed to Step 15, so the two datasets are
methodologically comparable):
  - `create_val_test_split.py --root <holographic> --seed 42`
    shuffles the 677-entry val.txt and halves it -> 339 val / 338 test.
    Train is never touched. Byte-identical recipe to holo, which also produced
    339/338 from its own 677.
  - `create_symlinks.py --root <holographic> --variants clean noisy verynoisy
     --splits val test --allow-replace-files`
    rebuilds val_/test_ dirs for all three variants.
  - `verify_splits.py --root <holographic> --variants clean noisy verynoisy`
    -> train 6101 / val 339 / test 338, all pairwise intersections 0, exit 0.

Scripts generalized in the process (all three were hardcoded to
holo_image_dataset and to clean+noisy only):
  - `--root`, `--variants`, `--splits` args added throughout
  - create_val_test_split.py now backs up val.txt -> val.txt.bak and REFUSES to
    run if test.txt already exists (a second run would halve an already-halved
    val set, silently shrinking it to ~169). --force overrides.
  - create_val_test_split.py now writes a trailing newline. The Step 15 version
    used '\n'.join(...) with none, which is the exact cause of the dropped-last-
    image bug hit in Step 19.
  - create_symlinks.py refuses to delete a real file unless the same filename
    exists in the master variant dir (i.e. content is recoverable), and requires
    --allow-replace-files to do so at all. Needed because holographic's
    val_clean/val_noisy held real COPIES (677 each), not symlinks, unlike holo.
    Verified all 677x3 were present in the master dirs before rebuilding.
  - verify_splits.py now exits non-zero on mismatch/overlap so it can gate.
  - Only val/test were rebuilt; train_* dirs (6101 real copies) left untouched
    since the train split did not change.

Evaluation run as job 1757484 (`eval_test_verynoisy.sh`, ~5 min on a V100) using
**net_g_292000.pth — the BEST checkpoint by val PSNR (22.4460 dB), not the final
300k (22.4374 dB)**, on the new 338-image test split.

RESULTS (338 held-out images):
  noisy input   PSNR 12.3544 dB   SSIM 0.3353
  denoised      PSNR 22.4047 dB   SSIM 0.7999
  improvement        +10.0504 dB       +0.4645
  full image    PSNR 22.4047 +/- 3.0523   SSIM 0.7999 +/- 0.0766
  masked (fg)   PSNR 18.3131 +/- 2.9174   SSIM 0.5438 +/- 0.1361  (39.5% coverage)

  Test 22.405 vs val 22.446 -> 0.04 dB gap, so the model generalizes cleanly.

  SHARPNESS -- the model over-smooths. Pred/GT ratios: Laplacian var 0.283,
  Sobel grad 0.730, HF energy fraction 0.216. The prediction keeps only ~22% of
  the GT's high-frequency energy. The +10 dB is partly bought by blurring away
  genuine fine structure along with the noise -- the classic L1 over-smoothing
  failure mode. This is the most promising target for the DINOv2 work: a
  semantic prior is exactly the kind of thing that could restore detail the
  pixel loss has no incentive to keep.

METHODOLOGICAL CAVEAT (important, do not lose this):
Step 15 carved holo's test split BEFORE the 300k training run, so validation
during that run used only the 339 val images and the test set was genuinely
held out. Here the order is reversed -- Exp 2 trained to completion using ALL
677 images as its validation set, and the test half is being carved afterwards.
So the test images influenced CHECKPOINT SELECTION (they were never trained on;
they are not in train.txt). Practical impact is small: the top-5 checkpoints
span 22.4305-22.4460 dB, a 0.016 dB spread, so the choice is near-arbitrary.
But the resulting test number is NOT a fully clean held-out estimate and must be
described that way in the thesis. A clean number would require either selecting
the checkpoint on the 339-image val half only, or re-running training with the
split in place first (as Step 15/17 did).

Side effect: the training config's val dataroots now resolve to 339 images
instead of 677. Any FUTURE run on this dataset validates on the smaller set --
which is the correct behaviour, but means val PSNR from future runs is not
directly comparable to Exp 2's logged val curve.

## Step 19b — Checkpoint prune, verynoisy run (2026-07-22)

The completed verynoisy experiment was occupying 44 GB: 151 model checkpoints
(15 GB) plus 150 training states (30 GB), saved every 2000 iters.

KEPT (2 files, 201 MB total):
  net_g_128000.pth   mid-run reference point
  net_g_292000.pth   BEST by val PSNR (22.4460 dB); produced the Step 19a test
                     numbers -- this is the checkpoint the thesis reports

DELETED: 149 other .pth files and ALL 150 .state files. Training reached 300k and
TRAINING_DONE is set, so the resume states had no remaining purpose. This also
removes net_g_300000.pth (the final checkpoint, 22.4374 dB) and net_g_latest.pth
-- a deliberate call, since 292k is both better and the one actually evaluated.
Note net_g_latest.pth was NOT byte-identical to net_g_300000.pth (different
md5), so it was a distinct file rather than a duplicate.

Both survivors were verified to load (494 tensors each) before AND after the
deletion. Freed ~44 GB.

CONSEQUENCE: training can no longer be resumed or extended for this experiment
-- there are no .state files left. Re-running would mean starting from iter 0.
The two kept .pth files are inference-only weights, which is all that is needed
to reproduce the reported test metrics.

NOTE for future runs: `save_checkpoint_freq: 2e3` produces ~150 checkpoints and
~45 GB per 300k run. Worth pruning as soon as a run completes rather than
letting several runs accumulate.

## TODO
- [x] Step 17-run: fresh 4-stage run driven to 300k (noisy baseline; peak 33.473 dB @ 224k)
- [x] Step 19: verynoisy baseline trained to 300k (peak 22.446 dB @ 292k)
- [x] Step 19a: test split carved for holographic_image_dataset (339 val / 338 test, seed 42)
- [x] Step 19a-eval: test metrics recorded (22.405 dB full / 18.313 dB masked, net_g_292000.pth)
- [ ] Investigate over-smoothing: pred keeps only ~22% of GT high-frequency energy
- [ ] Step 20 (July): DINOv2 injection into bottleneck
- [ ] Step 21 (July): ablation with/without DINOv2
- [ ] Step 22 (Aug): write thesis chapter

## Step 19c — Masked-metric analysis on the verynoisy test set (2026-07-22)

Re-ran masked_metrics.py on the 338-image test set with --save_mask_viz to add
the mask visualization Exp 1 had but Exp 2 was missing. Metrics reproduced
exactly (they were already computed inside the Step 19a eval job):

  full image    PSNR 22.4047 +/- 3.0523   SSIM 0.7999 +/- 0.0766
  masked (fg)   PSNR 18.3131 +/- 2.9174   SSIM 0.5438 +/- 0.1361
  coverage 39.5% mean (13.9%-80.3% range), threshold 0.01, dilate 3

New analysis from the per-image CSV:
  - full-minus-masked PSNR gap 4.09 +/- 1.24 dB (0.95-8.54). The object region is
    consistently much harder than the full frame; ~60% of pixels are easy empty
    background, so the full-image number flatters the model. Report the masked
    figure as the honest one.
  - corr(mask_frac, psnr_mask) = +0.13, i.e. essentially uncorrelated. The masked
    score is NOT an artifact of object size -- large objects are not easier.
  - worst cases by masked PSNR: 3752.png (8.88 dB), 3796.png (11.69), 0616.png
    (11.71). Starting points for qualitative failure analysis.
  - masked SSIM 0.5438 vs full 0.7999: on the object itself structural fidelity
    is only moderate, consistent with the over-smoothing finding in Step 19a.

Note masked_metrics.py is CPU-only, so this ran on the login node -- no job needed.
Artifacts archived to experiment_results/exp2_verynoisy/.

## Steps 20-26 — E1 DINOv2-FiLM guidance: ATTEMPTED AND ABANDONED (2026-08-06 - 08-09)

E1 added a frozen DINOv2 ViT-B/14 to Restormer as FiLM modulation at the
bottleneck and the three decoder stages, in two arms (lqDINO = DINO sees the
noisy input; renderDINO = DINO sees the aligned render). It was launched,
cancelled, fixed and relaunched three times, and failed every time:

  - Attempt 1: FiLM runaway, |gamma| reached 325. Trained 73k iterations with
    nothing logging the modulation.
  - Attempt 2: gamma/beta bounded + modulation logging + a 4k launch gate.
    Both arms FAILED the gate (14.8 / 6.7 dB vs the 19.6 dB baseline); gamma
    pinned at the bound.
  - Attempt 3: FiLM warmup/ramp + raw_scale, 16k gate. Warmup delayed the
    runaway but did not prevent it. Both arms failed again.

Total cost ~10 GPU-hours across three gate rounds -- the gate is what stopped
this becoming three 3-GPU-day runs.

The E1 training code (RestormerDINO, FiLMHead, the two model wrappers, the four
configs, the chain/gate launchers), the pre-registration design.md and the
full E1_DINO_report.md write-up were REMOVED from this branch. The complete E1
tree, including all three attempts and the detailed per-step log that used to
occupy this space, is preserved on the `dino_prior` branch.

What survives here is the DINO *analysis* line, which is unaffected by the
training failure: the frozen extractor (basicsr/models/archs/dinov2_feature_extractor.py,
renamed from restormer_dino_arch.py once the FiLM half was stripped), the
triplet dataset (basicsr/data/radar_render_triplet_dataset.py, renamed from
paired_image_uint16_render_dataset.py), the data-only config
(Deraining_Holo/Options/DINO_analysis_data.yml, renamed from
Holo_DINOv2_renderDINO_Restormer.yml), and dino_analysis_phases/. Those measure
whether a DINO prior carries usable signal at all -- the question E1 assumed
the answer to. See dino_analysis_phases/DINO_ANALYSIS_DEVLOG.md.

NOTE: this log is otherwise append-only. This entry is a deliberate exception --
418 lines of E1 detail were compressed here rather than left in place. Use
`git log dino_prior -- DEVLOG.md` to read the original.

## Step 27 — CORRECTION: the ray counts were documented backwards (2026-08-09)

User correction: **clean = 1e7 rays, verynoisy = 1e5 rays**. The docs had it the
wrong way round from the very beginning (DEVLOG Step 1 / CONTEXT.md problem
statement said input 1e7, target 1e6). It was also physically backwards -- more
rays means less Monte-Carlo noise, so the CLEAN image must be the HIGH ray count.

Verified empirically before changing anything, PSNR against `clean` over 30 val
images:
    noisy       29.55 +/- 3.21 dB
    verynoisy   13.11 +/- 2.39 dB
i.e. clean > noisy > verynoisy in quality, consistent with a ladder where fewer
rays = more noise. The dataset's three variants (clean / noisy / verynoisy) match
a 1e7 / 1e6 / 1e5 ladder.

`noisy`'s exact count is [UNCONFIRMED] -- the user stated clean and verynoisy
only; 1e6 is inferred from the ladder and is marked as an assumption, not a fact.

SCOPE OF THE ERROR: labels and prose only. Every experiment trained
degraded -> clean using the actual files, so no result, metric or conclusion
changes. Exp 1, Exp 2 and all of E1 are unaffected. What IS affected is anything
written up describing the data, including thesis text.

Corrected in: CONTEXT.md (problem statement, with the reasoning) and the data
example figure, now at
experiment_results/exp2_verynoisy/figures/data_example_verynoisy_vs_clean.png.
The other files corrected at the time (HANDOVER.md, design.md,
render_radar_similarity.py) were removed with E1; see the Steps 20-26 entry.
DEVLOG Step 1's original line is left as written -- this entry is the correction,
since the log is append-only.

## Step 28 — E1 purged from the working branch; repo cleanup (2026-08-10)

WHY. E1 (DINOv2 FiLM guidance) failed three times and is not being continued
(see the Steps 20-26 entry). The DINO *analysis* line continues, and the next
experiment starts from a tree that is not full of a dead one. The E1 code was
not wrong so much as answered a question we had not yet asked: it assumed a DINO
prior carries usable signal for this data. Phases 0/1/2 are what actually test
that assumption, so they stay and everything built on top of the assumption goes.

BRANCHES. Nothing was destroyed in git. `dino_prior` is a complete, working
snapshot of E1 at 00cf082 and was pushed before any deletion. All cleanup
happened on a new branch `dino_e2` cut from it. To read any removed file:
`git show dino_prior:<path>`.

REMOVED (tracked, all recoverable from dino_prior):
  - arch/model: the FiLM half of restormer_dino_arch.py (FiLMHead, _film,
    RestormerDINO, _StubExtractor; 415 -> 162 lines), image_clean_dino_model.py,
    image_clean_render_model.py
  - configs: Holo_DINOv2_{lq,render}DINO_{Restormer,GATE}.yml (3 of 4 deleted,
    the 4th survives as a data-only spec, see KEPT)
  - launchers/tooling: train_holo_chain_{lq,render}DINO.sh, gate_{lq,render}DINO.sh,
    gate_verdict.py, make_gate_configs.py, sanity_check_dino_film.py,
    compute_dino_feat_mean.py
  - one-off scripts, each referenced only by E1 docs also deleted here:
    verify_dino_weights.py, debug_dino_inputs.py, analyze_dino_features.py,
    render_radar_similarity.py
  - docs: HANDOVER.md, exp3_dino_film/{design.md, E1_DINO_report.md}
  - upstream Denoising/ (23 files). Never used here. Motion_Deblurring/ and
    Defocus_Deblurring/ went in Steps 3 and 5; Deraining/ was copied to
    Deraining_Holo/ in Step 5 and is likewise gone. Denoising/ was the last
    upstream task dir still sitting there. Dataset_GaussianDenoising still lives
    in basicsr/data/paired_image_dataset.py and is untouched.

KEPT, because the analysis depends on them and they are reusable, and RENAMED so
the filename states the function rather than the dead experiment:
  restormer_dino_arch.py            -> basicsr/models/archs/dinov2_feature_extractor.py
  paired_image_uint16_render_dataset.py -> basicsr/data/radar_render_triplet_dataset.py
  Holo_DINOv2_renderDINO_Restormer.yml  -> Deraining_Holo/Options/DINO_analysis_data.yml
All three are pure renames (0 insertions / 0 deletions). Suffixes are deliberate:
the extractor drops _arch.py because it registers nothing (basicsr's arch registry
was importing it for no reason; arch modules 2 -> 1), the dataset KEEPS _dataset.py
so the data registry still auto-scans it.

dino_analysis_phases/ (renamed from dino_analysis/) was otherwise left alone.
Rename-aware diff dino_prior..HEAD over the folder: 13 of the 14 tracked files
are 0 insertions / 0 deletions, and the 14th, visualize_dino_spatial_pca.py, is
+11 / -7 -- seven replaced lines (the two basicsr imports, DEFAULT_OPT, the
POOLED_MEANS path, and three provenance strings) plus a four-line explanatory
comment above POOLED_MEANS. Phases 1 and 2 needed no edits at all; they inherit
through this module. No logic changed anywhere.

TWO THINGS THAT NEARLY WENT WRONG, both caught by running the code rather than
reading it:
  1. Deleting Holo_DINOv2_renderDINO_Restormer.yml broke Phase 1 instantly -- it
     is the default --opt for all three analysis scripts, not just a training
     config. It was restored at the same path (later renamed) as a data-only spec
     holding datasets.val plus five dino_* keys. Phase 2 asserts three of those
     values against phase1_metadata.json, so they must not drift.
  2. Deleting exp3_dino_film/dino_feat_mean_*.pt was reported at the time as
     harmless. It was not. report_pooled_mean_incompatibility() reads those two
     vectors and its return value is written to metadata as
     centering.existing_pooled_means_examined; with the files gone the field
     silently became [] where the committed runs recorded two entries. No number
     or assertion was affected, but a rerun would no longer match the record.
     Both tensors were restored byte-identical to
     experiment_results/dino_pooled_means_reference/. They are 3072-d POOLED
     vectors (4 layers x 768, over 300 crops) and are NOT used for centering --
     build_extractor passes feat_mean=None and spatial centering uses the 768-d
     per-layer means in dino_analysis_phases/dino_spatial_layer_means.pt. They
     exist only so the analysis can record WHY the pooled vectors are unusable
     spatially. DO NOT DELETE THEM AGAIN.

DELETED FROM DISK, NOT RECOVERABLE. experiments/ and tb_logger/ are gitignored,
so the E1 commits never touched them and no branch holds them -- dino_prior does
NOT have these:
  experiments/Holo_DINOv2_lqDINO_verynoisy       24G
  experiments/Holo_DINOv2_renderDINO_verynoisy   15G
  experiments/Holo_DINOv2_{lq,render}DINO_GATE   1.8G each
  experiments/Holo_chain_state_{lq,render}DINO{,_v2}, experiments/Holo_gate_state
  tb_logger/Holo_DINOv2_*
Worktree 44G -> 1.6G. Every E1 checkpoint, training state, training log and
TensorBoard curve is gone. The surviving record of E1 is the Steps 20-26 entry
above plus the code on dino_prior. KEPT: Holo_Baseline_Restormer (701M),
Holo_Baseline_Restormer_verynoisy (201M), their chain states, both baseline
tb_logger dirs, compare_noisy_vs_verynoisy.png.

VERIFIED after every commit, not just at the end: Phase 1 (--max-samples 4) and
Phase 2 (--max-samples 6) both exit 0 on CPU against the real DINOv2 checkpoint
(load_state_dict reports "All keys matched"), Phase 2's phase1-metadata
consistency check passes, Phase 1 reproduces Block 6 at +0.6728 centered
1e5<->1e7 -- identical before and after the renames and the disk purge -- and a
fresh Phase 1 run records config_read: Options/DINO_analysis_data.yml. Smoke runs
were always written to a scratch --out-dir so the committed 339-sample outputs
were never overwritten.

KNOWN STALE, left deliberately:
  - README.md keeps 4 links to Denoising/README.md that now dangle (upstream
    boilerplate, not ours to maintain)
  - dino_analysis_phases/DINO_ANALYSIS_DEVLOG.md names the old yml (inside the
    do-not-touch folder)
  - Step 3 above says "Kept: Deraining/, Denoising/" -- true when written; this
    log is append-only
  - committed metadata JSONs under dino_analysis_phases/*/outputs/ record the OLD
    module paths in config_read / pairing_source. That is correct: they are
    provenance of what those runs actually read. Rewriting them would falsify the
    record. Phase 2 compares dino_model / dino_checkpoint / dino_hub_source only.

## Step 29 — Phase 3: the restoration experiments (2026-08-11 → 2026-08-13)

WHY. Phases 0/1/2 measured the DINO prior in representation space (patch-wise
cosine). Nothing before Phase 3 tested whether that signal converts into
restoration quality. This step builds and launches the experiments that do.
It deliberately does NOT resurrect E1-FiLM (Steps 20-26): the fusion here is a
zero-initialized residual projection, and none of the three recorded FiLM
failure modes are reachable from it.

### 29a — Work Order 1: verification before any experiment existed

Three questions, answered without creating a single config or launching a
single training run.

1. **Is the LR scheduler mechanically independent of progressive cropping?**
   Yes. `CosineAnnealingRestartCyclicLR` steps on iteration count alone; the
   progressive block at `basicsr/train.py:241-270` only changes the crop and the
   mini-batch. The 92k restart is therefore inherited for comparability with the
   existing radar recipe, not motivated by a crop transition — and since both
   arms share it, it cannot confound the comparison. Written into the configs as
   a comment so the reason survives the run.

2. **One shared DINO extractor.** `phase3_restoration/scripts/dino_shared.py` is
   now the ONLY sanctioned extraction path for Phase 3 — the layer re-check, the
   centering means, both architectures and the evaluation scripts all import it.
   Verified to reproduce the Phase-1 representation. `extract_blocks` returns a
   **dict keyed by block index**, never a list, because
   `get_intermediate_layers` returns ASCENDING block order regardless of the
   order indices are passed in — that was the Phase-2 bug (commit `4240521`) and
   the data structure now makes it unrepresentable.

3. **Layer re-check at the actual fixed-128 training scale** (val n=339). On the
   pre-registered primary criterion, centered same-scene 1e5<->1e7:
   B3 0.6091, **B6 0.6694**, B9 0.5482, B12 0.2337. **B6 locked (0-indexed 5).**
   DOCUMENTED TENSION, recorded rather than smoothed over: B3 wins the
   same-vs-different *scene advantage* (+0.1966 vs +0.1453), and that gap is
   WIDER at 128 than at 256. This is a criterion conflict, not a measurement
   error. A **B3 run under an identical recipe is pre-registered** so it is
   settled empirically instead of by argument.

FIRST REAL FAILURE OF THE STEP, and a reusable lesson: the initial layer sweep
was run in the background on the login node and died after ~7 min at ~800% CPU
having produced only the means. That is the login-node watchdog (exit 143), not
a code bug. Resubmitted through SLURM, it finished in **64 s**. An interim
"the run completed" claim made before checking was wrong and was corrected at
the time. All non-trivial compute goes through SLURM from here.

### 29b — Work Order 2: E0-Fixed and E1-addition-noisy

Two arms, one permitted difference.

  E0    `Holo_E0_fixed128_baseline` — stock Restormer, fixed 128 crops, no DINO
  E1-N  `Holo_E1_addition_noisy_fixed128_spatial_B6_latent` — identical, plus
        centered spatial DINOv2 B6 from the SAME 1e5 crop, injected at the latent

THE FUSION. `P = nn.Conv2d(768, 384, 1)` with zero weight AND zero bias, added
to `inp_enc_level4` immediately before the 8 latent Transformer blocks:
`guided = inp_enc_level4 + P(D_centered)`. It starts as an exact no-op, yet
`dL/dW = dL/d(guided) (x) D` is non-zero on the first backward, so the branch
grows if it is useful and stays at zero if it is not. No learnable alpha, no
gate, no multiplicative path, no cross-attention, no concatenation, no decoder
injection.

FAIRNESS, verified mechanically rather than asserted:
  - parameter delta = 768x384 + 384 = **295,296** exactly
  - step-0 outputs identical to E0
  - trunk initialisation bit-identical: the stock `__init__` builds the whole
    trunk FIRST, then an **RNG fence** snapshots/restores the CPU and CUDA
    generator states around ViT + `P` construction, so every later draw (data
    order, crop positions, augmentation flags) stays aligned with E0's

THE N.1 HAZARD. `REPO_INVESTIGATION_REPORT.md` section N.1 names the
highest-probability silent bug in this repo: `train.py:241-270` sub-crops and
subsamples ONLY `lq`/`gt` and passes only `{'lq','gt'}` to the model, so any
third aligned tensor is silently misaligned. E1-N avoids it by construction —
the DINO input is derived from `inp_img` INSIDE `forward`, so there is one
stream and nothing to misalign. The smoke test proves it element-wise with
`torch.equal` rather than trusting the paragraph.

CHECKPOINTS. The frozen ViT is kept OUT of every checkpoint (86M params x ~150
checkpoints would be ~60 GB of duplicated frozen weights): `state_dict` strips
the `dino_ext.` prefix and `load_state_dict` re-injects the live weights before
delegating, so `strict=True` remains meaningful for everything that IS trained.
Measured: 105.8 MB, 498 entries, 0 DINO entries.

MEANS. Production centering means computed on the TRAIN split only, >=1000
images (~256k tokens at 128, ~1024k at 256), accumulated float64 and stored
float32 — one position-independent [768] vector per (domain, regime). This is
NOT ImageNet pixel normalisation; it is feature centering, and the arch refuses
a mean whose metadata records the wrong block, input size or split.

### 29c — The gate, and the amendment it needed

Continuous abort gate, evaluated every optimizer step, logging
`dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio` to
`experiments/<name>/dino_stability.csv`.

**E1 aborted at iteration 4**, on `injection_ratio 0.5555 > 0.5`. This was NOT
repaired by changing the design. It was reported, and the user chose a
diagnostic: a throwaway experiment identity with the abort disabled but every
measurement and trigger still evaluated. Result: the ratio **plateaus at 2.3-6.6
and drifts DOWN after iteration 1000** — a startup transient, not a runaway.

The diagnosis is mechanical, not hand-waving. `P` is zero-initialized, so
`injection_ratio` MUST rise from 0; and AdamW's first steps move approximately
`lr` per weight regardless of gradient magnitude, so the early ratio says
nothing about steady-state behaviour. A second probe measured the stock latent
blocks' own residual scales on the E0 diagnostic checkpoint for reference.

GATE AMENDMENT (2026-08-12), the one functional change of the step:
  - `injection_ratio_max` 0.5 -> **10.0**
  - ratio rules enforced from iteration **5000** (still measured and logged from
    iteration 1)
  - **NaN/Inf hard-stop unchanged and NOT windowed** — active from iteration 1
This changes WHEN the gate is valid, not how much injection is tolerated in
steady state.

KNOWN GAP, recorded rather than quietly patched: rule 2 watches the *ratio*, so
**co-inflation is not gated**. E1-noisy went from latent 215 / projected 872 at
1k to latent 2310 / projected 5237 at 41k with the ratio flat. A proposed rule
(abort if either norm exceeds 5x its own 5k reference) is NOT implemented,
because adding it changes an experiment definition and would require a new
identity.

### 29d — E1-addition-render: the source ablation

A third arm where the ONLY change is the tensor DINO receives: the aligned
**render** instead of the noisy 1e5 crop. Restormer still gets 1e5; the target
is still 1e7. The render is DINO input and nothing else.

THE ALIGNMENT PROBLEM, and why the solution needed no edit to `train.py`. The
render must receive the identical crop and geometric augmentation as the LQ/GT
pair — exactly the N.1 hazard above. Rather than adding a third tensor and
teaching `train.py` to crop it, `Dataset_PairedImage_uint16_RenderStacked` packs
the render as **channel 1 of the LQ tensor**, so the existing sub-crop is ONE
slice hitting both channels. `RestormerDinoSpatialRender` then splits
`radar = inp_img[:, 0:1]` for Restormer and `render = inp_img[:, 1:2]` for DINO,
with the global residual taken against the radar.

**Zero edits to `basicsr/train.py`** — which was the point. E0 and E1-noisy
re-read that file at every resume, and both were mid-flight.

NEW MEANS, not reused: the render's DINO statistics differ from the noisy
heatmap's (clean geometry on black vs a noisy radar field). Same method, same
count, train split only:

  1e5_B6_train128_dino224_mean.pt      norm 43.5031
  1e5_B6_eval256_dino448_mean.pt       norm 41.2580
  render_B6_train128_dino224_mean.pt   norm 56.8906
  render_B6_eval256_dino448_mean.pt    norm 61.6951

cosine(1e5, render) = **0.1799** at train128, **0.3288** at eval256 — the two
domains sit in genuinely different places, so centering the render against the
1e5 mean would have been the wrong distribution.

WORTH STATING IN THE THESIS: the render is normally available in this pipeline,
so this arm is a **usable method, not merely an oracle**.

The one-line widening of the parent arch's `dino_source` validation was made
while E1-noisy was in flight; its full 61/61 verification harness was re-run
afterwards to confirm the in-flight arm was unaffected.

### 29e — Operational: chaining, and never overwriting a log

Each arm now has its OWN chain script — `chain_E0.sh`,
`chain_E1_addition_noisy.sh`, `chain_E1_addition_render.sh` — sourcing a shared
`chain_core.sh` so the LOGIC cannot drift between arms while the IDENTITY stays
separate: distinct job names (`p3chain_E0` / `p3chain_E1addN` / `p3chain_E1addR`),
log directories, chain-state directories and TensorBoard trees. Each job queues
its successor with `--dependency=afterany` BEFORE training starts, which is the
only thing that survives the walltime SIGKILL.

Three bugs found and fixed in this machinery, all of which would have destroyed
records rather than merely failed:
  1. SLURM `--output` pointed INTO `experiments/<name>/`, which BasicSR RENAMES
     to `_archived_<timestamp>` on a fresh start — the log would have been
     carried off mid-write. Moved to `results/<EXP>/logs/chain_%j.out`.
  2. `STABILITY_FAILURE.json` was opened with `'w'`, so a later event could
     destroy an earlier record. Now every failure is archived as
     `STABILITY_FAILURE_<rule>_iter<N>_job<J>.json` and the un-suffixed file is
     only a FLAG for the chain driver.
  3. The "don't start a second trainer" guard matched on SLURM job NAME, so it
     could not see a trainer launched by a different script. Replaced with an
     experiment-keyed lock file — placed in `CHAIN_DIR`, never `EXP_DIR`, for
     reason (1).

HARDWARE. An rtx3080 attempt OOMed (job 1773194): Restormer at 128^2 x batch 8
exceeded 9.6 GB with the **stock E0 trunk alone** — not a Phase-3 defect. Arms
were moved to v100, then all three consolidated onto **a100**, so per-iteration
timing is comparable across the whole Phase-3 table. Two claims made during this
triage were wrong and were corrected explicitly at the time: "v100 has 4 free of
16" (tg071 is DRAINED) and "a100 has 0 pending jobs" (`PrivateData=jobs` means
`squeue` shows only my own jobs — it is not evidence about a partition's load).

### 29f — Status at the time of writing (2026-08-13 ~00:30 CEST)

| arm | job | node | iter | best val PSNR |
|---|---|---|---|---|
| E0 | 1774669 | tg092 | 287,000 / 300k | 22.0749 @ 268k |
| E1-addition-noisy | 1774876 | tg097 | 50,000 | 21.0481 @ 24k |
| E1-addition-render | 1774839 | tg095 | 62,000 | **23.2106 @ 60k** |

Iteration-matched validation PSNR:

| iter | E0 | E1-noisy | E1-render |
|---|---|---|---|
| 4,000 | 19.4023 | 19.6836 | 21.2728 |
| 20,000 | 19.2156 | 19.6787 | 22.3753 |
| 40,000 | 20.8640 | 20.4713 | 22.6431 |
| 48,000 | 20.1320 | 20.4536 | 22.9463 |
| 60,000 | 20.7795 | -- | **23.2106** |

Two readings, both stated without softening:

  1. **E1-render is far ahead.** +2.43 dB over E0 at matched 60k, and its 60k
     checkpoint already beats E0's BEST checkpoint at 268k by +1.14 dB — well
     outside the pre-registered "+0.30 dB = meaningful" threshold.
  2. **E1-noisy is not separating from E0.** Ahead at 4k/20k, behind at 40k,
     slightly ahead at 48k. That is oscillation, not signal. On current evidence
     the noisy-source arm tracks the baseline — which, if it holds, is itself a
     result: it would say the gain comes from the render's clean geometry rather
     than from DINO features per se.

CAVEATS THAT TRAVEL WITH THOSE NUMBERS. Mid-training **validation** PSNR, 8-bit
path, single seed, arms at very different maturity (287k vs 50-62k). The
pre-registered decision rule is on **test** PSNR of the best-val checkpoint after
300k. The test split has not been touched. These numbers are not a thesis table.

Stability at last check: noisy ratio ~2.63 (5k reference 4.0398), render ~1.03
(5k reference 1.1656). Both far inside the cap of 10. No gate has fired since
the amendment.

### 29g — Open, and deliberately not started

  - **Global/pooled arm of E1** — pool the B6 patch grid to [B,768,1,1], broadcast
    back, then project; reuse the 1e5 means (centering and pooling commute, since
    the mean is position-independent). Work order issued 2026-08-13. NOT built.
  - co-inflation gate rule (29c) — proposed, not implemented
  - B3 run under an identical recipe — pre-registered
  - concat-then-project fusion variant — pre-registered
  - token-shuffle control for the 0.524 different-scene floor — not run
  - **E0-Fixed over-smoothing re-characterisation** — `scripts/run_evaluate.sh`,
    wired in, must run once E0 finishes. NOT OPTIONAL: the project's motivating
    number (HF-energy ratio **0.216**) and the 22.446/22.405/18.313 dB figures
    were all measured on the OLD progressive baseline. E0-Fixed never trains at
    256 and should be EXPECTED to score lower in absolute PSNR — that is to be
    reported, not explained away. Until this runs, the motivation and the results
    describe two different models.

NOT COMMITTED. The whole Phase-3 body of work is untracked: the three new
`basicsr/` modules, the new dataset, and `dino_analysis_phases/phase3_restoration/`.
Three live runs read those files from disk, and both E1 arms pick up an arch edit
at their NEXT RESUME rather than the next iteration — so an edit lands silently
hours later. `experiments/` and `tb_logger/` remain gitignored, which is exactly
how the E1-FiLM runs became unrecoverable in Step 28.

Session handover for all of the above: **HANDOVER.md** (rewritten this step; the
previous file of that name documented the abandoned FiLM experiment and was
deleted in `6216542`).

## Step 30 — Phase 3 results: the DINO source decides the sign (2026-08-13 → 2026-08-14)

All three original arms reached 300,000 iterations and were evaluated. This is
the entry that records what Phase 3 actually found.

### 30a — The result

Validation split, n=339, full256 protocol, 16-bit evaluation path,
best-validation checkpoint per arm. Selection never read the test split.

| arm | DINO reads | best val @ iter | PSNR | vs E0 | improves |
|---|---|---|---|---|---|
| E0-Fixed | -- | 22.0749 @268k | 22.077 | baseline | -- |
| E1-addition-noisy | 1e5 radar | 21.4678 @128k | **21.469** | **-0.608** | 100/339 (29.5%) |
| E1-addition-render | render | 24.1171 @204k | **24.120** | **+2.043** | **290/339 (85.5%)** |

| metric | E0 | E1-noisy | E1-render |
|---|---|---|---|
| PSNR object mask | 17.978 | 17.479 | 19.893 |
| SSIM whole / mask | 0.783 / 0.569 | 0.762 / 0.547 | 0.818 / 0.647 |
| HF energy ratio | 0.200 | 0.280 | 0.336 |
| Laplacian / Sobel ratio | 0.284 / 0.756 | 0.370 / 0.765 | 0.449 / 0.873 |

E1-render clears the pre-registered "+0.30 dB = meaningful" threshold by nearly
7x, and does it broadly rather than through outliers: median +1.946 dB, best
+9.19, worst regression only -3.72 (against the noisy arm's -10.14).

It also attacks the weakness the whole project was built on. HF energy ratio
0.200 -> 0.336 is a **68% relative improvement** in retained high-frequency
energy, with Laplacian ratio up 58%. The over-smoothing is measurably reduced,
not traded away for PSNR.

### 30b — Why the NEGATIVE arm is the load-bearing one

E1-addition-noisy landing at **-0.608 dB** is what makes the render result
usable. The two arms are the same code with one tensor swapped: identical
parameter count (+295,296 over E0), identical seed, schedule, crop, augmentation,
optimizer, LR, fusion, gate settings and evaluation. They land on opposite sides
of the baseline.

So the gain cannot be attributed to added capacity, nor to "adding DINO
features" generically. It belongs to the DINO **input**.

**State the claim narrowly.** The render is a clean view of the same object and
carries the target's geometry almost directly. The defensible sentence is:
*a clean geometric view of the object, delivered through frozen DINO features,
substantially improves restoration, while the identical mechanism fed the noisy
radar makes it worse.* Not "DINO features help". The render IS normally
available in this pipeline, so this stays a usable method rather than an oracle,
but the source of the advantage must not be overstated.

### 30c — Three findings that a headline number would hide

**1. E1-noisy is SHARPER while being less accurate.** HF ratio 0.280 against the
baseline's 0.200, Laplacian 0.370 against 0.284 — more high-frequency content
recovered, and still 0.61 dB worse. It is not an over-smoothing failure; it adds
structure the target does not contain. Visible directly on val image 4467, where
it synthesises a bright cross with no counterpart in the 1e7 target. This is a
HALLUCINATION failure and should be written up as a different mode from blurring.

**2. Both priors help most where the baseline is worst.** On E0's hardest decile
(34 images): E1-render **+3.07 dB, wins 31/34** — well above its own +2.043
average; E1-noisy **+0.478 dB, wins 20/34** — positive, despite being negative
overall. The noisy prior carries something usable when the observation is nearly
hopeless and interferes on ordinary images.

**3. E1-noisy peaked early.** Best val at 128k (43% of the schedule) then a slow
decline to 21.2154 at 300k, against E0 peaking at 268k and E1-render at 204k.
Early peak plus decline is the signature of a prior the network later has to work
around. Observed, not investigated.

No gate rule fired on any arm for the whole 300k. Both arms trained cleanly; the
noisy arm simply converged to a worse answer, which is the useful kind of
negative result — it is about the prior, not the optimisation.

### 30d — Baseline re-characterisation (the open item from Step 29)

Run on E0-Fixed's best checkpoint. **HF energy ratio 0.200** against **0.216**
for the old progressive baseline -- marginally worse, not better. The premise
holds and the motivation now describes the same model as the results.
**Quote 0.200 from here, never 0.216.**

The Sobel/Laplacian gap is its own finding: gradient ratio 0.756 against
Laplacian ratio 0.284. First-order edges (silhouette, bright structural bars)
survive reconstruction; second-order detail does not. The radial power spectrum
shows it directly -- the prediction tracks the target only below ~0.1 of the
maximum radius, then falls up to an order of magnitude short to Nyquist.

NOT to be misquoted: the crop128 HF ratio of 0.910 has a standard deviation of
0.475 (on a 128 crop the HF band is narrow and often nearly empty). full256 is
the figure of record. And the 16-bit evaluation PSNR is not comparable to the
8-bit training-time val PSNR, despite 22.077 landing near 22.075 by coincidence.

### 30e — The fourth arm

**global-render** (`Holo_global_addition_render_fixed128_B6_latent`), built
2026-08-13: identical to E1-render except the centered DINO grid is pooled to
`[B,768,1,1]` and broadcast back, so every spatial position receives the same
vector. It asks whether the render's contribution is its SPATIAL LAYOUT or just
a global descriptor of "what object is present".

Implementation is a subclass overriding `dino_prior` and nothing else, so
`forward`, the radar/render split, the crop alignment and the checkpoint handling
are inherited unmodified. Verified 61/61: parameter count identical to E1-render
(26,419,348), zero-init identity **max diff 0.0** against both E0 and E1-render,
centering/pooling commutation to 2.4e-06 relative 2.5e-07, spatial variance
driven 7.627 -> exactly 0, and `P(broadcast)` still spatially constant under a
random 1x1 projection.

A 6000-iteration smoke run with the gate ON completed clean. injection_ratio
settled at **0.529** mean over the enforcement window against E1-render's ~1.03,
confirming the registered prediction that pooling would shrink it. Notably the
ORIGINAL 0.5 cap would have aborted this arm within ~10 iterations -- a second,
independent confirmation that the Step-29 gate amendment fixed a real
misdiagnosis.

Launched 2026-08-14, now at **88k/300k**, best val 20.5911 @60k -- below the
baseline so far. If that holds it means the render's value is genuinely spatial,
and pooling to one vector is worse than injecting no prior at all.

### 30f — Two tooling bugs found by running the evaluation

Both in `predict_phase3.py`, both would have silently prevented the render arm
from being evaluated at all:

1. `is_dino` tested `type == 'RestormerDinoSpatial'` by exact string, so the
   `...Render` and `...GlobalRender` subclasses never had `set_dino_mode` called
   and would have run the eval256 images in the train128 regime. Now
   `startswith('RestormerDinoSpatial')`.
2. The model input was hard-coded to `[1,1,H,W]`. The render arms need the
   stacked `[1,2,H,W]` (radar ch0, render ch1). Now detected from
   `dino_source == 'render'` in the config, with the render cropped by the SAME
   manifest window as the radar under crop128.

Verified after the fix by the eval logs themselves: each arm loaded its own
centering mean (`1e5_B6_eval256...` vs `render_B6_eval256...`) in eval256 mode.

### 30g — A log-parsing trap that produced one wrong claim

I reported the two E1 arms as "at 182k, chains stopped" when both had in fact
FINISHED. Two causes compounding: `grep 'iter:'` matches **`total_iter: 300000`**
in the config dump at the head of every training log, and `train_*.log` files do
not sort chronologically under a shell glob, so `tail -1` read an older log.

**Use the checkpoint files as ground truth for progress**
(`ls experiments/<name>/models/`), and anchor any log regex as
`(?<!total_)iter:\s*([\d,]+),`.

### 30h — New tooling

  scripts/make_single_arm_figures.py   4-panel input/prediction/target/error for
                                       ONE arm; cases by score (best/median/worst)
                                       plus fixed representatives
  scripts/make_three_arm_figures.py    E0 / E1-noisy / E1-render side by side on
                                       the SAME images, cases keyed on **E0's**
                                       per-image PSNR so the selection cannot
                                       flatter the guided arms
  scripts/make_smoke6k_global_render.py + run_smoke6k_global_render.sh
  scripts/chain_global_addition_render.sh, smoke_tests_global_render.py

Figures of record: `results/comparisons/three_arm_{harsh,median}_full256_val.png`.

### 30i — Still open

  - **FINAL TEST EVALUATION** -- now unblocked for the three completed arms. The
    test split remains UNTOUCHED so all arms can be scored in one pass under one
    protocol. Not run.
  - global-render to 300k (88k as of this entry)
  - the pooled ablation on the 1e5 source (requested, never implemented)
  - B3 run, concat-then-project variant, token-shuffle control -- all still
    pre-registered and unrun
  - co-inflation gate rule -- still not implemented
  - E1-noisy's early peak at 128k -- unexplained
  - **Phase 3 is still entirely UNTRACKED in git**

## Step 31 — Final test evaluation: the locked split, read once (2026-08-14)

The pre-registered conditions were met before the split was touched: all three
arms had completed 300,000 iterations, and each checkpoint was selected on
**validation PSNR alone** (E0 268k, E1-noisy 128k, E1-render 204k). The test
split had never been read. Jobs 1776745-1776750, v100, three arms x two
protocols.

Note the contrast with the OLD verynoisy baseline (Step 19a), where the split
was carved AFTER training so the test half influenced checkpoint selection.
Phase 3 does not carry that caveat.

### The numbers (test, n=338)

| metric | E0 | E1-noisy | E1-render |
|---|---|---|---|
| **PSNR full256** | 21.873 | **21.296 (-0.577)** | **24.081 (+2.208)** |
| **PSNR crop128** | 19.546 | **19.062 (-0.484)** | **22.259 (+2.713)** |
| PSNR object mask (full256) | 17.599 | 17.210 | 19.673 |
| SSIM whole / mask (full256) | 0.783 / 0.560 | 0.762 / 0.540 | 0.822 / 0.643 |
| HF energy ratio (full256) | 0.218 | 0.275 | 0.328 |
| Laplacian / Sobel (full256) | 0.286 / 0.764 | 0.370 / 0.779 | 0.438 / 0.871 |

Per-image against E0 (full256): E1-render improves **298/338 (88.2%)**, median
+2.159, worst -2.81, best +8.19. E1-noisy improves 111/338 (32.8%), median
-0.517, worst -7.43, best +4.10. On crop128: render 299/338 (88.5%), median
+2.540, best +12.19.

### Verdict against the thresholds frozen before any result existed

  < +0.10 dB   no meaningful improvement
  +0.10..+0.30 marginal / promising
  > +0.30 dB   MEANINGFUL

**E1-addition-render: MEANINGFUL on both protocols** (+2.208 full256, +2.713
crop128), roughly 7-9x the threshold.
**E1-addition-noisy: negative on both** (-0.577, -0.484).

Test CONFIRMED validation (+2.043 / -0.608) rather than overturning it, which is
what the pre-registration existed to make checkable.

**The matched-128 escape clause does NOT fire.** It was registered as: "if E1
improves on matched-128 but not on full-256, that indicates the prior is useful
in-distribution and that full-image scale/context transfer is the limiter -- not
that the prior is useless." E1-render improves on BOTH, and by MORE at crop128.
So there is no scale-transfer limitation to invoke; the prior works in both
regimes, slightly better in the regime it trained in. The clause remains
unused -- worth recording, because it would have been available as an excuse had
the result gone the other way.

### E1-render is the best model this project has produced

  old progressive baseline (Exp 2), test, best-val ckpt   22.405 dB
  E0-Fixed, test, best-val ckpt                           21.873 dB
  E1-addition-render, test, best-val ckpt                 24.081 dB

+1.676 dB over the previous best. And E0-Fixed landing 0.532 dB BELOW the old
baseline is the registered prediction confirmed, not a problem: the README stated
in advance that E0-Fixed should score lower in absolute PSNR because it never
trains at the evaluation resolution. The Phase-3 comparison is internal -- every
arm shares E0-Fixed's recipe exactly -- so the absolute offset does not affect it.

### A framing correction the test numbers forced

Step 30d recorded E0-Fixed at HF ratio **0.200** against the old baseline's
**0.216** and called it "marginally worse". The test evaluation makes the
apples-to-apples comparison available and it is different:

  old progressive baseline (TEST)   0.216
  E0-Fixed (TEST)                   0.218
  E0-Fixed (VAL)                    0.200

The old 0.216 was measured on TEST. Test-to-test the two baselines are
**indistinguishable**; the 0.200-vs-0.216 gap was a val-vs-test artefact, not a
difference between models. The conclusion is unchanged and cleaner: both
baselines over-smooth to the same degree, so the motivating weakness transfers
exactly. **Quote the figure matched to the split of whatever sits beside it.**

Also not to be read: the crop128 HF ratios (0.909 / 0.921 / 0.886) are all near
0.9 with std ~0.47. On a 128 crop the HF band is narrow and often nearly empty,
so the statistic is unstable at that scale -- full256 is the figure of record.
Note this makes E1-render's crop128 HF ratio LOOK lower than E0's; it is noise,
not a reversal, and the Laplacian ratio at the same scale (0.400 vs 0.305) moves
the other way.

### Still open

  - global-render at 90k/300k; it will need its OWN test pass when it finishes.
    Selection is per-arm on validation, so adding it later contaminates nothing.
  - the pooled ablation on the 1e5 source (requested, never implemented)
  - B3 run, concat-then-project variant, token-shuffle control
  - co-inflation gate rule
  - E1-noisy's early peak at 128k -- unexplained
  - **Phase 3 is still entirely UNTRACKED in git**

---

## Step 32 — affm/dinolight evaluated; the F_sa finding; aca-L6-nosa built (2026-08-25 → 2026-08-26)

Three things happened and they are separable. **(a)** affm-render and
dinolight-render finished 300k and were finally evaluated on validation.
**(b)** Verifying dinolight's numbers surfaced a structural observation about
the ACA block that had not been noticed before. **(c)** A new arm,
`aca-L6-nosa`, was built to test it. It is **built and smoke-passed, NOT
submitted.**

### (a) The two live arms, evaluated — VALIDATION ONLY

Checkpoints selected on the 8-bit training-time validation curve, as
pre-registered. The test split was **not** read for either arm.

| arm | best iter | val PSNR (8-bit, selection) | top-5 spread |
|---|---|---|---|
| affm-render | 224,000 | 24.402 | 0.0525 dB |
| dinolight-render | 272,000 | 24.3411 | 0.0369 dB |

Neither peaked at 300k. affm's final is 24.335, BELOW its 224k best; dinolight's
final 24.3069 is 3rd of its top 5. **The earlier "both still climbing at 300k"
reading did not survive parsing the full curve** — that came from reading the
tail of a noisy series, and the top-5 spreads (0.05 and 0.04 dB) say the
selected checkpoint is barely distinguishable from four neighbours.

Then the real evaluation, uint16 path, n=339, jobs 1793530-1793533:

| arm | full256 PSNR | psnr_mask | ssim | ssim_mask | hf |
|---|---|---|---|---|---|
| E0-fixed | 22.077 | 17.978 | 0.7828 | 0.5686 | 0.200 |
| addition-render | 24.120 | 19.893 | 0.8178 | 0.6473 | 0.336 |
| concat-render | 24.143 | 19.963 | 0.8192 | 0.6512 | 0.322 |
| global-render | 20.585 | 17.077 | 0.7341 | 0.5425 | 0.264 |
| **affm-render** | **24.404** | **20.154** | **0.8243** | **0.6569** | 0.315 |
| **dinolight-render** | **24.344** | 20.113 | 0.8235 | 0.6522 | 0.274 |

Paired per-image against addition-render (full256 val, n=339):

| arm | mean delta | 95% CI | better on | wilcoxon p |
|---|---|---|---|---|
| affm-render | **+0.284 dB** | [+0.176, +0.393] | 219/339 (64.6%) | 1.0e-09 |
| dinolight-render | **+0.224 dB** | [+0.105, +0.344] | 198/339 (58.4%) | 1.3e-04 |
| concat-render | +0.023 dB | [-0.072, +0.118] | 181/339 (53.4%) | 0.49 |

concat reproducing as a clean tie is the control that says the method works.

**HOW TO STATE THIS, AND HOW NOT TO.** Both gaps are statistically real — the
CIs exclude zero comfortably. **Neither clears the pre-registered +0.30 dB
"meaningful" bar**: affm at +0.284 falls short and its CI straddles 0.30, so it
is unresolved in both directions. affm-render's registered prediction ("not
>0.10 dB") is **FALSIFIED** — the CI lower bound is +0.176. dinolight's ("not
>0.30 dB") **HOLDS**.

**The capacity confound differs sharply between the two and must not be
averaged over.** affm-render is +298,372 params vs addition-render's +295,296 —
within 1.04%, which is the entire reason AFFM was chosen over a concat. Its
result is NOT attributable to capacity. dinolight-render is +1,352,849, ~4.6x,
so its +0.224 IS capacity confounded.

Cleanest defensible sentence available today: *adding DINO layers {3,6,9,12}
with a per-position softmax fusion, at essentially zero parameter cost, gives
+0.284 dB over the single-layer {6} addition arm on validation.*

Note the ordering: **affm-render, with the SIMPLE fusion, beats dinolight-render,
with the sophisticated one** (+0.284 vs +0.224). That is what motivated (b).

### The verification pass on dinolight, and a metadata trap

Before trusting 24.344 it was re-derived four ways. All four agree:

  - **provenance** — `predict_metadata.json` confirms `net_g_272000.pth`, the
    selected checkpoint, git commit 7873004.
  - **metrics recomputed from the raw PNGs**, independently of
    `masked_metrics.py`: mean 24.344305 vs 24.344305, max per-image deviation
    **7.1e-15 dB** (full) and **exactly 0** (mask).
  - **aggregation** — every summary mean and median matches the per-image CSV
    to 0.0e+00 across all eight metrics, n=339 both sides.
  - **the gate is alive** — `alpha_logit` = -1.8470 -> alpha = **0.1362**, up
    from the 0.1192 it was initialised at.

**THE TRAP, RECORDED SO NOBODY RE-DISCOVERS IT AS A BUG.**
`predict_metadata.json` reports `block_1indexed: 6` and a single mean path
`render_B6_eval256_dino448_mean.pt` for dinolight-render — which reads as if the
evaluation used ONE layer instead of {3,6,9,12}. **It did not.** Those are the
legacy scalar fields written verbatim at `predict_phase3.py:229-231`, and they
do not describe the multi-layer path. The checkpoint carries all eight per-layer
mean buffers (`mu_b3/b6/b9/b12` x `train128/eval256`) plus four `affm.score`
convs, and `predict_phase3.py:90` loads with `strict=True`, so a single-layer
model would have thrown on unexpected keys rather than loading quietly. **This
is a metadata reporting gap, not a computational one** — but it is a live trap
for anyone reading that JSON later, and it applies to every multi-layer arm.

### (b) THE F_sa FINDING — the ACA block contains a redundant MDTA

`DinoAca` computes two attentions and sums them before one output conv:

    guided = project_out(F_sa + alpha * F_ca) + F

`F_ca` is the point of the block: the latent attending to the DINO prior.
**`F_sa` is the latent attending to ITSELF — and that is Restormer's own MDTA,
step for step**: 1x1 + 3x3 depthwise projections, L2-normalise along the token
axis, per-head multiplicative temperature, channel x channel softmax. Compare
`dino_aca.py:_attend` with `restormer_arch.py` `Attention.forward`.

The ACA is applied to `inp_enc_level4`, whose output goes **straight into
`self.latent`** — EIGHT transformer blocks that each already run exactly that
operation. **So F_sa is a ninth MDTA immediately in front of eight more, with
its own weights, trained from scratch.**

Measured on dinolight-render `net_g_272000.pth`:

| piece | params |
|---|---|
| ACA self-attention half (`to_q/k/v`, `temperature_sa`) | **452,742** |
| ACA cross-attention half (`to_*_cross`, `temperature_ca`) | 452,742 |
| shared (norms, `project_out`, `alpha`) | 148,993 |
| **ACA total** | **1,054,477** |
| the 8 latent blocks it feeds — MDTA attention alone | 4,801,600 |

F_sa is **43% of the fusion block** and roughly a third of the arm's entire
+1,352,849 delta over E0.

**What the trained model did with it.** `dino_aca.py:170` already logs
`aca_ca_to_sa_ratio = ||alpha*F_ca|| / ||F_sa||`. Over dinolight's 300k it rose
from 0 (alpha and P are both zero-init, so F_ca starts at exactly 0) to:

    iter 150,000   7.75      iter 300,000   7.94
    mean over the last 100k iterations:  8.157   (n=100 unique points)

The DINO half ends up carrying **~8x the magnitude** of the self half.

**THIS IS A MAGNITUDE, NOT A CAUSAL RESULT, AND THE DISTINCTION IS THE WHOLE
POINT.** A small-norm term can still matter: F_sa and alpha*F_ca are summed and
pushed through ONE shared `project_out`, so F_sa is also the baseline the cross
term is added onto. The ratio **motivates** an ablation; it does not substitute
for one. Do not write "F_sa does nothing" — the defensible sentence is:

> In dinolight-render the self-attention branch converges to ~1/8 the magnitude
> of the gated cross-attention branch, while duplicating an operation the
> trunk's eight latent blocks already perform, at 43% of the fusion block's
> parameter cost.

**THIS IS NOT A CLAIM THAT DINOLight IS IMPLEMENTED WRONGLY.**
dinolight-render reproduces a published block faithfully and must keep doing so;
that is what the arm is for. The redundancy is a property of the published
design on our architecture, which is a finding, not a bug.

It also sharpens the capacity story: dinolight spends ~1.35M parameters for
+0.224 dB while affm-render spends ~298k for +0.284 dB, and a third of
dinolight's extra capacity goes to a branch the model itself down-weights 8:1.

### (c) aca-L6-nosa — BUILT AND SMOKE-PASSED, NOT SUBMITTED

    addition-render   guided = F + P(D)
    aca-L6            guided = project_out(F_sa + alpha * F_ca) + F
    aca-L6-nosa       guided = project_out(       alpha * F_ca) + F

Files: `basicsr/models/archs/dino_aca_nosa.py` (`DinoAcaNoSa`),
`basicsr/models/archs/restormer_aca_l6_nosa_render_arch.py`,
`configs/aca_render_fixed128_L6_nosa_latent.yml`,
`scripts/chain_aca_render_fixed128_L6_nosa_latent.sh`,
`scripts/smoke_tests_aca_nosa.py`.

**`dino_aca.py` WAS NOT TOUCHED, DELIBERATELY.** Three ACA arms were RUNNING
when this was built and a chain script re-reads the arch on the NEXT RESUME,
silently, hours later. `DinoAcaNoSa` SUBCLASSES `DinoAca` and deletes the SA
half after construction, which buys two things:

  1. `_attend`, `_split`, `attn_shape`, both LayerNorms and `project_out` are
     INHERITED, so the cross path is provably the code aca-L6 runs.
  2. The parent draws the SA projections FIRST and the cross projections
     SECOND, so building-then-deleting leaves `to_*_cross` holding the **SAME
     draws aca-L6 gives them**. Verified: max abs diff **0.000e+00**. The two
     arms start from an identical cross branch, which makes this one-factor at
     initialisation and not merely similar.

**A RISK THAT WAS CHECKED, AND MUST BE CHECKED AGAIN FOR ANY NEW ARM.**
`basicsr/models/archs/__init__.py` scans for `*_arch.py` and imports EVERY
match. A new arch file that fails to import would kill the registry import for
**every running arm's next resume**. The scan was re-run explicitly: 13 modules
import, and all five existing arch classes still resolve.

Smoke: `smoke_tests_aca_nosa.py`, **29/29 passed**, CPU, no dataset needed.

  - step-0 output **bit-identical to E0** at BOTH train128 and eval256
    (`0.000e+00`), 494/494 trunk tensors byte-identical
  - **zero** `to_q.` / `to_k.` / `to_v.` / `temperature_sa` keys in
    `state_dict` and in `named_parameters`
  - param delta vs aca-L6 is **exactly -452,742**, and the symmetric difference
    of the two key sets minus the SA keys is **0** — nothing else moved
  - E0 26,124,052 | aca-L6 114,054,305 | **nosa 113,601,563**
  - cross branch identical to aca-L6 (`0.000e+00`), `project_out` still zero,
    alpha 0.11920
  - `attn_shape` (6,64,64) at BOTH 16x16 and 32x32 tokens — the
    scale-invariance property crossattn-render lacked

**THREE COMPARISONS, THREE CAVEATS — NEVER CONFLATE THEM.**

| against | factors | caveat |
|---|---|---|
| **aca-L6** | ONE (F_sa), identical cross init | the comparison it was built for; nosa is SMALLER, so a win is not a capacity artefact |
| addition-render | add vs gated cross-attention | smaller capacity gap than aca-L6's ~4.6x, still NOT parameter-matched |
| dinolight-render | THREE (layer count, AFFM, F_sa) | meaningless as an ablation |

One deliberate asymmetry: `aca_ca_to_sa_ratio` is emitted as **exactly 0.0**
rather than dropped, so the observation series keeps the same shape across the
ladder and a reader greping for it sees "absent" rather than "missing".

**Not run:** the 6k integration smoke. aca-L6's integration run covers the
shared `DinoAca` code path, but the affm precedent is that integration bugs live
where the architectural test cannot reach.

### (d) The AFFM learned layer weights — recorded, because the log is gitignored

affm-render's per-position softmax across {3,6,9,12} (the four weights sum to 1
at each of the 256 positions). Mean/std over the LAST 100 logged points
(iter > 200k). Uniform would be 0.2500.

| depth | mean | std | min | max |
|---|---|---|---|---|
| B3 | 0.2161 | 0.0358 | 0.1453 | 0.2884 |
| B6 | **0.2568** | **0.0155** | 0.2303 | 0.2812 |
| B9 | **0.3261** | 0.0182 | 0.2996 | 0.3678 |
| B12 | 0.2010 | 0.0342 | 0.1359 | 0.2749 |

  - **No depth is ever discarded.** Whole-run minimum across all four is
    **0.1028** (B3). The network keeps all four for all 300k.
  - **The mid-depths are the stable core.** B6 and B9 carry the most weight and
    have roughly HALF the std of B3 and B12. B6's weight is the tightest of the
    four.
  - **DO NOT rank B9 above B6 as a finding.** These weights are
    NON-STATIONARY -- B12 ran 0.3592 at 151k to 0.1729 at 299k. Report a
    windowed mean, state the window, and claim no strict ordering among the two
    wandering depths.

**WHY THIS MATTERS BEYOND THE NUMBERS.** The reflex reading of Step 32(a) is
that the multi-layer result invalidates the Phase-1/2 depth study. It does not:

  1. The project's best TEST-split model, addition-render (24.081, +2.208,
     298/338 improved), uses **B6 alone** -- the depth Phase 1/2 selected. The
     multi-layer gain is +0.284 dB on VALIDATION ONLY and does not clear the
     pre-registered +0.30 bar.
  2. The learned weighting independently places **B6 in the stable core**.

The progression is best-single-depth -> best-combination, not a reversal. See
`THESIS_STORY.md` Q6a for the sentence to write.

### Still open

  - **aca-L6-nosa is not submitted.** One arm, ~37.5 h on a100 = two jobs.
  - aca-L6 / aca-L36 / aca-L6912 all RUNNING as of 2026-08-26, successors
    queued (1794252-1794254). None evaluated.
  - **The test split is unread for affm-render, dinolight-render and the whole
    ACA ladder.** Read it once, for all arms together.
  - addition-render has **no `crop128_val`** cell — only `crop128_test`. Job
    1794421 queued to fill it; until it lands the crop128 val comparison
    cannot include addition-render.
  - priorquery-render: DROPPED at 90k, zero metrics, stale `RUNNING_JOB` lock
    (1785021) that MUST stay so the dead arm cannot auto-resume.
  - HANDOVER.md line 97 still lists global-render as *in flight* 90k/300k. It
    is DONE at 300k and evaluated (20.589, -1.284 vs E0); line 167 is correct
    and line 97 is stale.

---

## Step 33 — The ACA ladder finished, and the operator question is answered (2026-08-29 → 08-31)

**Phase 3 is COMPLETE.** Every arm except priorquery-render is trained and
evaluated on both splits and both protocols.

### What finished

aca-L6, aca-L36 and aca-L6912 all reached **300,000** iterations (jobs
1793518/27/28 timing out at 24 h, resumed by 1794252/53/54, ~38.7 h total each).
No `CHAIN_ABORTED`, no `STABILITY_FAILURE`.

**Nothing selected their checkpoints, because nothing was asked to.**
`chain_core.sh` manages training only — it writes `TRAINING_DONE` and exits. It
never calls `select_best_checkpoint.py`. concat-render got selected
automatically because a deferred dispatcher was submitted with
`--dependency=afterany`; the ACA arms had no such dispatcher attached. Selection
was run by hand on 08-31. **Attach a dispatcher, or plan to run selection
manually — the chain will not do it.**

Related, and worth one line in the methods: **BasicSR has no best-model
tracking at all** (zero hits for `best_metric` / `save_best` / `is_best`).
Validation runs on its own every `val_freq` and logs a PSNR; nothing compares
scores. Also `save_checkpoint_freq` is 2000 while `val_freq` is 4000, so only
~75 of the 151 saved checkpoints per arm are ever scored, and every selected
iteration is a multiple of 4000.

### Selected checkpoints (VALIDATION only, as pre-registered)

| arm | best iter | val PSNR (8-bit) | top-5 spread |
|---|---|---|---|
| addition-render | 204,000 | 24.1171 | 0.0921 |
| aca-L6 | 236,000 | 24.1921 | 0.1177 |
| aca-L36 | 296,000 | 24.2355 | 0.0125 |
| aca-L6912 | 224,000 | 24.2192 | 0.0265 |
| dinolight-render | 272,000 | 24.3411 | 0.0369 |
| affm-render | 224,000 | 24.4020 | 0.0525 |

**This table covers only the six arms selected in THIS step.** The other five
evaluated arms were selected earlier and their iterations are scattered across
Steps 30-32; `concat-render`'s was never written down anywhere. The complete
list, recovered and re-verified in Step 35, is in that step. Use it, not this
table, when a full-study figure is needed.

### THE TEST RESULT — n=338, uint16, jobs 1799708-1799717

| arm | dep | full256 | vs E0 | crop128 | vs E0 | psnr_mask | ssim |
|---|---|---|---|---|---|---|---|
| E0-Fixed | — | 21.873 | — | 19.546 | — | 17.599 | 0.7829 |
| E1-addition-noisy | 1 | 21.296 | -0.577 | 19.062 | -0.484 | 17.210 | 0.7622 |
| global-render | 1 | 20.589 | -1.284 | 19.352 | -0.194 | 16.947 | 0.7327 |
| addition-render | 1 | 24.081 | +2.208 | 22.259 | +2.713 | 19.673 | 0.8220 |
| concat-render | 1 | 24.065 | +2.193 | 22.473 | +2.927 | 19.741 | 0.8227 |
| aca-L6 | 1 | 24.111 | +2.238 | 21.955 | +2.409 | 19.761 | 0.8211 |
| aca-L36 | 2 | 24.196 | +2.323 | 22.128 | +2.583 | 19.895 | 0.8238 |
| aca-L6912 | 3 | 24.071 | +2.199 | 21.870 | +2.324 | 19.648 | 0.8215 |
| dinolight-render | 4 | **24.390** | +2.517 | 22.053 | +2.507 | **19.972** | 0.8276 |
| affm-render | 4 | 24.311 | +2.439 | **22.303** | +2.758 | 19.923 | **0.8279** |

Paired per-image against addition-render:

| arm | test full256 | p | test crop128 | p |
|---|---|---|---|---|
| concat-render | -0.016 | 0.98 | +0.214 | 1.7e-04 |
| **aca-L6** | **+0.030** | **0.22 NULL** | **-0.304** | 1.3e-06 |
| aca-L36 | +0.115 | 0.011 | -0.130 | 0.032 |
| aca-L6912 | -0.010 | 0.88 | -0.389 | 4.3e-08 |
| dinolight-render | +0.309 | 9.9e-08 | -0.206 | 0.0054 |
| affm-render | +0.230 | 4.3e-07 | +0.044 | 0.14 |

### FINDING 6 — THE FUSION OPERATOR BUYS NOTHING

`aca-L6` is ONE FACTOR from addition-render — same depth {6}, same layer, same
mean, same injection point, only the operator differs. **+0.030 dB, p=0.22, not
significant, for ~4.6x the parameters.** Confirmed independently on both
splits: validation +0.075 (p=0.075, n.s.), test +0.030 (p=0.22, n.s.).

This is the arm the whole ACA ladder existed to produce, and it settles the
question dinolight-render could not: **the gain came from the depth count, not
the fusion operator.**

### FINDING 7 — THE ATTENTION ARMS PAY A crop128 PENALTY

**Every ACA arm, dinolight included, is SIGNIFICANTLY WORSE than
addition-render on crop128** while gaining (or not) on full256. affm-render is
the only multi-depth arm that loses nowhere. **Never report full256 alone for
these arms** — it is the single most important reporting rule in the project.

### aca-L6-nosa: BUILT, SMOKE-RUN, AND IT CANNOT TRAIN

The 6k integration smoke (job 1794461, a100, 49 min) returned rc=0 but
`projected_norm`, `injection_ratio` and `aca_injected_norm` were **exactly
0.0000e+00 at every print**, with CA entropy pinned at ln(64) = 4.1589
(uniform) and alpha moving only by weight decay.

**Diagnosis, verified on a real backward pass:**

    gradient reaching project_out:   aca-L6 (with F_sa) 1.754    nosa 0.000

With `P` and `project_out` both zero-initialised, the cross branch outputs zero,
so `project_out`'s input is zero — and a conv's weight gradient is proportional
to its input. `project_out` therefore receives zero gradient forever, and
nothing upstream of it ever learns. **`F_sa` is what breaks that deadlock in the
working arms**, which is the "three-step gradient staircase" the arch docstring
describes.

**This is a finding, not just a bug: F_sa is LOAD-BEARING FOR OPTIMISATION, not
only for representation.** A naive "remove F_sa" ablation is impossible under
double zero-init. The fix — initialise `P` normally, keep `project_out` at zero
— preserves the step-0 identity with E0 but makes the arm differ from aca-L6 in
two ways, so it is no longer a one-factor ablation.

The architectural suite passed 29/29 because it only checks that step 0 equals
E0 — which is true; the arm simply never leaves step 0. **The 6k integration
smoke existed for exactly this and it worked: the arm was never submitted, so
the cost was 49 minutes rather than 38 hours.** Do NOT write "F_sa does
nothing", and do NOT claim dinolight is implemented wrongly.

### The AFFM weight MAPS (job 1794607) — AFFM is genuinely spatial

The training log carries only position-AVERAGED weights. The maps recover the
rest, at 300k, std computed ACROSS POSITIONS:

| depth | mean | std over positions |
|---|---|---|
| B3 | 0.2567 | 0.1590 |
| B6 | 0.2458 | 0.1053 |
| B9 | 0.3001 | 0.1049 |
| B12 | 0.1973 | 0.1126 |

Variation of 0.10-0.16 against means of 0.20-0.30 is **large**: the per-position
choice is real, not a flat global mix. **AFFM is nothing like the pooled
global-render arm** — global pools over SPACE, AFFM sums over LAYERS per
position. They collapse orthogonal axes, which is why they land on opposite
sides of baseline.

### Test-split honesty

The test split was read in **several passes** — the original three arms on
08-14, then concat / crossattn / global, then these five on 08-31. The
pre-registration asked for one pass. **Checkpoint selection never touched
test**: every arm was selected on validation alone, verified for all eleven
evaluated arms. Report it that way; do not claim a single clean read.

### Still open

  - priorquery-render: DROPPED at 90k, zero metrics, stale `RUNNING_JOB` lock
    (1785021) that MUST stay so the dead arm cannot auto-resume.
  - No capacity control for the ACA arms. **aca-L6 partly closes this**: a 4.6x
    arm landing level with addition-render says the extra capacity is roughly
    neutral at this data scale.
  - No seed replication, deliberately (see `THESIS_STORY.md` scope decisions).
    Cross-split replication substitutes: affm +0.284 val / +0.230 test,
    aca-L6 null on both.
  - `HANDOVER.md` line 97 still lists global-render as *in flight* 90k/300k. It
    is DONE at 300k and evaluated. Line 167 is correct; line 97 is stale.

---

## Step 34 — Phase 5: DINO does not see a crop the way it sees the full frame (2026-09-01)

Phase 3 trains on 128 crops (DINO at 224 -> 16x16 tokens) and evaluates on full
256 frames (DINO at 448 -> 32x32). Every arm therefore gets a prior computed in
one context and used in another. Step 33's Finding 7 (every attention arm loses
on crop128) recorded its mechanism as INTERPRETATION, not result. This step
measures the untested half: does the PRIOR ITSELF differ between the regimes?

**It does, substantially.** Scripts in `dino_analysis_phases/phase5_crop_context/`.
Figures and JSON are in its `results/` dir, WHICH IS GITIGNORED -- the numbers
below are the record.

### The interaction DSGIR predicts, confirmed (n=339, render vs clean)

`deg_crop - deg_full`: how much worse degraded-vs-clean agreement is inside a
crop than inside a full frame.

| layer | mean delta | 95% CI | wilcoxon p | worse in crop |
|---|---|---|---|---|
| B3 | -0.0192 | [-0.0201,-0.0184] | 2.8e-57 | 337/339 |
| **B6** | **-0.0020** | [-0.0032,-0.0008] | 3.3e-04 | **193/339** |
| B9 | -0.0180 | [-0.0204,-0.0156] | 6.8e-33 | 269/339 |
| B12 | -0.0284 | [-0.0345,-0.0223] | 1.0e-15 | 225/339 |

Negative and significant at every depth, largest at the deepest -- matching
DSGIR's "particularly pronounced at deep semantic levels".

### DSGIR Fig-10 replica: cosine by crop ratio (all ratios resized to 224)

| layer | Full | 0.8 | 0.5 | 0.2 |
|---|---|---|---|---|
| B1 | 0.9556 | 0.9204 | 0.9031 | 0.9561 |
| B4 | 0.9004 | 0.8187 | 0.7856 | 0.8984 |
| B8 | 0.8254 | 0.7625 | 0.7269 | 0.8182 |
| B12 | 0.5512 | 0.4899 | 0.4085 | 0.5151 |

Monotonic decline Full -> 0.8 -> 0.5 at EVERY layer, and the gap widens with
depth: Full-minus-0.5 is 0.053 at B1 and **0.143 at B12**.

  **THE 0.2 RATIO REVERSES, AND MUST BE EXPLAINED RATHER THAN DROPPED.** 0.2 of
  a 256 frame is 51 px upsampled 4.4x to 224, which smooths the degradation away
  so both images look alike. A small-source-image artifact, not a contradiction.
  Report the 1.0-0.5 range -- which is the range training actually uses -- and
  state the artifact.

### The aligned measurement, on this project's REAL pipeline

full 448 (32x32) vs 128 crop 224 (16x16), token-aligned because 256/32 = 8 px
per token.

| layer | position cosine | border - centre | NN top-1 |
|---|---|---|---|
| B1 | 0.7987 | -0.0639 | 0.123 |
| B3 | 0.7591 | -0.0549 | 0.106 |
| B6 | 0.7081 | -0.0578 | 0.111 |
| B9 | 0.6568 | -0.1373 | 0.114 |
| B12 | 0.6721 | -0.1261 | 0.156 |

  * **BORDER MINUS CENTRE IS NEGATIVE AT EVERY LAYER.** The drift concentrates
    at the crop BORDERS. DSGIR asserts "absence of global contextual support";
    this MEASURES it, spatially. **Goes beyond their paper.**
  * **NN position retrieval 0.045-0.156** against chance 1/1024 = 0.001. About
    100x chance, yet 85-95% of crop tokens still cannot identify their own
    position in the full frame.
  * **Context separability**: a logistic regression on the raw 768-d features
    tells crop from full at **0.97-1.00 image-level**, 0.78-0.99 patch-level,
    and it SURVIVES centring.
  * Raw crop-vs-full cosine is 0.66-0.76 but falls to **0.48-0.55 once the layer
    mean is removed**. Raw cosine is inflated by the shared mean, so the true
    shift is LARGER than it looks, and the arms' regime-specific centring does
    NOT compensate.

### B6 IS THE MOST CROP-ROBUST DEPTH — three independent measurements

  1. the interaction is -0.0020, against -0.018 to -0.028 at every other depth
  2. context separability is 0.859, the LOWEST of the seven layers measured
  3. Phase 1/2 selected B6 for an unrelated reason (cross-source consistency)

**This independently vindicates the Phase-1/2 layer choice**, and it pairs with
the AFFM weights, where B6 carries the tightest weight of the four.

### What this settles, and what it does NOT

It establishes that the PRIOR ITSELF differs between regimes -- previously
untested. **It does NOT fully explain Finding 7 on its own**: every arm receives
the same shifted prior, including the additive arms that show no protocol split.
It makes the existing mechanism more grounded (attention pools statistics across
the grid, so a context-shifted prior plausibly perturbs it more than a plain
addition) but does not replace it.

### DSGIR, now READ rather than second-hand

Deng, Tian, Zhao, Liu, Neurocomputing 696 (2026) 134106. Corrections to what the
repo previously assumed:

  * **Fig. 8 is NOT the crop analysis** -- it is t-SNE of degradation-TYPE
    representations from their DSE module. Fig. 10 is the crop analysis; Fig. 12
    is the KDE of similarity distributions.
  * CSA adapts DINOv2 layers {9,10,11,12} with a **zero-initialised residual
    projection** -- the same device this project uses at `P`.
  * CSA trains with **hybrid preprocessing**: each iteration is, with equal
    probability, a full image resized to 224 OR a random 224 crop. That is how
    they make DINO robust to both regimes -- a cheap idea this project could
    adopt without changing any architecture.
  * Content priors are injected **hierarchically**: z^(12) at the latent stage,
    z^(8)/z^(4)/z^(1) into successive DECODER stages.
  * **Their content prior is a GLOBAL VECTOR** (SGFM emits channel-wise affine
    gamma/beta broadcast spatially), exactly like Perceive-IR's 1x768.

**BOTH reference methods buy multi-level decoder injection by giving up SPATIAL
structure, and on this data a global prior is 1.284 dB BELOW the no-DINO
baseline (global-render).** That is the architectural trade-off to state, and it
is why this project's design diverges from theirs rather than looking like an
oversight.

---

## Step 35 — Checkpoint provenance audit; Phase 5 folded into the chapter; the misalignment control (2026-09-02)

Three separable things. **(a)** A full audit of which checkpoint every arm was
evaluated at, because the record was scattered and one arm's was missing
entirely. **(b)** The Phase-5 crop measurement written into `PHASE3_CHAPTER.md`,
where it had never appeared. **(c)** A render-misalignment dose-response control,
built and verified.

### (a) CHECKPOINT PROVENANCE — the complete list, verified three ways

The selected iterations were spread across Steps 30, 31, 32 and 33 and no single
place held them all. Recovered and cross-checked by three independent routes:
the checkpoint files on disk, the `weights` field of every
`predict_metadata.json`, and a fresh parse of the training logs for the
best-validation point. **All three agree for all eleven evaluated arms.**

| arm | selected iter | val PSNR (8-bit) | ckpts on disk | recorded in DEVLOG before this step |
|---|---|---|---|---|
| E0-fixed | 268,000 | 22.0749 | 151 | Step 30a |
| E1-addition-noisy | 128,000 | 21.4678 | 151 | Step 30a |
| addition-render | 204,000 | 24.1171 | 151 | Step 30a / 33 |
| **concat-render** | **292,000** | **24.1417** | 151 | **NOWHERE — recovered here** |
| global-render | 60,000 | 20.5911 | 151 | Step 30e (as an in-flight figure) |
| aca-L6 | 236,000 | 24.1921 | 151 | Step 33 |
| aca-L36 | 296,000 | 24.2355 | 151 | Step 33 |
| aca-L6912 | 224,000 | 24.2192 | 151 | Step 33 |
| dinolight-render | 272,000 | 24.3411 | 151 | Step 32a / 33 |
| affm-render | 224,000 | 24.4020 | 151 | Step 32a / 33 |
| crossattn-render | 4,000 | 18.5674 | 89 (stopped 178k) | Step 33 — EXCLUDED from the narrative |
| priorquery-render | none | — | 45 (stopped 90k) | never evaluated |

**Every arm used the SAME checkpoint for all four evaluation cells**
(full256/crop128 x val/test), confirmed from the metadata. No cell is quietly
running a different model.

Two things this audit surfaced:

  * **`concat-render` at 292,000 (val 24.1417) was never written down.** Step 33's
    selection table covers only the six arms selected in that step, and concat
    was selected automatically much earlier by its deferred dispatcher, so it
    fell through every record. The number is now here. Note the coincidence trap:
    the OLD verynoisy baseline of Step 19 also peaked at 292k, at 22.4460 — a
    different arm and a different value. Do not conflate them.
  * **`global-render` peaked at 60,000 of 300,000** and never beat it — one fifth
    of the way through. That is the same early-peak-then-decline shape
    `E1-noisy` shows at 128k, and these are precisely the two arms that land
    BELOW baseline. Consistent with Step 30c's reading of a prior the network
    later has to work around. Observed, not investigated.

### (b) PHASE 5 WRITTEN INTO THE CHAPTER

Step 34's measurement existed only in this devlog. `PHASE3_CHAPTER.md` still
described the crop128 mechanism as interpretation at section 7.3, and a grep for
"phase 5" across the chapter and `PROSE_ARGUMENTS.md` returned nothing.

New **section 7.4, "The prior under cropping — a direct measurement"**, placed
immediately after the protocol-split finding it supports: the interaction table,
the Fig-10 crop-ratio replica with the 0.2 artifact stated rather than dropped,
the spatial border-minus-centre result, NN position retrieval, context
separability, the centring caveat, and the three-way B6 vindication. Old 7.4
became 7.5.

**The honest limit is written INTO the section, not left to a reader:** every arm
receives the same shifted prior, including the additive arms that show no
protocol split, so this grounds 7.3's mechanism without closing it.

Also: section 5.3 now points forward to it; section 9 gains the regime-mismatch
limitation (measured, not corrected — DSGIR's hybrid preprocessing is the
obvious remedy and was not applied); section 10 gains a closing paragraph.

**Section 11 was restructured, and this was overdue.** It listed *seed
replication* as priority #2 and the *F_sa ablation* as #3 — both decisions
already taken and recorded in section 9 and section 6.11 of the same document.
As written the chapter recommended, in its final section, work it had already
explained why it would not do. Now split into **11.1 Open** (parameter-matched
control, a middle point on the additive depth curve, DSGIR-style hybrid
crop/full preprocessing, the co-inflation gate) and **11.2 Considered and
declined, with the reason**. Same facts, and they now read as decisions.

### (c) THE RENDER-MISALIGNMENT CONTROL — built and verified

**Why.** Render specificity currently rests on ONE binary point: shuffle the
renders between images and the arm loses 9.094 dB (job 1776802, validation,
n=339). That is a cliff with nothing on it. It shows a completely wrong render is
catastrophic; it says nothing about how accurately the render must be registered,
which is the first question a reader asks of a guided method.

`predict_phase3.py` gains `--render-shift K`, default 0, single code path.
Displaces the render K pixels along x before it reaches DINO. Design choices,
each deliberate:

  * **Zero fill, never `np.roll`.** Wrapping would reintroduce object structure
    on the opposite edge and understate the damage. The renders sit on a black
    background, so zero is also the physically correct fill.
  * **Displaced BEFORE the crop**, so crop128 sees the same misalignment
    full256 does.
  * **Only the DINO input moves.** Radar, target and crop window are untouched,
    so the single factor is render-to-radar registration.
  * Refuses to run on a non-render arm rather than silently doing nothing.
  * `render_shift_px` is written into `predict_metadata.json`.

**Verification before spending GPU time** (job 1800934, 36 s, 6 images):

    shift 0 identical to the arm's RECORDED predictions   6/6   (bit-exact)
    shift 8 differs from shift 0                          6/6

The first check is the load-bearing one: it proves this is the same computation
the arm's own evaluation ran, not a re-implementation that lands nearby. A unit
test on the shift function itself covers zero-fill, absence of wrap, and
shape/dtype preservation.

**Configuration.** addition-render at its selected checkpoint (204,000), because
that is the arm the 9.094 dB shuffle control was measured on, so the endpoint and
the dose-response sit on the same model. **Validation split, n=339** — this
control was not pre-registered, so it does not touch the locked test split, and
validation is what the shuffle control used. Both protocols. k = 0, 1, 2, 4, 8,
16, with k=0 included as the self-check described above.

Sweep submitted as job 1801014.

---

## Step 36 — The misalignment curve, the in-regime test, and a correction (2026-09-02 → 09-03)

Two inference-only experiments on `addition-render` at its selected checkpoint
(204,000), validation split n=339, plus the disk cleanup. **One result in this
step corrects a claim made earlier in the same session — see (c).**

### (a) RENDER MISALIGNMENT — the dose-response (job 1801014)

`--render-shift k` displaces the render along x before it reaches DINO. Zero
fill, not `np.roll`. Radar, target and crop window untouched.

**k=0 reproduces the arm's recorded evaluation exactly** — 24.120 full256 and
22.187 crop128 — so this is the arm's own inference path, not a lookalike.

| shift | full256 | vs aligned | worse on | p | crop128 | vs aligned |
|---|---|---|---|---|---|---|
| 0 | 24.120 | — | — | — | 22.187 | — |
| 1 | 23.920 | -0.200 | 237/339 | 9.7e-17 | 21.931 | -0.256 |
| 2 | 23.355 | -0.765 | 279/339 | 5.9e-41 | 21.386 | -0.801 |
| **4** | 21.914 | **-2.207** | 332/339 | 1.0e-56 | 20.155 | **-2.032** |
| 8 | 19.659 | -4.461 | 337/339 | 2.8e-57 | 18.136 | -4.051 |
| 16 | 17.081 | -7.039 | **339/339** | 2.6e-57 | 15.661 | -6.526 |

**ONE DINO TOKEN = 8 RADAR PIXELS. Read the table in tokens, not pixels.**

  * **Half a token (4 px) erases the whole benefit.** The arm is +2.043 over E0
    on this split; a 4 px shift costs -2.207.
  * **One token (8 px) makes the prior HARMFUL** — 19.659 against E0's 22.077,
    i.e. **2.4 dB BELOW no prior at all**. Same territory as E1-addition-noisy,
    same underlying reason.
  * 1 px, an eighth of a token, is already significant.
  * 16 px reads -7.039 against the mismatched-render control's -9.094: **the
    binary control is the ENDPOINT of this curve.**

Both protocols agree (-2.207 / -2.032 at 4 px), so this is not a scale artifact.
Written up as chapter §7.6.

  LIMIT TO STATE: pure translation on one axis. Rotation, scale error and
  non-rigid misregistration are NOT tested.

### (b) IN-REGIME FULL-FRAME INFERENCE (jobs 1801041, 1801052)

Two protocols added to `predict_phase3.py`, both emitting a full 256 prediction
scored against the REAL 256 target so every row is comparable:

| condition | in regime | full res | PSNR | vs full256 | p |
|---|---|---|---|---|---|
| full256 | no | yes | 24.120 | — | — |
| resize128 | yes | **no** | 11.108 | -13.012 | 2.6e-57 |
| tiled, no overlap | yes | yes | 24.081 | -0.039 | **0.57 NULL** |
| tiled ov64, box | yes | yes | 24.764 | +0.643 | — |
| tiled ov64, ramp | yes | yes | **24.813** | **+0.693** | 2.6e-25 |

**resize128 fails on RESOLUTION, not on regime.** Diagnostic on a 60-image
subset: the raw 1e5 input scores 12.683 against the target; downscale-then-
upscale with NO MODEL scores 9.357; the model output scores 11.252. The resize
destroys 3.3 dB before inference and the model adds +1.9 back. It works and
cannot undo resampling. Do not report this as the model failing.

### (c) THE CORRECTION — "in-regime is worth ~0.7 dB" WAS WRONG

Stated during the session, before the control ran. **It is wrong.**
Non-overlapping tiles are fully in-regime at full resolution and land at
**-0.039 dB, p=0.57 — a clean null.** Being in-regime buys NOTHING here.

The +0.693 comes from OVERLAP, and the confound control separates why. A box
window at the same overlap averages the identical tiles with equal weight
(verified: identical tiles-per-pixel coverage, min 1 max 4), keeping the
self-ensembling and removing the border down-weighting:

    averaging          (box  - no overlap)   +0.682 dB  311/339  p=2.2e-50
    border weighting   (ramp - box)          +0.049 dB  212/339  p=8.4e-06

**~93% of the gain is ordinary prediction averaging**, which would help almost
any model and has nothing to do with the DINO prior. The Phase-5 border
prediction IS confirmed in direction and IS significant, but it is **0.049 dB**.
Report it as a confirmed prediction of modest size. **Do NOT present the tiling
result as a vindication of the crop measurement.**

Consequence for planning: a ~38 h retraining run at the evaluation resolution
can no longer be justified by the crop measurement alone. §11.1 has been
qualified accordingly.

**What still stands:** overlapped tiled inference gives **+0.693 dB for no
retraining and no parameters**, larger than the depth finding, at 9 forward
passes per image instead of 1. An inference recipe, not an architectural result;
chapter §7.7 reports it separately from the arm comparisons.

### (d) DISK CLEANUP — 497 GB → 7.9 GB

`experiments/` held 497 GB. Two thirds was `training_states/` — resume data at
**twice** the size of the models, for arms that have all finished.

Kept per arm: the validation-selected checkpoint, the final 300k, and
`net_g_latest`. crossattn keeps 4k/176k/178k (all three referenced by the
Phase-4 diagnosis outputs); priorquery keeps 90k as the dropped arm's only
record. Both OLD baseline directories untouched.

**All 34 keepers were loaded and md5'd BEFORE any deletion** — the script aborts
without deleting if a keeper is missing or unreadable — and re-verified
byte-identical afterwards, 34/34. Deleted 1,610 checkpoints and 1,634 states.

Untouched: training logs and `dino_stability.csv` (the checkpoint-selection
provenance), and the stale `RUNNING_JOB` locks for crossattn and priorquery.

Manifest committed at `dino_analysis_phases/phase3_restoration/KEPT_CHECKPOINTS.json`,
NOT left in `experiments/`, which is gitignored.

## Step 37 — Wave 1 of the follow-up arms: the additive depth-curve middle points (2026-09-03)

Two new arms submitted, both **config-only** changes of the finished
affm-render {3,6,9,12} arm. Also: a code review of the whole pipeline, a
tracked paired-test script, and a measurement that retires the co-inflation
gate proposal.

### (a) The two arms

The additive operator carries the depth finding (+0.230 test full256, p=4.3e-07)
but had only the endpoints {6} and {3,6,9,12}. The new arms are the nested
middle points, so the four arms trace one curve with one factor moving:

| arm | experiment | layers | params over E0 | job |
|---|---|---|---|---|
| affm-L36 | `Holo_affm_render_fixed128_spatial_L36_latent` | {3,6} | +296,834 | **1802091** |
| affm-L369 | `Holo_affm_render_fixed128_spatial_L369_latent` | {3,6,9} | +297,603 | **1802092** |

Each config differs from `affm_render_fixed128_spatial_L3691_latent.yml` in
three places only: the name, `dino_layers`, and the per-layer mean maps
(unused layers removed). Same class, recipe, seed, gate, injection. Chain
scripts `chain_affm_render_L36.sh` / `_L369.sh` are the affm chain script with
the identity substituted; a100, two chained 24 h jobs expected.

Pre-registered predictions are in each arm's devlog: {3,6} expected at
+0.05..+0.20 vs addition-render; {3,6,9} at +0.15..+0.25, close to the
four-depth arm because B12 carried the smallest learned weight.

**Smoke:** `smoke_tests_affm_subset.py` (new, reads the layer set from the
config) — **34/34 passed for both arms**, CPU: parameter delta exact, trunk
byte-identical to E0, step-0 output identical to E0 (max dev 0.0), softmax
exactly uniform at 1/L, one-step gradient staircase clears at step 2, eval256
path 448 -> 32x32 with no interpolation. No 6k GPU integration run: only the
length of one list changed against an arm that ran 300k iterations through the
same code.

Both jobs were PD (Priority) at submission: every a100 node carried a
"Reboot ASAP" drain flag from 14:10 that day.

### (b) THE CO-INFLATION GATE — measured against the finished arms, and dropped

HANDOVER §5 and chapter §9/§11.1 list an unimplemented rule: abort if
`latent_norm` or `projected_norm` exceeds 5x its own 5k reference, because the
crossattn arm inflated both ~15x while the ratio stayed flat. It was going to be
added to the new arms. Replayed first against every finished arm's
`dino_stability.csv`:

| arm | latent growth vs 5k ref | projected growth |
|---|---|---|
| addition-render | 9.12x | 8.19x |
| affm-render | 9.72x | 6.70x |
| concat-render | 9.07x | 7.40x |
| global-render | 9.99x | 6.11x |
| E1-addition-noisy | 12.53x | 6.21x |
| aca-L6 | 15.77x | 2.28x |
| dinolight-render | 15.86x | 2.03x |

**Every healthy arm, the best model included, grows its latent norm 9-16x over
the run**, peaking around iteration 180k-200k. The 5x rule would have aborted
all of them, and crossattn's ~15x sits inside the healthy range. Norm growth
under this recipe is normal, not a failure signature; the rule cannot
discriminate and is **not adopted**. The new arms keep the gate byte-identical
to every other arm's. Chapter §9 and §11.1 and HANDOVER §5 updated to say so.

### (c) Code review — no result-changing bug found

Read end to end: the stacked dataset, the train.py sub-crop, both model
wrappers, the base and AFFM/ACA/global archs, `predict_phase3.py`, checkpoint
selection, the metric scripts, and the E0-vs-addition config diff.

  * The stacked dataset draws crop and flip in the SAME RNG order as the stock
    dataset (top, left, flag), so data order and augmentation are identical to
    E0's for the same seed. The train.py sub-crop slices both channels together.
    The config diff holds only the intended keys. EMA off, mixup off, clip
    identical.
  * Render loading is identical in training and prediction (BGR channel 0,
    /255). Checkpoints are saved before validation at the same iteration, so the
    selected checkpoint is the validated one.
  * **8-bit selection vs 16-bit reporting: a non-issue, measured.** 67% of clean
    pixels are below one 8-bit level, so this was a real worry. Recomputed on
    every arm's saved full256 validation predictions: |PSNR8 - PSNR16| <= 0.003
    dB for all eight arms, and the ranking is identical. One methods sentence.
  * **The paired-test procedure was untracked.** No script in the repo computed
    the Wilcoxon p-values in Steps 32-33. Recomputed from the tracked per-image
    CSVs, they reproduce exactly (aca-L6 full256 test +0.030 p=0.224; affm
    +0.230 p=4.30e-07; concat -0.016 p=0.978). The procedure is now
    `scripts/paired_compare.py` (mean delta, 5000-resample bootstrap CI, Wilcoxon,
    better-on count; writes `results/comparisons/paired/*.json`).
  * Known and unchanged: cudnn benchmark mode (non-deterministic, equal for all
    arms); the log-parse regex in `select_best_checkpoint.py` also matches
    `total_iter:` in the startup dump, harmless because a training `iter:` line
    always precedes the first validation, and Step 35 cross-checked selection
    three ways.

### Still open (waves 2 and 3, not started)

Gated-render and gated-noisy (a per-position, per-channel gate on the
projected prior; the light test of "the model decides"), post-latent injection,
B3-alone and B9-alone, the offline crop-gap correction measurement, and the
inference-only misalignment extensions.

## Step 38 — global-render re-read: it overfits, and the global vector is only a partial scene ID (2026-09-03)

Prompted by a fair objection: global-render pools the B6 patch tokens, not the
last-layer CLS token the reference methods use, so is it a fair test of a
global prior? Two measurements, no training.

### (a) The memorisation signature, from the existing logs

| arm | train l_pix, last 10k | val PSNR best | val at 300k | drop |
|---|---|---|---|---|
| E0 | 0.0587 | 22.075 @268k | 22.034 | 0.04 |
| addition-render | 0.0307 | 24.117 @204k | 24.008 | 0.11 |
| **global-render** | **0.0444** | **20.591 @60k** | **19.734** | **0.86** |
| addition-noisy | 0.0471 | 21.468 @128k | 21.215 | 0.25 |

global-render fits the TRAINING set 24% better than E0 and generalises 2.3 dB
worse, collapsing 0.86 dB from a 60k peak. That is overfitting, not a prior the
network "works around". The noisy arm shows the same shape, milder.

### (b) Is the global vector a scene fingerprint? (job 1802097, v100, 103 s)

`scripts/global_vector_identifiability.py`: two independent random 128 crops
per training scene, drawn as train.py draws them, DINO at 224; query crop A
against the gallery of every scene's crop B, n = 6,101, chance top-1 = 0.00016.

| vector | top-1 | top-5 | median rank | same-scene cos | hardest other scene |
|---|---|---|---|---|---|
| pooled B3 | 0.089 | 0.147 | 283 | 0.649 | 0.923 |
| **pooled B6** (global-render's prior) | **0.177** | 0.260 | **109** | 0.616 | 0.860 |
| pooled B12 | 0.323 | 0.443 | 12 | 0.584 | 0.735 |
| **CLS B12** (the papers' descriptor) | **0.395** | 0.526 | **4** | 0.632 | 0.735 |

Reading, both halves:

  * The pooled B6 vector of a random crop is **1,000x chance** at identifying
    its scene, but it is NOT a unique key: the true scene sits at median rank
    109 of 6,101, and on average some OTHER scene's crop is closer (0.860) than
    the same scene's other crop (0.616). So "the network memorised scene IDs"
    is too strong. The honest sentence: the broadcast vector carries enough
    scene-specific information to overfit on, and the loss curves show it did.
  * **The CLS token is MORE identifying, not less** — 2.2x the top-1, median
    rank 4. A CLS-broadcast arm would hand the network a sharper fingerprint
    than global-render did. Direction of the prediction: it overfits at least
    as much. It remains a prediction; the arm was not run.

### What changes in the write-up

global-render stays the spatial ablation (same prior, positions removed, one
factor from addition-render). Its sentence changes from "a global prior is
worse than no prior" to: removing the positions turns the benefit into
overfitting (lower train loss, worse validation, 0.86 dB collapse), the
per-image vector is a partial scene identifier (top-1 1,000x chance, median
rank 109), and the literature's CLS descriptor is a sharper identifier still,
so it was not tested and is not expected to help. Chapter §6.5 updated.

## Step 39 — The crop-gap is mostly a learnable transform (2026-09-04)

Follow-up to Step 34, in feature space, no training arm. Question: is the
crop-versus-full DINO drift a SIMPLE transform of the crop features that a tiny
module could undo before the prior reaches the projection?

**Setup** (`phase5_crop_context/crop_gap_correction.py`, job 1802212, v100,
6.5 min, 21 GB RAM; a first attempt, 1802131, ran out of the 23 GB per-GPU cap
after the linear fit and the script was rewritten to stream). Every training
scene: full render -> DINO 448 -> 32x32 tokens (the eval regime); one
token-aligned random 128 crop -> DINO 224 -> 16x16 tokens (the train regime);
target = the 16x16 block of the full grid at the crop position. Crop tokens
centred with the train128 mean, full tokens with the eval256 mean, i.e. exactly
the two tensors an arm receives in its two regimes. Fits on 4,880 scenes,
scores on 1,221 HELD-OUT scenes. Border = 2-token ring.

**A0, the raw gap, all 6,101 scenes** (cosine crop token vs full token at the
same place; wrong-place floor = the token half a block away):

| layer | all | border | centre | wrong-place |
|---|---|---|---|---|
| B3 | 0.483 | 0.420 | 0.532 | -0.116 |
| **B6** | 0.549 | 0.505 | 0.582 | -0.179 |
| B9 | 0.524 | 0.457 | 0.576 | -0.182 |
| B12 | 0.548 | 0.490 | 0.594 | -0.021 |

(Step 34 reported 0.708 for B6 on raw tokens; this is the CENTRED cosine, as
Step 34 also noted centring lowers it to ~0.5. Consistent.)

**Corrections on B6, held-out scenes:**

| candidate | params | cos all | cos border | rel-L2 | R² | own-place top-1 | gap closed |
|---|---|---|---|---|---|---|---|
| A0 nothing | 0 | 0.547 | 0.504 | 0.911 | 0.08 | 0.215 | 0% |
| A1 per-channel affine | 1.5k | 0.633 | 0.596 | 0.750 | 0.42 | 0.276 | 19% |
| **A2 linear 768x768** | 591k | **0.871** | 0.858 | 0.480 | 0.75 | 0.405 | **71.5%** |
| A2p + per-position bias | +197k | 0.872 | 0.860 | 0.478 | 0.76 | 0.406 | 71.7% |
| **A3 A2 + two 3x3 convs** | +3.5M | **0.940** | 0.931 | 0.324 | 0.88 | **0.566** | **86.8%** |

Readings:

  * **The drift is largely a fixed linear transform of the token.** One 1x1
    linear map, fit in closed form, removes 71.5% of the gap on scenes it never
    saw, explains 75% of the target variance, and nearly doubles the chance that
    a corrected token is closest to its OWN position (0.215 -> 0.405). It is not
    pulling everything toward the mean.
  * **Per-channel rescaling is not enough** (19%): the gap is a rotation/mixing
    of channels, not a statistics offset — which is why centring (Step 34) could
    not fix it.
  * **A fixed per-position offset adds nothing** (71.5 -> 71.7%). The border
    drift is not "the same shift at every border token"; it depends on content.
  * **Neighbourhood context gets to 87%.** Two 3x3 convs on top of the linear map
    (they see the zero-padded edge, so border and centre can be treated
    differently) close 86.8%, own-place top-1 0.566. Border closes almost as
    well as centre for every candidate (86.1 vs 87.5%).
  * The held-out conv loss was still falling at epoch 30 (1.05 vs train 0.91),
    so 87% is a floor for this size, not a ceiling.

**Decision rule from the plan (>50% closed on held-out with own-place retrieval
kept): MET.** A one-factor arm is justified: addition-render + the FROZEN
correction applied in the train128 regime only (at eval256 the full frame is
already in the target space). Not built; awaiting the go. Saved for that use:
`results/crop_gap_correction/A2_linear_B6.pt` and `A3_conv_B6.pt` (gitignored
folder; regenerable in 7 minutes from the tracked script).

**Caveat carried forward, unchanged:** Step 36's non-overlap tiling was fully
in-regime and a PSNR null (-0.039, p=0.57), so removing the drift may buy no
PSNR. What this step establishes is a feature-space fact for chapter 7.4: the
regime shift is real, large, and mostly a learnable transform rather than lost
information.

## Step 40 — The additive depth curve is a THRESHOLD, not a ramp (2026-09-07)

Wave 1 (Step 37) finished while the session was away. Both arms trained 300k on
a100 (jobs 1802091/1802092 hit the 24 h walltime and their chained successors
1802220/1802221 completed, 14.3 h and 14.6 h), were selected on validation
alone, and were evaluated on both splits and both protocols (jobs 1805059-66).

### Selection (validation PSNR, 8-bit, as every other arm)

| arm | best iter | val PSNR | top-5 spread | final (300k) |
|---|---|---|---|---|
| affm {3,6} | 220,000 | 24.2583 | 0.053 | 24.216 |
| affm {3,6,9} | 224,000 | 24.3804 | 0.039 | 24.332 |

No stability failure in either run. 151 checkpoints each, 75 validation points.

### THE CURVE, paired against addition-render (B6 alone), n=339 val / 338 test

| depths | val full256 | p | test full256 | p | verdict |
|---|---|---|---|---|---|
| {3,6} | **+0.140** | 6.9e-03 | **-0.012** | 0.96 | **NULL — sign flips across splits** |
| {3,6,9} | **+0.262** | 3.1e-07 | **+0.255** | 2.6e-06 | **replicates** |
| {3,6,9,12} | +0.284 | 1.0e-09 | +0.230 | 4.3e-07 | replicates |

crop128, same comparison: {3,6} +0.106 (p=0.18) val / +0.066 (p=0.19) test;
{3,6,9} +0.127 (p=0.053) val / +0.057 (p=0.32) test. **Neither loses anywhere**,
which extends the Finding-7 property of the additive family to the whole ladder.

### {3,6,9} vs {3,6,9,12} — B12 CONTRIBUTES NOTHING

| cell | delta | p |
|---|---|---|
| val full256 | -0.022 | 0.51 |
| val crop128 | -0.080 | 0.59 |
| test full256 | +0.025 | 0.92 |
| test crop128 | +0.013 | 0.80 |

Four cells, all null, deltas on both sides of zero. **The fourth depth is
inert.**

### THE FINDING, and it is cleaner than the trend we expected

**The depth effect is a THRESHOLD at three depths, not a graded ramp.**

  * ONE extra depth buys NOTHING. {3,6} is +0.140 on validation and **-0.012 on
    test** — on opposite sides of zero, the same signature concat-render showed
    (+0.023 / -0.016). That is what a true null looks like, and it is now the
    second one this measurement has produced, which is worth citing when
    defending the positives.
  * THREE depths gets the WHOLE effect: +0.262 val, +0.255 test, both
    significant, both replicating.
  * The FOURTH depth adds nothing (four null cells).

So the recipe is {3,6,9}: **+0.255 dB on test for +2,307 parameters
(+0.78% over addition-render)**. It matches the four-depth arm at three
quarters of the (already trivial) cost.

**This independently confirms the learned-weight reading of Step 32d.** The
{3,6,9,12} arm gave B12 the smallest and least stable share (0.201 mean, std
0.034); removing B12 entirely costs nothing measurable. Two unrelated methods —
what the network learned to weight, and what an ablation removes — agree.

**PRE-REGISTRATION SCORECARD, reported as required.**

  * {3,6}: predicted +0.05..+0.20 on test. Actual **-0.012. PREDICTION WRONG.**
    The band was set from the four-depth result on the assumption of a graded
    curve; the curve is not graded. Recorded as a miss, not retrofitted.
  * {3,6,9}: predicted +0.15..+0.25 and "close to the four-depth arm". Actual
    +0.255, marginally above the band, and statistically indistinguishable from
    the four-depth arm. Essentially correct.

### What changes in the write-up

THESIS_STORY Q6 and chapter §6.8/§7.2 gain the ladder. The headline sentence
becomes: reading DINO at THREE depths, combined per position and injected by
plain addition, is worth +0.255 dB on the locked test split for +0.78%
parameters; a second depth alone is a null and a fourth adds nothing. The
"middle point on the additive depth curve" item in chapter §11.1 is CLOSED.

## Step 41 — Wave 2 built and submitted: the gate, the injection point, the single depths (2026-09-07)

Five arms, all one-factor changes, all smoke-passed, all submitted. Three need
new code; two are config-only. Predictions are pre-registered in each arm's
devlog, written before submission.

| arm | experiment | the one change | params vs addition-render | job |
|---|---|---|---|---|
| gated-render | `Holo_gated_render_fixed128_B6_latent` | a per-position, per-channel gate on P(D) | +295,296 | 1805344 |
| gated-noisy | `Holo_gated_noisy_fixed128_B6_latent` | the same gate, on the arm whose prior HURTS | +295,296 | 1805345 |
| postlatent-render | `Holo_postlatent_render_fixed128_B6` | prior added AFTER the 8 latent blocks | **0** | 1805346 |
| addition-render B3 | `Holo_addition_render_fixed128_B3_latent` | single depth B3 instead of B6 | **0** | 1805347 |
| addition-render B9 | `Holo_addition_render_fixed128_B9_latent` | single depth B9 instead of B6 | **0** | 1805348 |

### (a) THE GATE — the degree of freedom no arm has varied

`basicsr/models/archs/dino_gate.py`, imported by both gated arms so there is one
implementation:

    g      = sigmoid( G( concat[ F , P(D) ] ) )      [B, 384, h, w], in (0,1)
    guided = F + g * P(D)

`G` is one `Conv2d(768 -> 384, 1x1)`. The switch at each position and channel is
decided from BOTH the network's own feature there and the prior there.

**Why this is not a fourth operator null waiting to happen.** Three arms varied
HOW the prior is mixed (concat, ACA at matched depth, the ACA ladder) and all
were nulls. **None varied HOW MUCH arrives, per position.** AFFM cannot: its
weights sum to 1 by construction, so it selects the depth mix but never the
total. `DinoAca` has a gate, but ONE SCALAR for the whole image. The recurring
objection to plain addition — that it hands the network everything everywhere —
has therefore never actually been tested. It is now.

**Initialisation.** `G` is zero-init, so the gate starts EXACTLY 0.5 at every
position and channel: deterministic and seed-independent, the same property
AFFM's uniform 1/L start has. Step-0 equality with E0 comes from `P`, not from
the gate.

**The staircase runs the OPPOSITE way from `DinoAca`'s, and this is what stops
it deadlocking.** At step 1, d(g·P(D))/dP = g = 0.5, non-zero, so `P` learns
immediately; d(g·P(D))/dG carries a factor P(D) = 0, so `G` gets exactly zero
and learns from step 2. In `aca-L6-nosa` BOTH paths to the loss passed through a
zero-initialised weight, which is why nothing upstream ever moved. Verified on a
real backward, not argued.

### (b) POST-LATENT — the limitation the chapter has carried since §4.5

One line of the forward pass moves. The tensor on either side of the latent
stage has the same shape, so `P` is unchanged and the parameter count is
**EXACTLY addition-render's**. The chapter argues "before" strictly contains
"after" because of the residual path; the code hard-refuses any other point and
§9 lists it as untested. It is now testable. The eight latent blocks hold 55.0%
of the network and under this arm never see the prior.

Config uses `dino_injection: post_latent`. The parent's guard accepts only
`latent`, so the subclass pops the key, validates it, hands the parent the
string it demands, and overwrites the attribute — four lines, confined to that
class, so a YAML typo still cannot silently relocate an injection anywhere else.

Monitoring caveat: `latent_norm` measures the latent OUTPUT here, so this arm's
5,000-iteration gate reference is NOT comparable with the other arms'.

### (c) B3 AND B9 — the oldest open item, and its counterpart

Config-only, same class, same parameter count. **The B3 run was pre-registered
in the Phase-3 README and never run**; the criterion conflict behind it (B3 wins
the same-vs-different-scene advantage, B6 wins raw correspondence) has stood
unresolved since Phase 2. B9 is included because the four-depth AFFM arm gave it
the LARGEST learned share (0.326), the only evidence on record that a different
single depth might beat B6. Neither is an AFFM or an ACA arm: AFFM needs two or
more depths to have anything to choose between, and Finding 6 showed the
attention operator is inert.

### (d) SMOKE — one file, three families, 123 checks

`scripts/smoke_tests_wave2.py` reads the family from the config and checks the
shared contract plus the family-specific part.

    gated-render        30/30      gated-noisy        29/29
    postlatent-render   24/24      addition B3        20/20
    addition B9         20/20

Every arm: step-0 output identical to E0 (max deviation 0.0), trunk
byte-identical to E0 for seed 100, exact parameter delta, DINO frozen with no
gradient, eval256 path 448 -> 32x32 with no feature-grid interpolation.
Gate arms additionally: the map is [B, 384, g, g] (per position AND channel, not
a scalar), every value strictly inside (0,1), exactly 0.5 at init, and the
staircase in the correct direction. Post-latent additionally: the latent stage
receives an unguided input and the prior is added to `self.latent(F)`, both
verified by recomputation.

### Predictions, all registered before submission

  * gated-render: does NOT beat addition-render by >0.10 dB. Every operator
    change on this data has been a null and the render prior is uniformly
    useful, so there is little for a spatial switch to exploit. The gate
    STATISTICS are the real deliverable either way.
  * gated-noisy: recovers at least half the 0.577 dB deficit (lands above
    21.60). If it does not, the source finding gets stronger, not weaker.
  * postlatent: worse than addition-render by >0.30 dB, still well above E0.
  * B3, B9: neither beats B6 by >0.10 dB; the depth COUNT is the lever, not the
    depth CHOICE.

## Step 42 — Wave 2 evaluated: three predictions wrong, and two earlier claims corrected (2026-09-11)

Four of the five wave-2 arms finished 300k (chain successors 1806147, 1806902,
1806903, 1807116), were selected on validation alone, and were evaluated on both
splits and both protocols (jobs 1810388-1810403). B9 is at 298k and is added in
a later step. **This step corrects two things this project has previously
written down — (c) and (d) below. Read them before quoting Step 40 or the B6
narrative.**

### Selection (validation PSNR, 8-bit)

| arm | selected | val PSNR | top-5 spread | final 300k |
|---|---|---|---|---|
| gated-render | 140,000 | 24.2109 | 0.064 | 24.140 |
| gated-noisy | **72,000** | 21.0927 | 0.106 | 20.932 |
| postlatent-render | 292,000 | 24.4318 | 0.014 | 24.426 |
| addition-render B3 | 256,000 | 24.3493 | 0.039 | 24.298 |

gated-noisy peaked at 72k and declined — the same early-peak shape as
addition-noisy (128k) and global-render (60k), the other two arms below E0.

### Test split, n = 338, uint16

| arm | full256 | crop128 | object PSNR | SSIM |
|---|---|---|---|---|
| E0-fixed | 21.873 | 19.546 | 17.599 | 0.7829 |
| addition-noisy | 21.296 | 19.062 | 17.210 | 0.7622 |
| **gated-noisy** | **21.162** | **18.781** | 17.131 | 0.7636 |
| addition-render (B6) | 24.081 | 22.259 | 19.673 | 0.8220 |
| **gated-render** | 24.123 | **22.007** | 19.811 | 0.8126 |
| **addition-render B3** | **24.309** | 22.253 | **19.976** | 0.8270 |
| **postlatent-render** | **24.387** | 22.267 | 19.891 | 0.8270 |
| affm {3,6,9} (reference, Step 40) | 24.336 | 22.316 | 19.951 | 0.8261 |

### Paired against each arm's own reference (Wilcoxon, bootstrap 95% CI)

| comparison | val full256 | test full256 | val crop128 | test crop128 |
|---|---|---|---|---|
| gated-render − addition-render | +0.094 (p=0.056) | +0.042 (p=0.15) | **−0.277** (4.1e-05) | **−0.252** (1.6e-05) |
| gated-noisy − addition-noisy | **−0.375** (4.3e-08) | **−0.134** (0.016) | −0.260 (0.009) | −0.281 (0.0012) |
| postlatent − addition-render | **+0.313** (2.1e-07) | **+0.306** (1.8e-08) | +0.088 (0.88) | +0.008 (0.55) |
| B3 alone − addition-render | **+0.231** (1.1e-06) | **+0.228** (9.7e-06) | +0.135 (0.36) | −0.006 (0.71) |

And against the best additive depth arm, affm {3,6,9}:

| comparison | val full256 | test full256 | test crop128 |
|---|---|---|---|
| postlatent − affm {3,6,9} | +0.051 (p=0.46) | +0.051 (p=0.57) | −0.049 (0.51) |
| B3 alone − affm {3,6,9} | −0.032 (p=0.69) | −0.027 (p=0.65) | −0.063 (0.10) |
| B3 alone − affm {3,6} | +0.091 (p=0.023) | **+0.240** (2.7e-06) | −0.072 (0.13) |

### (a) The gate: a null on the full frame, a penalty on the crop

gated-render is +0.042 on test (n.s.) and +0.094 on validation (n.s.): **selection
buys nothing on the full frame**, extending Finding 6 from "how the prior is
mixed" to "how much of it arrives". It **loses significantly on crop128 on both
splits** (−0.252, −0.277). It is the first additively-injected arm to pay the
protocol penalty that until now separated the attention family from the additive
one. Observation, not mechanism: the gate, like the attention blocks, makes the
injection depend multiplicatively on the latent feature F; plain addition and
the AFFM arms do not. (concat also reads F, but linearly, and does not pay it.)

**What the gate did, measured on six validation frames at the selected
checkpoint, full256:** it is binary — 57% of position-channel values below 0.1,
40% above 0.9 — and genuinely spatial: EVERY one of the 384 channels is on at
some positions and off at others within the same frame (median per-channel
spatial std 0.44), and the pattern moves between frames. No channel is shut
everywhere. It lets through about 68% of the prior's magnitude. The training-log
statistics agree (frac_closed 0.55-0.59 from 20k on). So the network did use a
per-position switch — and it still bought nothing.

### (b) The gate does NOT rescue a bad prior. The source finding is stronger.

gated-noisy is **worse than addition-noisy** on both splits and both protocols
(test −0.134, val −0.375) and **0.711 dB below E0** on test. The gate did not
close on the noisy prior: 50% of values below 0.1, 47% above 0.9, about 71% of
the prior's magnitude let through — slightly MORE than the render gate. The
explanation "the network is forced to take a bad prior everywhere and cannot
ignore it" is therefore **not supported**: given a switch, it did not switch the
prior off. The damage is in what the prior contains, or in the optimisation
trajectory it induces (early peak at 72k, then decline), not in the architecture
lacking an off-switch.

### (c) CORRECTION — the injection point matters, in the opposite direction from §4.5

postlatent-render — the prior added AFTER the eight latent blocks, identical
parameter count — is **+0.306 dB on test (p=1.8e-08) and +0.313 on validation
(p=2.1e-07)** over addition-render, and loses nowhere (crop +0.008, +0.088).
Chapter §4.5 argued that injecting before the latent stage "strictly contains"
what injecting after would provide. **On this data that argument is empirically
wrong.** Injecting where only the decoder sees the prior is better, at zero
parameter cost, by a margin comparable to the best result in the study.

### (d) CORRECTION — the depth-ladder reading of Step 40 does not survive

Step 40 concluded that the depth effect is "a THRESHOLD at three depths" and that
depth COUNT is the lever. **Two measurements now contradict it.**

  * **B3 ALONE matches the three-depth arm**: 24.309 against 24.336, −0.027,
    p=0.65, and −0.032 on validation. A single depth reaches the "three-depth"
    tier.
  * **The two-depth null has a mechanism, and it is not "one extra depth is not
    enough".** affm {3,6} learned to put **0.847 of its weight on B6 and 0.153 on
    B3** (last 100k iterations, std 0.04). It converged onto the WEAKER of its two
    depths. B3 alone beats that arm by +0.240 on test (p=2.7e-06).

Learned AFFM weights, last 100k iterations, for the record:

| arm | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| {3,6} | 0.153 | **0.847** | — | — |
| {3,6,9} | 0.237 | 0.282 | **0.481** | — |
| {3,6,9,12} | 0.216 | 0.257 | **0.326** | 0.201 |

What the data now supports, stated as narrowly as it deserves:

  * **addition-render at B6 is the LOW point of the render-guided arms, not the
    reference ceiling.** Several different single changes lift it into a tier
    around +0.23 to +0.31 dB on test — B3 instead of B6, three depths via AFFM,
    injection after the latent stage, and dinolight — and **the members of that
    tier are statistically indistinguishable from one another.**
  * **Depth COUNT is not established as the lever.** The four-depth and
    three-depth gains are consistent with "adding depths moves the prior away
    from B6", and B3 alone does the same.
  * **AFFM's learned weighting is not a reliable depth selector.** In the one
    arm where the choice was between a better and a worse depth, it chose the
    worse one.
  * **The B6 lock was suboptimal for restoration.** THESIS_STORY Q6a lists three
    "vindications" of B6 — cross-source feature consistency (Phase 1/2), the
    tightest learned AFFM weight, and crop robustness (Phase 5). **None of the
    three is restoration quality**, and on restoration quality B3 beats B6 by
    +0.228 on test, replicated on validation (+0.231). Phase 2's own
    interpretation had called the same-vs-different-scene advantage — which B3
    wins — "the criterion to trust". It was the right one.

Single-seed caveat, applied evenly: every claim above rests on one seed per arm.
The positive ones replicate across the two independent splits at p < 1e-5 on
test; the differences WITHIN the +0.25 tier are ~0.05 dB and not significant,
and are not claimed.

### Pre-registration scorecard

| arm | prediction | result | verdict |
|---|---|---|---|
| gated-render | does not beat addition-render by >0.10 | +0.042 (n.s.) | **correct** |
| gated-noisy | recovers ≥ half the deficit, lands > 21.60 | 21.162, worse than addition-noisy | **WRONG** |
| postlatent-render | worse than addition-render by > 0.30 | **better** by +0.306 | **WRONG, sign reversed** |
| addition-render B3 | within 0.10 of B6 | +0.228 above B6 | **WRONG** |
| addition-render B9 | within 0.10 of B6 | pending | — |

Three of four wrong, recorded as misses. Each was reasoned from the project's
own earlier conclusions — the forced-injection explanation, §4.5's residual-path
argument, and the three-way B6 vindication — and each of those conclusions is
what this step corrects.

### What changes in the write-up (NOT done here — the framing is the author's call)

THESIS_STORY and PHASE3_CHAPTER carry correction banners pointing here. The
following need rewriting before they are quoted: the "one sentence" and "ending"
of THESIS_STORY; Q6 and Q6a; chapter §4.5 (injection point), §6.8 and §7.2a (the
depth reading), §7.5, §9 ("one injection point" is no longer a limitation), §10
(conclusion). Findings 1 (source), 2 (spatial), 6 (operator, now including the
gate) and 7 (protocol split, now with a first additive-family member) stand.

## Step 43 — B9 alone: the last wave-2 arm, and what the AFFM weights were NOT telling us (2026-09-11)

B9 finished 300k (chain count 2, no stability failure), selected on validation
at **236,000 (val 24.2425, top-5 spread 0.064)**, evaluated on both splits and
protocols (jobs 1810420-1810423).

**Test, n = 338:** full256 **24.140**, crop128 22.176, object PSNR 19.843,
SSIM 0.8236. Validation full256 24.247.

| B9 alone minus | val full256 | test full256 | test crop128 |
|---|---|---|---|
| B6 (addition-render) | +0.127 (p=0.035) | +0.059 (p=0.095) | −0.083 (0.25) |
| B3 alone | −0.104 (p=0.30) | **−0.168 (p=0.032)** | −0.077 (0.64) |
| affm {3,6,9} | −0.136 (p=0.079) | **−0.195 (p=5.7e-04)** | −0.140 (0.11) |

**Reading.** B9 over B6 does not replicate: significant on validation, not on
test. Report it as unresolved, not as a gain. B3 beats B9 on test. So the
single-depth ordering on restoration is **B3 first; B9 and B6 not separable.**
Pre-registered prediction (within 0.10 of B6): +0.059 on test — **correct**
(val +0.127 is marginally outside).

**What this does to the AFFM-weight evidence, stated plainly.** The learned AFFM
weights gave B9 the LARGEST share in both the three- and four-depth arms (0.481,
0.326) and B3 the smallest or near-smallest (0.237, 0.216; 0.153 in {3,6}). On
restoration quality as a single depth, **B3 is the best and B9 is no better than
B6. The learned weights anti-predict which single depth restores best.** They
describe what the fused prior leans on inside a softmax mixture, not which depth
is most useful on its own, and should not be cited as evidence about depth
quality — which is how THESIS_STORY Q6a used them ("B6 has the tightest
weight").

**Which earlier feature-space criterion picked the winner.** Phase 1/2 ranked
depths two ways at the training scale (WO1 Task 1.3). Raw same-scene
correspondence: B6 > B3 > B9 — wrong about the winner. Same-vs-different-scene
ADVANTAGE: **B3 (+0.197) > B6 (+0.145) > B9 (+0.125) — right about the winner**,
with the B6/B9 order inside restoration noise. Phase 2's interpretation had
called the advantage "the one to trust". It was.

**Scorecard, wave 2 complete:** gated-render correct, gated-noisy wrong,
postlatent wrong (sign reversed), B3 wrong, B9 correct. Two of five.

## Step 44 — Wave 3 scoped down, three claims of mine corrected, and the stacking arm submitted (2026-09-11)

Wave 3 was proposed as five arms and **narrowed to a sequence in a ChatGPT-assisted design review, relayed and accepted by the author**.
Their scoping and their corrections are adopted in full and recorded here
because two of the corrections are errors in my own reasoning, one of which is
contradicted by this repo's own notes.

### (a) THREE CORRECTIONS TO WHAT I PROPOSED

**1. "The reference papers give up the spatial grid to inject at multiple
stages" is WRONG.** It holds for Perceive-IR (prior is 1x768) and DSGIR (SGFM
emits channel-wise affine broadcast spatially) — the two papers Step 34 was
discussing when it wrote "BOTH reference methods". **DINOLight is also in the
reference set, keeps SPATIAL features (H/14 x W/14 x 768), and injects at every
scale**, and this repo already recorded that: the dinolight-render devlog lists
"injection points: every scale" for them against "one" for us. I generalised a
correctly-scoped sentence to a paper it does not cover.

  CONSEQUENCES. A hierarchical spatial injection arm is **not broadly novel** —
  it is closer to what DINOLight actually does than our single-point
  `dinolight-render` arm is. The contribution would be its **controlled
  adaptation and evaluation on radar**, not the idea. And the claim in
  THESIS_STORY (~line 250, ~523) and PROSE_ARGUMENTS (~line 86) that "no paper in
  the reference set does what these attention arms do" is too broad for the same
  reason; the narrow claim that survives is the one about SPATIAL-TOKEN
  cross-attention. A correction banner now says so.

**2. "A global prior is fatal on this data" is too broad.** What was measured is
that OUR pooled-B6 broadcast integration underperforms (global-render, −1.284 dB,
and Step 38 shows it overfits). That does not invalidate other global
conditioning designs, and it should not be used to dismiss them.

**3. Token repetition IS nearest-neighbour upsampling.** Calling it "exact
coverage, not interpolation" is misleading. It is piecewise-constant
upsampling; it preserves token boundaries, which is why it is still the right
first choice, but **the project's "never interpolate the feature grid" condition
would NOT still hold** for any decoder arm that uses it. If such an arm is built,
the mapping is documented as a new, named mapping with its own justification —
not as a continuation of the original rule. Matching spatial ratios across the
two regimes also does not by itself guarantee crop/full feature consistency.

### (b) AN OBSERVATION THAT MAY EXPLAIN THE WHOLE +0.25 TIER

Raised in the ChatGPT-assisted review from post-latent's numbers and checked across the tier.
Against addition-render, on test:

| arm | full256 | crop128 |
|---|---|---|
| postlatent-render | **+0.306** | +0.008 |
| addition-render B3 | **+0.228** | −0.006 |
| affm {3,6,9} | **+0.255** | +0.057 (n.s.) |
| affm {3,6,9,12} | **+0.230** | +0.044 (n.s.) |

**Every member of the tier gains ONLY on the out-of-regime protocol and is flat
in the regime it trained in.** That is a shared signature, and it is more
consistent with these arms addressing one common weakness — plausibly robustness
to the train-to-eval scale shift measured in Phase 5 — than with four
independent improvements to restoration quality.

  STATED AS AN OBSERVATION, NOT A FINDING. The two protocols score different
  targets, so the magnitudes are not directly comparable; what is comparable is
  the sign and significance of paired deltas within a protocol, and those are
  what show the pattern. Post-latent's +0.306/+0.008 in particular **motivates
  investigating placement and context; it does NOT establish that the prior's
  benefit occurs exclusively in the decoder**, which is how I first read it.

### (c) THE AGREED WAVE-3 SEQUENCE

1. **postlatent-B3 — SUBMITTED, job 1810432.** The stacking test.
2. **B6 at decoder level 3 only** — the next location comparison, with a smaller
   adapter, **holding depth at B6 so the location is not confounded**.
3. **Either dual injection or hierarchical injection**, chosen once (2) is known,
   depending on whether the open question is interaction or multi-scale delivery.
4. **post-latent + AFFM three-depth — DEFERRED.** B3 alone weakens the case for
   depth mixing.

**Dual injection is downgraded from prerequisite to supporting experiment.** My
claim that it "cleanly separates" late-arrival from unguided-latent does not
hold: it also changes capacity, total guidance strength and the optimisation
path, so if it loses to post-latent alone that is *consistent with* early
guidance being unhelpful but does not prove the latent blocks must stay
unguided. Its projections would be INDEPENDENT (two separate `P`), which is what
the quoted +295,296 assumed; a shared projection is a different arm.

Also deferred, both by decision: applying the Step-39 feature-gap correction
inside restoration (the correction itself is already measured; using it is a
separate experiment), and disk cleanup, which is to be decided separately from
the scientific plan and only against an exact deletion manifest.

### (d) THE ARM THAT WENT OUT

`Holo_postlatent_render_fixed128_B3`, config-only from the post-latent class and
the existing B3 means, **parameter count identical to addition-render's**, smoke
24/24, submitted as job 1810432. It is **NOT a one-factor arm** against
addition-render and will not be reported as one; its references are its two
constituents.

**The reporting rule for a null is fixed in advance**, in the ChatGPT review's
words: if it does not beat both constituents, write **"the improvements did not
combine under this training recipe"** — NOT "the same ceiling", and no statement
about a performance bound. One combined run cannot establish a limit; it can
only fail to show addition. "Config-only" lowered implementation risk, not
experimental risk. Full pre-registration in the arm's devlog.

## Step 45 — The ACA question, attacked from both ends: checkpoint interventions and a later location (2026-09-12)

Scoped in a ChatGPT-assisted design review, accepted by the author, around three questions: is the trained model actually
using cross-attention, was ACA placed at the wrong point, and does explicit
cross-attention add anything over an equally sized self-attention block. Their
reading of the implementation is correct and is the premise of all of it:
`DinoAca` is `project_out(F_sa + alpha*F_ca) + F`, **self-attention AND gated
cross-attention behind one shared output projection**, so every "ACA versus
addition" number this project has ever reported tests the WHOLE BLOCK, never the
cross path on its own.

### (a) CHECKPOINT INTERVENTIONS — built and verified, job 1810441 (v100, inference only)

Three conditions on the SAME trained aca-L6 weights (validation-selected
checkpoint 236,000), validation split, both protocols:

| condition | what changes | question |
|---|---|---|
| `none` | nothing | reference, and a bit-exact SELF-CHECK |
| `no_cross` | `alpha * F_ca` forced to zero, `F_sa` untouched | does the trained model depend on the cross path? |
| `uniform_cross` | the cross channel-attention matrix replaced by a uniform one, its values `V'` untouched | does the LEARNED mixing matter, given the prior still gets through? |

`uniform_cross` keeps the row sums at 1, as a softmax's are. **That does NOT
preserve the output magnitude** (corrected 2026-09-14, Step 48): averaging the
value channels changes the norm, and on the trained checkpoint the uniform cross
term measured 1.153x the as-trained one (below). What it removes is the learned
selectivity of the mixing; it remains distinct from `no_cross`, which removes
the term entirely.

**Implementation.** `dino_aca.py` gains an inference-only `aca_intervention`
attribute (default `'none'`) and `_attend` gains a `uniform` flag;
`predict_phase3.py` gains `--aca-intervention`, which REFUSES on an arm with no
ACA block rather than silently doing nothing, and records the condition in
`predict_metadata.json`. A guard raises if an intervention is set while the
module is in `train()` mode — training with one would be a different experiment
needing its own identity.

**Verified before the job was submitted, because this edits a file five finished
arms depend on:**

  * the full ACA smoke suite still passes **31/31** with the default, and the
    default path re-runs bit-exactly, so `'none'` is a strict no-op;
  * `no_cross` drives `alpha*F_ca` to **exactly** zero;
  * `uniform_cross` leaves a non-zero cross term, `||uniform|| / ||as-trained||
    = 1.1532`;
  * both change the output (max abs 8.1e-02 and 5.0e-02 on a real checkpoint);
  * the train-mode guard raises.

The job additionally runs a **bit-exact self-check inside the job, on the same
device**: the `none` predictions are md5-compared against the arm's recorded
validation predictions, and the sweep aborts if they differ. A CPU rerun would
differ in the last bits for reasons unrelated to the intervention, which is why
the check is not done here.

**INTERPRETATION RULES, fixed in advance.** Disabling the cross path hurts but
uniform mixing does not → the prior information matters and the specific learned
mixing may contribute little. Both hurt → consistent with the model using both
the prior and its mixing. Disabling the cross path HELPS → the branch may be
counterproductive under that evaluation context. **In every case these are
interventions on a trained model: they establish DEPENDENCE, not how a model
trained without the component would perform.** That sentence goes next to the
table wherever it is quoted.

### (b) aca-L6-postlatent — SUBMITTED, job 1810442

The SAME `DinoAca` block, imported not copied, moved after the latent stage.
Identical parameter count to aca-L6 (the tensor either side of the latent stage
has the same shape). One factor moves. Smoke **31/31** including the scale check.

It fills the empty cell of a 2x2 that has so far only ever been varied one axis
at a time (test full256):

| fusion \ location | before latent | after latent |
|---|---|---|
| addition, B6 | 24.081 | **24.387** |
| ACA, B6 | 24.111 | **this arm** |

**THE DECISIVE COMPARISON IS AGAINST postlatent-render, AT THE SAME LOCATION —
not against the weaker before-latent addition reference.** Comparing it to
addition-render would let a location gain masquerade as an operator gain, which
is precisely what the 2x2 exists to prevent.

Registered prediction: it does not beat postlatent-render by more than 0.10 dB
and still loses on crop128. Outcomes and what each licenses are in the arm's
devlog. Note the standing confound if it does win: this arm is ~4.6x
addition-render's added parameters.

### (c) DEFERRED, and conditional — the matched refinement pair

If the question survives (a) and (b), the next design is to PRESERVE the
successful additive path and let attention refine it: `U = F + P(D)`, then at the
same post-latent location compare **cross-attention** (queries from `U`, keys and
values from the projected prior) against a **matched self-attention control**
(queries, keys and values all from `U`), at equal widths, heads, projection
structure, parameter count and training budget. That pair is what would isolate
"specific value from the cross connection" from "more capacity spent on the
combined features" — which no existing arm can do, because `DinoAca` does not
preserve a direct additive path. Two full runs; not started. The extra output
projection would be zero-initialised, and because the additive path lets `P`
learn immediately it would not reproduce the `aca-L6-nosa` double-zero deadlock
— but gradient flow would still be verified on a real backward before any run.

### (d) SEQUENCING PRESERVED

`postlatent-B3` (job 1810432) continues untouched; it changes depth under
additive guidance and answers none of the ACA questions. The agreed wave-3
location sequence — decoder level 3 at B6, then either dual or hierarchical
injection — is unchanged and still sits behind its result.

## Step 46 — What the trained ACA block actually uses: strong dependence on the cross path, a smaller cost from uniform mixing (2026-09-12)

Job 1810441, v100, inference only. aca-L6 at its validation-selected checkpoint
(236,000), validation split, n=339, both protocols.

**SELF-CHECK PASSED, and it is load-bearing.** The `none` condition produced
predictions **bit-identical to the arm's recorded validation predictions,
339/339 on BOTH protocols**, and its metric reproduces the recorded mean to four
decimals (full256 24.1949, crop128 22.1040). This is the arm's own computation,
not a re-implementation that lands nearby.

### The numbers

**full256 / val, n = 339**

| condition | PSNR | delta | 95% CI | worse on | wilcoxon p |
|---|---|---|---|---|---|
| none (as trained) | 24.1949 | — | — | — | — |
| **no_cross** | **17.6790** | **−6.5158** | [−6.808, −6.209] | 337/339 | 2.8e−57 |
| **uniform_cross** | 24.0486 | **−0.1462** | [−0.186, −0.107] | 234/339 | 3.5e−15 |

**crop128 / val, n = 339**

| condition | PSNR | delta | 95% CI | worse on | wilcoxon p |
|---|---|---|---|---|---|
| none (as trained) | 22.1040 | — | — | — | — |
| **no_cross** | **16.3381** | **−5.7659** | [−6.122, −5.416] | 331/339 | 1.0e−55 |
| **uniform_cross** | 22.0305 | **−0.0735** | [−0.128, −0.018] | 194/339 | 2.8e−03 |

### (a) Question 1 — is the trained model using cross-attention? YES, overwhelmingly

Zeroing `alpha * F_ca` costs **6.52 dB** on full256 and 5.77 on crop128, and hurts
**337 of 339** images. The branch is emphatically not vestigial: `alpha` could
have decayed toward zero over 236,000 iterations and the model could have fallen
back on plain self-attention, which is the clean negative outcome `dino_aca.py`
was written to permit. It did not.

**TWO LIMITS ON HOW FAR THAT NUMBER GOES, and they matter.**

  * **It is not "cross-attention is worth 6.5 dB over addition".** In this arm the
    cross branch is the ONLY route by which the DINO prior reaches the network,
    so `no_cross` is closer to "delete the prior at inference" than to an
    operator comparison. For scale: it lands at 17.679, which is **4.40 dB BELOW
    the no-DINO baseline E0** (22.077 on this split) and 3.48 dB below it on
    crop128. A model trained without a prior reaches 22.077; a model trained WITH
    one and then robbed of it at inference reaches 17.679. Those are different
    quantities and the second is not a performance claim about anything.
  * `F_sa` and `project_out` were trained with the cross term present, so removing
    it also shifts the input distribution `project_out` sees. The measurement
    establishes **dependence**, not the performance of a model trained without
    the component.

### (b) Question 2 — does the LEARNED channel mixing matter? MEASURABLY, and much less than the branch

Replacing the cross attention matrix with a uniform one — keeping the learned
value projections `V'`, so the prior still passes through (row sums stay 1, but
that does NOT preserve magnitude: the term measured 1.153x the trained one) —
costs **0.146 dB** on full256 and 0.074 on
crop128. Both are statistically significant (p = 3.5e−15, 2.8e−03) and both are
small.

**The supported statement, and only this:** the trained model depends strongly
on the cross branch, and making that branch's channel mixing uniform causes a
smaller, measurable degradation than removing the branch.

**The two drops are NOT to be divided into a percentage contribution.** PSNR
differences are log-ratios of MSE, and interventions on a trained model do not
decompose its performance into independent parts; `no_cross` also leaves the
model outside anything it was trained on (below E0), so its drop is not "the
value" of the branch. An earlier version of this paragraph did divide them
("about 2%"); that figure is WITHDRAWN (Step 47). Nor does this bound what a
differently placed or differently trained ACA could contribute.

### (c) Why this coheres with Finding 6 rather than contradicting it

aca-L6 beat addition-render by +0.030 dB, p = 0.22 — a null, at 4.6x the ADDED
parameters (about 4% more TOTAL trainable parameters; see Step 47). The
interventions are consistent with that: the trained model leans heavily on the
branch that delivers the prior, and less on how the branch mixes it, and
delivering the prior is also what plain addition does. That is a consistency,
not an explanation, and it does not show that the mixing is worthless in a
model trained differently.

  This is consistent with, and does not prove, the operator null. The
  interventions describe one trained model. What settles the operator question
  at the better location is `aca-L6-postlatent` (job 1810442) against
  `postlatent-render`, and what would isolate the cross connection itself is the
  deferred matched refinement pair.

### Pre-registered interpretation, applied

The rule set in the ChatGPT-assisted review was: "disabling cross-attention hurts, but uniform mixing
does not → prior information matters; the specific learned mixing may contribute
little." **That is close to the case observed, with one refinement: uniform mixing
does hurt, significantly, and by much less than removing the branch.** Reported as
"small but non-zero", not as "does not matter".

## Step 47 — Provenance and precision corrections to Steps 44-46, and the rules for closing the study (2026-09-14)

A ChatGPT-assisted review of Steps 44-46, relayed by the author, found errors in
my records. Each is corrected IN PLACE in those steps (so a reader quoting them
does not propagate the error) and recorded here.

### (a) PROVENANCE — the reviews were ChatGPT's, not the supervisor's

Steps 44, 45 and 46 attributed five things to "the supervisor": the narrowing of
wave 3 to a sequence, the tier observation, the null-reporting rule, the scoping
of the ACA questions, and the intervention interpretation rule. **All five came
from ChatGPT-assisted design reviews that the author relayed and accepted.** The
wording now says so. **No record in this repository implies supervisor approval
of any wave-3 or ACA-branch design.** The separate, older statement that the
supervisor specified a DINO prior (THESIS_STORY, "SCOPE DECISIONS") comes from
the author's own handover prompt and is the author's claim; it is unchanged.

### (b) WITHDRAWN — "the learned mixing is about 2% of the cross path's value"

Step 46 divided the uniform-mixing drop (0.146 dB) by the no-cross drop
(6.516 dB). That is not a contribution percentage: PSNR differences are
log-ratios of MSE, interventions on a trained model do not decompose performance
into independent parts, and the no-cross condition leaves the model outside
anything it was trained on (4.40 dB below E0). **The supported statement is only:
the trained model depends strongly on the cross branch, and uniform channel
mixing causes a smaller, measurable degradation than removing it.** It places no
upper limit on what a differently placed or differently trained ACA could
contribute. The same division appears in the message of commit e001bc9, which
cannot be amended; this step supersedes it.

### (c) PARAMETER RATIOS MUST NAME THEIR DENOMINATOR — a project-wide fix

"4.6x the parameters" is an ADDED-parameter ratio. Against the 26,124,052-
parameter trunk, in TOTAL trainable parameters (frozen DINO excluded from both):

| arm | total trainable | vs addition-render |
|---|---|---|
| addition-render | 26,419,348 | — |
| aca-L6 | 27,473,825 | **+3.99%** total (4.57x added) |
| dinolight-render | 27,476,901 | +4.00% total |
| multi-level addition (proposed) | 26,640,820 | +0.84% total |
| multi-level ACA (proposed) | 28,034,587 | +6.11% total; **+5.2% over multi-level addition** (3.70x added) |

The phrase occurs 12 times in THESIS_STORY, 6 in PHASE3_CHAPTER, 2 in HANDOVER
and 2 in the published results grid, always without a denominator. Each must say
"added parameters", and the capacity argument should be restated with both
figures: the attention arms roughly quadruple what the prior branch adds, which
is about 4% of the network. That weakens nothing in Finding 6 — a null at +4%
total capacity is still a null — but "4.6x the parameters" overstates the
network-size difference by two orders of magnitude and must not be quoted. Runtime
and memory cannot be read from either ratio. A banner line in THESIS_STORY now
flags it for the rewrite.

### (d) THE STACKING VERDICT — precise wording, and the pre-registration left untouched

Three of four cells were in when I wrote, in conversation, that the record would
read "the improvements did not combine". That was premature and overstrong: the
crop-test cell was still queued, and a test point estimate of **+0.094 dB**
(p = 0.022, 95% CI [−0.004, +0.190]) is a positive observation just 0.006 dB
under a registered +0.10 threshold, not evidence of no combination. When all four
cells are in, the verdict will separate:

  * the OBSERVED additional gain (the paired mean, per split and protocol);
  * whether it is RELIABLE (the paired interval, and cross-split replication —
    the project's standing substitute for seeds; validation read +0.002);
  * whether it EXCEEDS the registered practical threshold.

If warranted the wording is **"the combination did not demonstrate an additional
improvement exceeding the registered threshold."** The arm's pre-registration is
NOT edited — changing a registered rule after seeing results would defeat its
purpose — so this refinement is recorded here, and the verdict step will cite it.

### (e) BEFORE ANY FINAL MULTI-LEVEL RUN IS FUNDED

The proposed final pair (multi-level addition and multi-level ACA at the same
three B6 locations) is not approved; it is to be decided once, after the ACA
post-latent evaluation completes. If funded, two things are fixed first:

  * **"Clear improvement" is defined in advance** — the primary metric, the
    practical threshold, the paired-uncertainty requirement, cross-split
    replication, and how a full-frame gain with a crop128 penalty is reported.
  * **Every injection site is monitored**, not one: non-finite checks at all three
    sites, and each site's own norms and ratio recorded. The post-latent ratio
    rule stays the one the gate enforces, so it remains comparable with the
    single-site arm, but one monitored site cannot certify the other two.

The expectation to register, as a hypothesis and not a mechanism claim: limited
incremental benefit, because the decoder stages receive the same prior again;
repeated delivery can still change how easily those stages use it, which is
exactly what the pair would test. And a loss for single-location ACA would
LOWER the expectation for multi-level ACA without ANSWERING it — the two arms ask
different questions.

## Step 48 — The two FINAL experiments: implemented, verified, integration-smoked (2026-09-14)

**Authorisation.** The author authorised exactly two final training runs — a
matched pair of fusion operators across two injection layouts — to close the
architecture-training study. The brief was drafted in a ChatGPT-assisted design
discussion and accepted by the author. No further training arms follow.

|  | post-latent only | post-latent + decoder 3 + decoder 2 |
|---|---|---|
| addition, B6 | postlatent-render (done) | **A: `Holo_multilevel_addition_render_fixed128_B6`** |
| ACA, B6 | aca-L6-postlatent (evaluation queued) | **B: `Holo_multilevel_aca_render_fixed128_B6`** |

B's submission is explicitly NOT conditional on how aca-L6-postlatent scores.

### State check, before anything was built

  * **Evaluations of the two preceding arms:** 3 of 8 cells complete; 5
    (postlatent-B3 crop128/test and all four aca-L6-postlatent cells) queued on
    v100 as jobs 1812371-1812375. They own their evaluations and were NOT
    duplicated; a passive watcher follows them. No partial result is used to
    choose anything.
  * **No equivalent work exists** under any name: no multi-level or
    decoder-injection experiment directory, config or architecture.
  * **Disk:** `/home/woody` at 735.9 GB of a 1000 GB soft quota (1500 hard).
    The two arms will add roughly 45-47 GB each, to about 830 GB. No deletion is
    needed, and none was done.

### Implementation — isolated, no shared implementation touched

  * `basicsr/models/archs/dino_multilevel.py` — the layout, the
    nearest-neighbour expansion (shape-checked, refuses to broadcast), and the
    per-site monitoring. Not an `*_arch.py`, so outside the registry.
  * `restormer_multilevel_addition_render_arch.py` — subclasses
    `RestormerPostLatentRender`; adds `P_dec3` 768->192 and `P_dec2` 768->96,
    zero weight and bias, in an RNG fence.
  * `restormer_multilevel_aca_render_arch.py` — subclasses
    `RestormerAcaL6PostLatentRender`; adds the same two projections and two
    independent `DinoAca` instances at widths 192 and 96 (6 heads, alpha logit
    −2, zero-init `project_out`, trunk bias and LayerNorm conventions), in an RNG
    fence. `DinoAca` is imported, not copied.
  * Configs, chain scripts and pre-registration devlogs for both, derived from
    the two single-site references by changing only name, class and
    `dino_injection: 'post_latent+dec3+dec2'`.

**The layout.** The centred B6 render prior is extracted ONCE per input and
reused. Site pl after the latent blocks at the native grid; site d3 at the input
of decoder level 3 and site d2 at the input of decoder level 2, both AFTER skip
concatenation and channel reduction. No pre-latent and no decoder-level-1
injection. **The decoder sites receive the prior by NEAREST-NEIGHBOUR
upsampling** (2x, 4x) of each stage's projection applied at the native grid — a
new mapping, documented as such; the original no-resizing condition does not
hold there, and matching expansion factors across regimes does not prove
crop/full invariance.

**Monitoring.** The stability gate keeps its three keys at the post-latent site,
exactly as each single-site reference measures them. All three sites are
recorded separately — feature norm, update norm, ratio, finiteness (plus
||alpha·F_ca|| for ACA) — both as logged observations and inside
`last_dino_stats`, so any stability record names the site. Any non-finite value
at a site propagates to the loss, where the unwindowed NaN rule stops the run.
No new ratio threshold was invented for the decoder sites.

### Verification — `smoke_tests_multilevel.py`, real data, CPU

CPU because a100 TF32 has previously broken exact step-0 equality spuriously.

| check | A (addition) | B (ACA) |
|---|---|---|
| result | **42/42** | **48/48** |
| added trainable | +516,768 (expected) | +1,910,535 (expected) |
| total trainable | 26,640,820 | 28,034,587 |
| frozen DINO, separate | 86,580,480 | 86,580,480 |
| trunk vs E0 (seed 100) | 494/494 identical | 494/494 identical |
| step-0 output vs E0, real crops | max dev 0.0 | max dev 0.0 |
| DINO extractions per forward | 1 (train128), 1 (eval256) | 1, 1 |
| shapes train128 pl / d3 / d2 | 384x16² / 192x32² / 96x64² | same |
| shapes eval256 pl / d3 / d2 | 384x32² / 192x64² / 96x128² | same |
| decoder mapping | exact NN replication, on non-zero projections | same |

**Learning, on a real multi-step backward with the recipe's AdamW:**

  * A: every projection has non-zero gradient on the FIRST backward (P 9.3e-03,
    P_dec3 5.2e-02, P_dec2 3.2e-01). No staircase.
  * B, the three-step staircase, verified at EACH stage separately:

| stage | step 1: only project_out | step 2: P and feature path | step 3 |
|---|---|---|---|
| pl (384) | project_out 5.0e-05, P = alpha = 0 | P 5.7e-04 | every group live |
| d3 (192) | project_out 5.3e-04, P = alpha = 0 | P 1.3e-03 | every group live |
| d2 (96) | project_out 3.4e-03, P = alpha = 0 | P 1.3e-02 | every group live |

No dead branch at any stage; the `aca-L6-nosa` double-zero deadlock does not
recur, because the self-attention path keeps each `project_out` live from step 1.

### Integration smoke — submitted

Bounded real-data runs through `basicsr/train.py`, 2,000 iterations, throwaway
`SMOKE2K_*` identities, validation on all 339 full frames at 1k and 2k, a100:
jobs **1812529** (A) and **1812530** (B). The gate's ratio rules start at 5,000
by design, so in 2,000 iterations they are measured, not enforced; the NaN rule
is active from iteration 1.

**The two training chains are NOT yet submitted.** Per the brief they wait for
the integration smokes and for the five queued evaluation cells.

### A correction made while implementing

The brief's interpretation rules state that uniform attention does **not**
preserve output magnitude. Steps 45 and 46 said it did, contradicting their own
measurement (the uniform cross term was 1.153x the trained one). Both are
corrected in place. **The same inaccurate sentence remains in a code comment in
`basicsr/models/archs/dino_aca.py`** (in `_attend`, the `uniform` branch). It is
left unedited because the brief forbids changing shared implementations used by
existing experiments; it is a comment with no behavioural effect, and is flagged
here for a later authorised fix.

## Step 49 — Both integration smokes pass; submission gated on the queued evaluations (2026-09-14)

Bounded real-data runs through `basicsr/train.py`, 2,000 iterations each,
throwaway `SMOKE2K_*` identities, a100:

| | A: multi-level addition (1812529) | B: multi-level ACA (1812530) |
|---|---|---|
| exit | rc 0, 17m41s | rc 0, 18m07s |
| in-loop eval256 validation, 339 frames | ran at 1k and 2k | ran at 1k and 2k |
| stability failures | none | none |
| checkpoint at 2,000 | written | written |
| every site finite, every logged step | yes | yes |
| post-latent gate ratio at 500 / 1k / 1.5k / 2k | 0.93 / 0.69 / 0.75 / 0.79 | 0.35 / 0.26 / 0.21 / 0.19 |
| site ratios at 2k, pl / d3 / d2 | 0.79 / 1.06 / 1.26 | 0.23 / 0.48 / 0.62 |
| ACA alpha at 2k, pl / d3 / d2 | — | 0.121 / 0.121 / 0.123 (init 0.119) |

**Observations only, no conclusions.** The additive arm's update at the two
decoder sites already exceeds the stage feature in norm by 2k iterations (1.06
and 1.26), rising with resolution; the post-latent site sits near the level the
single-site additive arm reached (~1.0). This is recorded because the gate's
ratio rule only watches the post-latent site; the decoder sites have no rule and
none is invented, but their trajectories will be reported. The ACA gates have
barely left their initial value in 2k iterations, as expected.

The validation PSNR at 1k and 2k (about 20.1 and 20.6 for both) is a smoke
artefact of a 2k-iteration run and is not to be quoted or compared with anything.

Throughput is ~0.45 s/iteration in the logs, consistent with the ~39 h of the
earlier single-site arms: two chained 24 h jobs per arm.

**Submission.** The brief requires the preceding evaluations to finish first.
All five remaining cells (1812371-1812375) are still queued on v100. A persistent
watcher submits both chains automatically once those jobs leave the queue, and
ONLY if all five report `EVAL DONE`; if any does not, it submits nothing and
reports which. It also refuses if either chain's state directory already exists.

## Step 50 — The two preceding arms, fully evaluated: stacking is sub-additive, and ACA loses to addition at the post-latent location (2026-09-14)

All eight cells of the two wave-3 arms are in (jobs 1812368-1812375, every one
`EVAL DONE`). Both arms are 2x2 designs with an existing arm in each other cell,
so each is analysed with `final_matched_pair.py` as a 2x2 rather than as loose
pairwise tests. Outputs: `results/comparisons/stacking_2x2.json` and
`results/comparisons/aca_location_2x2.json`.

### (a) postlatent-B3 — depth x location

Test full256 cells: B6-before 24.081 · B6-after 24.387 · B3-before 24.309 ·
**B3-after 24.480**.

| comparison | test full256 | 95% CI | p | val full256 | test crop128 |
|---|---|---|---|---|---|
| B3-after − B6-after (the better constituent) | **+0.094** | [−0.004, +0.190] | 0.022 | +0.002 (p=0.79) | −0.055 |
| B3-after − B3-before (the other constituent) | +0.172 | [+0.057, +0.284] | 0.0065 | +0.084 (n.s.) | −0.041 |
| interaction (B3 location gain − B6 location gain) | **−0.134** | [−0.257, −0.003] | 0.024 | **−0.229** (p=2.6e-4) | −0.049 |

**Verdict, with the three fields kept separate** (Step 47):

  * OBSERVED against the better constituent: **+0.094 dB**.
  * RELIABLE: **no** — the interval includes zero and validation reads +0.002.
  * EXCEEDS the +0.10 threshold: **no**.

**Wording of record: "the combination did not demonstrate an additional
improvement exceeding the registered threshold."** Not "did not combine", and no
statement about a limit. The arm's pre-registered prediction (it would not beat
post-latent B6 by more than 0.10 dB) held.

**What IS reliable is the interaction, and it is negative on both splits.**
Moving the injection after the latent stage is worth +0.306 at B6 but only
+0.172 at B3: the two changes are SUB-ADDITIVE. Their gains overlap.

B3-after's 24.480 is the highest single test number in the study, and it is
**not a new best**: it is not reliably separable from post-latent B6. No crop128
trade-off anywhere in this 2x2.

### (b) aca-L6-postlatent — operator x location

Test full256 cells: addition-before 24.081 · **addition-after 24.387** ·
ACA-before 24.111 · ACA-after 24.139.

| comparison | test full256 | 95% CI | p | val full256 | test crop128 |
|---|---|---|---|---|---|
| ACA-after − ACA-before (location, for ACA) | +0.028 | [−0.088, +0.140] | 0.74 | +0.101 (p=0.07) | +0.044 |
| **ACA-after − addition-after (operator, same location)** | **−0.247** | [−0.356, −0.142] | **2.8e−05** | **−0.138** (p=0.027) | **−0.268** |
| interaction (ACA location gain − addition location gain) | **−0.277** | [−0.414, −0.141] | 1.6e−06 | **−0.213** (p=1.3e−3) | +0.036 |

**Reading.**

  * Moving the ACA block after the latent stage does **not reliably help it**
    (+0.028 on test, n.s.).
  * **At the post-latent location, the ACA block is significantly WORSE than plain
    addition** — −0.247 dB on test, replicated on validation (−0.138), and worse
    on crop128 as well (−0.268). This is the decisive comparison the arm was built
    for, and it is reliable on every criterion.
  * The interaction is significant and replicated: addition gains 0.28 dB more
    from the later location than ACA does.

**Capacity, with its denominator.** ACA-after has 4.57x addition-after's ADDED
parameters but only **+3.99% of TOTAL trainable parameters**. A loss at slightly
more capacity cannot be attributed to having too little.

**What this does to Finding 6.** At the pre-latent location the operator
comparison was a null (+0.030, p=0.22). At the post-latent location it is a
significant, replicated deficit for ACA. So the operator's effect DEPENDS ON
LOCATION, and where it is measurable, addition wins. The claim concerns the
tested ACA BLOCK (self-attention + gated channel cross-attention + shared output
projection), not cross-attention in isolation.

The arm's pre-registered prediction (it would not beat post-latent addition by
more than 0.10 dB and would still lose on crop128) held, and the outcome is
stronger than predicted: it loses on the full frame as well.

**This does not change the final pair.** Experiment B's submission was specified
as not conditional on this result; a single-location loss lowers the expectation
for multi-level ACA without answering whether repeated access changes it.

### Submission record — the two final chains

The gate set in Step 49 was met: all five queued cells reported `EVAL DONE`, and
neither final experiment had a pre-existing chain state or experiment
directory. The persistent watcher then submitted both chains on a100 at
2026-09-14 14:36:23:

| experiment | first job | successor |
|---|---|---|
| A `Holo_multilevel_addition_render_fixed128_B6` | **1812561** | 1812563 |
| B `Holo_multilevel_aca_render_fixed128_B6` | **1812562** | 1812564 |

Both started immediately. Expected about 39 h each across two chained jobs.
Afterwards, per the brief: validation-only checkpoint selection, both protocols
on validation and test, the four-cell comparison with A-D, and a final
recommendation. No further training arm follows.

## Step 51 — A radar-only residual refiner on the frozen E0: a full-frame gain that comes from darkening, and no recovery of weak structure (2026-09-15)

**Outside the DINO study.** One isolated curiosity experiment,
`Holo_E0_frozen_noisy_output_residual_refiner`, proposed in a **ChatGPT-assisted
discussion** and authorised by the author as one implementation plus one
training run, no sweep. It does not touch the two final chains (1812561/2),
which keep running. Code, pre-registration, fixed cases and smoke record:
`dino_analysis_phases/refiner_e0/` (commits bc0bf7c, 2c1c36e, ac9902d, all
before any result).

### What was run

`Y = Y0 + Refiner([X, Y0])`: X the noisy 1e5 frame, Y0 the output of the
**frozen** E0-Fixed baseline (checkpoint 268,000 from its recorded validation
selection, md5-checked). No render, no DINO, no target-derived input; E0 never
trained. The refiner is a fixed small U-Net (2→16→32→64, bilinear decoder,
zero-initialised signed 1x1 output), **118,129 trainable parameters = 0.45% of
E0's 26,124,052** (denominator: E0's total trainable parameters). Full 256x256
frames throughout; E0's raw float32 full-frame outputs cached before clamping.
L1 on the unbounded output, AdamW 1e-4 cosine to 1e-6, batch 16, grad clip 1.0,
float32 with TF32 off, seed 100, validation every 500 updates on all 339 frames,
selection on validation full256 PSNR (uint16 convention).

* **Cache**: job 1813080 (v100). Quantising the val cache reproduces E0's
  recorded full256 predictions **bit-exactly, 339/339**; the test cache, made
  at evaluation, **338/338**. 16.6% of E0's raw output pixels are slightly
  negative (min −0.030), essentially all background.
* **Smoke**: 26/26 on real data — refiner input is exactly [X, Y0]; initial
  delta exactly 0 and Y == Y0; only the output conv has gradient at step 1,
  every layer from step 2; E0 unchanged, no gradient, not in the optimizer;
  metric path identical to predict_phase3 + skimage.
* **Training**: job 1813217 (v100; first submitted to rtx3080 as 1813124 and
  cancelled before starting, at the author's request). **Early stop at 8,500
  updates** (5 non-improving checks, ≥ 3,000), 22.3 epochs, **5.5 min** wall
  including validation, peak 1.26 GiB. **Selected update 6,000** (not the
  identity). Training L1 barely moved (0.0318 at 100 → 0.0309).
* **Evaluation**: job 1813693 (v100), 2 min 36 s, both splits, unchanged
  `masked_metrics.py`. The E0 side reproduces E0's recorded per-image PSNR with
  max difference 0.0 dB on both splits.

### Results — full256, B = E0 + refiner minus A = E0

Paired 95% bootstrap intervals and Wilcoxon are **sample-level uncertainty for
this one trained run, not training-seed robustness.**

| metric | val (n=339) | test (n=338) | test better / worse |
|---|---|---|---|
| whole-image PSNR | **+0.139** [+0.075, +0.206], p=4.3e-04 | **+0.193** [+0.124, +0.264], p=7.6e-06 | 189 / 149 |
| foreground PSNR (gt > 0.01) | **−0.167** [−0.220, −0.111], p=3.9e-09 | **−0.127** [−0.183, −0.071], p=8.1e-06 | 124 / 214 |
| whole-image SSIM | +0.0057 | +0.0069 [+0.0056, +0.0082] | 246 / 92 |
| foreground SSIM | −0.0105 | **−0.0106** [−0.0121, −0.0091] | 61 / 277 |
| background MAE | 0.01058 → 0.00734 | 0.01101 → 0.00763 | 338 / 0 |
| background RMSE | 0.0423 → 0.0338 | 0.0434 → 0.0346 | 338 / 0 |
| background pixels > 0.05 | 5.5% → 3.9% | 5.8% → 4.2% | 338 / 0 |

Absolute test values: E0 21.8725 → 22.0654 whole-image; foreground 17.599 →
17.473. Validation 22.0767 → 22.2153 (the validation figure is the one
selection picked from 17 checks, so it is optimistic; test is the clean read).

**What the correction does, measured on the quantised outputs (test; val
the same to two digits):** it **darkens**. 95.5% of foreground pixels get
darker and 2.8% brighter; the mean signed foreground change is −0.022, and the
foreground is darkened on net in **338/338** images (val 339/339). In the
background, 0.4% of pixels get brighter. E0 already under-predicts the
foreground on average (mean E0 − target −0.013 on test), so the darkening adds
foreground error while removing E0's residual background haze. The correction
is weakly aligned with the true residual (correlation +0.17 over the frame,
+0.11 in the foreground). Why L1 training settles on this is not measured and
is not claimed.

**Sidelobes: none reintroduced.** In the background the correction is
anti-correlated with (X − Y0), mean −0.17 (val −0.19): it moves the background
away from the noisy frame, and no background pixel indicator got worse on any
image.

**Recovery of weak, measurement-supported structure: not observed.** In the
228 validation recovery windows fixed before training (E0 retained ≤ 60% of a
weak target structure that the blurred noisy frame still shows), retention
FELL from 0.42 to 0.35 and window PSNR fell by −0.63 dB (median −0.54),
improving in 6 of 228 windows. In the 191 control windows (no clear noisy
support) the pattern is the same: 0.33 → 0.23, −0.66 dB, 14 of 191 improved.
**There is no recovery-versus-control contrast**, which is what
measurement-supported recovery would have required. The four pre-declared
recovery cases (6689, 1918, 2708, 2886) show no restored stripe in any
refined panel; the correction there is small and negative. The two typical
cases gain slightly (2906: window 21.18 → 21.57 dB; 1076: 30.06 → 30.12).
Post hoc, the validation PSNR changes range from −1.38 dB (0977) to +2.04 dB
(0802), median +0.09.

### Verdict, with the fields kept separate

  * **Practical target (+0.10 dB, test full256):** OBSERVED +0.193; RELIABLE
    yes (interval excludes zero, validation +0.139 same sign); EXCEEDS yes.
  * **But it is a TRADE-OFF, reported as one:** the whole-frame gain comes with
    a significant, replicated **foreground loss** (PSNR −0.127 test, −0.167
    val; foreground SSIM worse on 277/338). The gain is background cleanup; the
    object, where the structures are, gets worse.
  * **On the question the experiment asked — recovering measurement-supported
    structures E0 weakened — the answer from this run is no.** Nothing in the
    window analysis or the panels shows it, and the refiner removes more of
    those structures than E0 did. Absence of recovery here is a result for this
    refiner, loss and recipe; it does not show that no second stage could do it.

### Pre-registered guesses, scored

| guess | result | verdict |
|---|---|---|
| test full256 +0.1 to +0.5 dB | +0.193 | **correct** |
| recovery windows gain more than control windows | both lose, −0.63 vs −0.66 dB | **wrong** |
| no sidelobe signature (background corr ≤ 0) | −0.17 | **correct** |

The registered confound — that a full-frame gain could come from something
other than structure recovery — is what happened, though the observed route is
darkening, not the full-frame scale deficit that was named. Stated as an
observation.

### Limitations, as registered

The refiner was trained on E0 outputs for images E0 had seen (as 128 crops);
the pipeline gets extra supervised training; this does not show E0 trained
further, or at 256, would not do the same; one run cannot separate the value of
access to X from the architecture (no Y0-only control); single seed, and the
bilinear backward is not bit-deterministic. Only full256 was evaluated, by the
brief's design; the project's both-protocols rule concerns the DINO arms.
No control or follow-up run is started.

## Step 52 — Foreground-balanced refiner: a real foreground gain, and still no recovery of missing structure (2026-09-15)

**One follow-up to Step 51, requested by the author**, outside the DINO study
(the brief was drafted in a ChatGPT-assisted discussion). Question: did
refiner_e0 fail to recover weak structure *partly because* its loss and its
checkpoint selection favoured the background? `Holo_E0_frozen_noisy_output_
fgbalanced_refiner`, code in `dino_analysis_phases/refiner_fgbal/`; the
pre-registration (commit 4553203) was written before training. **Exploratory:
the test split was already inspected for refiner_e0, so the test read below is
not an untouched confirmatory evaluation.** No other run was started. The final
chains 1812561/2 were not touched.

### What changed, and what did not

Identical to refiner_e0, checked mechanically: frozen E0 (268,000), the same
verified caches (sha256), the same U-Net (118,129 parameters), AdamW 1e-4 with
cosine to 1e-6, 10,000-update budget, batch 16, seed 100, clip 1.0, float32,
validation every 500, no augmentation. The initialisation was **bit-identical**
to refiner_e0's step 0 on the GPU node (22/22 tensors).

Changed:

* **Loss:** 0.5·mean_fg|e| + 0.5·mean_bg|e|, with region means per image over
  the image's own pixels. The foreground is gt > 0.01 (the study's mask),
  checked on training targets: it contains the faint band, and nothing
  structured lies below it.
* **Selection:** validation foreground PSNR among checkpoints whose validation
  background MAE and RMSE are no worse than E0's; E0 is the fallback.
* **Early stopping:** follows the selection rule (derived).
* **Checkpoints:** saved at every check.

Jobs: training **1813720** (v100, 6.8 min, full 10,000 updates; all 20 checks
eligible). Evaluation **1813729 failed** after 42 s on a bug of mine (the metrics
folder was created after `masked_metrics.py` needed it), on validation, before
any test prediction. Fixed, dry-run on validation on CPU, and rerun as
**1813745** (1 min 16 s). The GPU window results equal the dry run at printed
precision.

### Selection: both rules, both runs (validation foreground PSNR)

| | original rule (max whole-image PSNR) | new rule (eligible foreground PSNR) |
|---|---|---|
| refiner_e0 (whole-image L1) | update 6,000: **17.812** | none eligible beats E0 → **E0, 17.978** |
| fg-balanced (region L1) | update 6,000: **18.130** | update 8,500: **18.145** (selected) |

**Matched update 6,000:** 17.812 vs 18.130 (+0.318). Through that update the
two runs share bit-identical initial weights, identical logged learning rates
at all 60 logging points, identical caches and preprocessing, and identical
optimiser, clipping, seeded batch-order and schedule code (only a comment
differs). Batch order is not logged, so its identity is inferred from the code.
The GPU node differed (tg074 / tg072, both v100). On this run the selection rule
moves the new run by 0.015 dB, so **most of the foreground difference comes
with the loss weighting, not the selection rule**. It is one seed: nothing here
measures whether that gap would repeat across seeds.

### Results — A frozen E0 · B refiner_e0 · C fg-balanced (full256)

Paired differences with bootstrap 95% CIs and Wilcoxon (refiner_e0's
procedure). **The intervals reflect image sampling only; each refiner is one
training run and no training-seed variability is measured.**

| metric | split | A | B | C | C − A [95% CI] | C better / worse | C − B |
|---|---|---|---|---|---|---|---|
| whole-image PSNR | val | 22.077 | 22.215 | 22.316 | +0.240 [+0.200, +0.281] | 254 / 85 | +0.101 |
| | test | 21.873 | 22.065 | 22.154 | **+0.282** [+0.241, +0.323] | 267 / 71 | +0.089 |
| foreground PSNR | val | 17.978 | 17.812 | 18.145 | +0.167 [+0.129, +0.206] | 234 / 105 | +0.333 |
| | test | 17.599 | 17.473 | 17.809 | **+0.210** [+0.175, +0.245] | 257 / 81 | +0.336 |
| foreground SSIM | val | 0.5686 | 0.5581 | 0.5723 | +0.0037 [+0.0029, +0.0045] | 255 / 84 | +0.0142 |
| | test | 0.5598 | 0.5492 | 0.5636 | **+0.0038** [+0.0030, +0.0046] | 251 / 87 | +0.0144 |
| whole-image SSIM | val | 0.7828 | 0.7884 | 0.7794 | −0.0034 [−0.0039, −0.0028] | 86 / 253 | −0.0090 |
| | test | 0.7829 | 0.7898 | 0.7789 | **−0.0040** [−0.0045, −0.0034] | 71 / 267 | −0.0108 |
| background MAE | val | 0.01058 | 0.00734 | 0.01009 | −0.00049 [−0.00082, −0.00022] | 120 / 219 | +0.00275 |
| | test | 0.01101 | 0.00763 | 0.01059 | −0.00043 [−0.00074, −0.00015] | **113 / 225** | +0.00296 |
| background RMSE | val | 0.0423 | 0.0338 | 0.0389 | −0.0034 [−0.0041, −0.0029] | 258 / 81 | +0.0051 |
| | test | 0.0434 | 0.0346 | 0.0399 | −0.0035 [−0.0042, −0.0029] | 263 / 75 | +0.0053 |
| background pixels > 0.05 | test | 5.79% | 4.17% | 6.52% | +0.73 points | **33 / 305** | |
| background pixels > 0.10 | test | 3.97% | 2.67% | 3.73% | −0.24 points | 174 / 162 | |

Validation agrees in sign with test on every line. Where the correction goes
(test; validation within 0.01): mean signed foreground change −0.010, 22.1% of
foreground pixels brighter, 8.8% of background pixels brighter; the correlation
of the correction with (X − Y0) in the background is **+0.063** (val +0.049), a
weak tendency towards the noisy frame; the correlation with the true residual
in the foreground is +0.27.

### Fixed windows (validation; chosen before refiner_e0 was trained)

Scored from quantised outputs. Target correlation was **defined in every
window** (0 undefined in all three methods and both groups), so the two
conventions for constant windows (exclude / score 0) give identical numbers.
Retention is an **intensity** measure, not evidence of geometric recovery.

| | recovery (228) E0 / B / C | C − E0 [95% CI] | control (191) E0 / B / C | C − E0 [95% CI] |
|---|---|---|---|---|
| window PSNR | 16.86 / 16.23 / 16.83 | −0.03 [−0.08, +0.01] | 16.83 / 16.17 / 16.83 | +0.00 [−0.06, +0.06] |
| window MAE | 0.100 / 0.108 / 0.102 | +0.0014 [+0.0009, +0.0019] | 0.113 / 0.120 / 0.114 | +0.0003 [−0.0007, +0.0014] |
| target correlation | 0.728 / 0.703 / 0.740 | +0.012 [+0.008, +0.016] | 0.485 / 0.467 / 0.501 | +0.016 [+0.009, +0.022] |
| retention (intensity) | 0.421 / 0.352 / 0.415 | −0.006 [−0.010, −0.003] | 0.332 / 0.231 / 0.325 | −0.007 [−0.014, −0.000] |
| false additions (> target + 0.05) | 3.2% / 1.9% / 3.6% | +0.3 points | 7.6% / 4.2% / 7.9% | +0.3 points (n.s.) |
| missing-structure fill* | 1.3% / 0.7% / 5.4% | +4.1 points (224/228) | 2.3% / 0.8% / 7.2% | +4.8 points (173/188) |

\* pixels with target > 0.05 where E0 < 0.02 (E0 removed them): Σprediction /
Σtarget. The balanced refiner lifts 19% (recovery) and 24% (control) of those
pixels above 0.02.

**Contrast (recovery − control) of C − E0:** correlation −0.003, window PSNR
−0.036, retention +0.001. **None.**

**The groups do not start from the same place**, which limits what the contrast
can show. By construction, noisy support is 0.71 against 0.06. E0 also starts
higher in the recovery group (correlation 0.73 against 0.49, retention 0.42
against 0.33) at similar target brightness (0.158, 0.145). Control windows
therefore have more room to gain correlation, and both groups were chosen where
E0 failed. A null contrast is consistent with no measurement-specific recovery
but is a weak test of it. The absolute numbers carry the main message: window
PSNR does not move in either group.

### Local corrections inside the windows (added analysis, author-requested)

`analyze_windows_local.py`, written before any fg-balanced window output was
viewed and not part of the pre-registration. Descriptive. A pixel counts as
changed if |C − E0| > 0.005.

| per window, mean | recovery | control |
|---|---|---|
| pixels brightened | 16.2% (mean +0.020) | 20.8% (+0.019) |
| … of which error fell | 77% | 69% |
| … target-supported (target > E0 and > 0.01) | 12.4% of pixels | 14.7% |
| … unsupported | 3.8% | 6.1% |
| … overshooting the target by > 0.05 | 1.4% | 2.3% |
| pixels darkened | 18.0% (mean −0.022) | 23.1% (−0.021) |
| … of which error fell | 20% | 29% |
| … darkening weak structure (target > E0) | 14.7% | 15.4% |
| … justified (E0 > target) | 3.3% | 7.7% |
| existing structure (target > 0.05, E0 ≥ 0.02): mean change | −0.008 | −0.006 |

refiner_e0, for comparison, brightened no window pixel and darkened 31%
(recovery) and 43% (control); 25% and 30% of pixels were darkened weak
structure.

**Reading, without forcing one explanation.** The evidence is mixed:

* **target-supported brightening exists**: most brightening reduces error,
  but it is small (+0.02) and fills about 5% of the missing intensity;
* it happens **as much in control windows**, where the noisy frame gives no
  support, so it is not shown to come from the measurement;
* **suppression of weak structure continues** on about 15% of window pixels,
  and there it mostly increases error;
* **some unsupported signal** appears: 4–6% of window pixels, overshoot
  1.4–2.3%, more background pixels above 0.05, and a weak positive background
  correlation with the noisy residual;
* **not mainly a brightness change on existing structure**: the mean change
  there is −0.008.

The net effect on the windows is close to zero. The balanced objective
removed most of refiner_e0's *extra* suppression (retention back to E0's level
from 0.35); it did not add structure beyond E0.

### Figures (validation, same pre-declared cases, identical limits)

`refiner_fgbal/results/figures/`: `cases_val_enlarged_all_methods.png` (full
frame, then 96x96 enlargements: noisy | target | E0 | refiner_e0 | fg-balanced
| both corrections | three error maps; images gamma 0.5 on [0,1], corrections
±0.15, errors ±0.30), `cases_val_three_way.png`, and
`posthoc_val_three_way.png` (post hoc, labelled).

What they show:

* **Recovery cases 6689, 1918, 2708, 2886:** the stripes E0 erased are absent
  in all three outputs; the error maps are practically identical. The
  balanced correction inside the windows is near zero. Correlation moves both
  ways (2886: 0.67 → 0.71; 2708: 0.31 → 0.20) while intensity stays near zero,
  so at these levels correlation tracks tiny changes and is ambiguous, not
  evidence of structure.
* **Control 2547:** the balanced refiner adds a faint positive rim along the
  lower object edge, outside the missing bar; the correlation falls −0.11 →
  −0.21. **Control 4158:** unchanged.
* **Typical cases:** small gains for both refiners (2906: 21.18 → 21.52; 1076:
  30.06 → 30.23); the balanced refiner darkens less than refiner_e0.
* **Post hoc:** the recurring pattern is interior darkening with a thin bright
  rim at object outlines — the likely source of the extra background pixels
  above 0.05. The worst frame (3508, foreground −1.07 dB) has target pillars
  that no method recovers.

### Verdict

| question | verdict |
|---|---|
| better overall denoising? | **mixed**: whole-image PSNR up (+0.28 test, +0.24 val, reliable) but whole-image SSIM down (−0.004, worse on 267/338) |
| better foreground reconstruction? | **supported**: foreground PSNR +0.21 test / +0.17 val, above the +0.10 target and reliable; foreground SSIM +0.004 on both splits |
| recovery of previously missing structure? | **unsupported**: missing structures stay missing in every case shown; window PSNR does not move; missing fill 5% of intensity, equally in control windows. **This foreground-balanced refiner did not improve structure recovery under the tested setup.** |
| acceptable background preservation? | **uncertain**: the registered constraint is met (mean MAE and RMSE no worse than E0, on both splits, and RMSE better on 263/338), but per image MAE is worse on 225/338, pixels above 0.05 increase on 305/338, and the correction leans slightly towards the noisy frame |

On the question asked: objective alignment explains **refiner_e0's extra
foreground suppression** (on one seed, mostly the loss). It does **not**
explain the absence of weak-structure recovery, which persists once the
objective is balanced. None of this shows the information is absent from the
measurement, that refinement is impossible, or that a larger network would
recover it.

Pre-registered guess ("E0 fallback, or a small foreground gain carried mainly
by brightening, with no recovery-over-control contrast"): **half right**.
There was no contrast, but there was no fallback, and the gain was not carried
by brightening.

**Recommendation: record as a limited result and stop.** A small, cheap
foreground refinement (118,129 parameters, 0.45% of E0's trainable
parameters) with a background cost that is small in mean but widespread per
image, and no structure recovery. It stays outside the thesis's main line; at
most a short note on objective alignment. No further run is proposed.

## Step 53 — The two FINAL arms evaluated: multi-level injection HURTS both operators, and the training study closes (2026-09-16)

Both final chains finished 300,000 iterations (chain count 2, `TRAINING_DONE`,
no stability failure, every injection site finite at the last logged step; site
ratios settled at pl/d3/d2 = 0.35/0.79/0.78 for addition and 0.21/0.59/0.43 for
ACA, having started above 1.0 at the decoder sites in the 2k smoke).
**Validation-only selection** (the test split never drove it): multi-level
addition **236,000** (in-loop 8-bit val 23.928, top-5 spread 0.084), multi-level
ACA **276,000** (24.126, spread 0.035). Eight evaluation cells, jobs
1814322-1814329 on v100, all `EVAL DONE`; a100 was checked at the author's
request and was four days out, so the pair kept the same evaluation hardware as
every earlier arm. Analysis: `final_matched_pair.py` with its registered
defaults -> `results/comparisons/final_matched_pair.json`.

### The 2x2 cells (uint16 convention)

full256 PSNR, foreground PSNR in brackets:

| | post-latent only | + decoder 3 + decoder 2 |
|---|---|---|
| **addition, val** | **24.433** (20.106) | 23.932 (19.662) |
| **addition, test** | **24.387** (19.891) | 23.869 (19.457) |
| **ACA, val** | 24.295 (20.002) | 24.128 (19.940) |
| **ACA, test** | 24.139 (19.699) | 23.961 (19.610) |

crop128 PSNR: val 22.276 / 22.274 / 22.058 / 22.170; test 22.267 / 22.196 /
21.999 / 22.111 (same cell order).

### The registered comparisons (paired per image, full256 primary)

| | test full256 | 95% CI | p | val full256 | test crop128 |
|---|---|---|---|---|---|
| **A** ml_add - pl_add | **-0.518** | [-0.623, -0.417] | 6.5e-20 | -0.501 | -0.071 (n.s.) |
| **B** ml_aca - pl_aca | **-0.178** | [-0.285, -0.067] | 5.4e-05 | -0.168 | +0.112 (n.s.) |
| **C** ml_aca - ml_add | +0.092 | [-0.023, +0.209] | 0.055 | +0.195 | -0.084 (n.s.) |
| **D** difference-in-differences | **+0.339** | [+0.209, +0.477] | 3.3e-06 | +0.333 | +0.184 |

Verdict fields kept separate (primary test full256, threshold +0.10 dB):

  * **A: observed -0.518, reliable (CI excludes zero, validation same sign),
    and it is a LOSS, not an improvement.** Adding the two decoder sites to
    post-latent addition costs half a dB. No crop128 trade-off: crop is flat.
  * **B: observed -0.178, reliable, also a loss.** The same layout change costs
    ACA less than it costs addition.
  * **C: observed +0.092, NOT reliable** (interval includes zero, p = 0.055),
    and it does not exceed +0.10. At the multi-level layout the ACA block is
    **not shown to beat addition**; validation reads +0.195, test +0.092, so the
    two splits do not agree in magnitude and the claim is not made.
  * **D: observed +0.339, reliable, exceeds the threshold.** The interaction is
    the one clear positive: **addition loses more from the extra sites than ACA
    does.** That is a statement about how the two operators respond to the
    layout, not about either being good.

### What this settles

  * **Repeated delivery of the same prior does not help; it hurts.** Both
    operators are worse with the prior added again at decoder levels 3 and 2.
    The registered expectation was "limited incremental benefit"; the direction
    is worse than that, and it is recorded as a miss in magnitude, not a
    confirmation.
  * **Finding 6 is unchanged.** The operator still buys nothing that survives
    both splits: at the single post-latent site ACA was significantly WORSE than
    addition (-0.247, Step 50); at the multi-level layout the difference is not
    reliable (+0.092, p = 0.055). Nowhere in this study does the ACA block
    reliably beat plain addition.
  * **The best arm is unchanged: post-latent addition at B6** (test 24.387),
    with postlatent-B3 (24.480) not reliably separable from it (Step 50).
  * **Capacity, with denominators.** Multi-level addition adds 516,768
    parameters over E0 (total 26,640,820, +0.84% over post-latent addition's
    26,419,348); multi-level ACA adds 1,910,535 (total 28,034,587, +6.11% over
    it). Both spend more capacity for a reliable loss, so the losses cannot be
    blamed on too little capacity.
  * **Both protocols reported, as always.** The full256 losses do not appear on
    crop128, where every cell sits within about 0.28 dB and no comparison is
    reliable. The +0.25 tier's signature - gains only on full256 - now has a
    mirror image: these losses are also full256-only.

### Final configuration, recommended on VALIDATION evidence and complexity

**`postlatent-render` (B6 prior, single injection after the eight latent blocks,
plain addition).** Validation full256 24.433, the best of the four cells here
and level with postlatent-B3 (+0.002, Step 50); crop128 22.276, losing nowhere.
It is also the cheapest and simplest of the candidates: one injection site, one
zero-initialised 1x1 projection, 295,296 added parameters (+1.13% of E0's
26,124,052 total trainable), no attention block, one DINO extraction. Every
alternative tested either costs more for no reliable gain (ACA, +3.99% total) or
is reliably worse (multi-level, either operator). B3 instead of B6 is an equally
defensible choice on the same evidence; B6 is kept for continuity with the depth
study, and the two are not separable.

**Single seed**, as throughout: the intervals measure image-to-image variation,
and cross-split replication is the substitute for seed repeats. Every comparison
above replicates in sign on validation.

**The architecture training study is closed. No further training arm follows.**
