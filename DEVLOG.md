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
