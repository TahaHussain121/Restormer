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

## Step 20 — E1 centering + pre-launch decisions (2026-08-06)

Pre-launch changes to E1 (DINOv2 FiLM guidance). Still NOT trained; nothing was
submitted to the scheduler.

**1. Centering the FiLM input (applied).** The FiLM head now receives
`pooled − mean` instead of the raw pooled DINO vector.
  - New arch kwarg / config field `dino_feat_mean` -> a .pt path. Loaded by
    `load_dino_feat_mean()` (fails loudly on a missing file or width mismatch)
    and registered as a BUFFER in `DINOv2Extractor`; subtracted at the end of
    `forward`. Zeros when unset, so the state_dict keys are identical either way
    and "no centering" is a genuine no-op. Being a buffer it is saved into the
    checkpoint, so a chained resume cannot silently pick up a different mean.
  - Vectors built by `Deraining_Holo/compute_dino_feat_mean.py` (new), seed 0,
    300 crops drawn through the arm's OWN dataset class (exact training data
    path: same random crop, same geometric augs, same loader/value range),
    TRAIN SPLIT ONLY. Crop sizes drawn in proportion to the progressive
    schedule's iteration counts -> 92/64/48/96 crops at 128/160/192/256, so the
    mean matches the crop-size mix the run will actually see.
  - Measured over those 300 crops:
        renderDINO  ||mean|| 102.234   mean||residual|| 32.626   offset 90.0%
        lqDINO      ||mean||  85.628   mean||residual|| 35.401   offset 84.9%
    (Lower than the 95.6% quoted in the earlier full-frame single-size analysis
    -- mixing crop sizes adds real variance and lowers the offset share. Not a
    contradiction, a different sampling distribution.)
  - FIXED mean, deliberately NOT BatchNorm: the schedule drops the batch to 2 at
    256px and a two-sample mean is noise, not a mean.
  - Justification is measured, not assumed: raw pooled features are object-blind
    in the corrected render<->radar test (d ~ 0.03, n.s. at 128; null at 256);
    centered they are not (d = +0.25/+8.6 sigma at 128, +0.17/+6.0 sigma at 256).
  - .pt files committed under experiment_results/exp3_dino_film/ (14 KB each,
    with full provenance metadata) -- the gitignore exception covers them.

**2. Identity at init RE-VERIFIED after the change -- PASSED.**
`sanity_check_dino_film.py` gained a second part that builds each arm from its
ACTUAL yml with the REAL frozen DINOv2 and the REAL mean vector:
        lqDINO      max|film_on - film_off| = 0.000e+00
                    max|film_model - baseline| = 0.000e+00
        renderDINO  max|film_on - film_off| = 0.000e+00
                    max|film_model - baseline| = 0.000e+00
plus: extractor output == raw_pooled - feat_mean (max residual 0.00e+00), and
the buffer is byte-equal to the .pt the yml names. Identity could not break by
construction (gamma=beta=0 whatever the feature) -- confirmed rather than assumed.

**3. No raw-feature baseline arm.** Deliberate: ~3 GPU-days to confirm an
already-measured null. Written up as "raw pooled features were measured
object-blind (d ~ 0.03, n.s.); centering was adopted before training rather than
ablated". Stated limitation: E1 cannot attribute a gain to centering
specifically. Optional later row if GPU time frees up.

**4. Both arms run.** lqDINO kept -- matches the published recipes, needs no
render at inference, and flat-lqDINO vs non-flat-renderDINO would itself be a
result.

**5. Registered in the pre-registration BEFORE training** (design.md
"Amendments made before training"):
  - the DINO signal weakens with crop size in both arms (renderDINO +0.249@128
    -> +0.167@256; lqDINO +0.183@128 -> null@256), and the schedule's last 96k
    iterations -- the finest-reconstruction phase whose weights are kept -- run
    at 256. Recorded as a NAMED CANDIDATE EXPLANATION to be invoked only if E1
    underperforms. Schedule deliberately NOT changed.
  - the two arms' d values measure different things (lqDINO = noise robustness
    within the radar domain; renderDINO = cross-domain correspondence) and are
    not comparable scores.

**6. Launch drivers created** (did not exist before):
`train_holo_chain_{lqDINO,renderDINO}.sh`, mirrored from the verynoisy chain
driver, each with its own experiment dir and Holo_chain_state_* bookkeeping so
the arms can run concurrently. Partition defaults to a100 (CONTEXT.md's stated
default; the Exp 2 baseline ran v100). NEITHER HAS EVER BEEN SUBMITTED -- they
are untested against the scheduler.

Still open: the [SOURCE NEEDED] layer-recipe citation, and `test_holo.py` cannot
yet load/pass the render for arm-A test-time eval (needed before results, not
before launch).

## Step 21 — E1 LAUNCHED, both arms (2026-08-06)

Submitted after the Step 20 changes and the isolation audit below.

  job 1771016  holo_renderDINO  a100  -> Holo_DINOv2_renderDINO_verynoisy
  job 1771017  holo_lqDINO      v100  -> Holo_DINOv2_lqDINO_verynoisy

Both self-chaining (sbatch --dependency=afterany), 23 h walltime, MAX_CHAIN=8,
300k iters each. Submitted ONCE per arm -- do NOT resubmit; each job queues its
own successor and basicsr auto-resumes from the latest .state.

Different GPUs per arm is a deliberate user choice (renderDINO carries the
measured cross-domain signal, so it got the faster card). THESIS NOTE: this
affects wall-clock only -- same code, seed 100, schedule, data and split -- but
per-arm training time is not a like-for-like comparison and must be stated.

ISOLATION AUDIT (done before submitting):
  - experiments_root comes from the yml `name`, so each arm owns
    experiments/Holo_DINOv2_{arm}_verynoisy + tb_logger/<same name>. Neither
    existed; neither collides with Holo_Baseline_Restormer{,_verynoisy}.
    (train.py line ~165 uses opt['name'] for tb, not logger.tb_logger_dir --
    the config field is inert, but the name-derived path is unique anyway.)
  - basicsr's mkdir_and_rename archiving runs ONLY when resume_state is None,
    and only on the arm's own dir => chained resumes never archive, and Exp 2's
    directory cannot be reached from these runs.
  - auto-resume scans experiments/<name>/training_states/, keyed on name, so
    each arm can only resume itself.
  - experiments/Holo_chain_state_{lqDINO,renderDINO}/ created BEFORE submitting:
    SLURM opens the --output file at job start, so the in-job mkdir -p would
    have been too late for job #1 of each chain.
  - datasets read-only and complete: train_/val_ x clean/verynoisy/
    renders_blackbg all present at 6101/339.
  - quota: 285G used of 954G soft on /home/woody; two 300k runs ~90G of
    checkpoints. Prune both as soon as they finish (Step 19b cost 44G/run).
  - MAX_CHAIN raised 5 -> 8 so a slower-per-step run cannot silently stop short.

PRE-LAUNCH SMOKE (CPU, both arms): real forward+backward+grad-clip step
completes; DINO gets 0 gradients (frozen); FiLM head and backbone both get
gradients; loss finite. Pre-clip grad norms are large (7e6-9e7) but so is the
Exp 2 baseline's at the same seed (2.1e7; 3e6-2e7 across seeds 0/1/2) -- inherent
to Restormer at init, not introduced by DINO/FiLM, and the reason upstream clips
to 0.01.

WATCH: tail experiments/Holo_chain_state_{renderDINO,lqDINO}/slurm_chain_<job>.out
and experiments/Holo_DINOv2_{arm}_verynoisy/*.log. A crash inside 1800 s writes
CHAIN_ABORTED and cancels the successor rather than burning the chain.

## Step 22 — E1 CANCELLED: the FiLM head runs away (2026-08-07)

Both arms launched in Step 21 were cancelled by hand ~13:35 CEST after 14 h
(lqDINO, 73k iters) and 5 h 41 (renderDINO, 44k iters). They were NOT crashing --
they were training a degenerate model.

  scancel 1771126 1771018   (queued successors, killed FIRST so the afterany
                             dependency could not launch a replacement)
  scancel 1771016 1771017   (running jobs)

SYMPTOM. Training loss tracked the Exp 2 baseline almost exactly (41k: renderDINO
5.40e-2, lqDINO 6.11e-2, baseline 5.43e-2) while validation was catastrophic:

  iter     renderDINO   lqDINO   Exp 2 baseline
   4000       7.80 dB   3.22 dB      19.62 dB
  44000       4.73 dB   4.96 dB      ~20.5 dB
  72000            --   5.61 dB      ~20.7 dB

The noisy input alone is 12.35 dB, so both arms were far worse than doing
nothing, and flat rather than climbing.

CAUSE (measured from the checkpoints, not inferred). |gamma| and |beta| grow
monotonically from the start; the model is already broken by iter 2000:

  lqDINO   iter   |gamma|max  |beta|max  film W_last norm  val PSNR
            2000       30.72       7.24             9.54    -74.3
           10000      125.44      33.94            17.23   -138.7
           40000      255.68      64.02            23.08   -167.1
           72000      325.50      88.37            24.94   -172.4

(1+gamma)*F+beta with gamma ~325 is a ~326x feature amplification; output range
at 256px reached +/-2.4e9. renderDINO is the same failure, milder: gamma 4.6-7.3,
output range +/-700.

WHY TRAIN LOOKED FINE. Stage 1 trains at 128px; validation runs at native 256.
The network co-adapted to a knife-edge solution that only survives its training
crop size. Measured on lqDINO@72k, one val image:
    128 crop, FiLM ON  -> 20.55 dB   (healthy)
    256 full, FiLM ON  -> -172.4 dB  (garbage)
    256 full, FiLM OFF ->   -1.4 dB  (backbone alone is also distorted -- it has
                                      co-adapted to compensate for huge gamma)

WHY NOTHING BOUNDED IT. `use_grad_clip: true` clips to 0.01 over all 28.5M
params, but AdamW normalises per-parameter, so a uniformly rescaled gradient
yields nearly the same update -- the clip does essentially nothing to constrain
the FiLM head. Nothing in the design bounds gamma. This is a design flaw in the
FiLM block, independent of which image DINO looks at (both arms show it).

ON THE CENTERING CHANGE (Step 20). Probably not the cause, and deliberately not
claimed as ruled out. Centering SHRINKS the FiLM input (norm ~30 residual vs
~110 raw), which would if anything reduce gamma for the same weights; and both
arms are centered yet differ ~70x in gamma. The mechanism points at unbounded
gamma under Adam. Confirming this would need a training run, which has not been
done.

VERIFICATION GAP (own it). Step 20/21 checks covered identity at t=0 and a single
train step. NOTHING tested stability OVER training. A few hundred iterations
logging |gamma|max would have caught this in minutes and saved ~20 GPU-hours.
Any future launch gate must include a short run that watches the modulation
magnitude, not just an init-time identity check.

STATE ON DISK. Kept for now, deliberately: experiments/Holo_DINOv2_lqDINO_verynoisy
(24 G) and .../renderDINO_verynoisy (15 G), plus full logs. CHAIN_ABORTED +
WHY_ABORTED.txt written into both chain-state dirs so a resubmit cannot silently
resume a broken run. Prune both once the fix is settled and the evidence is
written up.

NEXT (one change, not yet implemented): bound the FiLM modulation -- gamma through
a tanh (and/or a separate smaller LR / stronger weight decay for the FiLM head) --
add a |gamma|max diagnostic to the training log, verify over a few hundred iters,
then relaunch as a FRESH experiment dir. The Step 20 pre-registration stands
otherwise; this is a stability fix, not a change to what is being studied.

## Step 23 — FiLM stability fix + a launch gate (2026-08-07)

Response to the Step 22 failure. ONE substantive change to the model; the rest is
observability and a gate so a broken run can never again consume GPU-days
unnoticed. The Step 20 pre-registration is otherwise unchanged -- what is being
studied (which image DINO looks at) is untouched.

**1. BOUNDED MODULATION (the fix).** `FiLMHead.forward` now squashes both
outputs: `gamma = gamma_scale * tanh(raw)`, `beta = beta_scale * tanh(raw)`,
with `film_gamma_scale = film_beta_scale = 0.5` in both configs, so
`(1+gamma)` is confined to [0.5, 1.5].

Why bounding and not something else. FiLM has a SCALE DEGENERACY: rescaling a
feature map and letting later layers undo it leaves the loss unchanged, so gamma
sits on a flat direction with no restoring force. AdamW then walks it outward at
~lr per step. The two guards already in the recipe do essentially nothing --
grad-clip is near-invariant under Adam (which normalises per-parameter), and
decoupled weight decay contributes lr*wd = 3e-8/step. Alternatives considered and
rejected as primary: a smaller FiLM LR only SLOWS the drift (300k iters is a long
time); stronger weight decay sets a large equilibrium rather than a bound;
normalising the FiLM input improves conditioning but leaves the degeneracy.
Bounding removes the failure mode by construction.

0.5 is a judgement call, not a measured optimum, and is written up as such: 0.1
would be so weak that a null result is uninterpretable, 1.0 permits gamma = -1
(zeroing a channel outright). tanh(0) = 0, so the zero-init identity survives
exactly.

**2. OBSERVABILITY.** `RestormerDINO.forward` records detached
`film_g_absmax / film_b_absmax / film_g_std` (std ACROSS the batch -- near zero
means gamma is a constant rescale, not input-dependent guidance; the failed
lqDINO run was 71% constant). New model `ImageCleanModelDINO` (subclass of
ImageCleanModel, new file -- basicsr core still unmodified) copies them into
log_dict so they reach the log and TensorBoard every print_freq.
`ImageCleanModelRender` now inherits from it. lqDINO's `model_type` changed
`ImageCleanModel -> ImageCleanModelDINO`; training behaviour is identical.

**3. LAUNCH GATE (the part that actually matters).** `Holo_DINOv2_{arm}_GATE.yml`
(generated by `make_gate_configs.py` FROM the arm config, so they cannot drift)
plus `gate_{arm}.sh`: 4000 iters, val every 1000, ~30-45 min, own experiment dir,
no chaining. PASS = val PSNR >= 18 dB at iter 4000. Reference points: the Exp 2
baseline hit 19.62 dB at its first validation; the two broken arms were at 3.22
and 7.80 dB. This gate would have caught Step 22 in half an hour rather than
~20 GPU-hours. The gate script wipes its own dir first so it can never resume a
previous attempt, and prints an explicit PASS/FAIL verdict.

**4. v2 EXPERIMENT NAMES.** Both arms renamed to
`Holo_DINOv2_{arm}_verynoisy_v2` (+ matching tb_logger and
`Holo_chain_state_{arm}_v2`). NECESSARY, not cosmetic: basicsr auto-resumes from
the highest .state under `experiments/<name>/`, and the v1 dirs are full of the
broken run's states. Without the rename a relaunch would silently continue the
failed run. The v1 dirs are left intact as evidence.

VERIFIED (CPU, before any GPU time):
  - sanity_check_dino_film.py ALL PASS, including a new bound check that drives
    the FiLM head with weights ~N(0,50) and features scaled to 1e6:
    worst |gamma| = 0.500000, worst |beta| = 0.500000. The runaway is now
    impossible by construction, not by hoping the optimizer behaves.
  - zero-init identity still exact for both arms with real DINOv2 + real means:
    max|film_on - film_off| = max|film_model - baseline| = 0.000e+00.
  - end-to-end through basicsr create_model + 3 real optimize_parameters steps:
    both arms log film_g_absmax (0.017 / 0.013), within bound, film_g_std > 0.

NOT YET RUN: the gates. Nothing chains to 300k until a gate passes.

## Step 24 — Both gates FAIL: bounding stopped the explosion, not the failure (2026-08-07)

Gate jobs 1771540 (renderDINO, a100) and 1771541 (lqDINO, v100), 4000 iters each,
~35 min. Both rc=0, no crash. Both FAIL the 18 dB criterion.

  arm          val@1k   val@2k   val@3k   val@4k   Exp2 baseline @4k
  renderDINO   15.12    14.00    12.88    14.82        19.62
  lqDINO       12.68     7.93     7.15     6.69        19.62

The bound HELD exactly as designed -- max |gamma| 0.49992 / 0.49998, no ±2e9
outputs, no divergence. That failure mode is gone. But the model is still worse
than the baseline, and lqDINO gets steadily WORSE over the run.

WHAT THE gamma LOG SHOWS (this is why the logging was added):
  |gamma|max reaches ~0.31 by iter 200, ~0.48 by iter 400, and is pinned at
  0.4995-0.4999 from ~iter 1000 to the end. It saturates the tanh rail almost
  immediately and stays there.
  film_g_std (input-dependence) DECAYS: renderDINO 0.073 @400 -> 0.011 @4000.
  So gamma degenerates into a near-CONSTANT +/-0.5 mask -- and once tanh is
  saturated its gradient vanishes, so the head can no longer modulate on the
  input even in principle. Worst of both worlds: a large fixed multiplicative
  distortion the backbone must spend its capacity undoing.

READING (not softened). Bounding was necessary and it worked, but it was not
sufficient. The real problem is a RACE: at iter 200 the backbone is still random,
and the fastest available loss reduction for a randomly-initialised FiLM head is
the scale degeneracy. It takes it, saturates, and the backbone spends the rest of
training compensating for a constant distortion. Capping the magnitude only caps
how bad the distortion is; it does not stop the head from going straight to it.

Structural note for the thesis: FiLM/adapter conditioning in the literature
(ControlNet, T2I-Adapter) attaches to a PRETRAINED, frozen or near-frozen
backbone. Here the backbone is trained from scratch simultaneously with the
conditioning head, which is what creates the race. That mismatch is the honest
framing of these two failures.

No GPU is running. Nothing was chained. Cost of this iteration: ~35 min x 2,
which is the point of the gate.
