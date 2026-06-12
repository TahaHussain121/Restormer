# Restormer Radar CFR — Project Context
> For Claude Code: Read this at the start of every session. Update the "Current State" and "Last Change" sections after every meaningful change. Do not rewrite the History section — only append to it.

## Who / What
Person: Taha — MSc Data Science, FAU Erlangen-Nürnberg, graduating Sept 2026.
Thesis: Radar CFR heatmap denoising using DINOv2-guided Restormer. Supervised by Prof. Belagiannis at FAU, company supervisor Christian at FiveD (Erlangen).
Repo being adapted: https://github.com/swz30/Restormer

## Problem Statement
Denoising holographic backprojection heatmaps from an SFCW MIMO radar.
- Input (degraded): CFR heatmap at 1e7 ray count — noisy, strong side lobes
- Target (clean): CFR heatmap at 1e6 ray count — cleaner reference
- Data: ShapeNetCore.v2 chairs, rendered via shapenet_radar package
- Image format: single-channel grayscale PNG, 256x256
- Pairing: fully supervised — clean/noisy pairs exist

Using deraining config as template (not denoising) because side lobes are structured, object-dependent artifacts — closer to rain streaks than Gaussian noise.

DINOv2 injection planned for July. NOT now. Get baseline Restormer working first.

## Architecture Decisions
- Backbone: Restormer with deraining config as starting point
- Loss: L1 (current config; original Charbonnier from deraining baseline)
- inp_channels: 1, out_channels: 1 (CFR is single-channel)
- Everything else in yml identical to deraining baseline
- basicsr/ is NOT to be modified (except already-committed uint16 loader fixes)
- Progressive schedule CAPPED at native resolution (gt_size 256, patches <=256).
  Images are 256x256; gt_size 384 padded them to 384 via cv2.BORDER_REFLECT,
  injecting mirrored (physically fake) side-lobe structure into a large share of
  crops at ALL stages. So the 6-stage 128->384 schedule is truncated to 4 stages
  128->256: gt_sizes [128,160,192,256], iters [92000,64000,48000,96000],
  mini_batch_sizes [8,5,4,2]. *** Apply this same cap to the future
  denoising-schedule ablation config. ***

## Environment: HPC (current)
Cluster: tinygpu — FAU HPC (SLURM v25.11.2)
Login node: no GPU; submit jobs via sbatch

Partitions:
  a100      — A100, 8 nodes × 4 GPUs, 24 h walltime  ← default going forward
  v100      — V100-PCIE-32GB, 4 nodes × 4 GPUs, 24 h walltime
  rtx3080   — RTX 3080, 6 nodes × 8 GPUs, 24 h walltime
  work      — default/mixed (RTX 2080Ti + RTX 3080), 24 h walltime

Dataset path: /home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset/
  clean/        — 6778 uint16 PNGs (256×256, single-channel)
  noisy/        — 6778 uint16 PNGs
  splits/       — train.txt (6101), val.txt (339), test.txt (338)
  train_clean/, train_noisy/ — symlinks (6101 each), all resolve OK
  val_clean/,   val_noisy/   — symlinks (339 each),  all resolve OK
  test_clean/,  test_noisy/  — symlinks (338 each),  all resolve OK

  *** DO NOT touch test set until final model evaluation. ***
  test.txt must be committed to git for reproducibility.

Venv: /home/woody/iwnt/iwnt174h/thesis_dino/code/venv/
  PyTorch 2.5.1+cu121  (works on cluster CUDA 12.x drivers — confirmed)
  Activation (inside sbatch): module load python && conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
  Direct call without activating: /home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python

Job script: train_holo.sh (repo root) — sbatch train_holo.sh to submit
Logs: experiments/Holo_Baseline_Restormer/slurm_<JOBID>.log

Training workflow (HPC):
  Submit:  sbatch train_holo.sh
  Monitor: squeue -u $USER   or   tail -f experiments/Holo_Baseline_Restormer/slurm_<JOBID>.log
  Resume:  edit resume_state in yml to experiments/Holo_Baseline_Restormer/training_states/ITER.state
           then resubmit: sbatch train_holo.sh
  State files: experiments/Holo_Baseline_Restormer/training_states/ITER.state (every 2k iters)
  Note: max walltime is 23 h. The 300k run will NOT fit one job — expect ~3 jobs.

  Auto-resume: basicsr/train.py (lines ~138-149) ALWAYS scans training_states/
  and resumes from the highest .state, overriding resume_state in the yml.
  So resume needs NO yml edit — just resubmit the same command.

  Self-chaining resume: Deraining_Holo/train_holo_chain.sh. Submit ONCE after a
  walltime kill; each job queues its successor (sbatch --dependency=afterany) and
  auto-resumes. Stops on TRAINING_DONE marker (300k done), after MAX_CHAIN=5 jobs,
  or via the crash guard (python exits < 1800s without the 300k ckpt => scancel
  successor + CHAIN_ABORTED marker). Markers + chain SLURM logs live in
  experiments/Holo_chain_state/ (NOT the experiment dir — keeps it separate from
  basicsr's fresh-start archiving): CHAIN_COUNT, TRAINING_DONE, CHAIN_ABORTED.
  To restart after an abort: rm experiments/Holo_chain_state/CHAIN_ABORTED.

  Effective-batch caveat (thesis): upstream Restormer deraining assumes 8 GPUs
  => effective batch 64. We train on 1 GPU => effective batch 8. Report this.

## Previous environment (fived08 — no longer used)
Machine: fived08 at FiveD, RTX 4070 Ti SUPER (16 GB), PyTorch 2.6.0+cu124
Training was tmux-based, no job scheduler.
A 300k-iter run was started and later cancelled before producing a usable checkpoint.
All config work happened on fived08; the actual first completed training run was on the HPC.

## Current State
- [x] Restormer repo cloned
- [x] Motion_Deblurring/ and Defocus_Deblurring/ deleted
- [x] Deraining_Holo/ directory created (copy of Deraining/)
- [x] yml modified: inp_channels=1, out_channels=1, name=Holo_Baseline_Restormer (controls experiments_root), dataset paths set
- [x] uint16 loading fixed — imfrombytes_uint16 + Dataset_PairedImage_uint16
- [x] val split created — ~6101 train / ~677 val, seed=42, symlink dirs
- [x] val split into val + test — 339 val / 338 test (seed=42); test locked
- [x] 1-batch smoke test passed — iter1 l_pix=7.23e-02, iter2 l_pix=8.14e-02, val PSNR=20.72 dB
- [x] Full training config finalized — 75k iters (~50 epochs), checkpoints every 2k iters, TensorBoard enabled
- [x] 75k baseline training run done — val PSNR 32.2650 dB — NOW OBSOLETE (progressive schedule never activated; all iters at 128px)
- [x] test_holo.py created — uint16 inference + heatmap visualization (inferno) + PSNR/SSIM output
- [x] reflect-padding contamination found (gt_size 384 padded 256 imgs -> mirrored fake structure in many crops at all stages); job 1694322 cancelled at iter ~28k
- [x] obsolete runs DELETED (75k run + 26-28k reflect-pad run + archives + tb_logger); numbers preserved in DEVLOG Step 17
- [x] yml schedule truncated to native 256: 4 stages, gt_size 256, gt_sizes [128,160,192,256], iters [92000,64000,48000,96000], mini_batch [8,5,4,2]
- [~] Fresh 4-stage 300k retrain SUBMITTED via chain script — v100, job 1695465 (PD as of submit), self-chaining to 300k
- [ ] Confirm first log (gt_size 256, 4-stage dump, fresh iter 0, finite loss)
- [ ] Evaluate retrained model on val set, then test_holo.py on locked test set
- [ ] DINOv2 injection into bottleneck (July)

Config naming convention:
  Holo_Baseline_Restormer.yml       — pure Restormer, no DINOv2
  Holo_Baseline_Restormer_test.yml  — inference-only config (basicsr/test.py)
  Holo_DINOv2_Restormer.yml         — DINOv2 cross-attention injection (July)

Manual commands log: COMMANDS.md at repo root — all commands to run by hand are recorded there.

Last change: 2026-06-11 — RESTART. Found reflect-padding contamination (gt_size 384 on 256 imgs); cancelled job 1694322 at iter ~28k, truncated schedule to native 256 (4 stages), deleted all obsolete runs/tb_logger (numbers in DEVLOG Step 17), decoupled chain bookkeeping into experiments/Holo_chain_state/, and submitted fresh run via chain script (job 1695465, v100, PD). Awaiting first-log confirmation.

## Known Issues (must fix before smoke test)
1. ~~uint16 → uint8 truncation~~ FIXED via imfrombytes_uint16 + Dataset_PairedImage_uint16
2. ~~Channel mismatch~~ FIXED — same fix as above
3. ~~No val split~~ FIXED — 90/10 split, seed=42, symlink dirs created

## What NOT to do
- Do not add DINOv2 injection yet
- Do not change the loss function
- Do not modify anything inside basicsr/ beyond the already-committed uint16 loader
