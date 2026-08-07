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
- Loss: L1Loss (mean reduction, weight 1) — verified from train.pixel_opt.type in the
  yml and self.cri_pix / l_pix in basicsr/models/image_restoration_model.py. This matches
  the upstream Restormer deraining recipe. (Earlier docs said "Charbonnier" — that was wrong.)
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
- [x] Fresh 4-stage 300k retrain COMPLETE (noisy baseline) — peak val 33.473 dB @ 224k, final 33.405 dB @ 300k
- [x] Evaluated on locked test set (224k ckpt) — full 33.50 dB / 0.9458, masked 29.31 dB / 0.8739 (n=338)
- [x] Verynoisy baseline COMPLETE (Step 19) — holographic_image_dataset, clean GT + verynoisy LQ, mixup OFF,
      experiment Holo_Baseline_Restormer_verynoisy — peak val 22.446 dB @ 292k, final 22.437 dB @ 300k, no overfitting
- [x] Test split carved for holographic_image_dataset (Step 19a) — 339 val / 338 test, seed 42, same recipe as Step 15
- [x] Verynoisy evaluated on the 338-image test set with net_g_292000.pth (best val, not final):
      full 22.405 dB / 0.7999, masked 18.313 dB / 0.5438, +10.05 dB over the noisy input
      CAVEAT: split was carved AFTER training, so the test half influenced checkpoint selection
      (not the weights). Exp 1's split predated training and is clean. See DEVLOG Step 19a.
- [x] Verynoisy checkpoints pruned (Step 19b) — kept net_g_292000.pth (best) + net_g_128000.pth only;
      all other .pth and ALL training states deleted (44 GB -> 201 MB). This run can no longer be resumed.
- [ ] Over-smoothing is the headline weakness: prediction retains only ~22% of GT high-frequency energy
- [ ] DINOv2 injection into bottleneck (July)

Config naming convention:
  Holo_Baseline_Restormer.yml       — pure Restormer, no DINOv2
  Holo_Baseline_Restormer_test.yml  — inference-only config (basicsr/test.py)
  Holo_DINOv2_Restormer.yml         — DINOv2 cross-attention injection (July)

Manual commands log: COMMANDS.md at repo root — all commands to run by hand are recorded there.

Experiment results: Deraining_Holo/experiment_results/ — per-run figures, metrics CSVs and
results.md write-up. Tracked in git via a .gitignore exception, because experiments/, results/
and *.png are otherwise all ignored. Put anything worth keeping there, not in experiments/.

Dataset note: TWO datasets are now in play.
  holo_image_dataset/         — original; clean/noisy; HAS a locked test split (Step 15)
  holographic_image_dataset/  — newer; clean/noisy/verynoisy/renders; train.txt+val.txt ONLY,
                                no test split. train_/val_verynoisy built as symlinks from
                                splits/*.txt (see DEVLOG Step 19).

Last change: 2026-08-07 — E1 CANCELLED, both arms broken (DEVLOG Step 22): FiLM runaway, |gamma|->325,
val PSNR 3-8 dB vs baseline 19.6 dB, broken by iter 2000. Needs a FiLM stability fix before relaunch.
Prior (2026-08-06): E1 LAUNCHED, both arms (DEVLOG Step 21): job 1771016 renderDINO on a100,
job 1771017 lqDINO on v100, self-chaining to 300k. Earlier the same day: E1 CENTERING added + pre-launch decisions settled (DEVLOG Step 20).
The FiLM head is now fed `pooled − mean` (fixed per-arm mean over 300 training crops, registered
buffer, config field `dino_feat_mean`); identity-at-init re-verified with real DINOv2 + real means
(max diff 0.000e+00, both arms); both arms confirmed; no raw-feature arm (predicted null, saves
~3 GPU-days); crop-size signal decay registered in advance as a candidate explanation if E1
underperforms; per-arm self-chaining launch scripts created. STILL NOT TRAINED — nothing submitted.

Prior (2026-08-05): E1 (DINOv2 FiLM guidance) built, wired, and VERIFIED but NOT trained.
Two arms (lqDINO, renderDINO), frozen DINOv2 ViT-B/14, FiLM at bottleneck+decoder, zero-init identity.
Offline DINO cache wired + fails loudly; weights verified genuinely loaded; input/alignment/pipeline
verified with real images. Feature-separation analysis: pooled DINO feature is ~95% shared offset,
weak object signal after centering. Pending: centering decision, which arm(s), test-time render eval.
** For the full picture read HANDOVER.md at repo root ** (this file is the project primer; HANDOVER.md
is the E1/DINO handover; DEVLOG.md is the step log up to Step 19c).

Prior (2026-07-21): Verynoisy baseline complete (DEVLOG Step 19) — full 300k, peak val 22.446 dB @ 292k;
test 22.405 dB full / 18.313 dB masked; over-smoothing (HF ratio 0.216) is the weakness E1 targets.

## Known Issues (must fix before smoke test)
1. ~~uint16 → uint8 truncation~~ FIXED via imfrombytes_uint16 + Dataset_PairedImage_uint16
2. ~~Channel mismatch~~ FIXED — same fix as above
3. ~~No val split~~ FIXED — 90/10 split, seed=42, symlink dirs created

## What NOT to do
- Do not add DINOv2 injection yet
- Do not change the loss function
- Do not modify anything inside basicsr/ beyond the already-committed uint16 loader
