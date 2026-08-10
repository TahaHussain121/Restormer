# Restormer Radar CFR — Project Context
> For Claude Code: Read this at the start of every session. Update the "Current State" and "Last Change" sections after every meaningful change. Do not rewrite the History section — only append to it.

## Who / What
Person: Taha — MSc Data Science, FAU Erlangen-Nürnberg, graduating Sept 2026.
Thesis: Radar CFR heatmap denoising using DINOv2-guided Restormer. Supervised by Prof. Belagiannis at FAU, company supervisor Christian at FiveD (Erlangen).
Repo being adapted: https://github.com/swz30/Restormer

## Problem Statement
Denoising holographic backprojection heatmaps from an SFCW MIMO radar.
- Target (clean): CFR heatmap at **1e7** ray count — cleaner reference
- Input (degraded): fewer rays => more Monte-Carlo noise + strong side lobes.
  `verynoisy` = **1e5** rays (the Exp 2 / E1 input). `noisy` = the intermediate
  level (Exp 1's input) — its exact count is [UNCONFIRMED], assumed 1e6.
  *** CORRECTED 2026-08-09 (user). These numbers were previously written the
  WRONG WAY ROUND (input 1e7 / target 1e6). More rays = LESS noise, so the
  clean target is the HIGH ray count. Verified empirically: PSNR against clean
  is 29.55 dB for `noisy` and 13.11 dB for `verynoisy` (n=30 val images), i.e.
  clean > noisy > verynoisy in quality. Only the LABELS were wrong — every
  experiment trained degraded -> clean and is unaffected. ***
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
- [x] E1 (DINOv2 FiLM injection) ATTEMPTED AND ABANDONED — failed 3x, code removed from this
      branch, preserved on `dino_prior`. See DEVLOG "Steps 20-26".
- [ ] DINO prior analysis (Phase 0/1/2 done, Phase 3 open) — dino_analysis_phases/

Config naming convention:
  Holo_Baseline_Restormer.yml       — pure Restormer, no DINOv2
  Holo_Baseline_Restormer_test.yml  — inference-only config (basicsr/test.py)
  DINO_analysis_data.yml            — NOT a training config. Data-only spec
                                      (datasets.val + five dino_* keys) read by
                                      dino_analysis_phases/ as its default --opt.

Manual commands log: COMMANDS.md at repo root — all commands to run by hand are recorded there.

Experiment results: Deraining_Holo/experiment_results/ — per-run figures, metrics CSVs and
results.md write-up. Tracked in git via a .gitignore exception, because experiments/, results/
and *.png are otherwise all ignored. Put anything worth keeping there, not in experiments/.

Dataset note: TWO datasets are now in play.
  holo_image_dataset/         — original; clean/noisy; HAS a locked test split (Step 15)
  holographic_image_dataset/  — newer; clean/noisy/verynoisy/renders; train.txt+val.txt ONLY,
                                no test split. train_/val_verynoisy built as symlinks from
                                splits/*.txt (see DEVLOG Step 19).

Last change: 2026-08-10 — E1 REMOVED from this branch (branch `dino_e2`). The DINOv2-FiLM
training experiment failed three times and is not being continued; its arch, model wrappers,
configs, launchers, gate scripts, design.md and E1_DINO_report.md were deleted here and are
preserved on the `dino_prior` branch. What was KEPT, because the DINO analysis line depends
on it and it is reusable for the next experiment — all three RENAMED after the E1 strip, so
the filename now states the function rather than the dead experiment:
  basicsr/models/archs/dinov2_feature_extractor.py  — was restormer_dino_arch.py. Loads the
                                                 frozen DINOv2 offline (strict=True) and owns
                                                 dino_preprocess / dino_denormalize. Holds no
                                                 Restormer subclass and registers nothing, so
                                                 it deliberately does NOT end in _arch.py.
  basicsr/data/radar_render_triplet_dataset.py — was paired_image_uint16_render_dataset.py.
                                                 Aligned 1e5 / 1e7 / render triplets; keeps the
                                                 _dataset.py suffix so the registry scans it.
  Deraining_Holo/Options/DINO_analysis_data.yml — was Holo_DINOv2_renderDINO_Restormer.yml.
                                                 Data-only spec, not a training config.
  dino_analysis_phases/                        — only the 3 import lines and 3 provenance
                                                 strings in visualize_dino_spatial_pca.py were
                                                 touched, for the renames. No logic changed.
Verified after the removal and after the renames: Phase 1 and Phase 2 both run end-to-end on
CPU against the real DINOv2 checkpoint, and Phase 1 still reproduces the Block 6 result.

NOTE: metadata JSONs already committed under dino_analysis_phases/*/outputs/ record the OLD
paths in their "config_read" / "pairing_source" fields. That is correct — they are provenance
records of what those runs actually read, and were deliberately not rewritten. Phase 2's
consistency check compares dino_model / dino_checkpoint / dino_hub_source only, so the stale
path strings do not affect it.

The two E1 pooled feature means SURVIVE the E1 removal, at
  Deraining_Holo/experiment_results/dino_pooled_means_reference/dino_feat_mean_{lq,render}DINO.pt
They are 3072-d (4 layers x 768) vectors mean-POOLED over patch tokens, computed over 300
training crops for E1's FiLM centering. The analysis never uses them for centering
(build_extractor passes feat_mean=None; spatial centering uses its own 768-d per-layer means
in dino_analysis_phases/dino_spatial_layer_means.pt). They are kept because
report_pooled_mean_incompatibility() reads them to RECORD why the pooled vectors cannot be
reused spatially — the result lands in metadata as centering.existing_pooled_means_examined.
Delete them and that field silently becomes [] on every future run, which would not match the
committed runs. Do not delete.

Removed 2026-08-10: the upstream Denoising/ task directory (Gaussian + real image denoising,
23 files). Nothing in this project referenced it, and Deraining/, Motion_Deblurring/ and
Defocus_Deblurring/ were already deleted back in DEVLOG Step 1. Dataset_GaussianDenoising
still lives in basicsr/data/paired_image_dataset.py and is untouched. Upstream README.md
still links to Denoising/README.md — those 4 links now dangle; README is upstream boilerplate
and was deliberately left alone.

Prior (2026-08-10): Phase 2 (DINO prior source: 1e5 radar vs render) complete after the
block/feature alignment fix. Phase 1: Block 6 leads, Block 9 does not generalise (339/339
val triplets). See dino_analysis_phases/DINO_ANALYSIS_DEVLOG.md and the per-phase devlogs.

Prior (2026-07-21): Verynoisy baseline complete (DEVLOG Step 19) — full 300k, peak val 22.446 dB @ 292k;
test 22.405 dB full / 18.313 dB masked; over-smoothing (HF ratio 0.216) is the weakness the
DINO line targets.

## Known Issues (must fix before smoke test)
1. ~~uint16 → uint8 truncation~~ FIXED via imfrombytes_uint16 + Dataset_PairedImage_uint16
2. ~~Channel mismatch~~ FIXED — same fix as above
3. ~~No val split~~ FIXED — 90/10 split, seed=42, symlink dirs created

## What NOT to do
- Do not resurrect E1 FiLM guidance without a reason the three recorded failures do not
  already cover (DEVLOG "Steps 20-26"; full detail on the `dino_prior` branch)
- Do not touch dino_analysis_phases/ — Phase 0/1/2 results are committed and Phase 2
  asserts against phase1_metadata.json
- Do not change the loss function
- Inside basicsr/, the only project-owned files are the uint16 loader
  (utils/img_util.py + data/paired_image_uint16_dataset.py),
  data/radar_render_triplet_dataset.py and models/archs/dinov2_feature_extractor.py.
  Everything else is stock Restormer — add new archs/models/datasets as NEW files;
  the registry auto-scans *_arch.py / *_model.py / *_dataset.py.
