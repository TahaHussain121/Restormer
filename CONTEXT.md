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
- Loss: Charbonnier (unchanged from deraining baseline)
- inp_channels: 1, out_channels: 1 (CFR is single-channel)
- Everything else in yml identical to deraining baseline
- basicsr/ is NOT to be modified

## Current State
- [x] Restormer repo cloned
- [x] Motion_Deblurring/ and Defocus_Deblurring/ deleted
- [x] Deraining_Holo/ directory created (copy of Deraining/)
- [x] yml modified: inp_channels=1, out_channels=1, name=Holo_Baseline_Restormer (controls experiments_root), dataset paths set
- [x] uint16 loading fixed — imfrombytes_uint16 + Dataset_PairedImage_uint16
- [x] val split created — 6101 train / 677 val, seed=42, symlink dirs
- [x] 1-batch smoke test passed — iter1 l_pix=7.23e-02, iter2 l_pix=8.14e-02, val PSNR=20.72 dB
- [x] Full training config finalized — 75k iters (~50 epochs), checkpoints every 2k iters, TensorBoard enabled
- [ ] Full training run launched (kill old session, fresh start with updated yml)
- [ ] Baseline PSNR/SSIM logged

Config naming convention:
  Holo_Baseline_Restormer.yml   — pure Restormer, no DINOv2, train now
  Holo_DINOv2_Restormer.yml     — DINOv2 injection placeholder, implement July

Training workflow:
  Kill:    tmux kill-session -t restormer_baseline
  Start:   tmux new-session -d -s restormer_baseline 'CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Deraining_Holo/Options/Holo_Baseline_Restormer.yml --launcher none 2>&1 | tee experiments/Holo_Baseline_Restormer/train.log'
  Attach:  tmux attach -t restormer_baseline
  Resume:  edit yml resume_state to training_states/ITER.state, then rerun Start
  TB:      tmux new-session -d -s tensorboard 'bash Deraining_Holo/launch_tensorboard.sh'
  SSH tunnel: ssh -L 6006:localhost:6006 taha.hussain@ad.five-d.ai@fived08

Last change: 2026-06-10 — training config finalized: 75k iters, 2k checkpoint freq, TB enabled, resume workflow documented.

## Known Issues (must fix before smoke test)
1. ~~uint16 → uint8 truncation~~ FIXED via imfrombytes_uint16 + Dataset_PairedImage_uint16
2. ~~Channel mismatch~~ FIXED — same fix as above
3. ~~No val split~~ FIXED — 90/10 split, seed=42, symlink dirs created

## What NOT to do
- Do not add DINOv2 injection yet
- Do not change the loss function
- Do not modify anything inside basicsr/
