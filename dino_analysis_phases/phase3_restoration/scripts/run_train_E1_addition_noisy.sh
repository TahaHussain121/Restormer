#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=E1_B6_latent
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_E1_addition_noisy_fixed128_spatial_B6_latent/metadata/slurm_%j.out
#
# =============================================================================
# E1-N-Fixed — E0-Fixed plus frozen DINOv2 B6 spatial guidance from the same
# noisy 1e5 crop, zero-initialized 1x1 residual projection at the latent.
# 300k iterations, fixed 128x128 crops, batch 8, from scratch.
#
# HARDWARE. v100 (32 GB), chosen deliberately for this arm plus ~0.35 GB of frozen ViT.
# Use the SAME GPU model as the E0 run -- the comparison should not straddle
# two hardware generations.
#
# STABILITY. The in-run gate (image_restoration_dino_model.py) hard-stops on
# NaN/Inf, on injection_ratio > 0.5, and on >10x growth versus the ~5k
# reference. A stop writes experiments/<name>/STABILITY_FAILURE.json and
# appends to the E1 devlog. Every ~10k iterations also run the cross-run gate:
#
#   python dino_analysis_phases/phase3_restoration/scripts/stability_gate.py
#
# which checks E1 val PSNR against E0 at the same iteration (>1.0 dB below =
# trigger) and E1-vs-E0 training loss divergence.
#
# A trigger is an OPTIMIZATION / STABILITY FAILURE. It is NOT a verdict on the
# DINO prior, and it is NOT answered by editing this experiment: the YAML is
# frozen once the run starts, and any design change needs a new identity.
#
# AUTO-RESUME. Identical mechanism to E0 -- resubmitting continues, never
# restarts. Before the FIRST submission:
#   ls experiments/Holo_E1_addition_noisy_fixed128_spatial_B6_latent   # must be "No such file"
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

python -u dino_analysis_phases/phase3_restoration/scripts/write_run_metadata.py \
    --config dino_analysis_phases/phase3_restoration/configs/E1_addition_noisy_fixed128_spatial_B6_latent.yml

python -u basicsr/train.py \
    -opt dino_analysis_phases/phase3_restoration/configs/E1_addition_noisy_fixed128_spatial_B6_latent.yml \
    --launcher none
