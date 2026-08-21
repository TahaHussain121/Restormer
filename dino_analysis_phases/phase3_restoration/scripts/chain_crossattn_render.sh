#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_xatt
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_crossattn_render_fixed128_spatial_B6_latent/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment:
# Holo_crossattn_render_fixed128_spatial_B6_latent
#
#   sbatch dino_analysis_phases/phase3_restoration/scripts/chain_crossattn_render.sh
#
# Submit ONCE. Each job queues its successor with --dependency=afterany BEFORE
# training starts, so the chain survives the walltime SIGKILL. It stops itself
# on completion, on a stability failure, or on an early crash.
#
# Its own job name (p3chain_xatt), log directory, chain-state directory and
# TensorBoard tree, so no file it writes can collide with E0-Fixed,
# addition-1e5, addition-render, global-addition-render or concat-render. The
# lock in chain_core.sh is keyed by experiment name, so this arm also cannot
# start a second trainer against another arm's directory.
#
# All shared logic lives in chain_core.sh so behaviour cannot drift between
# arms; only the identity below is per-experiment.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_crossattn_render_fixed128_spatial_B6_latent
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/crossattn_render_fixed128_spatial_B6_latent.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_crossattn_render.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="crossattn-render (DINO<-render, radar queries DINO K/V)"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
