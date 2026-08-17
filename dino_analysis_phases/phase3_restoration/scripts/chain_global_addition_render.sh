#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_globR
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_global_addition_render_fixed128_B6_latent/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment:
#   Holo_global_addition_render_fixed128_B6_latent
#
#     sbatch dino_analysis_phases/phase3_restoration/scripts/chain_global_addition_render.sh
#
# Submit ONCE. Each job queues its successor with --dependency=afterany BEFORE
# training starts, so the chain survives the walltime SIGKILL. It stops itself
# on completion, on a stability failure, or on an early crash.
#
# This experiment has its own job name (p3chain_globR), its own log directory
# (results/Holo_global_addition_render_fixed128_B6_latent/logs/), its own
# chain-state directory and its own TensorBoard tree. No file written by this
# arm can collide with, or overwrite, a file written by another arm or by an
# earlier job of this same arm.
#
# All shared logic lives in chain_core.sh so behaviour cannot drift between
# arms; only the identity below is per-experiment.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_global_addition_render_fixed128_B6_latent
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/global_addition_render_fixed128_B6_latent.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_global_addition_render.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="global-render (DINO<-render, POOLED)"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
