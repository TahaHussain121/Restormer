#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_affm
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_affm_render_fixed128_spatial_L3691_latent/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment:
#   Holo_affm_render_fixed128_spatial_L3691_latent
#
#     sbatch dino_analysis_phases/phase3_restoration/scripts/chain_affm_render.sh
#
# Submit ONCE. Each job queues its successor with --dependency=afterany BEFORE
# training starts, so the chain survives the walltime SIGKILL. It stops itself
# on completion (300k), on a stability failure, or on an early crash.
#
# Its own job name, log directory, chain-state directory and TensorBoard tree.
# No file written by this arm can collide with any other arm's.
#
# PARTITION: a100, deliberately. Every finished arm trained on a100 with TF32
# enabled; Volta has no TF32, so a v100 run would be the only arm in true fp32
# -- an uncontrolled difference in a comparison designed to have exactly one.
# ARM_PART is read by every successor too, so an sbatch --partition override
# would move only the FIRST job and silently split the run across two numeric
# regimes. Do not override it.
#
# All shared logic lives in chain_core.sh so behaviour cannot drift between
# arms; only the identity below is per-experiment.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_affm_render_fixed128_spatial_L3691_latent
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/affm_render_fixed128_spatial_L3691_latent.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_affm_render.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="affm-render (DINO<-render, layers {3,6,9,12}, AFFM fusion)"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
