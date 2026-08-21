#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_pq
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_priorquery_render_fixed128_spatial_B6_latent/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment:
# Holo_priorquery_render_fixed128_spatial_B6_latent
#
#   sbatch dino_analysis_phases/phase3_restoration/scripts/chain_priorquery_render.sh
#
# Submit ONCE. Each job queues its successor with --dependency=afterany BEFORE
# training starts, so the chain survives the walltime SIGKILL. It stops itself
# on completion, on a stability failure, or on an early crash.
#
# Its own job name (p3chain_pq), log directory, chain-state directory and
# TensorBoard tree. Nothing it writes can collide with E0-Fixed, addition-1e5,
# addition-render, global-addition-render, concat-render or crossattn-render --
# and crossattn-render in particular may still be TRAINING, so the chain lock in
# chain_core.sh (keyed by experiment name) is what prevents this arm from ever
# touching that arm's directory.
#
# All shared logic lives in chain_core.sh; only the identity below is per-arm.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_priorquery_render_fixed128_spatial_B6_latent
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/priorquery_render_fixed128_spatial_B6_latent.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_priorquery_render.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="priorquery-render (DINO queries, radar K/V -- Perceive-IR direction)"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
