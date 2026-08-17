#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_E0
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_E0_fixed128_baseline/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment: Holo_E0_fixed128_baseline
#
#     sbatch dino_analysis_phases/phase3_restoration/scripts/chain_E0.sh
#
# Submit ONCE. Each job queues its successor with --dependency=afterany BEFORE
# training starts, so the chain survives the walltime SIGKILL. It stops itself
# on completion, on a stability failure, or on an early crash.
#
# This experiment has its own job name (p3chain_E0), its own log directory
# (results/Holo_E0_fixed128_baseline/logs/), its own chain-state directory and its own
# TensorBoard tree. No file written by this arm can collide with, or overwrite,
# a file written by another arm or by an earlier job of this same arm.
#
# All shared logic lives in chain_core.sh so behaviour cannot drift between
# arms; only the identity below is per-experiment.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_E0_fixed128_baseline
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/E0_fixed128_baseline.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_E0.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="E0-Fixed baseline"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
