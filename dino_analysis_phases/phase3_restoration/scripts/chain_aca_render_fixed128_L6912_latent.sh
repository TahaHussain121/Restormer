#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1-00:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_acaL6912
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_aca_render_fixed128_L6912_latent/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment: Holo_aca_render_fixed128_L6912_latent
#
#     sbatch dino_analysis_phases/phase3_restoration/scripts/chain_aca_render_fixed128_L6912_latent.sh
#
# Submit ONCE. Each job queues its successor with --dependency=afterany BEFORE
# training starts, so the chain survives the walltime SIGKILL. It stops itself
# on completion (300k), on a stability failure, or on an early crash.
#
# NO DEADLINE GUARD, DELIBERATELY. Jobs resume after the maintenance window, so
# an arm that does not finish beforehand is not wasted. (dinolight-render does
# carry a guard, from when the deadline was believed to be hard -- that
# asymmetry is recorded in the manifest and is NOT to be "fixed" while it runs.)
#
# WALLTIME is the a100 maximum (MaxTime=1-00:00:00) and governs every job in
# the chain, because each successor is `sbatch $SELF`. At ~0.46 s/iter a 300k
# run is ~38.3 h, so EVERY arm needs at least two jobs.
#
# PARTITION: a100. ARM_PART is read by every successor, so an sbatch
# --partition override would move only the FIRST job and silently split the run
# across TF32 (a100) and non-TF32 (v100) numeric regimes. Do not override it.
#
# THE LOG PATH carries the experiment name AND `%j`, so a resubmit can never
# overwrite an earlier job's log.
#
# ON FAILURE: if the gate fires, chain_core writes CHAIN_ABORTED and cancels the
# successor. Do NOT resubmit that arm -- record why. A failure in one arm never
# blocks another, because each arm has its own chain state directory.
#
# All shared logic lives in chain_core.sh so behaviour cannot drift between
# arms; only the identity below is per-experiment.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_aca_render_fixed128_L6912_latent
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/aca_render_fixed128_L6912_latent.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_aca_render_fixed128_L6912_latent.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="aca-L6912 (render, layers {6,9,12}, AFFM + ACA)"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
