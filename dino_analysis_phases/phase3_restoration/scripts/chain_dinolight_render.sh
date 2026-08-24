#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1-00:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3chain_dinol
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_dinolight_render_fixed128_L3691_aca_latent/logs/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for ONE experiment:
#   Holo_dinolight_render_fixed128_L3691_aca_latent
#
#     sbatch dino_analysis_phases/phase3_restoration/scripts/chain_dinolight_render.sh
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
# WALLTIME IS THE a100 PARTITION MAXIMUM (MaxTime=1-00:00:00). Every successor
# is `sbatch $SELF`, so this line governs every job in the chain, not just the
# first.
#
# THIS ARM NEEDS AT LEAST TWO JOBS. The addition arms manage ~7,700 iters/h on
# a100 (177,000 in 22.97 h), so 300k is ~39 h for them. This arm carries two
# transposed attentions and six MDTA-style projections on top, so expect it to
# be SLOWER -- see the devlog for the s/iter measured in the 6k smoke and the
# projection that follows from it. The chain is REQUIRED, not insurance.
#
# HARD DEADLINE 2026-08-28. The cluster goes down for maintenance then and a
# stalled chain at that point is fatal, so the guard below stops the chain
# before chain_core.sh can queue another successor.
#
# THE GUARD LIVES HERE, IN THIS ARM'S OWN WRAPPER, NOT IN chain_core.sh. That
# file is sourced by every arm including the affm run that is training RIGHT
# NOW, and a change to it would be picked up by that run's NEXT RESUME, hours
# later and silently. Per-arm duplication is the correct trade.
#
# All shared logic lives in chain_core.sh so behaviour cannot drift between
# arms; only the identity below is per-experiment.
# =============================================================================

unset SLURM_EXPORT_ENV

EXP=Holo_dinolight_render_fixed128_L3691_aca_latent
CONFIG=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/configs/dinolight_render_fixed128_L3691_aca_latent.yml
SELF=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_dinolight_render.sh
ARM_PART=a100
ARM_GRES=gpu:a100:1
JOB_LABEL="dinolight-render (DINO<-render, layers {3,6,9,12}, AFFM + gated channel ACA)"

# --- HARD DEADLINE GUARD (runs BEFORE chain_core, so no successor is queued) --
DEADLINE=2026-08-28
CHAIN_DIR=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/experiments/Phase3_chain_state_$EXP
if [ "$(date +%Y-%m-%d)" \> "$DEADLINE" ] || [ "$(date +%Y-%m-%d)" = "$DEADLINE" ]; then
    mkdir -p "$CHAIN_DIR"
    {
      echo "job $SLURM_JOB_ID started $(date -Is) on/after the $DEADLINE cluster"
      echo "maintenance deadline. The chain STOPS here: no successor queued, no"
      echo "training started. Whatever checkpoint is on disk is the final one."
    } > "$CHAIN_DIR/DEADLINE_REACHED"
    cat "$CHAIN_DIR/DEADLINE_REACHED"
    exit 0
fi
echo "[chain] deadline guard OK: $(date +%Y-%m-%d) is before $DEADLINE"

source /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/dino_analysis_phases/phase3_restoration/scripts/chain_core.sh
