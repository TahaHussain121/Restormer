#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_chain
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for the Phase-3 300k runs. One script, both arms.
#
#     sbatch dino_analysis_phases/phase3_restoration/scripts/run_train_chain.sh E0
#     sbatch dino_analysis_phases/phase3_restoration/scripts/run_train_chain.sh E1
#     sbatch dino_analysis_phases/phase3_restoration/scripts/run_train_chain.sh E1R
#
# Submit ONCE per arm. Each job queues its own successor with
# `--dependency=afterany`, so the chain survives a walltime kill, a node
# failure, or any non-zero exit -- the successor is already queued before
# training starts, which is the only thing that works when the job is SIGKILLed
# at the walltime and no trailing code ever runs.
#
# To attach the chain to an ALREADY RUNNING job (so it takes over the moment
# that job is killed):
#     sbatch --dependency=afterany:<JOBID> .../run_train_chain.sh E0
#
# Same mechanism as Deraining_Holo/train_holo_chain_verynoisy.sh, plus one
# guard that script did not need:
#
#   STABILITY GUARD (E1). If the training process wrote STABILITY_FAILURE.json,
#   the chain STOPS. An OPTIMIZATION / STABILITY FAILURE must be investigated
#   and reported, never answered by silently restarting into the same abort.
#
# Nothing here edits a config, a threshold or a checkpoint. Resuming is
# BasicSR's own mechanism: train.py:138-149 scans training_states/ and resumes
# from the highest .state, overriding the yml. A genuine clean rerun needs a
# NEW experiment identity -- this script never deletes anything.
# =============================================================================

unset SLURM_EXPORT_ENV

ARM=${1:?usage: sbatch run_train_chain.sh <E0|E1>}
REPO=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
P3=$REPO/dino_analysis_phases/phase3_restoration
SCRIPT=$P3/scripts/run_train_chain.sh

# PER-ARM HARDWARE. The #SBATCH header above is only the default for a job
# submitted without an override; these values are what every SUCCESSOR in the
# chain is submitted with, so an arm cannot silently drift onto another GPU
# type halfway through its run.
#   E0 -> a100 (already running there since job 1773732)
#   E1 -> v100 (chosen deliberately; see the devlog hardware note)
#   E1R -> a100 (render arm)
# rtx3080 (10 GB) and rtx2080ti (11 GB) are NOT viable: Restormer at 128^2 x
# batch 8 exceeded 9.6 GB on an rtx3080 with the stock E0 trunk alone (measured,
# job 1773194).
case "$ARM" in
  E0) EXP=Holo_E0_fixed128_baseline
      CONFIG=$P3/configs/E0_fixed128_baseline.yml
      SIBLING_NAMES="E0_fixed128"
      ARM_PART=a100; ARM_GRES=gpu:a100:1 ;;
  E1) EXP=Holo_E1_addition_fixed128_spatial_B6_latent
      CONFIG=$P3/configs/E1_addition_fixed128_spatial_B6_latent.yml
      SIBLING_NAMES="E1_B6_latent"
      ARM_PART=v100; ARM_GRES=gpu:v100:1 ;;
  E1R) EXP=Holo_E1_render_fixed128_spatial_B6_latent
      CONFIG=$P3/configs/E1_render_fixed128_spatial_B6_latent.yml
      SIBLING_NAMES="E1_render"
      ARM_PART=a100; ARM_GRES=gpu:a100:1 ;;
  *)  echo "[chain] unknown arm '$ARM' (expected E0, E1 or E1R)"; exit 2 ;;
esac

if [ -n "$SLURM_JOB_PARTITION" ] && [ "$SLURM_JOB_PARTITION" != "$ARM_PART" ]; then
    echo "[chain] WARNING: this job is on partition '$SLURM_JOB_PARTITION' but"
    echo "[chain] arm $ARM is configured for '$ARM_PART'. Training will still"
    echo "[chain] run, but successors will be submitted to $ARM_PART -- the arm"
    echo "[chain] would then straddle two GPU types. Cancel and resubmit with"
    echo "[chain]   sbatch --partition=$ARM_PART --gres=$ARM_GRES $SCRIPT $ARM"
fi

EXP_DIR=$REPO/experiments/$EXP                       # basicsr-managed
CHAIN_DIR=$REPO/experiments/Phase3_chain_state_$EXP  # our bookkeeping only
DONE_FILE=$CHAIN_DIR/TRAINING_DONE
ABORT_FILE=$CHAIN_DIR/CHAIN_ABORTED
COUNT_FILE=$CHAIN_DIR/CHAIN_COUNT
STAB_FILE=$EXP_DIR/STABILITY_FAILURE.json
FINAL_CKPT=$EXP_DIR/models/net_g_300000.pth
MAX_CHAIN=6          # E0 needs ~2 jobs at 0.42 s/iter, E1 ~3 at 0.55 s/iter
MIN_RUNTIME=1800     # a healthy job runs ~23h; anything under 30 min is a crash

cd "$REPO"
mkdir -p "$CHAIN_DIR"
echo "[chain] arm=$ARM experiment=$EXP job=$SLURM_JOB_ID on $(hostname)"

# --- 1. already finished? ----------------------------------------------------
if [ -f "$DONE_FILE" ]; then
    echo "[chain] TRAINING_DONE present -> finished. Exiting, no successor."
    exit 0
fi
if [ -f "$FINAL_CKPT" ]; then
    echo "[chain] $FINAL_CKPT already exists -> 300k complete. Writing TRAINING_DONE."
    touch "$DONE_FILE"
    exit 0
fi

# --- 2. a previous job flagged a crash --------------------------------------
if [ -f "$ABORT_FILE" ]; then
    echo "[chain] CHAIN_ABORTED present -> a previous job died early. Exiting."
    echo "[chain] Investigate, then 'rm $ABORT_FILE' before restarting the chain."
    exit 0
fi

# --- 3. STABILITY GUARD: never restart into the same abort ------------------
if [ -f "$STAB_FILE" ]; then
    echo "[chain] STABILITY_FAILURE.json present -> OPTIMIZATION / STABILITY"
    echo "[chain] FAILURE recorded at:"
    cat "$STAB_FILE"
    echo "[chain] The chain STOPS. This is investigated and reported, not"
    echo "[chain] restarted. Any design change requires a NEW experiment identity."
    touch "$ABORT_FILE"
    exit 0
fi

# --- 4. refuse to run a second trainer against the same experiment ----------
# Two processes writing one experiment directory would corrupt the checkpoints
# and the training states. Check for any of MY running jobs for this arm.
for NAME in $SIBLING_NAMES p3_chain; do
    OTHERS=$(squeue -h -u "$USER" -n "$NAME" -t RUNNING -o "%i" 2>/dev/null \
             | grep -v "^${SLURM_JOB_ID}$" || true)
    for J in $OTHERS; do
        if scontrol show job "$J" 2>/dev/null | grep -q "$EXP\|Command=.*$ARM"; then
            echo "[chain] job $J is ALREADY training $EXP -> refusing to start a"
            echo "[chain] second trainer against the same experiment directory."
            echo "[chain] (Submit with --dependency=afterany:$J instead.)"
            exit 1
        fi
    done
done

# --- 5. safety cap on chained jobs ------------------------------------------
COUNT=0
[ -f "$COUNT_FILE" ] && COUNT=$(cat "$COUNT_FILE")
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNT_FILE"
echo "[chain] chain job #$COUNT of max $MAX_CHAIN"
if [ "$COUNT" -gt "$MAX_CHAIN" ]; then
    echo "[chain] exceeded the cap of $MAX_CHAIN jobs -> stopping. Investigate."
    exit 1
fi

# --- 6. queue the successor BEFORE training (survives a SIGKILL) ------------
SUCC=$(sbatch --partition="$ARM_PART" --gres="$ARM_GRES" \
       --dependency=afterany:"$SLURM_JOB_ID" "$SCRIPT" "$ARM" | awk '{print $NF}')
echo "[chain] queued successor job $SUCC on $ARM_PART (afterany:$SLURM_JOB_ID)"

# --- 7. environment ---------------------------------------------------------
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

# --- 8. report the resume point (no yml edit; basicsr auto-resumes) ---------
LATEST=$(ls -t "$EXP_DIR"/training_states/*.state 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "[chain] latest state on disk: $LATEST -> AUTO-RESUME"
else
    echo "[chain] no .state on disk -> START FRESH"
fi

python -u "$P3/scripts/write_run_metadata.py" --config "$CONFIG" \
    --run-id "$SLURM_JOB_ID"

# --- 9. train ---------------------------------------------------------------
START_TS=$(date +%s)
python -u basicsr/train.py -opt "$CONFIG" --launcher none
TRAIN_RC=$?
ELAPSED=$(( $(date +%s) - START_TS ))
echo "[chain] training exited rc=$TRAIN_RC after ${ELAPSED}s"

# --- 10. finished? ----------------------------------------------------------
if [ -f "$FINAL_CKPT" ]; then
    echo "[chain] net_g_300000.pth present -> 300k complete. Writing TRAINING_DONE."
    touch "$DONE_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null \
        && echo "[chain] cancelled now-unneeded successor $SUCC"
    exit 0
fi

# --- 11. stability failure during THIS job ----------------------------------
if [ -f "$STAB_FILE" ]; then
    echo "[chain] STABILITY_FAILURE.json written during this job -> stopping the chain."
    touch "$ABORT_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null \
        && echo "[chain] cancelled successor $SUCC"
    exit 1
fi

# --- 12. crash guard --------------------------------------------------------
if [ "$ELAPSED" -lt "$MIN_RUNTIME" ]; then
    echo "[chain] exited after ${ELAPSED}s (< ${MIN_RUNTIME}s) without finishing"
    echo "[chain] -> treating as a persistent crash. Aborting the chain."
    touch "$ABORT_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null \
        && echo "[chain] cancelled successor $SUCC"
    exit 1
fi

echo "[chain] walltime kill after ${ELAPSED}s -> successor $SUCC will resume."
