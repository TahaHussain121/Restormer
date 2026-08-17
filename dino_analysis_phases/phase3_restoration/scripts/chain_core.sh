# =============================================================================
# Shared body of the Phase-3 self-chaining resume drivers.  SOURCED, not run.
#
# Each experiment has its OWN thin wrapper (chain_E0.sh, chain_E1_addition_noisy.sh,
# chain_E1_addition_render.sh) carrying its own #SBATCH job-name, log directory,
# partition and GRES. They all source this file so the LOGIC cannot drift
# between arms while the IDENTITY stays completely separate.
#
# The wrapper must set, before sourcing:
#   EXP        BasicSR experiment name (drives every output path)
#   CONFIG     absolute path to the YAML
#   SELF       absolute path to the wrapper itself (for successor submission)
#   ARM_PART   partition for every job of this arm
#   ARM_GRES   gres for every job of this arm
#   JOB_LABEL  short human label used in messages
#
# NOTHING HERE IS EVER OVERWRITTEN:
#   SLURM stdout      results/<EXP>/logs/chain_<jobid>.out     unique per job
#   BasicSR train log experiments/<EXP>/train_<EXP>_<stamp>.log unique per job
#   run metadata      results/<EXP>/metadata/run_metadata[_resume_N].json
#   stability CSV     experiments/<EXP>/dino_stability.csv     APPEND only
#   stability record  experiments/<EXP>/STABILITY_FAILURE_<rule>_iter<N>_job<J>.json
#   devlog            appended, never rewritten
# The only files rewritten by design are the chain's own counters/flags
# (CHAIN_COUNT, TRAINING_DONE, CHAIN_ABORTED, STABILITY_FAILURE.json), which
# are state markers, not records -- every record has an archived twin.
# =============================================================================

REPO=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
P3=$REPO/dino_analysis_phases/phase3_restoration

EXP_DIR=$REPO/experiments/$EXP                       # basicsr-managed
CHAIN_DIR=$REPO/experiments/Phase3_chain_state_$EXP  # our bookkeeping only
LOG_DIR=$P3/results/$EXP/logs
DONE_FILE=$CHAIN_DIR/TRAINING_DONE
ABORT_FILE=$CHAIN_DIR/CHAIN_ABORTED
COUNT_FILE=$CHAIN_DIR/CHAIN_COUNT
STAB_FILE=$EXP_DIR/STABILITY_FAILURE.json
FINAL_CKPT=$EXP_DIR/models/net_g_300000.pth
MAX_CHAIN=${MAX_CHAIN:-6}
MIN_RUNTIME=${MIN_RUNTIME:-1800}

cd "$REPO"
mkdir -p "$CHAIN_DIR" "$LOG_DIR"
echo "[chain] $JOB_LABEL  experiment=$EXP  job=$SLURM_JOB_ID  node=$(hostname)"
echo "[chain] partition=$SLURM_JOB_PARTITION  config=$CONFIG"

# --- 1. already finished? ----------------------------------------------------
if [ -f "$DONE_FILE" ]; then
    echo "[chain] TRAINING_DONE present -> finished. Exiting, no successor."
    exit 0
fi
if [ -f "$FINAL_CKPT" ]; then
    echo "[chain] $FINAL_CKPT exists -> 300k complete. Writing TRAINING_DONE."
    touch "$DONE_FILE"
    exit 0
fi

# --- 2. a previous job flagged a crash --------------------------------------
if [ -f "$ABORT_FILE" ]; then
    echo "[chain] CHAIN_ABORTED present -> a previous job died early. Exiting."
    echo "[chain] Investigate, then 'rm $ABORT_FILE' before restarting."
    exit 0
fi

# --- 3. STABILITY GUARD: never restart into the same abort ------------------
if [ -f "$STAB_FILE" ]; then
    echo "[chain] STABILITY_FAILURE.json present -> OPTIMIZATION / STABILITY"
    echo "[chain] FAILURE already recorded:"
    cat "$STAB_FILE"
    echo "[chain] The chain STOPS. Investigated and reported, never restarted."
    touch "$ABORT_FILE"
    exit 0
fi

# --- 4. refuse a second trainer against the same experiment directory -------
# A LOCK, not a job-name match: the lock is keyed by the experiment, so it also
# catches a trainer launched by a different script (an older driver, a manual
# sbatch of run_train_*.sh). Two processes writing one experiment directory
# would corrupt the checkpoints and the training states.
# The lock lives in CHAIN_DIR, never in EXP_DIR -- BasicSR's make_exp_dirs
# RENAMES an existing experiments/<name>/ on a fresh start, which would carry a
# lock file off into an _archived_ directory.
LOCK=$CHAIN_DIR/RUNNING_JOB
if [ -f "$LOCK" ]; then
    OTHER=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$OTHER" ] && [ "$OTHER" != "$SLURM_JOB_ID" ] \
       && squeue -h -j "$OTHER" -t RUNNING -o "%i" 2>/dev/null | grep -q .; then
        echo "[chain] job $OTHER is ALREADY training $EXP (lock: $LOCK)."
        echo "[chain] Refusing to start a second trainer on the same directory."
        echo "[chain] Wait for it, or attach with --dependency=afterany:$OTHER."
        exit 1
    fi
    echo "[chain] stale lock from job $OTHER (not running) -> taking over"
fi
echo "$SLURM_JOB_ID" > "$LOCK"

# --- 5. safety cap ----------------------------------------------------------
COUNT=0
[ -f "$COUNT_FILE" ] && COUNT=$(cat "$COUNT_FILE")
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNT_FILE"
echo "[chain] chain job #$COUNT of max $MAX_CHAIN"
if [ "$COUNT" -gt "$MAX_CHAIN" ]; then
    echo "[chain] exceeded the cap of $MAX_CHAIN jobs -> stopping. Investigate."
    exit 1
fi

# --- 6. queue the successor BEFORE training (survives the walltime SIGKILL) --
SUCC=$(sbatch --partition="$ARM_PART" --gres="$ARM_GRES" \
       --dependency=afterany:"$SLURM_JOB_ID" "$SELF" | awk '{print $NF}')
echo "[chain] queued successor job $SUCC on $ARM_PART (afterany:$SLURM_JOB_ID)"

# --- 7. environment ---------------------------------------------------------
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

# --- 8. resume point (no yml edit; basicsr auto-resumes from the highest) ----
LATEST=$(ls -t "$EXP_DIR"/training_states/*.state 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "[chain] latest state: $LATEST -> AUTO-RESUME"
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
    echo "[chain] net_g_300000.pth present -> 300k complete. TRAINING_DONE."
    touch "$DONE_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null \
        && echo "[chain] cancelled now-unneeded successor $SUCC"
    exit 0
fi

# --- 11. stability failure during THIS job ----------------------------------
if [ -f "$STAB_FILE" ]; then
    echo "[chain] STABILITY_FAILURE written during this job -> stopping the chain."
    touch "$ABORT_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null \
        && echo "[chain] cancelled successor $SUCC"
    exit 1
fi

# --- 12. crash guard --------------------------------------------------------
if [ "$ELAPSED" -lt "$MIN_RUNTIME" ]; then
    echo "[chain] exited after ${ELAPSED}s (< ${MIN_RUNTIME}s) without finishing"
    echo "[chain] -> persistent crash assumed. Aborting the chain."
    touch "$ABORT_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null \
        && echo "[chain] cancelled successor $SUCC"
    exit 1
fi

echo "[chain] walltime kill after ${ELAPSED}s -> successor $SUCC resumes."
