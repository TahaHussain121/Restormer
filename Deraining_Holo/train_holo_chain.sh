#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=holo_chain
#SBATCH --output=experiments/Holo_chain_state/slurm_chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for the Holo 300k baseline.
#
# Submit ONCE (after the current job dies at walltime):
#     sbatch Deraining_Holo/train_holo_chain.sh
# Each job:
#   - exits immediately if TRAINING_DONE marker exists
#   - increments CHAIN_COUNT; exits if it exceeds MAX_CHAIN (safety cap)
#   - queues its OWN successor with --dependency=afterany (runs after this
#     job ends for ANY reason: walltime kill, completion, or crash)
#   - runs the IDENTICAL training command as train_holo.sh
#   - touches TRAINING_DONE if the 300k checkpoint exists, which stops the chain
#
# NO yml edits. basicsr/train.py (lines ~138-149) unconditionally scans
# experiments/<name>/training_states/ and resumes from the highest .state,
# overriding resume_state in the yml. So resume is automatic and uses the
# exact same training code as the running job.
#
# Whole chain runs on v100 (matches the in-flight run and the queue reality —
# a100 was blocked). To flip to a100 later, change the two #SBATCH lines above.
#
# Chain bookkeeping (markers + SLURM logs) lives in experiments/Holo_chain_state/,
# NOT in the experiment dir. basicsr archives a pre-existing experiment dir at a
# fresh start, so keeping markers/logs separate lets basicsr create the experiment
# dir cleanly and keeps CHAIN_COUNT persistent across the whole chain.
# =============================================================================

unset SLURM_EXPORT_ENV

REPO=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
SCRIPT=$REPO/Deraining_Holo/train_holo_chain.sh
EXP_DIR=$REPO/experiments/Holo_Baseline_Restormer      # basicsr-managed (states, models)
CHAIN_DIR=$REPO/experiments/Holo_chain_state           # our bookkeeping (markers, logs)
DONE_FILE=$CHAIN_DIR/TRAINING_DONE
ABORT_FILE=$CHAIN_DIR/CHAIN_ABORTED
COUNT_FILE=$CHAIN_DIR/CHAIN_COUNT
FINAL_CKPT=$EXP_DIR/models/net_g_300000.pth
MAX_CHAIN=5
MIN_RUNTIME=1800   # seconds. Training exiting faster than this, without producing
                   # the 300k checkpoint, is treated as a crash (a walltime kill
                   # runs ~23h, so this can never falsely trigger on a healthy run).

cd "$REPO"
mkdir -p "$CHAIN_DIR"

# --- 1. stop if training already complete -----------------------------------
if [ -f "$DONE_FILE" ]; then
    echo "[chain] TRAINING_DONE present -> training finished. Exiting, no successor."
    exit 0
fi

# --- 1b. stop if a previous job flagged a crash -----------------------------
if [ -f "$ABORT_FILE" ]; then
    echo "[chain] CHAIN_ABORTED present -> a previous job crashed early. Exiting."
    echo "[chain] Investigate, then 'rm $ABORT_FILE' before restarting the chain."
    exit 0
fi

# --- 2. safety cap on number of chained jobs --------------------------------
COUNT=0
[ -f "$COUNT_FILE" ] && COUNT=$(cat "$COUNT_FILE")
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNT_FILE"
echo "[chain] chain job #$COUNT of max $MAX_CHAIN"
if [ "$COUNT" -gt "$MAX_CHAIN" ]; then
    echo "[chain] exceeded cap of $MAX_CHAIN jobs -> stopping. Investigate manually."
    exit 1
fi

# --- 3. queue successor (starts after THIS job ends for any reason) ---------
SUCC=$(sbatch --dependency=afterany:"$SLURM_JOB_ID" "$SCRIPT" | awk '{print $NF}')
echo "[chain] queued successor job $SUCC (dependency afterany:$SLURM_JOB_ID)"

# --- 4. environment (identical to train_holo.sh) ----------------------------
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

# --- 5. report resume point (no yml edit; basicsr auto-resumes from latest) --
LATEST=$(ls -t "$EXP_DIR"/training_states/*.state 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "[chain] latest state on disk: $LATEST  (basicsr will auto-resume from it)"
else
    echo "[chain] no .state on disk -> fresh start (resume_state stays ~)"
fi

# --- 6. run the IDENTICAL training command ----------------------------------
START_TS=$(date +%s)
python basicsr/train.py -opt Deraining_Holo/Options/Holo_Baseline_Restormer.yml --launcher none
TRAIN_RC=$?
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "[chain] training process exited rc=$TRAIN_RC after ${ELAPSED}s"

# NOTE: steps 7-8 run ONLY if python returned on its own. A walltime kill
# terminates this script too, so the successor (afterany) just proceeds and
# resumes — which is the desired behaviour.

# --- 7. completion: 300k checkpoint exists -> stop the chain -----------------
if [ -f "$FINAL_CKPT" ]; then
    echo "[chain] net_g_300000.pth present -> 300k complete. Writing TRAINING_DONE."
    touch "$DONE_FILE"
    [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null && echo "[chain] cancelled now-unneeded successor $SUCC"
    exit 0
fi

# --- 8. crash guard: exited too fast without completing -> abort the chain ---
if [ "$ELAPSED" -lt "$MIN_RUNTIME" ]; then
    echo "[chain] training exited after ${ELAPSED}s (< ${MIN_RUNTIME}s) without the 300k"
    echo "[chain] checkpoint -> assuming a persistent crash. Aborting the chain."
    touch "$ABORT_FILE"
    if [ -n "$SUCC" ]; then
        scancel "$SUCC" 2>/dev/null && echo "[chain] cancelled queued successor $SUCC"
    fi
    exit 1
fi

echo "[chain] exited after ${ELAPSED}s without 300k ckpt -> successor will resume."
