#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=holo_lqDINO
#SBATCH --output=experiments/Holo_chain_state_lqDINO/slurm_chain_%j.out
#
# =============================================================================
# Self-chaining resume driver for E1 arm B: lqDINO (DINO sees the noisy LQ crop).
#
# Identical mechanism to train_holo_chain_verynoisy.sh, but targets the separate
# experiment "Holo_DINOv2_lqDINO_verynoisy" (clean GT + verynoisy LQ, mixup off,
# RestormerDINO + frozen DINOv2 FiLM with centered pooled features). Its own
# experiment name + chain-state dir means the Exp 2 baseline and the other DINO
# arm are never touched or resumed from -- the two arms can run concurrently.
#
# Partition set to a100 (CONTEXT.md: default going forward). The frozen ViT-B
# forward per step makes this heavier than the Exp 2 baseline; switch to v100 if
# a100 is congested, and record which was used.
#
# Submit ONCE:
#     sbatch Deraining_Holo/train_holo_chain_lqDINO.sh
# =============================================================================

unset SLURM_EXPORT_ENV

REPO=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
SCRIPT=$REPO/Deraining_Holo/train_holo_chain_lqDINO.sh
EXP_DIR=$REPO/experiments/Holo_DINOv2_lqDINO_verynoisy   # basicsr-managed (states, models)
CHAIN_DIR=$REPO/experiments/Holo_chain_state_lqDINO         # our bookkeeping (markers, logs)
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
# DINOv2 is loaded from the explicit local paths in the yml (compute nodes have
# no internet); TORCH_HOME is exported only as a belt-and-braces default.
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

# --- 5. report resume point (no yml edit; basicsr auto-resumes from latest) --
LATEST=$(ls -t "$EXP_DIR"/training_states/*.state 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "[chain] latest state on disk: $LATEST  (basicsr will auto-resume from it)"
else
    echo "[chain] no .state on disk -> fresh start (resume_state stays ~)"
fi

# --- 6. run the training command --------------------------------------------
START_TS=$(date +%s)
python basicsr/train.py -opt Deraining_Holo/Options/Holo_DINOv2_lqDINO_Restormer.yml --launcher none
TRAIN_RC=$?
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "[chain] training process exited rc=$TRAIN_RC after ${ELAPSED}s"

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
