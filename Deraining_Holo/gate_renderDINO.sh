#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=05:00:00
#SBATCH --export=NONE
#SBATCH --job-name=gate_renderDINO
#SBATCH --output=experiments/Holo_gate_state/gate_renderDINO_%j.out
#
# =============================================================================
# STABILITY GATE for E1 arm renderDINO -- 16000 iterations, ~2-3 h.
#
# Purpose: answer "is this run healthy?" BEFORE committing ~3 GPU-days. Two
# previous attempts (DEVLOG Steps 22, 24) trained a degenerate model while the
# training loss looked perfectly normal.
#
# Length follows the FiLM schedule: warmup 5k + ramp 5k, so FiLM is only fully
# on from iter 10k. 16k gives ~6k iterations of fully-on FiLM to judge.
#
# PASS = final val PSNR >= 18 dB  AND  >= (best warmup val - 1 dB).
# The second is the real test and it is free: during warmup film_ramp = 0, so
# the early validations ARE the plain baseline. The gate carries its own control.
#
# Runs on the SAME GPU type as the real arm. Own experiment dir, no chaining,
# does not launch the full run.
#
#     sbatch Deraining_Holo/gate_renderDINO.sh
# =============================================================================

unset SLURM_EXPORT_ENV
REPO=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
cd "$REPO"

module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

EXP=$REPO/experiments/Holo_DINOv2_renderDINO_GATE
# a gate must always start clean, never resume a previous attempt's states
rm -rf "$EXP" "$REPO/tb_logger/Holo_DINOv2_renderDINO_GATE"

echo "[gate] renderDINO: 16000 iters on a100"
python basicsr/train.py -opt Deraining_Holo/Options/Holo_DINOv2_renderDINO_GATE.yml --launcher none
echo "[gate] training exited rc=$?"

LOG=$(ls -t "$EXP"/*.log 2>/dev/null | head -1)
echo "[gate] ---------------- RESULT (renderDINO) ----------------"
python Deraining_Holo/gate_verdict.py "$LOG"
echo "[gate] ------------------------------------------------"
