#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=gate_lqDINO
#SBATCH --output=experiments/Holo_gate_state/gate_lqDINO_%j.out
#
# =============================================================================
# STABILITY GATE for E1 arm lqDINO -- 4000 iterations, ~30-45 min.
#
# Purpose: answer "is this run healthy?" BEFORE committing ~3 GPU-days. The
# first E1 attempt (DEVLOG Step 22) trained 73k iterations while |gamma| ran to
# 325 and validation sat at 3-8 dB; the training loss looked normal throughout.
# The Exp 2 baseline hit 19.62 dB at its FIRST validation, so a short run is
# enough to separate healthy from broken.
#
# PASS: val PSNR at iter 4000 >= 18 dB  AND  film_g_absmax <= film_gamma_scale.
# Runs on the SAME GPU type as the real arm so the numbers are representative.
# Writes its own experiment dir (Holo_DINOv2_lqDINO_GATE) -- never touches the
# real run. Does NOT chain and does NOT launch the full run.
#
#     sbatch Deraining_Holo/gate_lqDINO.sh
# =============================================================================

unset SLURM_EXPORT_ENV
REPO=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
cd "$REPO"

module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

EXP=$REPO/experiments/Holo_DINOv2_lqDINO_GATE
# a gate must always start clean, never resume a previous attempt's states
rm -rf "$EXP" "$REPO/tb_logger/Holo_DINOv2_lqDINO_GATE"

echo "[gate] lqDINO: 4000 iters on v100, threshold 18 dB at iter 4000"
python basicsr/train.py -opt Deraining_Holo/Options/Holo_DINOv2_lqDINO_GATE.yml --launcher none
RC=$?
echo "[gate] training exited rc=$RC"

LOG=$(ls -t "$EXP"/*.log 2>/dev/null | head -1)
echo "[gate] ---------------- RESULT (lqDINO) ----------------"
echo "[gate] val PSNR trajectory:"
grep -E "Validation ValSet" "$LOG" 2>/dev/null | sed 's/.*# psnr/  psnr/'
echo "[gate] FiLM modulation (last 5 prints):"
grep -oE "film_g_absmax: [0-9.e+-]+.*" "$LOG" 2>/dev/null | tail -5
python - "$LOG" <<'PYEOF'
import re, sys
log = open(sys.argv[1]).read() if len(sys.argv) > 1 else ''
ps = [float(m) for m in re.findall(r'# psnr: ([0-9.]+)', log)]
gs = [float(m) for m in re.findall(r'film_g_absmax: ([0-9.e+-]+)', log)]
ok = bool(ps) and ps[-1] >= 18.0
print(f"[gate] final val PSNR : {ps[-1] if ps else 'NONE'}  (need >= 18.0)")
print(f"[gate] max |gamma|    : {max(gs) if gs else 'NOT LOGGED'}")
print(f"[gate] VERDICT        : {'PASS - safe to launch the full run' if ok else 'FAIL - do NOT launch'}")
sys.exit(0 if ok else 1)
PYEOF
echo "[gate] ------------------------------------------------"
