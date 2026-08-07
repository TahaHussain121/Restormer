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
python - "$LOG" <<'PYEOF'
import re, sys
log = open(sys.argv[1]).read()
# val PSNR in order; val_freq is 2000 so val k is iter (k+1)*2000
ps = [float(x) for x in re.findall(r'# psnr: ([0-9.]+)', log)]
# per-print modulation stats
rows = re.findall(r'iter:\s+([\d,]+).*?l_pix: ([0-9.e+-]+)'
                  r'(?:.*?film_g_absmax: ([0-9.e+-]+))?'
                  r'(?:.*?film_g_std: ([0-9.e+-]+))?'
                  r'(?:.*?film_ramp: ([0-9.e+-]+))?', log)
print("  iter    val_psnr")
for i, p in enumerate(ps):
    it = (i + 1) * 2000
    tag = "(warmup: FiLM OFF = baseline)" if it <= 5000 else           "(ramping)" if it < 10000 else "(FiLM full)"
    print(f"  {it:>6}   {p:7.3f}  {tag}")
print("\n  modulation (every 2000 iters):")
for it, lp, gm, gs, rp in rows[::10]:
    print(f"  iter {it:>7}  l_pix={lp}  |g|max={gm or '-'}  g_std={gs or '-'}  ramp={rp or '-'}")

warm = [p for i, p in enumerate(ps) if (i + 1) * 2000 <= 5000]
base = max(warm) if warm else None
fin = ps[-1] if ps else None
c1 = fin is not None and fin >= 18.0
c2 = base is not None and fin is not None and fin >= base - 1.0
print(f"\n  baseline (best warmup val) : {base}")
print(f"  final val PSNR             : {fin}")
print(f"  [1] final >= 18.0          : {'PASS' if c1 else 'FAIL'}")
print(f"  [2] final >= baseline - 1  : {'PASS' if c2 else 'FAIL'}")
print(f"  VERDICT : {'PASS - safe to launch the full run' if (c1 and c2) else 'FAIL - do NOT launch'}")
sys.exit(0 if (c1 and c2) else 1)
PYEOF
echo "[gate] ------------------------------------------------"
