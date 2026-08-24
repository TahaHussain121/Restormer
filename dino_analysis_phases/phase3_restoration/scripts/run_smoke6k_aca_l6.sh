#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_sm6k_acaL6
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke6k_aca_l6_%j.out
#
# 6000-iteration SMOKE TRAINING run for aca-L6, GATE ON, plus peak VRAM.
#
# Closes the part of the work order the architectural suite could not: the
# basicsr INTEGRATION path, the gate ENFORCED across the iteration-5000
# boundary, and the two numbers that were previously only projected --
# measured s/iter and peak VRAM.
#
# Throwaway identity SMOKE6K_aca_L6 -- cannot touch the real experiment dir and
# never appends to the real devlog.
#
# ON v100, NOT a100: the two live arms hold the a100 GRES association
# (AssocGrpGRES), and this run is thrown away anyway -- it answers "does it
# train end to end", not "what does it score". The REAL run goes to a100.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
REAL=$P3/configs/aca_render_fixed128_L6_latent.yml
CFG=$P3/results/wo2_implementation/smoke6k_configs/SMOKE6K_aca_L6.yml
NAME=SMOKE6K_aca_L6

echo "===== peak VRAM (fresh process, one arm per process) ====="
for C in E1_addition_render_fixed128_spatial_B6_latent aca_render_fixed128_L6_latent; do
  echo "--- $C ---"
  python -u $P3/scripts/measure_peak_vram.py --config $P3/configs/$C.yml 2>&1 | tail -1
done

echo "===== 6k smoke training ====="
python -u $P3/scripts/make_smoke6k_aca.py --config $REAL
python -u basicsr/train.py -opt $CFG --launcher none
RC=$?
echo "[smoke6k] training exited rc=$RC"

EXP=experiments/$NAME
if [ -f "$EXP/dino_stability.csv" ]; then
  echo "[smoke6k] injection_ratio trajectory (every 500 iters):"
  awk -F, 'NR==1 || (NR>1 && ($1 % 500)==0) {printf "  iter %-6s loss %-13s latent %-13s projected %-13s ratio %s\n",$1,$2,$4,$5,$6}' $EXP/dino_stability.csv
fi
echo "[smoke6k] s/iter:"
grep -o "time (data): [0-9.]*" $EXP/train_*.log | awk -F': ' '{s+=$2;n++} END{printf "  %.4f s/iter over %d prints -> 300k = %.1f h\n", s/n, n, 300000*(s/n)/3600}'
echo "[smoke6k] validation points:"; grep "Validation ValSet" $EXP/train_*.log
echo "[smoke6k] alpha trajectory (aca_alpha, deduplicated):"
grep -o "dino/aca_alpha: [0-9.e+-]*" $EXP/train_*.log | awk '!seen[$2]++' | head -20
echo "[smoke6k] AFFM keys present? (aca-L6 has NO AFFM -- expect none):"
grep -c "affm_w_b" $EXP/train_*.log || true
echo "[smoke6k] stability failures (expect none):"; ls $EXP/STABILITY_FAILURE* 2>/dev/null || echo "  none"
exit $RC
