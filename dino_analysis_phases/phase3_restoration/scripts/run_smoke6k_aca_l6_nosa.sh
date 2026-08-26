#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_sm6k_acaL6nosa
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke6k_aca_l6_nosa_%j.out
#
# 6000-iteration SMOKE TRAINING run for aca-L6-nosa, GATE ON, plus peak VRAM.
#
# Closes the part the architectural suite (29/29) cannot reach: the basicsr
# INTEGRATION path, the gate ENFORCED across the iteration-5000 boundary, and
# the two numbers that are otherwise only projected -- measured s/iter and peak
# VRAM. On the affm arm the equivalent run caught two real bugs a 72-check
# architectural suite missed.
#
# ON a100, UNLIKE aca-L6's smoke WHICH RAN ON v100. Two reasons, both
# deliberate:
#   1. the a100 GRES block that forced aca-L6's smoke onto v100 is gone;
#   2. this arm's headline claim is that removing F_sa costs nothing and saves
#      parameters, so the s/iter and peak-VRAM numbers are worth measuring on
#      the CARD THE REAL RUN USES rather than converted from v100.
# It will QUEUE behind the three live a100 arms. That is expected.
#
# Throwaway identity SMOKE6K_aca_L6_nosa -- it cannot touch the real experiment
# dir and never appends to the real devlog.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
REAL=$P3/configs/aca_render_fixed128_L6_nosa_latent.yml
CFG=$P3/results/wo2_implementation/smoke6k_configs/SMOKE6K_aca_L6_nosa.yml
NAME=SMOKE6K_aca_L6_nosa

echo "===== peak VRAM (fresh process, one arm per process) ====="
# aca-L6 is the comparison that matters: same arm minus F_sa. addition-render
# is included because it is the ladder's other reference point. SAME METRIC
# throughout -- torch.cuda.max_memory_allocated, never nvidia-smi's
# whole-process figure. Mixing those two overstated the ACA's memory cost once
# already (31,450 vs 25,117 MiB, 2026-08-22).
for C in E1_addition_render_fixed128_spatial_B6_latent \
         aca_render_fixed128_L6_latent \
         aca_render_fixed128_L6_nosa_latent; do
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
echo "[smoke6k] s/iter (a100 -- the card the real run uses):"
grep -o "time (data): [0-9.]*" $EXP/train_*.log | awk -F': ' '{s+=$2;n++} END{printf "  %.4f s/iter over %d prints -> 300k = %.1f h\n", s/n, n, 300000*(s/n)/3600}'
echo "[smoke6k] validation points:"; grep "Validation ValSet" $EXP/train_*.log
echo "[smoke6k] alpha trajectory (aca_alpha, deduplicated):"
grep -o "dino/aca_alpha: [0-9.e+-]*" $EXP/train_*.log | awk '!seen[$2]++' | head -20

# ---- the three assertions specific to THIS arm ----------------------------
echo "[smoke6k] AFFM keys present? (single layer -- expect 0):"
grep -c "affm_w_b" $EXP/train_*.log || true
echo "[smoke6k] SA observation tags present? (F_sa is REMOVED -- expect 0):"
grep -cE "dino/aca_(temp|entropy)_sa_" $EXP/train_*.log || true
echo "[smoke6k] aca_ca_to_sa_ratio (expect EXACTLY 0.0 on every print):"
grep -o "dino/aca_ca_to_sa_ratio: [0-9.e+-]*" $EXP/train_*.log | awk '!seen[$2]++' | head -5
echo "[smoke6k] CA observation tags present? (expect 6 heads' worth):"
grep -oE "dino/aca_temp_ca_h[0-9]+" $EXP/train_*.log | sort -u | tr '\n' ' '; echo

echo "[smoke6k] stability failures (expect none):"; ls $EXP/STABILITY_FAILURE* 2>/dev/null || echo "  none"
exit $RC
