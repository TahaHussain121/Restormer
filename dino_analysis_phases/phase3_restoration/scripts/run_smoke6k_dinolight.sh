#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_sm6kDL
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke6k_dinolight_%j.out
#
# 6000-iteration SMOKE TRAINING run for dinolight-render, GATE ON, plus the
# trained-checkpoint SCALE CHECK that is the whole point of this arm's design.
#
# Throwaway identity SMOKE6K_dinolight_L3691_aca -- it cannot touch the real
# experiment directory and never appends to the real devlog. 6000 iterations
# passes ratio_rules_start_iter (5000), so the ratio rules are genuinely
# ENFORCED for the last 1000 iterations.
#
# ON v100, NOT a100: the affm arm holds the a100 GRES association, and rtx3080's
# 10 GB already OOM'd on the SMALLER arms. Fine here because this run is thrown
# away -- it answers "does it train end to end, and does it survive the scale
# change", not "what does this arm score". The REAL run stays on a100 with TF32.
#
# THIS DOES NOT START THE REAL EXPERIMENT.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
DS=/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset
NAME=SMOKE6K_dinolight_L3691_aca
CFG=$P3/results/wo2_implementation/smoke6k_configs/SMOKE6K_dinolight.yml
OUTD=$P3/results/wo2_implementation/smoke6k_dinolight_scale

python -u $P3/scripts/make_smoke6k_dinolight.py
python -u basicsr/train.py -opt $CFG --launcher none
RC=$?
echo "[smoke6k] training exited rc=$RC"

EXP=experiments/$NAME
CSV=$EXP/dino_stability.csv
if [ -f "$CSV" ]; then
    echo "[smoke6k] injection_ratio trajectory (every 500 iters):"
    awk -F, 'NR==1 || (NR>1 && ($1 % 500)==0) {printf "  iter %-6s loss %-13s latent %-13s projected %-13s ratio %s\n", $1,$2,$4,$5,$6}' "$CSV"
fi
echo "[smoke6k] iterations/second:"
grep -o "time (data): [0-9.]*" $EXP/train_*.log | awk -F': ' '{s+=$2;n++} END{printf "  %.4f s/iter over %d prints -> 300k = %.1f h\n", s/n, n, 300000*(s/n)/3600}'
echo "[smoke6k] validation points:"; grep "Validation ValSet" $EXP/train_*.log
echo "[smoke6k] stability failures (there should be none):"; ls $EXP/STABILITY_FAILURE* 2>/dev/null || echo "  none"

# ---------------- THE SCALE CHECK, on the trained 6k checkpoint -------------
CK=$EXP/models/net_g_6000.pth
if [ ! -f "$CK" ]; then echo "[scale] no 6k checkpoint -- skipping"; exit $RC; fi
for PROTO in crop128 full256; do
  EXTRA=""
  [ "$PROTO" = crop128 ] && EXTRA="--manifest $P3/results/crop_manifests/matched128_val.csv"
  python -u $P3/scripts/predict_phase3.py --config $CFG --weights $CK \
      --split val --protocol $PROTO $EXTRA \
      --out-root ${OUTD}_${PROTO} --device cuda || exit 1
  GT=$DS/val_clean
  [ "$PROTO" = crop128 ] && GT=${OUTD}_${PROTO}/${PROTO}_val/gt
  echo "===== SCALE CHECK: $PROTO / val, net_g_6000 ====="
  python -u Deraining_Holo/masked_metrics.py \
      --pred_dir ${OUTD}_${PROTO}/${PROTO}_val/raw --gt_dir $GT \
      --csv $P3/results/wo2_implementation/smoke6k_dinolight_${PROTO}_val.csv || exit 1
done
echo "[scale] BOTH protocols ran. Compare: crossattn's SAME-weights gap was"
echo "[scale] +2.00 dB at crop128 vs -7.56 dB at full256. Both numbers here are"
echo "[scale] from a 6k model and will be poor; what matters is that full256 is"
echo "[scale] not catastrophically below crop128."
exit $RC
