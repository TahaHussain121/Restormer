#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_sm6kA
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke6k_affm_render_%j.out
#
# 6000-iteration SMOKE TRAINING run for affm-render, GATE ON.
#
# Throwaway identity SMOKE6K_affm_render_L3691 -- it cannot touch
# experiments/Holo_affm_render_fixed128_spatial_L3691_latent, and it never
# appends to the real devlog. 6000 iterations passes ratio_rules_start_iter
# (5000), so the ratio rules are genuinely ENFORCED for the last 1000
# iterations rather than only measured.
#
# ON v100, NOT a100: priorquery-render holds the a100 GRES association, so an
# a100 job cannot start. That is fine HERE because this run is thrown away --
# it answers "does the arm train end to end and where do the numbers sit", not
# "what does this arm score". The REAL run must stay on a100 with TF32, like
# every finished arm; chain_affm_render.sh hardcodes that.
#
# THIS DOES NOT START THE REAL EXPERIMENT. Three questions before 300k a100
# hours are committed:
#   1. does the four-layer path run end to end through basicsr, including the
#      eval256 validation switch at 32x32 tokens?
#   2. where does injection_ratio settle, against addition-render's ~1.03?
#   3. do the AFFM weights move off the uniform 0.25, and in which direction?
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration

python -u $P3/scripts/make_smoke6k_affm_render.py
python -u basicsr/train.py \
    -opt $P3/results/wo2_implementation/smoke6k_configs/SMOKE6K_affm_render.yml \
    --launcher none
RC=$?
echo "[smoke6k] training exited rc=$RC"

EXP=experiments/SMOKE6K_affm_render_L3691
CSV=$EXP/dino_stability.csv
if [ -f "$CSV" ]; then
    echo "[smoke6k] injection_ratio trajectory (every 500 iters):"
    awk -F, 'NR==1 || (NR>1 && ($1 % 500)==0) {printf "  iter %-7s loss %-14s latent %-14s projected %-14s ratio %s\n", $1,$2,$4,$5,$6}' "$CSV"
else
    echo "[smoke6k] NO dino_stability.csv -- the run did not reach the first log"
fi

echo "[smoke6k] AFFM weight trajectory (from the training log):"
grep -o "affm_w_b3: [0-9.e+-]*\|affm_w_b6: [0-9.e+-]*\|affm_w_b9: [0-9.e+-]*\|affm_w_b12: [0-9.e+-]*\|iter: *[0-9,]*" \
    $EXP/train_*.log 2>/dev/null | paste - - - - - | awk 'NR%5==1' | tail -20

echo "[smoke6k] validation points:"
grep "Validation ValSet" $EXP/train_*.log 2>/dev/null

echo "[smoke6k] stability failures (there should be none):"
ls $EXP/STABILITY_FAILURE* 2>/dev/null || echo "  none"
exit $RC
