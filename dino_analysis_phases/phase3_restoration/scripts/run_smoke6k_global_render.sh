#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_sm6kG
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke6k_global_render_%j.out
#
# 6000-iteration SMOKE TRAINING run for global-render, GATE ON.
#
# Throwaway identity SMOKE6K_global_addition_render_B6_latent -- it cannot touch
# experiments/Holo_global_addition_render_fixed128_B6_latent, and it never
# appends to the real devlog. 6000 iterations passes ratio_rules_start_iter
# (5000), so the ratio rules are genuinely ENFORCED for the last 1000 iterations
# rather than only measured. ~0.45 s/iter on a100 -> ~45 min plus validation.
#
# This does NOT start the real experiment. It answers one question before 300k
# GPU-hours are committed: does the pooled prior behave, and where does
# injection_ratio settle relative to E1-render's ~1.03?
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration

python -u $P3/scripts/make_smoke6k_global_render.py
python -u basicsr/train.py \
    -opt $P3/results/wo2_implementation/smoke6k_configs/SMOKE6K_global_render.yml \
    --launcher none
RC=$?
echo "[smoke6k] training exited rc=$RC"

CSV=experiments/SMOKE6K_global_addition_render_B6_latent/dino_stability.csv
if [ -f "$CSV" ]; then
    echo "[smoke6k] injection_ratio trajectory (every 500 iters):"
    awk -F, 'NR==1 || (NR>1 && ($1 % 500)==0) {printf "  iter %-7s loss %-14s latent %-14s projected %-14s ratio %s\n", $1,$2,$4,$5,$6}' "$CSV"
else
    echo "[smoke6k] NO dino_stability.csv -- the run did not reach the first log"
fi
exit $RC
