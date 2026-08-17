#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080,v100,a100
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smoke_cat
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_concat_%j.out
#
# Smoke tests for concat-render. Inference / one-step only: no training, no
# experiment identity, no checkpoint written. Run this BEFORE the chain.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
python -u dino_analysis_phases/phase3_restoration/scripts/smoke_tests_concat_render.py \
    --device cuda || exit 1
echo "SMOKE CONCAT-RENDER DONE"
