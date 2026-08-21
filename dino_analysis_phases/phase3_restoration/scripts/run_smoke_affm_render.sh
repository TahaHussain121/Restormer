#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smoke_affm
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_affm_%j.out
#
# =============================================================================
# Smoke tests for affm-render. Inference / one-step only: no experiment
# directory, no checkpoint, no training state, nothing written under
# experiments/. Results land in results/wo2_implementation/.
#
# Pinned to rtx3080 rather than a100 so it cannot queue behind, or compete
# with, the training arms.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

python -u dino_analysis_phases/phase3_restoration/scripts/smoke_tests_affm_render.py \
    --device cuda --seed 100
