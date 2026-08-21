#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smoke_pq
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_priorquery_%j.out
#
# Smoke tests for priorquery-render. Inference plus ten optimizer steps on
# random tensors: no dataset write, no experiment identity, no checkpoint.
#
# PINNED TO V100: the step-0 check asserts bit-identity with E0, and cuDNN's
# TF32 default on Ampere perturbs paths that convolve F. Volta has no TF32, so
# the check measures the arithmetic rather than the accelerator.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
python -u dino_analysis_phases/phase3_restoration/scripts/smoke_tests_priorquery_render.py \
    --device cuda || exit 1
echo "SMOKE PRIORQUERY-RENDER DONE"
