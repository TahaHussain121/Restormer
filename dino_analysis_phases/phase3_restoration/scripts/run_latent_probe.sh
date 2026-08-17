#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_latprobe
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/latprobe_%j.out
#
# Inference-only probe: how big are the latent blocks' own residual branches,
# as a reference scale for E1's injection_ratio. Loads the 5k DIAG_E0 checkpoint,
# hooks the 8 latent blocks, no training, no experiment directory.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

python -u dino_analysis_phases/phase3_restoration/scripts/latent_residual_scales.py \
    --batches 5 --device cuda
