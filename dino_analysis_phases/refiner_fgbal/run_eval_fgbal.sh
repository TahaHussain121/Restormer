#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=fgbal_eval
#SBATCH --output=dino_analysis_phases/refiner_fgbal/results/logs/eval_%j.out
#
# Foreground-balanced refiner: evaluate the validation-selected model once,
# three-way against E0 and refiner_e0, then the fixed-case figures.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

test -f experiments/Holo_E0_frozen_noisy_output_fgbalanced_refiner/TRAINING_DONE || { echo "training not done"; exit 1; }
python -u dino_analysis_phases/refiner_fgbal/evaluate_fgbal.py || exit 1
echo "FGBAL EVAL DONE"
