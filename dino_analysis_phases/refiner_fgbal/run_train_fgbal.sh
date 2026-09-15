#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=fgbal_train
#SBATCH --output=dino_analysis_phases/refiner_fgbal/results/logs/train_%j.out
#
# Foreground-balanced refiner: the ONE training run. v100, as refiner_e0's
# training (1813217) ran. Submitted from the repo root.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

python -u dino_analysis_phases/refiner_fgbal/train_fgbal.py || exit 1
echo "FGBAL TRAIN DONE"
