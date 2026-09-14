#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --job-name=ref_train
#SBATCH --output=dino_analysis_phases/refiner_e0/results/logs/train_%j.out
#
# Refiner experiment, step 2: the ONE training run. A 118k-parameter refiner
# does not need an a100; float32 is enforced in code (TF32 off).

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

python -u dino_analysis_phases/refiner_e0/train_refiner.py || exit 1
echo "TRAIN DONE"
