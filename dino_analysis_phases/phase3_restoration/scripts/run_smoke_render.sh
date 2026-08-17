#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smokeR
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_render_%j.out
# E1-render smoke tests. Assertions only: no training, no experiment identity.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
python -u $P3/scripts/smoke_tests_render.py --device cuda
