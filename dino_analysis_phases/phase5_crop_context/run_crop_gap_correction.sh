#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --job-name=p5_gap
#SBATCH --output=dino_analysis_phases/phase5_crop_context/results/crop_gap_correction/run_%j.out
#
# Step 1 of the crop-gap correction plan (DEVLOG Step 39): can a small learned
# correction move crop-regime DINO features onto full-frame features? Feature
# space only, no training arm. See crop_gap_correction.py.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
python -u dino_analysis_phases/phase5_crop_context/crop_gap_correction.py --split train
