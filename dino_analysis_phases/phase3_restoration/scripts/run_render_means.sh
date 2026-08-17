#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_rmeans
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/render_means_%j.out
#
# Production centering means for the RENDER domain (E1-render), train split only.
# Same method, sample count and seed as the 1e5 means. Refuses to overwrite.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
S=dino_analysis_phases/phase3_restoration/scripts/compute_production_means.py
python -u $S --domain render --regime train128 --n-samples 1000 --seed 0 --device cuda
python -u $S --domain render --regime eval256  --n-samples 1000 --seed 0 --device cuda
