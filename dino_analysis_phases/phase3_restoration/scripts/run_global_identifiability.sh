#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_gid
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/global_identifiability/run_%j.out
#
# Feature-space check behind the global-render memorisation reading (DEVLOG
# Step 38): does a per-image global DINO vector of a random 128 crop identify
# its training scene? See global_vector_identifiability.py.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
python -u dino_analysis_phases/phase3_restoration/scripts/global_vector_identifiability.py --split train
