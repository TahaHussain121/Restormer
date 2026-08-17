#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_means
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/means_%j.out
#
# =============================================================================
# WORK ORDER 2, Step 2 -- the two PRODUCTION centering means (1e5, B6, train).
#
# A: 1e5_B6_train128_dino224_mean.pt   1000 train images, random 128 crop -> 224
# B: 1e5_B6_eval256_dino448_mean.pt    1000 train images, full 256        -> 448
#
# Inference only. Train split only. Refuses to overwrite an existing production
# mean (a mean is part of an experiment identity) unless --force is passed.
# Never touches the WO1 verification means or dino_spatial_layer_means.pt.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

SCRIPT=dino_analysis_phases/phase3_restoration/scripts/compute_production_means.py

python -u $SCRIPT --regime train128 --n-samples 1000 --seed 0 --device cuda
python -u $SCRIPT --regime eval256  --n-samples 1000 --seed 0 --device cuda
