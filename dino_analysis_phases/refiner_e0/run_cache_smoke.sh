#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=ref_cache
#SBATCH --output=dino_analysis_phases/refiner_e0/results/logs/cache_smoke_%j.out
#
# Refiner experiment, step 1: cache the frozen E0's raw float32 full-frame
# outputs for val and train, then run the real-data smoke checks. v100 because
# E0's recorded full256 predictions were made on v100, so the reproduction
# check can be bit-exact. Submitted from the repo root.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

R=dino_analysis_phases/refiner_e0
python -u $R/cache_e0_outputs.py --splits val train || exit 1
python -u $R/smoke_refiner.py || exit 1
echo "CACHE+SMOKE DONE"
