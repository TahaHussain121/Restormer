#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --job-name=ref_eval
#SBATCH --output=dino_analysis_phases/refiner_e0/results/logs/eval_%j.out
#
# Refiner experiment, step 3, only after selection is recorded: cache E0 on the
# test split (reproduction-checked against its recorded predictions), evaluate
# A = E0 and B = E0 + refiner on val and test, then the qualitative panels.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

R=dino_analysis_phases/refiner_e0
test -f experiments/Holo_E0_frozen_noisy_output_residual_refiner/TRAINING_DONE || { echo "training not done"; exit 1; }
python -u $R/cache_e0_outputs.py --splits test || exit 1
python -u $R/evaluate_refiner.py --splits val test || exit 1
python -u $R/make_panels.py || exit 1
echo "REFINER EVAL DONE"
