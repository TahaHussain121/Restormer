#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_verifyE1
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/verify_e1_%j.out
# Assertion harness ONLY (no train.py legs) against the RENAMED E1 config,
# to confirm the pre-launch checklist on the exact file that will be launched.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
python -u $P3/scripts/smoke_tests.py --device cuda \
    --e1 $P3/configs/E1_addition_noisy_fixed128_spatial_B6_latent.yml \
    --out $P3/results/wo2_implementation/verify_e1_renamed.json
