#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=E0_fixed128
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_E0_fixed128_baseline/metadata/slurm_%j.out
#
# =============================================================================
# E0-Fixed — 300k iterations, fixed 128x128 crops, batch 8, no DINO, from
# scratch. The fair control for E1-N-Fixed.
#
# HARDWARE. a100 (40 GB). NOT rtx3080: Restormer at 128^2 x batch 8 OOMs on a
# 10 GB card (measured, job 1773194 -- the stock E0 trunk alone exceeded 9.6 GB).
# a100 is faster if a slot is free; switch both #SBATCH lines together.
#
# WALLTIME. 23 h max, ~0.8 s/iter on V100 => expect 2-3 jobs for 300k.
# Checkpoints and training states are written every 2000 iters, so a walltime
# kill costs at most ~2k iterations.
#
# AUTO-RESUME. basicsr/train.py:138-149 scans
# experiments/Holo_E0_fixed128_baseline/training_states/ and resumes from the
# highest .state, overriding the yml. Resubmitting this script therefore
# CONTINUES the run; it never restarts it. A genuine clean rerun requires a new
# experiment identity (a new `name:`), never a deletion.
#
# Before the FIRST submission, confirm the directory does not exist:
#   ls experiments/Holo_E0_fixed128_baseline    # must be "No such file"
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

python -u dino_analysis_phases/phase3_restoration/scripts/write_run_metadata.py \
    --config dino_analysis_phases/phase3_restoration/configs/E0_fixed128_baseline.yml

python -u basicsr/train.py \
    -opt dino_analysis_phases/phase3_restoration/configs/E0_fixed128_baseline.yml \
    --launcher none
