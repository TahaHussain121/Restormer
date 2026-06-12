#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --export=NONE
#SBATCH --job-name=holo_baseline
#SBATCH --output=experiments/Holo_Baseline_Restormer/slurm_%j.out
#
# =============================================================================
# Holo Baseline Restormer — 300k progressive run.
# Currently set to v100 (a100 queue was full / GRES-limited). Switch the two
# #SBATCH lines above back to a100 when an A100 slot is available — it is faster.
#
# Max walltime is 23h. The full 300k-iter run will NOT fit in one job;
# expect 2-3 jobs total. Checkpoints + training states are saved every 2000
# iters, so a walltime kill loses at most ~2k iters of progress.
#
# RESUME after a walltime kill:
#   1. find the latest training state:
#        ls -t experiments/Holo_Baseline_Restormer/training_states/ | head -1
#   2. edit the yml, set:
#        path:
#          resume_state: experiments/Holo_Baseline_Restormer/training_states/<ITER>.state
#      (it is ~ for the very first job; point it at the .state for every resume)
#   3. resubmit:
#        sbatch Deraining_Holo/train_holo.sh
# =============================================================================

unset SLURM_EXPORT_ENV

module load python

conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

python basicsr/train.py -opt Deraining_Holo/Options/Holo_Baseline_Restormer.yml --launcher none
