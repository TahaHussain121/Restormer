#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=wo1_recheck
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo1_verification/slurm_%j.out
#
# =============================================================================
# WORK ORDER 1, Task 1.3 -- fixed-128 DINO layer re-check (inference only).
#
# WHY A JOB: this is ~1000 ViT-B/14 forwards. On the login node it ran at ~800%
# CPU and was SIGTERM'd by the node watchdog after ~7 minutes (exit 143).
# It must run on a compute node.
#
# Writes ONLY under phase3_restoration/results/wo1_verification/.
# Trains nothing, creates no config, no architecture, no experiment directory.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

# NOTE: on GPU the two extractor instances differ by float32 non-determinism
# (max abs ~4e-4 on features of norm ~1.4e3); the bitwise-identical result is the
# CPU one. Kept in a separate file so the two never overwrite each other.
python -u dino_analysis_phases/phase3_restoration/scripts/verify_phase1_equivalence.py \
    --device cuda \
    --out dino_analysis_phases/phase3_restoration/results/wo1_verification/phase1_equivalence_cuda.json

python -u dino_analysis_phases/phase3_restoration/scripts/layer_recheck_fixed128.py \
    --device cuda --split val --seed 0 --num-mean-samples 150 \
    --protocols random center
