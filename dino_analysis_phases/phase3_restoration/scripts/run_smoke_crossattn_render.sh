#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smoke_xatt
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_crossattn_%j.out
#
# Smoke tests for crossattn-render. Inference plus ten optimizer steps on random
# tensors: no dataset write, no experiment identity, no checkpoint.
#
# PINNED TO V100 for the same reason the concat smoke test is: the step-0 check
# asserts bit-identity with E0, and cuDNN's TF32 default on Ampere perturbs any
# path that convolves F. Volta has no TF32, so the check measures the
# arithmetic. It also measures peak VRAM for the addition arm and this arm on
# the same card, which is only comparable on one device.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
python -u dino_analysis_phases/phase3_restoration/scripts/smoke_tests_crossattn_render.py \
    --device cuda || exit 1
echo "SMOKE CROSSATTN-RENDER DONE"
