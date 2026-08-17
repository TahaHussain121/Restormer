#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smoke_cat
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_concat_%j.out
#
# Smoke tests for concat-render. Inference / one-step only: no training, no
# experiment identity, no checkpoint written. Run this BEFORE the chain.
#
# PINNED TO V100, ON PURPOSE. The step-0 check asserts that this arm's output
# IS E0's output, at a 1e-6 tolerance. That holds bit-exactly in true fp32
# (measured 0.000e+00 on V100 and on Ampere with TF32 disabled), but cuDNN runs
# convolutions in TF32 by default on Ampere and newer, and the fused conv sums
# 1152 input channels where E0 applies no convolution at all -- so only the
# concat side loses mantissa bits, and the check reads 2.409e-04 on an RTX 3080
# (job 1781843). Volta has no TF32, so V100 measures the arithmetic itself
# rather than the accelerator's precision mode. Training still runs on a100;
# TF32 there is the same regime every other Restormer conv already uses.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
python -u dino_analysis_phases/phase3_restoration/scripts/smoke_tests_concat_render.py \
    --device cuda || exit 1
echo "SMOKE CONCAT-RENDER DONE"
