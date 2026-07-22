#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=holo_vn_eval
#SBATCH --output=experiments/Holo_chain_state_verynoisy/eval_%j.out
#
# =============================================================================
# Evaluate the VERYNOISY baseline on the newly-carved held-out test split.
#
# Checkpoint: net_g_292000.pth -- the BEST by validation PSNR (22.4460 dB),
# not the final 300k (22.4374 dB).
#
# CAVEAT recorded in DEVLOG Step 19a: that "best" was chosen using the FULL
# 677-image val set, which at the time included the 338 images now designated
# test. So checkpoint selection saw the test images (the weights never trained
# on them). The top-5 checkpoints span only 0.016 dB, so the practical effect
# is negligible, but the number below is not a fully clean held-out estimate.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/Deraining_Holo

DS=/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset
OUT=./results/Holo_verynoisy_test_292k
CKPT=../experiments/Holo_Baseline_Restormer_verynoisy/models/net_g_292000.pth

python test_holo.py \
  --input_dir  $DS/test_verynoisy \
  --gt_dir     $DS/test_clean \
  --weights    $CKPT \
  --result_dir $OUT/

python masked_metrics.py \
  --pred_dir  $OUT/raw \
  --gt_dir    $DS/test_clean \
  --threshold 0.01 \
  --dilate    3 \
  --csv       $OUT/masked_metrics_per_image.csv
