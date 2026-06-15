#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=holo_eval
#SBATCH --output=experiments/Holo_chain_state/eval_%j.out
#
# Final evaluation of the best-val checkpoint (iter 224k) on the LOCKED test set.
# Outputs raw uint16 predictions + heatmap grids + mean PSNR/SSIM.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/Deraining_Holo

python test_holo.py \
  --input_dir  /home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset/test_noisy \
  --gt_dir     /home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset/test_clean \
  --weights    ../experiments/Holo_Baseline_Restormer/models/net_g_224000.pth \
  --result_dir ./results/Holo_test_224k/
