#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_tilewin
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/inregime/tilewin_%j.out
#
# THE CONFOUND CONTROL for the tiled-overlap gain.
#
# tiled128 at overlap 64 beat both full256 and non-overlapping tiles. Two
# explanations are confounded in that one number:
#
#   (a) BORDER AVOIDANCE. The ramp window down-weights tile borders, which is
#       where Phase 5 measured DINO's features drift most. This is the Phase-5
#       prediction.
#   (b) SELF-ENSEMBLING. Overlapping tiles average up to 4 predictions per
#       pixel, and averaging several predictions helps regardless of borders.
#
# A BOX window at the same overlap averages the SAME tiles with EQUAL weight, so
# it keeps (b) and removes (a). Verified: both windows give identical
# tiles-per-pixel coverage (min 1, max 4).
#
#   box  vs  overlap-0   -> the size of plain self-ensembling
#   ramp vs  box         -> the size of border avoidance, i.e. the Phase-5 part

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration
DS=/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset
EXP=Holo_E1_addition_render_fixed128_spatial_B6_latent
CONFIG=$P3/configs/E1_addition_render_fixed128_spatial_B6_latent.yml
WEIGHTS=experiments/$EXP/models/net_g_204000.pth
OUT=$P3/results/inregime

python -u $P3/scripts/predict_phase3.py \
    --config $CONFIG --weights $WEIGHTS --split val \
    --protocol tiled128 --tile-overlap 64 --tile-window box \
    --out-root $OUT/tiled_ov64_box --device cuda || exit 1

python -u Deraining_Holo/masked_metrics.py \
    --pred_dir $OUT/tiled_ov64_box/tiled128_val/raw --gt_dir $DS/val_clean \
    --csv $OUT/per_image_tiled_ov64_box_val.csv || exit 1

python -u Deraining_Holo/analyze_sharpness.py \
    --pred_dir $OUT/tiled_ov64_box/tiled128_val/raw --gt_dir $DS/val_clean \
    --noisy_dir $DS/val_verynoisy \
    --csv $OUT/sharpness_tiled_ov64_box_val.csv \
    --out $OUT/radial_tiled_ov64_box_val.png || exit 1

echo "TILE WINDOW CONTROL DONE"
