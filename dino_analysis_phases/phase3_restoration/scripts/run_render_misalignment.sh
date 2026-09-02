#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_misalign
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/render_misalignment/misalign_%j.out
#
# =============================================================================
# RENDER-MISALIGNMENT DOSE-RESPONSE — inference only, no training.
#
# WHY. The render-specificity claim currently rests on ONE binary point: give
# each image somebody else's render and the arm loses 9.094 dB (job 1776802,
# validation, n=339). That is a cliff with nothing on it. It says a completely
# wrong render is catastrophic; it says nothing about how ACCURATELY the render
# must be registered against the radar, which is the first question anyone
# deploying this asks.
#
# WHAT. Displace the render by k pixels along x before it reaches DINO, for
#   k = 0, 1, 2, 4, 8, 16
# and measure the loss at each. The radar, the target and the crop window are
# untouched, so the ONLY factor is render-to-radar registration.
#
# k=0 is included DELIBERATELY as a self-check: it must reproduce the arm's
# recorded validation numbers, which proves this path is the same computation
# as the arm's own evaluation and not a re-implementation that happens to be
# close.
#
# ARM. addition-render, at its validation-selected checkpoint (204,000).
# Chosen because it is the arm the 9.094 dB shuffle control was measured on, so
# the two controls sit on the same model and the same split.
#
# SPLIT. VALIDATION, n=339. This control was not pre-registered, so it does not
# touch the locked test split -- and validation is what the shuffle control
# used, so the dose-response and its endpoint are directly comparable.
#
# BOTH PROTOCOLS, as the project's reporting rule requires.
#
#   sbatch run_render_misalignment.sh
# =============================================================================

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
SPLIT=val

OUT=$P3/results/render_misalignment
mkdir -p $OUT

if [ ! -f "$WEIGHTS" ]; then echo "MISSING WEIGHTS $WEIGHTS"; exit 1; fi

for PROTOCOL in full256 crop128; do
  MANIFEST=""
  if [ "$PROTOCOL" = "crop128" ]; then
    MANIFEST="--manifest $P3/results/crop_manifests/matched128_${SPLIT}.csv"
  fi

  for K in 0 1 2 4 8 16; do
    CELL=$OUT/shift${K}/${PROTOCOL}_${SPLIT}
    echo "=== shift=${K}px  protocol=${PROTOCOL}  split=${SPLIT} ==="

    python -u $P3/scripts/predict_phase3.py \
        --config $CONFIG --weights $WEIGHTS --split $SPLIT \
        --protocol $PROTOCOL $MANIFEST \
        --render-shift $K \
        --out-root $OUT/shift${K} --device cuda || exit 1

    if [ "$PROTOCOL" = "crop128" ]; then
      GT_DIR=$CELL/gt
    else
      GT_DIR=$DS/${SPLIT}_clean
    fi

    # UNCHANGED metric script, default threshold/dilate -- identical to the
    # invocation in run_evaluate.sh, so these numbers are comparable to the
    # arm's own recorded evaluation.
    python -u Deraining_Holo/masked_metrics.py \
        --pred_dir $CELL/raw --gt_dir $GT_DIR \
        --csv $OUT/per_image_shift${K}_${PROTOCOL}_${SPLIT}.csv || exit 1
  done
done

echo "MISALIGNMENT SWEEP DONE"
