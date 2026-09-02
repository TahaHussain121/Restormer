#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_inregime
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/inregime/inregime_%j.out
#
# =============================================================================
# IN-REGIME FULL-FRAME INFERENCE — inference only, no training.
#
# THE QUESTION. Every arm trains on 128 crops and is evaluated on whole 256
# frames, so evaluation runs outside the regime the model and its prior were
# conditioned in. Section 7.4 of the chapter measures that the DINO prior really
# does differ between the two regimes. Does removing the mismatch at INFERENCE
# time recover anything?
#
# Four ways to produce a 256 prediction, all scored against the SAME real 256
# target so the numbers are directly comparable:
#
#   full256      whole frame, one pass.        out of regime, full resolution
#   resize128    frame downscaled to 128,      IN regime, HALF resolution
#                predicted, upscaled to 256
#   tiled128 ov0 four 128 tiles, stitched      IN regime, full resolution,
#                                              hard seams
#   tiled128 ov64 nine overlapping tiles,      IN regime, full resolution,
#                blended                       borders down-weighted
#
# WHAT EACH ONE IS FOR.
#
#   resize128 tests the obvious idea directly. Expect it to LOSE: downscaling
#   destroys the high-frequency content this project exists to restore, and the
#   upscale cannot bring it back. Reported either way.
#
#     THE TRAP THIS AVOIDS: scoring a 128 prediction against a DOWNSCALED 128
#     target would look like a large win and would be meaningless -- an easier
#     target, not a better model, and not comparable to any other number in the
#     study. The prediction is upscaled to 256 and scored against the real
#     target, exactly like every other full-frame number.
#
#   tiled128 is the version that keeps resolution. It also TESTS A PHASE-5
#   PREDICTION: if the crop drift concentrates at borders, then overlapping and
#   blending the tiles (which down-weights borders) should beat hard seams. That
#   turns Phase 5 from a descriptive measurement into one that predicts a
#   restoration outcome.
#
# ARM. addition-render at its validation-selected checkpoint (204,000) -- the
# same arm the misalignment sweep uses, so the two controls share a model.
# VALIDATION split, n=339: neither control was pre-registered, so neither
# touches the locked test split.
#
#   sbatch run_inregime_inference.sh
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

OUT=$P3/results/inregime
mkdir -p $OUT

if [ ! -f "$WEIGHTS" ]; then echo "MISSING WEIGHTS $WEIGHTS"; exit 1; fi

# name:protocol:overlap
for SPEC in "full256:full256:0" "resize128:resize128:0" \
            "tiled_ov0:tiled128:0" "tiled_ov64:tiled128:64"; do
  NAME=${SPEC%%:*}; REST=${SPEC#*:}
  PROTOCOL=${REST%%:*}; OV=${REST##*:}

  EXTRA=""
  if [ "$PROTOCOL" = "tiled128" ] && [ "$OV" != "0" ]; then
    EXTRA="--tile-overlap $OV"
  fi

  echo "=== $NAME  (protocol=$PROTOCOL overlap=$OV) ==="
  python -u $P3/scripts/predict_phase3.py \
      --config $CONFIG --weights $WEIGHTS --split $SPLIT \
      --protocol $PROTOCOL $EXTRA \
      --out-root $OUT/$NAME --device cuda || exit 1

  # UNCHANGED metric script, default threshold/dilate, and the REAL 256 target
  # directory in every condition.
  python -u Deraining_Holo/masked_metrics.py \
      --pred_dir $OUT/$NAME/${PROTOCOL}_${SPLIT}/raw \
      --gt_dir $DS/${SPLIT}_clean \
      --csv $OUT/per_image_${NAME}_${SPLIT}.csv || exit 1

  python -u Deraining_Holo/analyze_sharpness.py \
      --pred_dir $OUT/$NAME/${PROTOCOL}_${SPLIT}/raw \
      --gt_dir $DS/${SPLIT}_clean --noisy_dir $DS/${SPLIT}_verynoisy \
      --csv $OUT/sharpness_${NAME}_${SPLIT}.csv \
      --out $OUT/radial_${NAME}_${SPLIT}.png || exit 1
done

echo "IN-REGIME SWEEP DONE"
