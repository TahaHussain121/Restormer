#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_eval
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/eval_%j.out
#
# =============================================================================
# Phase-3 evaluation driver. ONE protocol, ONE split, ONE experiment per job.
#
#   sbatch run_evaluate.sh <CONFIG> <WEIGHTS> <SPLIT> <PROTOCOL>
#     CONFIG    phase3_restoration/configs/E{0,1}_*.yml
#     WEIGHTS   experiments/<name>/models/net_g_<best-val-iter>.pth
#     SPLIT     val | test
#     PROTOCOL  full256 | crop128
#
# Chain: predict_phase3.py  (predictions only, no metric defined here)
#     -> Deraining_Holo/masked_metrics.py     UNCHANGED  (full + object-only)
#     -> Deraining_Holo/analyze_sharpness.py  UNCHANGED  (HF/Laplacian/Sobel/
#                                                         radial power spectrum)
#     -> summarize_metrics.py  (aggregation only)
#
# BASELINE RE-CHARACTERISATION (Work Order 2, Step 7): running this for
# E0-Fixed is what re-measures the over-smoothing figures on the model the
# Phase-3 results actually describe. The project's motivating HF-energy ratio
# of 0.216 belongs to the OLD progressive baseline and does not describe
# E0-Fixed. Do not quote 0.216 next to a Phase-3 number.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

CONFIG=$1
WEIGHTS=$2
SPLIT=$3
PROTOCOL=$4
if [ -z "$PROTOCOL" ]; then
  echo "usage: sbatch run_evaluate.sh <CONFIG> <WEIGHTS> <val|test> <full256|crop128>"
  exit 2
fi

P3=dino_analysis_phases/phase3_restoration
DS=/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset
EXP=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['name'])")
OUT=$P3/results/$EXP
PRED=$OUT/predictions/${PROTOCOL}_${SPLIT}
MET=$OUT/metrics

mkdir -p $MET $OUT/visuals $OUT/metadata

MANIFEST=""
if [ "$PROTOCOL" = "crop128" ]; then
  MANIFEST="--manifest $P3/results/crop_manifests/matched128_${SPLIT}.csv"
fi

python -u $P3/scripts/predict_phase3.py \
    --config $CONFIG --weights $WEIGHTS --split $SPLIT \
    --protocol $PROTOCOL $MANIFEST --out-root $OUT/predictions --device cuda || exit 1

if [ "$PROTOCOL" = "crop128" ]; then
  GT_DIR=$PRED/gt
  NOISY_DIR=$PRED/input
else
  GT_DIR=$DS/${SPLIT}_clean
  NOISY_DIR=$DS/${SPLIT}_verynoisy
fi

python -u Deraining_Holo/masked_metrics.py \
    --pred_dir $PRED/raw --gt_dir $GT_DIR \
    --csv $MET/_raw_masked_${PROTOCOL}_${SPLIT}.csv || exit 1

python -u Deraining_Holo/analyze_sharpness.py \
    --pred_dir $PRED/raw --gt_dir $GT_DIR --noisy_dir $NOISY_DIR \
    --csv $MET/_raw_sharpness_${PROTOCOL}_${SPLIT}.csv \
    --out $OUT/visuals/radial_power_${PROTOCOL}_${SPLIT}.png || exit 1

python -u $P3/scripts/summarize_metrics.py \
    --experiment $EXP --protocol $PROTOCOL --split $SPLIT \
    --masked-csv $MET/_raw_masked_${PROTOCOL}_${SPLIT}.csv \
    --sharpness-csv $MET/_raw_sharpness_${PROTOCOL}_${SPLIT}.csv \
    --predict-metadata $PRED/predict_metadata.json || exit 1

echo "EVAL DONE  $EXP  $PROTOCOL/$SPLIT"
