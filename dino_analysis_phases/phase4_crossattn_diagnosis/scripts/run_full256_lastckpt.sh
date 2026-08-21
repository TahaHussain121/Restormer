#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --job-name=p4_xatt_full256
#SBATCH --output=dino_analysis_phases/phase4_crossattn_diagnosis/outputs/full256_lastckpt_%j.out
#
# =============================================================================
# The confirmatory run Task B's crop128 result demands: the SAME checkpoint
# (net_g_178000) on the SAME split, at full256 instead of crop128.
#
# crop128/val put net_g_178000 at 21.812 dB, +2.00 over E0 and +3.53 over the
# arm's own best-VALIDATION checkpoint. Either the arm is fine in-distribution
# and fails on scale transfer, or it is fine everywhere and only checkpoint
# selection went wrong. Only a full256 pass on this checkpoint separates them.
#
# INFERENCE ONLY, and deliberately routed through the project's own
# predict_phase3.py + masked_metrics.py rather than a re-implementation.
# --out-root points into phase4_crossattn_diagnosis/outputs/, NOT into
# results/<experiment>/predictions, so the published full256 predictions for
# this arm (produced from net_g_4000) are not touched. See the session handoff:
# run_evaluate.sh keys its output directory off the config NAME, which is
# exactly the overwrite hazard being avoided here.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration
P4=dino_analysis_phases/phase4_crossattn_diagnosis
DS=/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset
OUTROOT=$P4/outputs/full256_lastckpt

for CKPT in 178000 4000; do
  python -u $P3/scripts/predict_phase3.py \
      --config $P3/configs/crossattn_render_fixed128_spatial_B6_latent.yml \
      --weights experiments/Holo_crossattn_render_fixed128_spatial_B6_latent/models/net_g_${CKPT}.pth \
      --split val --protocol full256 \
      --out-root ${OUTROOT}_${CKPT} --device cuda || exit 1

  echo "===== masked metrics, net_g_${CKPT}, full256/val ====="
  python -u Deraining_Holo/masked_metrics.py \
      --pred_dir ${OUTROOT}_${CKPT}/full256_val/raw \
      --gt_dir $DS/val_clean \
      --csv $P4/outputs/full256_val_ckpt${CKPT}.csv || exit 1
done
