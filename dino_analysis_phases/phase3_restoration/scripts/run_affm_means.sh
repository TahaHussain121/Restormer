#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_affm_means
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/affm_means_%j.out
#
# =============================================================================
# The six new per-layer centering means for the AFFM arm, plus the two
# verification recomputes of B6.
#
#   render_B3_{train128_dino224,eval256_dino448}_mean.pt      NEW
#   render_B9_{train128_dino224,eval256_dino448}_mean.pt      NEW
#   render_B12_{train128_dino224,eval256_dino448}_mean.pt     NEW
#   render_B6_*                                               ALREADY EXISTS
#
# B6 is NOT recomputed into means/. It is recomputed into a scratch directory
# and compared element-wise against the file addition-render already trains
# with; compute_affm_means.py exits non-zero if they differ. The production B6
# mean is never touched -- compute_production_means.py refuses to overwrite a
# production mean without --force, and --force is never passed here.
#
# 1000 train images per mean, seed 0, float64 accumulation, float32 storage --
# the same code path, not a reimplementation.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration
MEANS=$P3/means
VERIFY_DIR=$P3/results/wo2_implementation/affm_b6_verify
mkdir -p "$VERIFY_DIR"

for REGIME in train128 eval256; do
  for BLOCK in 3 9 12; do
    echo "===== render B${BLOCK} ${REGIME} (NEW) ====="
    python -u $P3/scripts/compute_affm_means.py --block $BLOCK \
        --regime $REGIME --domain render --n-samples 1000 --seed 0 \
        --device cuda --outdir $MEANS || exit 1
  done
done

for REGIME in train128 eval256; do
  if [ "$REGIME" = train128 ]; then SUF=train128_dino224; else SUF=eval256_dino448; fi
  echo "===== render B6 ${REGIME} (VERIFY ONLY -- production file untouched) ====="
  python -u $P3/scripts/compute_affm_means.py --block 6 \
      --regime $REGIME --domain render --n-samples 1000 --seed 0 \
      --device cuda --outdir "$VERIFY_DIR" \
      --verify-against $MEANS/render_B6_${SUF}_mean.pt || exit 1
done

echo "===== all means present ====="
ls -la $MEANS/render_B*_mean.pt
