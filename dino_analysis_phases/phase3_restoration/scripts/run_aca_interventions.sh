#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_acaint
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/aca_interventions/aca_int_%j.out
#
# =============================================================================
# ACA CHECKPOINT INTERVENTIONS — inference only, no training.
#
# WHY. Every "ACA versus addition" number this project has measured compares the
# WHOLE DinoAca block — self-attention AND gated cross-attention behind one
# shared output projection — against plain addition. None of them asks whether
# the TRAINED model actually uses its cross path, or whether the learned channel
# mixing in that path does anything beyond delivering the prior at all.
#
# WHAT. Three conditions on the SAME trained weights:
#
#   none            the block exactly as trained            SELF-CHECK
#   no_cross        alpha * F_ca forced to zero, F_sa kept  does it depend on
#                                                           the cross path?
#   uniform_cross   the cross channel-attention matrix
#                   replaced by a uniform one, its values
#                   (V') untouched                          does the LEARNED
#                                                           mixing matter, given
#                                                           the prior still gets
#                                                           through?
#
# 'none' is included DELIBERATELY as a self-check: its predictions must be
# BIT-IDENTICAL to the arm's own recorded validation predictions. If they are
# not, this path is not the arm's computation and the table must not be used.
# The comparison is run inside this job, on the same device, because a CPU rerun
# would differ in the last bits for reasons that have nothing to do with the
# intervention.
#
# ARM. aca-L6 at its validation-selected checkpoint (236,000) — the one-factor
# ACA arm, so the block being interrogated is the one whose null result
# (+0.030 dB, p=0.22) motivated the question.
#
# SPLIT. VALIDATION, n=339. These controls were not pre-registered and do not
# touch the locked test split.
#
# BOTH PROTOCOLS, as the project's reporting rule requires.
#
# WHAT THIS CAN AND CANNOT SHOW. Interventions on a trained model establish what
# that model DEPENDS on. They do NOT show how a model trained without the
# component would perform — that needs its own run, and is not claimed here.
#
#   sbatch run_aca_interventions.sh
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration
DS=/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset

EXP=Holo_aca_render_fixed128_L6_latent
CONFIG=$P3/configs/aca_render_fixed128_L6_latent.yml
WEIGHTS=experiments/$EXP/models/net_g_236000.pth
SPLIT=val
RECORDED=$P3/results/$EXP/predictions

OUT=$P3/results/aca_interventions
mkdir -p $OUT

if [ ! -f "$WEIGHTS" ]; then echo "MISSING WEIGHTS $WEIGHTS"; exit 1; fi

for PROTOCOL in full256 crop128; do
  MANIFEST=""
  if [ "$PROTOCOL" = "crop128" ]; then
    MANIFEST="--manifest $P3/results/crop_manifests/matched128_${SPLIT}.csv"
  fi

  for COND in none no_cross uniform_cross; do
    CELL=$OUT/$COND/${PROTOCOL}_${SPLIT}
    echo "=== condition=${COND}  protocol=${PROTOCOL}  split=${SPLIT} ==="

    python -u $P3/scripts/predict_phase3.py \
        --config $CONFIG --weights $WEIGHTS --split $SPLIT \
        --protocol $PROTOCOL $MANIFEST \
        --aca-intervention $COND \
        --out-root $OUT/$COND --device cuda || exit 1

    if [ "$PROTOCOL" = "crop128" ]; then
      GT_DIR=$CELL/gt
    else
      GT_DIR=$DS/${SPLIT}_clean
    fi

    # UNCHANGED metric script, same invocation as run_evaluate.sh, so these
    # numbers are directly comparable to the arm's recorded evaluation.
    python -u Deraining_Holo/masked_metrics.py \
        --pred_dir $CELL/raw --gt_dir $GT_DIR \
        --csv $OUT/per_image_${COND}_${PROTOCOL}_${SPLIT}.csv || exit 1

    # ---- the self-check, on the same device that produced both sides -------
    if [ "$COND" = "none" ]; then
      python - "$CELL/raw" "$RECORDED/${PROTOCOL}_${SPLIT}/raw" <<'PYEOF'
import sys, os, hashlib
new, old = sys.argv[1], sys.argv[2]
if not os.path.isdir(old):
    print(f'  SELF-CHECK SKIPPED: no recorded predictions at {old}'); sys.exit(0)
def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()
names = sorted(f for f in os.listdir(new) if f.endswith('.png'))
same = sum(1 for f in names
           if os.path.isfile(os.path.join(old, f))
           and md5(os.path.join(new, f)) == md5(os.path.join(old, f)))
print(f'  SELF-CHECK intervention=none vs recorded: {same}/{len(names)} bit-identical')
if same != len(names):
    print('  SELF-CHECK FAILED -- this path is not the arm\'s own computation.')
    sys.exit(1)
PYEOF
      [ $? -ne 0 ] && exit 1
    fi
  done
done

echo "ACA INTERVENTION SWEEP DONE"
