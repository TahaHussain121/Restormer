#!/bin/bash -l
#
#SBATCH --partition=work,rtx3080,v100,a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_eval_dispatch_cat
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/Holo_concat_render_fixed128_spatial_B6_latent/logs/eval_dispatch_%j.out
#
# =============================================================================
# Deferred evaluation dispatcher for concat-render.
#
# Submitted with --dependency=afterany on the concat training chain, so it wakes
# up only after training has terminated. It does NO evaluation itself and needs
# GPU work of its own (TinyGPU requires a gres, so it takes the smallest slot
# available for the ~20 seconds it runs): it selects the best-VALIDATION checkpoint and then submits the four
# standard evaluation cells through the UNCHANGED run_evaluate.sh, exactly as
# the other four arms were evaluated.
#
# It REFUSES to evaluate anything if 300k did not complete -- a walltime kill
# mid-run leaves a resumed chain job still working, and evaluating a partial run
# would produce a number that looks final and is not.
#
# Checkpoint selection reads VALIDATION PSNR only. The test split never selects.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

P3=dino_analysis_phases/phase3_restoration
EXP=Holo_concat_render_fixed128_spatial_B6_latent
CONFIG=$P3/configs/concat_render_fixed128_spatial_B6_latent.yml
FINAL=experiments/$EXP/models/net_g_300000.pth
DONE=experiments/Phase3_chain_state_$EXP/TRAINING_DONE

echo "[dispatch] $(date)  checking whether $EXP actually finished 300k"
if [ ! -f "$FINAL" ] || [ ! -f "$DONE" ]; then
  echo "[dispatch] REFUSING TO EVALUATE."
  echo "[dispatch]   net_g_300000.pth present: $([ -f "$FINAL" ] && echo yes || echo NO)"
  echo "[dispatch]   TRAINING_DONE present:    $([ -f "$DONE" ] && echo yes || echo NO)"
  echo "[dispatch] The chain is probably still resuming. Re-submit this"
  echo "[dispatch] dispatcher once training reports TRAINING_DONE."
  exit 1
fi

mkdir -p $P3/results/$EXP/metadata
python -u $P3/scripts/select_best_checkpoint.py --name $EXP \
    --out $P3/results/$EXP/metadata/best_checkpoint.json || exit 1
BEST=$(python -c "import json;print(json.load(open('$P3/results/$EXP/metadata/best_checkpoint.json'))['best_iter'])")
W=experiments/$EXP/models/net_g_${BEST}.pth
if [ ! -f "$W" ]; then
  echo "[dispatch] best-val checkpoint $W is missing -- stopping."
  exit 1
fi
echo "[dispatch] best-VALIDATION checkpoint: iter $BEST -> $W"

for SPLIT in val test; do
  for PROT in full256 crop128; do
    J=$(sbatch --partition=rtx3080,v100,a100 --gres=gpu:1 \
               --job-name=p3_eval_cat \
               $P3/scripts/run_evaluate.sh $CONFIG $W $SPLIT $PROT | awk '{print $NF}')
    echo "[dispatch] submitted $J  $SPLIT/$PROT"
  done
done
echo "[dispatch] four evaluation cells queued for $EXP at iter $BEST"
