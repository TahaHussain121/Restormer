#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_diag
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/diag_%j.out
#
# =============================================================================
# DIAGNOSTIC: does E1's injection_ratio plateau, oscillate, or diverge?
#
# The integration smoke run aborted at iteration 4 on abort rule 1
# (injection_ratio 0.5555 > 0.5). Four iterations cannot tell "the branch
# switching on" from "a runaway". This runs 5000 iterations of the SAME recipe
# under a throwaway identity with the gate in DIAGNOSTIC mode (records triggers,
# does not stop), logging the ratio every iteration, plus the identical E0
# recipe for the loss reference that abort rule 4 compares against.
#
# NOT a fix, NOT a design change: LR, scheduler, crop, batch, seed, loss, block,
# means, injection and fusion are all exactly as specified. The real E0/E1
# identities are untouched and still unstarted.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration
D=$P3/results/wo2_implementation/diag_configs

echo "################ DIAG E1 (gate in diagnostic mode) ################"
python -u basicsr/train.py -opt $D/DIAG_E1.yml --launcher none
E1_RC=$?

echo "################ DIAG E0 (loss reference) ################"
python -u basicsr/train.py -opt $D/DIAG_E0.yml --launcher none
E0_RC=$?

echo "DIAG RESULT e1=$E1_RC e0=$E0_RC"
