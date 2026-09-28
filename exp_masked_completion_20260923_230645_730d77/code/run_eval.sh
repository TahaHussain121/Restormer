#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080,v100,a100
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --job-name=mcomp_ev
#SBATCH --output=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77/logs/eval_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77/logs/eval_%j.err
#
# Diagnostic evaluation of the validation-selected checkpoint: the fixed
# synthetic task, then the oracle-assisted actual-failure windows, then the
# fixed-case panel. Validation split only; the test split is not read.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

R=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77
export PYTHONPYCACHEPREFIX=$R/tmp/pycache PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$R/tmp TEMP=$R/tmp TMP=$R/tmp MPLCONFIGDIR=$R/tmp/mplconfig
export XDG_CACHE_HOME=$R/tmp/xdg_cache TORCH_HOME=$R/tmp/torch_home CUDA_CACHE_PATH=$R/tmp/cuda_cache
cd $R

test -f $R/TRAINING_DONE || { echo "training not done"; exit 1; }
python -u $R/code/evaluate_completion.py || exit 1
python -u $R/code/make_figure.py || exit 1
echo "MASKED COMPLETION EVAL DONE"
# Post-hoc SENSITIVITY run, clearly labelled: the pre-registered selection metric
# (mean in-mask PSNR on the synthetic task) oscillates because images whose mask
# sits on background reach very high PSNR; in-mask FOREGROUND PSNR and in-mask MAE
# improved monotonically to the last check. The final checkpoint is therefore
# evaluated as well, chosen on validation SYNTHETIC metrics only -- never on the
# actual-failure diagnostic. The primary result stays the update-2500 checkpoint.
python -u $R/code/evaluate_completion.py --checkpoint $R/checkpoints/ckpt_update_005000.pth --tag posthoc5000 || exit 1
echo "SENSITIVITY EVAL DONE"
