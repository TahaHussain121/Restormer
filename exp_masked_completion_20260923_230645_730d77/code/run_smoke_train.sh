#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=mcomp_tr
#SBATCH --output=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77/logs/smoke_train_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77/logs/smoke_train_%j.err
#
# ONE isolated diagnostic: real-data smoke checks, then the single bounded
# training run. v100, as both existing refiner training jobs used. Every
# scheduler and application output path is inside the experiment root.

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

R=/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77
export PYTHONPYCACHEPREFIX=$R/tmp/pycache
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$R/tmp TEMP=$R/tmp TMP=$R/tmp
export MPLCONFIGDIR=$R/tmp/mplconfig
export XDG_CACHE_HOME=$R/tmp/xdg_cache
export TORCH_HOME=$R/tmp/torch_home
export CUDA_CACHE_PATH=$R/tmp/cuda_cache
cd $R                      # never the repo root: nothing here writes relative paths

python -u $R/code/smoke_completion.py || { echo "SMOKE FAILED - not training"; exit 1; }
python -u $R/code/train_completion.py || exit 1
echo "MASKED COMPLETION TRAIN DONE"
