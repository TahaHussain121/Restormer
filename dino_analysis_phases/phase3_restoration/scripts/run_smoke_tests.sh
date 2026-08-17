#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_smoke
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke_%j.out
#
# =============================================================================
# WORK ORDER 2, Step 4 -- smoke tests. Short, safe, no full training.
#
#  1. smoke_tests.py            assertion harness (4.1-4.5), CUDA
#  2. smoke_tests.py --device cpu   the same harness bit-exactly on CPU, where
#                               step-0 equivalence must be EXACTLY 0.0
#  3. two real basicsr/train.py runs under SMOKE_* experiment names (12 iters),
#     proving the actual training entry point works end to end -- including the
#     E1 model wrapper, the stability logging and checkpoint writing. These
#     write to experiments/SMOKE_* and never touch the real experiment names.
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration

echo "################ smoke harness on CUDA ################"
python -u $P3/scripts/smoke_tests.py --device cuda \
    --out $P3/results/wo2_implementation/smoke_results_cuda.json
CUDA_RC=$?

echo "################ smoke harness on CPU (bit-exact) ################"
python -u $P3/scripts/smoke_tests.py --device cpu \
    --out $P3/results/wo2_implementation/smoke_results_cpu.json
CPU_RC=$?

echo "################ real train.py, 12 iters, SMOKE identities ################"
python -u basicsr/train.py -opt $P3/results/wo2_implementation/smoke_configs/SMOKE_E0.yml --launcher none
E0_RC=$?
python -u basicsr/train.py -opt $P3/results/wo2_implementation/smoke_configs/SMOKE_E1.yml --launcher none
E1_RC=$?

echo "SMOKE RESULT harness_cuda=$CUDA_RC harness_cpu=$CPU_RC train_e0=$E0_RC train_e1=$E1_RC"
