#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:25:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_vram_affm
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/vram_affm_%j.out
#
# Peak training-step VRAM for affm-render, ONE FRESH PROCESS PER ARM (measuring
# two arms in one process inherits the first one's allocations -- see
# measure_peak_vram.py's docstring).
#
# ON v100 (32 GB), WITH addition-render RE-MEASURED ALONGSIDE.
#
# FIRST ATTEMPT WAS rtx3080 AND IT OOM'd -- for BOTH arms, including the
# reference (job 1786912: "Tried to allocate 32.00 MiB ... 9.64 GiB capacity").
# A batch-8 128-crop training step of this model does not fit in 10 GB; the
# handoff's a100 figure for these arms is ~25 GB. That is a fact about the card,
# not about this arm, and it is why the measurement moved here.
#
# a100 is unavailable: priorquery-render holds the a100 GRES association, so a
# second a100 job cannot start. A number from one card is not comparable to a
# number from another (different cuDNN algorithm selection), so the reference
# arm is measured on the SAME card in the same job and the comparison is made
# RATIO-WISE against it. The absolute figures here are v100 figures.
#
# Inference/one-step only: no experiment directory, no checkpoint.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
for CFG in E1_addition_render_fixed128_spatial_B6_latent \
           affm_render_fixed128_spatial_L3691_latent; do
  echo "--- $CFG ---"
  python -u $P3/scripts/measure_peak_vram.py --config $P3/configs/$CFG.yml 2>&1 | tail -1
done
echo "VRAM AFFM DONE"
