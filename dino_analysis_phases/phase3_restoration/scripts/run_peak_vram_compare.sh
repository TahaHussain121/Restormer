#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_vram_cmp
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/vram_compare_%j.out
#
# Peak training-step VRAM, ONE FRESH PROCESS PER ARM, on the a100 the arms
# actually train on. Batch 8, crop 128 -- the real training shape.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
for CFG in E1_addition_render_fixed128_spatial_B6_latent \
           concat_render_fixed128_spatial_B6_latent \
           crossattn_render_fixed128_spatial_B6_latent; do
  echo "--- $CFG ---"
  python -u $P3/scripts/measure_peak_vram.py --config $P3/configs/$CFG.yml 2>&1 | tail -1
done
echo "VRAM COMPARE DONE"
