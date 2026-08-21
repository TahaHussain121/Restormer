#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:25:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_vram_pq
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/vram_priorquery_%j.out
#
# Peak training-step VRAM for priorquery-render, ONE FRESH PROCESS PER ARM.
#
# ON V100, NOT A100, AND WITH A REFERENCE ARM MEASURED ALONGSIDE. While
# crossattn-render is training it holds the a100 GRES association, so a second
# a100 job cannot start (AssocGrpGRES). A v100 number is not comparable to the
# a100 numbers measured earlier for addition/concat/crossattn (different card,
# different cuDNN algorithm selection), so addition-render is re-measured here
# on the SAME card and the comparison is made ratio-wise against it.
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
for CFG in E1_addition_render_fixed128_spatial_B6_latent \
           priorquery_render_fixed128_spatial_B6_latent; do
  echo "--- $CFG ---"
  python -u $P3/scripts/measure_peak_vram.py --config $P3/configs/$CFG.yml 2>&1 | tail -1
done
echo "VRAM PRIORQUERY DONE"
