#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p4_xatt_probe
#SBATCH --output=dino_analysis_phases/phase4_crossattn_diagnosis/outputs/interventions_%j.out
#
# =============================================================================
# Task B -- four inference-only interventions on crossattn-render.
#
# INFERENCE ONLY. Loads two crossattn checkpoints and E0's read-only, runs the
# val split under twelve conditions, writes ONLY under
# phase4_crossattn_diagnosis/outputs/. Trains nothing, resumes nothing, touches
# no experiment directory and no config.
#
# A job rather than the login node because it is ~4000 ViT-B/14 + Restormer
# forwards; the login-node watchdog SIGTERMs long CPU work (see WO1).
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

python -u dino_analysis_phases/phase4_crossattn_diagnosis/scripts/inference_interventions.py \
    --device cuda --split val --batch 8
