#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_misalign_smoke
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/render_misalignment/smoke_%j.out
#
# Integration smoke for the misalignment sweep: 6 images, shift 0 and shift 8,
# full256 only. Proves three things before 4 h of GPU is spent --
#   1. the render arm still loads and runs through the modified script
#   2. shift=0 is byte-identical to the arm's recorded predictions (the edit
#      did not perturb the normal path)
#   3. shift=8 actually changes the predictions (the control does something)
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv

cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

P3=dino_analysis_phases/phase3_restoration
EXP=Holo_E1_addition_render_fixed128_spatial_B6_latent
CONFIG=$P3/configs/E1_addition_render_fixed128_spatial_B6_latent.yml
WEIGHTS=experiments/$EXP/models/net_g_204000.pth
OUT=$P3/results/render_misalignment/_smoke
mkdir -p $OUT

for K in 0 8; do
  python -u $P3/scripts/predict_phase3.py \
      --config $CONFIG --weights $WEIGHTS --split val \
      --protocol full256 --render-shift $K --limit 6 \
      --out-root $OUT/shift${K} --device cuda || exit 1
done

python - <<'PYEOF'
import glob, os, cv2, numpy as np, json
base='dino_analysis_phases/phase3_restoration/results/render_misalignment/_smoke'
ref='dino_analysis_phases/phase3_restoration/results/Holo_E1_addition_render_fixed128_spatial_B6_latent/predictions/full256_val/raw'
a=sorted(glob.glob(f'{base}/shift0/full256_val/raw/*.png'))
b=sorted(glob.glob(f'{base}/shift8/full256_val/raw/*.png'))
assert a and len(a)==len(b), f'no predictions: {len(a)} vs {len(b)}'
same_as_ref=0; changed=0
for pa,pb in zip(a,b):
    ia=cv2.imread(pa, cv2.IMREAD_UNCHANGED); ib=cv2.imread(pb, cv2.IMREAD_UNCHANGED)
    if not np.array_equal(ia,ib): changed+=1
    rp=os.path.join(ref, os.path.basename(pa))
    if os.path.isfile(rp):
        ir=cv2.imread(rp, cv2.IMREAD_UNCHANGED)
        if np.array_equal(ia,ir): same_as_ref+=1
print(f'\nCHECK 2  shift0 identical to recorded predictions: {same_as_ref}/{len(a)}')
print(f'CHECK 3  shift8 differs from shift0:               {changed}/{len(a)}')
ok = (changed==len(a))
if os.path.isdir(ref):
    ok = ok and (same_as_ref==len(a))
    if same_as_ref!=len(a):
        print('  !! shift=0 does NOT reproduce the recorded predictions -- '
              'the edit changed the normal path. STOP.')
else:
    print('  (no recorded full256_val predictions on disk; check 2 skipped)')
if changed!=len(a):
    print('  !! shift=8 did not change every prediction -- the control is inert. STOP.')
print('\nSMOKE ' + ('PASSED' if ok else 'FAILED'))
raise SystemExit(0 if ok else 1)
PYEOF
