#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_inreg_smoke
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/inregime/smoke_%j.out
unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub
P3=dino_analysis_phases/phase3_restoration
EXP=Holo_E1_addition_render_fixed128_spatial_B6_latent
CONFIG=$P3/configs/E1_addition_render_fixed128_spatial_B6_latent.yml
WEIGHTS=experiments/$EXP/models/net_g_204000.pth
OUT=$P3/results/inregime/_smoke
for SPEC in "full256:full256:" "resize128:resize128:" "t0:tiled128:" "t64:tiled128:--tile-overlap 64"; do
  NAME=${SPEC%%:*}; R=${SPEC#*:}; PROTO=${R%%:*}; EXTRA=${R#*:}
  echo "--- $NAME ($PROTO $EXTRA)"
  python -u $P3/scripts/predict_phase3.py --config $CONFIG --weights $WEIGHTS \
     --split val --protocol $PROTO $EXTRA --limit 4 \
     --out-root $OUT/$NAME --device cuda || exit 1
done
python - <<'PYEOF'
import glob, os, cv2, numpy as np
B='dino_analysis_phases/phase3_restoration/results/inregime/_smoke'
ref='dino_analysis_phases/phase3_restoration/results/Holo_E1_addition_render_fixed128_spatial_B6_latent/predictions/full256_val/raw'
def load(name, proto):
    return {os.path.basename(p): cv2.imread(p, cv2.IMREAD_UNCHANGED)
            for p in sorted(glob.glob(f'{B}/{name}/{proto}_val/raw/*.png'))}
f=load('full256','full256'); r=load('resize128','resize128')
t0=load('t0','tiled128');    t64=load('t64','tiled128')
assert f and r and t0 and t64, 'a condition produced nothing'
ok=True
# every condition must output the full 256 frame
for n,d in [('full256',f),('resize128',r),('tiled ov0',t0),('tiled ov64',t64)]:
    shapes={v.shape for v in d.values()}
    print(f'{n:<12} n={len(d)} shapes={shapes}')
    if shapes!={(256,256)}: print('  !! not 256x256'); ok=False
# full256 must still reproduce the recorded predictions bit-exactly
same=sum(1 for k,v in f.items()
         if os.path.isfile(f'{ref}/{k}') and
         np.array_equal(v, cv2.imread(f'{ref}/{k}', cv2.IMREAD_UNCHANGED)))
print(f'\nfull256 identical to recorded: {same}/{len(f)}')
if same!=len(f): print('  !! the refactor changed the normal path. STOP.'); ok=False
# the three new conditions must differ from full256 and from each other
for n,d in [('resize128',r),('tiled ov0',t0),('tiled ov64',t64)]:
    diff=sum(1 for k in f if not np.array_equal(f[k], d[k]))
    print(f'{n:<12} differs from full256: {diff}/{len(f)}')
    if diff!=len(f): print('  !! inert'); ok=False
diff=sum(1 for k in t0 if not np.array_equal(t0[k], t64[k]))
print(f'ov0 vs ov64  differ: {diff}/{len(t0)}')
if diff!=len(t0): ok=False
print('\nSMOKE ' + ('PASSED' if ok else 'FAILED'))
raise SystemExit(0 if ok else 1)
PYEOF
