#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=p3_sm2kML
#SBATCH --output=dino_analysis_phases/phase3_restoration/results/wo2_implementation/smoke2k_multilevel_%j.out
#
# =============================================================================
# BOUNDED REAL-DATA INTEGRATION SMOKE for one final multi-level arm.
#
#   sbatch run_smoke2k_multilevel.sh <config>
#
# Runs the REAL entry point (basicsr/train.py) on the REAL data for 2,000
# iterations with a throwaway identity (SMOKE2K_*), so it can never land in or
# auto-resume from the real experiment directory, and never appends to the real
# devlog. It exercises what the unit smoke cannot: the stacked render dataset
# and the train.py sub-crop feeding three sites, the model wrapper's regime
# switch, the stability gate, checkpoint saving, and the in-training eval256
# validation on all 339 full frames (twice, at 1k and 2k).
#
# ONLY these keys change: name, total_iter, iters, print_freq,
# save_checkpoint_freq, val_freq, tb_logger_dir, dino_stability.log_freq,
# dino_stability.devlog and (ACA arm) dino_aca_stats_freq, the last only so the
# per-stage ACA statistics are observed more than once. Every threshold of the
# gate is copied verbatim. NOTE: the gate's RATIO rules start at iteration
# 5,000 by design, so in 2,000 iterations they are measured but not enforced;
# the NaN/Inf rule is active from iteration 1.
#
# On a100, the partition the real runs use, so the numeric regime (TF32)
# matches. Throwaway: it answers "does it train end to end and where do the
# per-site numbers sit", never "what does this arm score".
# =============================================================================

unset SLURM_EXPORT_ENV
module load python
conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

CONFIG=$1
[ -f "$CONFIG" ] || { echo "usage: sbatch run_smoke2k_multilevel.sh <config>"; exit 2; }
P3=dino_analysis_phases/phase3_restoration
OUTDIR=$P3/results/wo2_implementation/smoke2k_configs
mkdir -p $OUTDIR

SMOKE=$(python - "$CONFIG" "$OUTDIR" <<'PYEOF'
import sys, os
sys.path.insert(0, '.')
from basicsr.utils.options import ordered_yaml
import yaml
Loader, Dumper = ordered_yaml()
src, outdir = sys.argv[1], sys.argv[2]
with open(src) as f:
    c = yaml.load(f, Loader=Loader)
name = 'SMOKE2K_' + c['name'].replace('Holo_', '')
c['name'] = name
c['train']['total_iter'] = 2000
c['datasets']['train']['iters'] = [2000]
c['logger']['print_freq'] = 100
c['logger']['save_checkpoint_freq'] = 1000
c['logger']['tb_logger_dir'] = f'tb_logger/{name}'
c['val']['val_freq'] = 1000
c['dino_stability']['log_freq'] = 100
c['dino_stability']['devlog'] = os.path.abspath(os.path.join(outdir, f'{name}_devlog.md'))
if 'dino_aca_stats_freq' in c['network_g']:
    c['network_g']['dino_aca_stats_freq'] = 100
dst = os.path.join(outdir, f'{name}.yml')
with open(dst, 'w') as f:
    yaml.dump(c, f, Dumper=Dumper)
print(dst)
PYEOF
)
[ -f "$SMOKE" ] || { echo "failed to derive the smoke config"; exit 1; }
NAME=$(basename $SMOKE .yml)
echo "[smoke2k] $NAME  from $CONFIG"

python -u basicsr/train.py -opt $SMOKE --launcher none
RC=$?
echo "[smoke2k] training exited rc=$RC"

EXP=experiments/$NAME
LOG=$(ls -t $EXP/train_*.log 2>/dev/null | head -1)
echo "[smoke2k] gate keys at the post-latent site (every 500 iters):"
awk -F, 'NR==1 || (NR>1 && ($1 % 500)==0)' $EXP/dino_stability.csv 2>/dev/null
echo "[smoke2k] per-site norms / ratios / finiteness (every 500 iters):"
grep -E "iter: *[0-9,]*(500|000)," $LOG 2>/dev/null | grep -o -E "iter: *[0-9,]+|site_(pl|d3|d2)_(ratio|finite|update_norm): [0-9.e+-]+" | paste -sd' ' | sed 's/iter:/\niter:/g' | tail -6
echo "[smoke2k] validation (eval256, 339 frames):"
grep "Validation ValSet" $LOG 2>/dev/null
echo "[smoke2k] stability failures (there should be none):"
ls $EXP/STABILITY_FAILURE* 2>/dev/null || echo "  none"
[ -f $EXP/models/net_g_2000.pth ] && echo "[smoke2k] checkpoint at 2000 written" || echo "[smoke2k] NO 2000 checkpoint"
exit $RC
