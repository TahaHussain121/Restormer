"""Derive a 6000-iteration SMOKE TRAINING config for any ACA-ladder arm.

ONE generator, driven by --config, because the three ACA arms differ only in
their layer set. This is the run-length-limited twin of the REAL config: the
REAL entry point (basicsr/train.py) over the REAL config with only the run
length and the logging frequencies changed, so the whole INTEGRATION path is
exercised -- config parsing, the stacked render dataset, the model wrapper, the
DINO regime switch, the layer extraction, per-layer centering, AFFM (arms 2/3),
P, the ACA block, stability logging, the abort gate, checkpointing and the
eval256 validation switch.

THIS IS THE PART THE ARCHITECTURAL SMOKE CANNOT REACH. On the affm arm the
equivalent run caught two real bugs the 72-check architectural suite missed:
bare integer YAML keys crashing basicsr/utils/options.py at startup, and an
observation publish condition that never fired at freq 1.

THE GATE IS ON (`abort_on_trigger` left at its default) and 6000 iterations
passes `ratio_rules_start_iter: 5000`, so the ratio rules are genuinely
ENFORCED for the last 1000 iterations rather than only measured.

Renamed SMOKE6K_* so nothing can land in, or auto-resume from, the real
experiment directory.

Only these keys change: name, total_iter, iters, print_freq,
save_checkpoint_freq, val_freq, tb_logger_dir, dino_stability.log_freq,
dino_stability.devlog (a throwaway must never append to the real devlog) and
dino_aca_stats_freq. That last one is an OBSERVATION frequency only -- at the
real 5000 the alpha/AFFM/temperature series would have two points in the whole
run. The maths is identical on every forward regardless.
"""
import argparse, os, shutil, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_P3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_P3))
sys.path.insert(0, _REPO)
from basicsr.utils.options import ordered_yaml            # noqa: E402
import yaml                                               # noqa: E402

OUT = os.path.join(_P3, 'results', 'wo2_implementation', 'smoke6k_configs')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--aca-freq', type=int, default=100)
    ap.add_argument('--val-freq', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    L, D = ordered_yaml()
    opt = yaml.load(open(a.config), Loader=L)
    real = opt['name']
    stem = os.path.basename(a.config).replace('.yml', '')
    name = 'SMOKE6K_' + stem.replace('aca_render_fixed128_', 'aca_').replace('_latent', '')
    dst = os.path.join(OUT, f'{name}.yml')

    opt['name'] = name
    opt['train']['total_iter'] = a.iters
    opt['datasets']['train']['iters'] = [a.iters]
    opt['logger']['print_freq'] = 100
    opt['logger']['save_checkpoint_freq'] = a.iters
    opt['val']['val_freq'] = a.val_freq
    opt['logger']['tb_logger_dir'] = f'tb_logger/{name}'
    opt['dino_stability']['log_freq'] = 1
    opt['dino_stability']['devlog'] = None
    opt['network_g']['dino_aca_stats_freq'] = a.aca_freq
    if opt['name'] == real:
        raise SystemExit('refusing to write a smoke config carrying the real name')
    with open(dst, 'w') as f:
        yaml.dump(opt, f, Dumper=D, default_flow_style=False, sort_keys=False)
    print(f'  {name} -> {dst}')
    print(f'  {a.iters} iters, gate ON, ratio rules ENFORCED from 5000, '
          f'aca stats every {a.aca_freq} forwards, val every {a.val_freq}')
    for d in (os.path.join(_REPO, 'experiments', name),
              os.path.join(_REPO, 'tb_logger', name)):
        if os.path.isdir(d):
            shutil.rmtree(d); print(f'    removed stale {d}')
    print(f'  REAL config untouched: {a.config}')

if __name__ == '__main__':
    main()
