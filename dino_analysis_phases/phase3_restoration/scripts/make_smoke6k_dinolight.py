"""Derive the 6000-iteration SMOKE TRAINING config for dinolight-render.

The run-length-limited twin of the real config: the REAL entry point
(basicsr/train.py) over the REAL config with only the run length and the logging
frequencies changed, so the whole integration path -- config parsing, the
stacked render dataset, the model wrapper, the DINO regime switch, the
four-layer extraction, per-layer centering, the AFFM softmax, P, the ACA block,
stability logging, the abort gate, checkpointing and the eval256 validation
switch -- is exercised before 300k iterations are committed to it.

THE GATE IS ON (`abort_on_trigger` left at its default), and 6000 iterations
passes `ratio_rules_start_iter: 5000`, so the ratio rules are genuinely ENFORCED
for the last 1000 iterations rather than only measured.

Renamed to SMOKE6K_* so nothing can land in, or auto-resume from, the real
experiment directory.

Only these keys change: name, total_iter, iters, print_freq,
save_checkpoint_freq, val_freq, tb_logger_dir, dino_stability.log_freq,
dino_stability.devlog (a throwaway run must never append to the real devlog)
and dino_aca_stats_freq. The last is an OBSERVATION frequency only -- at the
real 5000 the alpha/AFFM/temperature/entropy series would have two points in the
whole smoke run; the maths is identical on every forward regardless.
"""

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
sys.path.insert(0, _REPO)

from basicsr.utils.options import ordered_yaml            # noqa: E402
import yaml                                               # noqa: E402

OUT = os.path.join(_PHASE3, 'results', 'wo2_implementation', 'smoke6k_configs')
SRC = os.path.join(_PHASE3, 'configs',
                   'dinolight_render_fixed128_L3691_aca_latent.yml')
DST_NAME = 'SMOKE6K_dinolight_L3691_aca'
ITERS = 6000
VAL_FREQ = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=ITERS)
    ap.add_argument('--name', default=DST_NAME)
    ap.add_argument('--out', default=os.path.join(OUT, 'SMOKE6K_dinolight.yml'))
    ap.add_argument('--aca-freq', type=int, default=100)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    Loader, Dumper = ordered_yaml()
    with open(SRC) as f:
        opt = yaml.load(f, Loader=Loader)

    real_name = opt['name']
    opt['name'] = args.name
    opt['train']['total_iter'] = args.iters
    opt['datasets']['train']['iters'] = [args.iters]
    opt['logger']['print_freq'] = 100
    opt['logger']['save_checkpoint_freq'] = args.iters
    opt['val']['val_freq'] = VAL_FREQ
    opt['logger']['tb_logger_dir'] = f'tb_logger/{args.name}'
    opt['dino_stability']['log_freq'] = 1
    opt['dino_stability']['devlog'] = None
    opt['network_g']['dino_aca_stats_freq'] = args.aca_freq

    if opt['name'] == real_name:
        raise SystemExit('refusing to write a smoke config carrying the real name')

    with open(args.out, 'w') as f:
        yaml.dump(opt, f, Dumper=Dumper, default_flow_style=False, sort_keys=False)
    print(f'  {args.name} -> {args.out}')
    print(f'  {args.iters} iters, gate ON, ratio rules enforced from 5000, '
          f'aca stats every {args.aca_freq} forwards')

    exp = os.path.join(_REPO, 'experiments', args.name)
    if os.path.isdir(exp):
        shutil.rmtree(exp)
        print(f'    removed stale smoke experiment dir {exp}')
    tb = os.path.join(_REPO, 'tb_logger', args.name)
    if os.path.isdir(tb):
        shutil.rmtree(tb)


if __name__ == '__main__':
    main()
