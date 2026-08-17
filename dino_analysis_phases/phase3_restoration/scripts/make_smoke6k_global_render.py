"""Derive the 6000-iteration SMOKE TRAINING config for global-render.

This is the run-length-limited twin of the real config: the REAL entry point
(basicsr/train.py) over the REAL config with only the run length and the logging
frequency changed, so the whole integration path -- config parsing, the stacked
render dataset, the progressive block, the model wrapper, the DINO regime
switch, the pooling, stability logging and the abort gate -- is exercised before
300k iterations are committed to it.

THE GATE IS ON. `abort_on_trigger` is left at its default (true), so this run
will stop exactly as the real run would. 6000 iterations is chosen so the run
passes `ratio_rules_start_iter: 5000` and the ratio rules are actually enforced
for the last 1000 iterations, rather than only measured.

The identity is renamed to SMOKE6K_* so nothing can land in, or auto-resume
from, `experiments/Holo_global_addition_render_fixed128_B6_latent`.

Only these keys change: name, total_iter, iters, print_freq,
save_checkpoint_freq, val_freq, tb_logger_dir, dino_stability.log_freq and
dino_stability.devlog (a throwaway run must never append to the real devlog).
Everything else -- crop, batch, optimizer, scheduler, seed, loss, means, block,
pooling, injection, fusion AND every gate threshold -- is copied verbatim.
"""

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
                   'global_addition_render_fixed128_B6_latent.yml')
DST_NAME = 'SMOKE6K_global_addition_render_B6_latent'
ITERS = 6000
VAL_FREQ = 2000          # three validation points, exercising the eval256 switch


def main():
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, 'SMOKE6K_global_render.yml')

    Loader, Dumper = ordered_yaml()
    with open(SRC) as f:
        opt = yaml.load(f, Loader=Loader)

    opt['name'] = DST_NAME
    opt['train']['total_iter'] = ITERS
    opt['datasets']['train']['iters'] = [ITERS]
    opt['logger']['print_freq'] = 100
    opt['logger']['save_checkpoint_freq'] = ITERS       # one checkpoint, not 3
    opt['val']['val_freq'] = VAL_FREQ
    opt['logger']['tb_logger_dir'] = f'tb_logger/{DST_NAME}'
    # every iteration, so the injection_ratio TRAJECTORY is readable afterwards
    opt['dino_stability']['log_freq'] = 1
    opt['dino_stability']['devlog'] = None              # never the real devlog
    # thresholds deliberately NOT touched: cap 10.0, ratio rules from 5000,
    # reference at 5000, growth 10x -- the gate behaves exactly as in the real run

    with open(dst, 'w') as f:
        yaml.dump(opt, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False)
    print(f'  {DST_NAME} -> {dst}')
    print(f'  {ITERS} iters, gate ON (abort_on_trigger default true), '
          f'ratio rules enforced from 5000')

    # a stale SMOKE dir would silently AUTO-RESUME the smoke run
    exp = os.path.join(_REPO, 'experiments', DST_NAME)
    if os.path.isdir(exp):
        shutil.rmtree(exp)
        print(f'    removed stale smoke experiment dir {exp}')
    tb = os.path.join(_REPO, 'tb_logger', DST_NAME)
    if os.path.isdir(tb):
        shutil.rmtree(tb)


if __name__ == '__main__':
    main()
