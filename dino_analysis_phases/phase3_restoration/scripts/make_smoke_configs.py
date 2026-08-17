"""Derive throwaway 12-iteration SMOKE configs from the real E0/E1 configs.

The point is to run the REAL entry point (basicsr/train.py) over the REAL
configs with only the run length changed, so the integration path -- config
parsing, dataset, the progressive block, the model wrapper, the DINO regime
switch, stability logging, checkpoint writing -- is exercised before 300k
iterations are committed to it.

The experiment identity is renamed to SMOKE_* so nothing can land in, or
auto-resume from, `experiments/Holo_E0_fixed128_baseline` or
`experiments/Holo_E1_addition_noisy_fixed128_spatial_B6_latent`.

Only these keys change: name, total_iter, iters, print_freq,
save_checkpoint_freq, val_freq, tb_logger_dir. Everything else -- crop, batch,
optimizer, scheduler, seed, loss, the whole DINO block -- is copied verbatim.
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

OUT = os.path.join(_PHASE3, 'results', 'wo2_implementation', 'smoke_configs')
ITERS = 12
VAL_FREQ = 8            # one validation pass, exercising E1's eval256 switch


def derive(src, dst, name):
    Loader, Dumper = ordered_yaml()
    with open(src) as f:
        opt = yaml.load(f, Loader=Loader)
    opt['name'] = name
    opt['train']['total_iter'] = ITERS
    opt['datasets']['train']['iters'] = [ITERS]
    opt['logger']['print_freq'] = 1
    opt['logger']['save_checkpoint_freq'] = ITERS
    opt['val']['val_freq'] = VAL_FREQ
    opt['logger']['tb_logger_dir'] = f'tb_logger/{name}'
    if 'dino_stability' in opt:
        opt['dino_stability']['log_freq'] = 1
        opt['dino_stability']['reference_iter'] = 4
        # a 12-iteration run must not append to the real experiment devlog
        opt['dino_stability']['devlog'] = None
    with open(dst, 'w') as f:
        yaml.dump(opt, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False)
    print(f'  {name} -> {dst}')


def main():
    os.makedirs(OUT, exist_ok=True)
    for src, dst, name in (
            ('E0_fixed128_baseline.yml', 'SMOKE_E0.yml', 'SMOKE_E0_fixed128_baseline'),
            ('E1_addition_noisy_fixed128_spatial_B6_latent.yml', 'SMOKE_E1.yml',
             'SMOKE_E1_addition_fixed128_spatial_B6_latent')):
        derive(os.path.join(_PHASE3, 'configs', src),
               os.path.join(OUT, dst), name)
        # a stale SMOKE experiment dir would silently AUTO-RESUME the smoke run
        exp = os.path.join(_REPO, 'experiments', name)
        if os.path.isdir(exp):
            shutil.rmtree(exp)
            print(f'    removed stale smoke experiment dir {exp}')
        tb = os.path.join(_REPO, 'tb_logger', name)
        if os.path.isdir(tb):
            shutil.rmtree(tb)


if __name__ == '__main__':
    main()
