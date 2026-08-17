"""Derive the DIAGNOSTIC configs for the injection-ratio question.

The 12-iteration integration smoke run aborted at iteration 4 on abort rule 1
(`injection_ratio 0.5555 > 0.5`). Four iterations cannot distinguish the
guidance branch SWITCHING ON from a RUNAWAY, and a fatal gate makes the
difference unobservable by construction. This produces two throwaway identities
that answer the question empirically:

    DIAG_E1_injection_ratio   E1 recipe, gate in DIAGNOSTIC mode
                              (abort_on_trigger: false -> measures and records
                              triggers but does not stop), ratio logged EVERY
                              iteration
    DIAG_E0_lossref           the identical E0 recipe, same length, so E1's loss
                              can be read against a real reference (this is
                              abort rule 4's comparison, measured rather than
                              assumed)

NOTHING ELSE CHANGES: same LR, scheduler, crop, batch, seed, loss, means, block,
injection and fusion as the real configs. Only `name`, run length, logging
frequency and `abort_on_trigger` differ. The real E0/E1 configs are untouched
and their identities have still never been started.
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

OUT = os.path.join(_PHASE3, 'results', 'wo2_implementation', 'diag_configs')
ITERS = 5000
VAL_FREQ = 2500          # two validation points -- is it learning at all?


def derive(src, dst, name, diagnostic):
    Loader, Dumper = ordered_yaml()
    with open(src) as f:
        opt = yaml.load(f, Loader=Loader)
    opt['name'] = name
    opt['train']['total_iter'] = ITERS
    opt['datasets']['train']['iters'] = [ITERS]
    opt['logger']['print_freq'] = 100
    opt['logger']['save_checkpoint_freq'] = ITERS      # one checkpoint, not 3
    opt['val']['val_freq'] = VAL_FREQ
    opt['logger']['tb_logger_dir'] = f'tb_logger/{name}'
    if diagnostic and 'dino_stability' in opt:
        opt['dino_stability']['abort_on_trigger'] = False   # measure, don't stop
        opt['dino_stability']['log_freq'] = 1               # every iteration
        opt['dino_stability']['devlog'] = None              # not the real devlog
    with open(dst, 'w') as f:
        yaml.dump(opt, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False)
    print(f'  {name} -> {dst}')

    exp = os.path.join(_REPO, 'experiments', name)
    if os.path.isdir(exp):
        shutil.rmtree(exp)          # a stale DIAG dir would silently auto-resume
        print(f'    removed stale diagnostic experiment dir {exp}')
    tb = os.path.join(_REPO, 'tb_logger', name)
    if os.path.isdir(tb):
        shutil.rmtree(tb)


def main():
    os.makedirs(OUT, exist_ok=True)
    derive(os.path.join(_PHASE3, 'configs',
                        'E1_addition_noisy_fixed128_spatial_B6_latent.yml'),
           os.path.join(OUT, 'DIAG_E1.yml'), 'DIAG_E1_injection_ratio', True)
    derive(os.path.join(_PHASE3, 'configs', 'E0_fixed128_baseline.yml'),
           os.path.join(OUT, 'DIAG_E0.yml'), 'DIAG_E0_lossref', False)


if __name__ == '__main__':
    main()
