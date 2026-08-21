"""Derive the 6000-iteration SMOKE TRAINING config for affm-render.

This is the run-length-limited twin of the real config: the REAL entry point
(basicsr/train.py) over the REAL config with only the run length and the logging
frequencies changed, so the whole integration path -- config parsing, the
stacked render dataset, the progressive block, the model wrapper, the DINO
regime switch, the four-layer extraction, the per-layer centering, the AFFM
softmax, the injection, stability logging, the abort gate, checkpointing and
the eval256 validation switch -- is exercised before 300k iterations are
committed to it.

THE GATE IS ON. `abort_on_trigger` is left at its default (true), so this run
stops exactly as the real run would. 6000 iterations is chosen so the run
passes `ratio_rules_start_iter: 5000` and the ratio rules are actually
ENFORCED for the last 1000 iterations, rather than only measured.

The identity is renamed to SMOKE6K_* so nothing can land in, or auto-resume
from, `experiments/Holo_affm_render_fixed128_spatial_L3691_latent`.

Only these keys change: name, total_iter, iters, print_freq,
save_checkpoint_freq, val_freq, tb_logger_dir, dino_stability.log_freq,
dino_stability.devlog (a throwaway run must never append to the real devlog)
and dino_affm_stats_freq.

WHY dino_affm_stats_freq IS LOWERED. At the real value of 5000 the AFFM weights
would be measured at forward 1 and forward 5001 -- two points in the whole smoke
run, which cannot show a trajectory. It is dropped to 100 here so the weights
are observed ~60 times and the early movement away from the uniform 0.25 is
readable. This changes ONLY what is observed, never the maths: the softmax and
the fused prior are computed identically on every forward regardless.

Every other key -- crop, batch, optimizer, scheduler, seed, loss, means, layer
set, injection, fusion AND every gate threshold -- is copied verbatim.
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
                   'affm_render_fixed128_spatial_L3691_latent.yml')
DST_NAME = 'SMOKE6K_affm_render_L3691'
ITERS = 6000
VAL_FREQ = 2000          # three validation points, exercising the eval256 switch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=ITERS)
    ap.add_argument('--name', default=DST_NAME)
    ap.add_argument('--out', default=os.path.join(OUT, 'SMOKE6K_affm_render.yml'))
    ap.add_argument('--affm-freq', type=int, default=100)
    ap.add_argument('--batch', type=int, default=0,
                    help='override batch_size_per_gpu AND mini_batch_sizes. '
                         'CPU integration smoke only -- it answers "is the '
                         'wiring correct", never "what does this arm score", '
                         'so the batch is not part of what is being checked')
    ap.add_argument('--cpu', action='store_true',
                    help='set num_gpu: 0 so basicsr routes to CPU. CPU '
                         'integration smoke only -- the real config and the '
                         '6k GPU smoke both keep num_gpu: 1')
    ap.add_argument('--no-val', action='store_true',
                    help='push val_freq beyond the run length. CPU integration '
                         'smoke only: validating 339 full-256 images is not '
                         'feasible without a GPU, and the eval256 path is '
                         'already covered by smoke_tests_affm_render.py')
    args = ap.parse_args()
    globals()['ITERS'] = args.iters
    globals()['DST_NAME'] = args.name

    os.makedirs(OUT, exist_ok=True)
    dst = args.out

    Loader, Dumper = ordered_yaml()
    with open(SRC) as f:
        opt = yaml.load(f, Loader=Loader)

    real_name = opt['name']
    opt['name'] = DST_NAME
    opt['train']['total_iter'] = ITERS
    opt['datasets']['train']['iters'] = [ITERS]
    opt['logger']['print_freq'] = min(100, max(1, ITERS // 10))
    opt['logger']['save_checkpoint_freq'] = ITERS       # one checkpoint, not 3
    opt['val']['val_freq'] = (ITERS * 100 if args.no_val else VAL_FREQ)
    opt['logger']['tb_logger_dir'] = f'tb_logger/{DST_NAME}'
    # every iteration, so the injection_ratio TRAJECTORY is readable afterwards
    opt['dino_stability']['log_freq'] = 1
    opt['dino_stability']['devlog'] = None              # never the real devlog
    # observation frequency only -- see the module docstring
    opt['network_g']['dino_affm_stats_freq'] = args.affm_freq
    if args.batch:
        opt['datasets']['train']['batch_size_per_gpu'] = args.batch
        opt['datasets']['train']['mini_batch_sizes'] = [args.batch]
        opt['datasets']['train']['num_worker_per_gpu'] = 0
    if args.cpu:
        opt['num_gpu'] = 0                              # basicsr -> torch.device('cpu')
    # thresholds deliberately NOT touched: cap 10.0, ratio rules from 5000,
    # reference at 5000, growth 10x -- the gate behaves exactly as in the real run

    if opt['name'] == real_name:
        raise SystemExit('refusing to write a smoke config that carries the '
                         'real experiment name')

    with open(dst, 'w') as f:
        yaml.dump(opt, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False)
    print(f'  {DST_NAME} -> {dst}')
    print(f'  {ITERS} iters, val_freq {opt["val"]["val_freq"]}, gate ON (abort_on_trigger default true), '
          f'ratio rules enforced from 5000')
    print(f'  layers {opt["network_g"]["dino_layers"]}, '
          f'affm stats every {opt["network_g"]["dino_affm_stats_freq"]} forwards')

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
