"""Checkpoint selection: highest VALIDATION PSNR. The test split never selects.

Both E0 and E1 are selected the same way, from the same parsed quantity (the
8-bit `use_image: true` validation PSNR logged every 4000 iterations). The test
set is not read here at all -- it cannot influence the choice even by accident.

Prints and records: best checkpoint path, best iteration, best val PSNR, the
val SSIM at that iteration, and the top-5 spread (the Exp-2 post-mortem showed
the top-5 spanning 0.016 dB, which is worth knowing before reading anything
into a "best" checkpoint).
"""

import argparse
import datetime
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))

ITER_RE = re.compile(r'iter:\s*([0-9,]+)')
PSNR_RE = re.compile(r'#\s*psnr:\s*([0-9.]+)')
SSIM_RE = re.compile(r'#\s*ssim:\s*([0-9.]+)')


def parse(exp_dir):
    """-> {iter: (psnr, ssim)} from the training logs."""
    out = {}
    logs = sorted(f for f in os.listdir(exp_dir)
                  if f.startswith('train_') and f.endswith('.log'))
    if not logs:
        raise SystemExit(f'no train_*.log in {exp_dir}')
    for name in logs:
        last_iter, psnr = None, None
        with open(os.path.join(exp_dir, name), errors='ignore') as f:
            for line in f:
                m = ITER_RE.search(line)
                if m:
                    last_iter = int(m.group(1).replace(',', ''))
                m = PSNR_RE.search(line)
                if m and last_iter is not None:
                    psnr = float(m.group(1))
                m = SSIM_RE.search(line)
                if m and psnr is not None and last_iter is not None:
                    out[last_iter] = (psnr, float(m.group(1)))
                    psnr = None
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='BasicSR experiment name')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    exp_dir = os.path.join(_REPO, 'experiments', args.name)
    curve = parse(exp_dir)
    if not curve:
        raise SystemExit(f'no validation entries parsed from {exp_dir}')

    ranked = sorted(curve.items(), key=lambda kv: -kv[1][0])
    best_iter, (best_psnr, best_ssim) = ranked[0]
    ckpt = os.path.join(exp_dir, 'models', f'net_g_{best_iter}.pth')
    exists = os.path.isfile(ckpt)

    rec = {
        'experiment': args.name,
        'selection_metric': 'validation PSNR (8-bit, use_image: true) — the '
                            'test split never drives selection',
        'best_iter': best_iter,
        'best_val_psnr': best_psnr,
        'val_ssim_at_best': best_ssim,
        'best_checkpoint': ckpt,
        'checkpoint_exists': exists,
        'n_val_points': len(curve),
        'top5': [{'iter': i, 'val_psnr': v[0], 'val_ssim': v[1]}
                 for i, v in ranked[:5]],
        'top5_spread_db': (ranked[0][1][0] - ranked[min(4, len(ranked) - 1)][1][0]),
        'final_iter': max(curve),
        'final_val_psnr': curve[max(curve)][0],
        'selected': datetime.datetime.now().astimezone().isoformat(),
    }
    print(json.dumps(rec, indent=2))
    if not exists:
        print(f'WARNING: {ckpt} is not on disk (checkpoints are saved every '
              f'2000 iters; validation runs every 4000, so this should exist)')

    out = args.out or os.path.join(_PHASE3, 'results', args.name, 'metadata',
                                   'best_checkpoint.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
