"""Cross-run abort rules 3 and 4, to be run every ~10k iterations FOR THE
ENTIRE RUN (not once at 20k).

Rules 1 and 2 (injection_ratio > 0.5, and >10x growth vs the ~5k reference) are
enforced inside the training step itself by
`basicsr/models/image_restoration_dino_model.py`, which hard-stops at any
iteration. They cannot be checked from outside often enough to matter.

Rules 3 and 4 are comparisons BETWEEN the two runs, so they live here:

  rule 3   E1 val PSNR is > 1.0 dB below E0 at approximately the same iteration
  rule 4   E1 training loss is clearly diverging relative to E0

A trigger is an OPTIMIZATION / STABILITY FAILURE. It is reported, appended to
the E1 devlog, and investigated. It is NOT evidence that the DINO prior is
useless, and it is NOT answered by changing the architecture, the layer, the
crop, the scheduler or the fusion.

This script only READS logs and WRITES a report. It never cancels a job -- that
decision stays with the operator.
"""

import argparse
import datetime
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))

VAL_RE = re.compile(r'Validation .*?# psnr: ([0-9.]+).*?# ssim: ([0-9.]+)',
                    re.S)
ITER_RE = re.compile(r'iter:\s*([0-9,]+).*?l_pix:\s*([0-9.eE+-]+)')


def read_val_curve(exp_dir):
    """-> {iter: psnr} parsed from the training logs, in iteration order."""
    curve = {}
    logs = sorted(f for f in os.listdir(exp_dir) if f.startswith('train_')
                  and f.endswith('.log'))
    for name in logs:
        last_iter = None
        with open(os.path.join(exp_dir, name), errors='ignore') as f:
            for line in f:
                m = ITER_RE.search(line)
                if m:
                    last_iter = int(m.group(1).replace(',', ''))
                m = re.search(r'# psnr: ([0-9.]+)', line)
                if m and last_iter is not None:
                    curve[last_iter] = float(m.group(1))
    return dict(sorted(curve.items()))


def read_loss_curve(exp_dir):
    curve = {}
    logs = sorted(f for f in os.listdir(exp_dir) if f.startswith('train_')
                  and f.endswith('.log'))
    for name in logs:
        with open(os.path.join(exp_dir, name), errors='ignore') as f:
            for line in f:
                m = ITER_RE.search(line)
                if m:
                    curve[int(m.group(1).replace(',', ''))] = float(m.group(2))
    return dict(sorted(curve.items()))


def nearest(curve, it, tol):
    best, bd = None, None
    for k in curve:
        d = abs(k - it)
        if d <= tol and (bd is None or d < bd):
            best, bd = k, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0-name', default='Holo_E0_fixed128_baseline')
    ap.add_argument('--e1-name', default='Holo_E1_addition_noisy_fixed128_spatial_B6_latent')
    ap.add_argument('--psnr-margin', type=float, default=1.0,   # rule 3
                    help='dB below E0 that triggers rule 3')
    ap.add_argument('--loss-ratio', type=float, default=1.5,    # rule 4
                    help='E1/E0 smoothed training loss ratio that triggers rule 4')
    ap.add_argument('--iter-tol', type=int, default=4000,
                    help='"approximately the same iteration" tolerance')
    ap.add_argument('--devlog', default=os.path.join(
        _PHASE3, 'devlogs', 'E1_addition_noisy_fixed128_spatial_B6_latent.md'))
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'comparisons', 'E0_vs_E1', 'stability_gate.json'))
    ap.add_argument('--append-devlog', action='store_true')
    args = ap.parse_args()

    e0_dir = os.path.join(_REPO, 'experiments', args.e0_name)
    e1_dir = os.path.join(_REPO, 'experiments', args.e1_name)
    for d in (e0_dir, e1_dir):
        if not os.path.isdir(d):
            raise SystemExit(f'{d} does not exist yet -- nothing to check')

    v0, v1 = read_val_curve(e0_dir), read_val_curve(e1_dir)
    l0, l1 = read_loss_curve(e0_dir), read_loss_curve(e1_dir)
    triggers, rows = [], []

    for it, p1 in v1.items():
        k = nearest(v0, it, args.iter_tol)
        if k is None:
            continue
        delta = p1 - v0[k]
        rows.append({'e1_iter': it, 'e0_iter': k, 'e1_psnr': p1,
                     'e0_psnr': v0[k], 'delta_db': delta})
        if delta < -args.psnr_margin:
            triggers.append(
                f'RULE 3: at iter {it} E1 val PSNR {p1:.4f} dB is '
                f'{-delta:.4f} dB below E0 ({v0[k]:.4f} dB at iter {k}), '
                f'more than the {args.psnr_margin} dB margin')

    loss_rows = []
    for it, x1 in l1.items():
        k = nearest(l0, it, args.iter_tol)
        if k is None or l0[k] <= 0:
            continue
        ratio = x1 / l0[k]
        loss_rows.append({'iter': it, 'e1_loss': x1, 'e0_loss': l0[k],
                          'ratio': ratio})
        if ratio > args.loss_ratio:
            triggers.append(
                f'RULE 4: at iter {it} E1 training loss {x1:.6e} is '
                f'{ratio:.2f}x E0 ({l0[k]:.6e} at iter {k})')

    stab = os.path.join(e1_dir, 'STABILITY_FAILURE.json')
    if os.path.isfile(stab):
        with open(stab) as f:
            rec = json.load(f)
        triggers.append(f'RULE 1/2 (in-run): {rec["reason"]} at iter {rec["iter"]}')

    report = {
        'checked': datetime.datetime.now().astimezone().isoformat(),
        'e0': args.e0_name, 'e1': args.e1_name,
        'psnr_margin_db': args.psnr_margin, 'loss_ratio_max': args.loss_ratio,
        'iter_tolerance': args.iter_tol,
        'n_val_points_compared': len(rows),
        'last_val_comparison': rows[-1] if rows else None,
        'last_loss_comparison': loss_rows[-1] if loss_rows else None,
        'triggers': triggers,
        'status': 'OPTIMIZATION / STABILITY FAILURE' if triggers else 'OK',
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != 'triggers'},
                     indent=2))
    for t in triggers:
        print(f'  !! {t}')
    print(f'wrote {args.out}')

    if triggers and args.append_devlog and os.path.isfile(args.devlog):
        with open(args.devlog, 'a') as f:
            f.write(f'\n\n## {report["checked"]} — OPTIMIZATION / STABILITY '
                    f'FAILURE (cross-run gate)\n\n')
            for t in triggers:
                f.write(f'- {t}\n')
            f.write('\nThis is an OPTIMIZATION / STABILITY failure, not a '
                    'verdict on the DINO prior. Investigate before changing '
                    'anything; any design change needs a new experiment '
                    'identity.\n')
        print(f'appended to {args.devlog}')

    sys.exit(2 if triggers else 0)


if __name__ == '__main__':
    main()
