"""E0 vs E1 comparison, against the PRE-REGISTERED thresholds.

Reads only the two experiments' own summary/per-image files and writes only
into results/comparisons/E0_vs_E1/. Protocols are compared like-for-like:
full256 against full256, crop128 against crop128, val against val, test against
test. Nothing is pooled across protocols.

PRE-REGISTERED, FROZEN before any result existed:

    Delta PSNR = E1 test PSNR - E0 test PSNR
      <  +0.10 dB   -> no meaningful PSNR gain
      +0.10..+0.30  -> marginal / promising
      >  +0.30 dB   -> meaningful improvement

PSNR is not the sole criterion. SSIM, object-only (masked) PSNR/SSIM, HF-energy
ratio, Laplacian variance and Sobel gradient are reported alongside it, because
a modest PSNR gain with clear high-frequency or object-structure recovery may
still be scientifically important.

MATCHED-128 INTERPRETATION RULE, also pre-registered: if E1 improves on
matched-128 but not on full-256, that indicates the prior is useful
in-distribution and that full-image scale/context transfer is the limiter -- it
does NOT mean the prior is useless.
"""

import argparse
import csv
import datetime
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)

THRESHOLDS = {'none': 0.10, 'marginal': 0.30}
METRICS = ['psnr_full', 'ssim_full', 'psnr_mask', 'ssim_mask',
           'hf_ratio', 'lap_ratio', 'grad_ratio']


def verdict(delta_db):
    if delta_db < THRESHOLDS['none']:
        return 'no meaningful PSNR improvement (< +0.10 dB)'
    if delta_db <= THRESHOLDS['marginal']:
        return 'marginal / promising (+0.10 .. +0.30 dB)'
    return 'meaningful improvement (> +0.30 dB)'


def load_per_image(exp, protocol, split):
    p = os.path.join(_PHASE3, 'results', exp, 'metrics',
                     f'{protocol}_{split}_per_image.csv')
    if not os.path.isfile(p):
        raise SystemExit(f'missing {p}')
    with open(p) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0', default='Holo_E0_fixed128_baseline')
    ap.add_argument('--e1', default='Holo_E1_addition_noisy_fixed128_spatial_B6_latent')
    ap.add_argument('--protocol', required=True, choices=['full256', 'crop128'])
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--outdir', default=os.path.join(
        _PHASE3, 'results', 'comparisons', 'E0_vs_E1'))
    args = ap.parse_args()

    a = load_per_image(args.e0, args.protocol, args.split)
    b = load_per_image(args.e1, args.protocol, args.split)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit('no shared images between the two experiments')
    if len(shared) != len(a) or len(shared) != len(b):
        print(f'WARNING: comparing {len(shared)} shared images '
              f'(E0 {len(a)}, E1 {len(b)})')

    os.makedirs(args.outdir, exist_ok=True)
    per = []
    for fn in shared:
        row = {'filename': fn, 'protocol': args.protocol, 'split': args.split}
        for m in METRICS:
            if m in a[fn] and a[fn][m] != '' and m in b[fn] and b[fn][m] != '':
                row[f'e0_{m}'] = float(a[fn][m])
                row[f'e1_{m}'] = float(b[fn][m])
                row[f'delta_{m}'] = float(b[fn][m]) - float(a[fn][m])
        per.append(row)

    csv_path = os.path.join(
        args.outdir, f'{args.protocol}_{args.split}_per_image_delta.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)

    summary = {'e0': args.e0, 'e1': args.e1, 'protocol': args.protocol,
               'split': args.split, 'n_images': len(shared),
               'pre_registered_thresholds_db': THRESHOLDS,
               'metrics': {},
               'created': datetime.datetime.now().astimezone().isoformat()}
    for m in METRICS:
        d = [r[f'delta_{m}'] for r in per if f'delta_{m}' in r]
        if not d:
            continue
        d = np.asarray(d)
        e0 = np.asarray([r[f'e0_{m}'] for r in per if f'e0_{m}' in r])
        e1 = np.asarray([r[f'e1_{m}'] for r in per if f'e1_{m}' in r])
        summary['metrics'][m] = {
            'e0_mean': float(e0.mean()), 'e1_mean': float(e1.mean()),
            'delta_mean': float(d.mean()), 'delta_median': float(np.median(d)),
            'delta_std': float(d.std(ddof=1)) if d.size > 1 else 0.0,
            'e1_better_fraction': float((d > 0).mean()),
        }

    if 'psnr_full' in summary['metrics']:
        dm = summary['metrics']['psnr_full']['delta_mean']
        summary['delta_psnr_db'] = dm
        summary['pre_registered_verdict'] = verdict(dm)
        if args.split != 'test':
            summary['verdict_note'] = (
                'The pre-registered thresholds are defined on the TEST split. '
                'This is the ' + args.split + ' split; the verdict string is '
                'informational only.')

    worst = sorted((r for r in per if 'delta_psnr_full' in r),
                   key=lambda r: r['delta_psnr_full'])
    summary['worst_regressions'] = worst[:5]
    summary['best_improvements'] = worst[-5:][::-1]
    summary['median_case'] = worst[len(worst) // 2] if worst else None

    out = os.path.join(args.outdir, f'{args.protocol}_{args.split}_comparison.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{args.e1}  vs  {args.e0}   [{args.protocol} / {args.split}, '
          f'n={len(shared)}]')
    for m, v in summary['metrics'].items():
        print(f'  {m:<12} E0 {v["e0_mean"]:+.4f}   E1 {v["e1_mean"]:+.4f}   '
              f'delta {v["delta_mean"]:+.4f}   E1 better on '
              f'{v["e1_better_fraction"] * 100:.1f}%')
    if 'delta_psnr_db' in summary:
        print(f'\n  Delta PSNR = {summary["delta_psnr_db"]:+.4f} dB  ->  '
              f'{summary["pre_registered_verdict"]}')
    print(f'wrote {out}\nwrote {csv_path}')


if __name__ == '__main__':
    main()
