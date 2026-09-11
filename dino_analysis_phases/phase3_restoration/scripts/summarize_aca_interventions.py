"""Summarise the ACA checkpoint-intervention sweep.

Reads the per-image CSVs written by run_aca_interventions.sh and reports, for
each protocol: mean PSNR under each condition, the change against the
as-trained condition, a paired Wilcoxon test of that change, a bootstrap 95%
interval on the mean change, and how many images got worse.

The `none` row is a SELF-CHECK, not a result: it must reproduce the arm's own
recorded validation figure. The driver additionally verifies bit-identity of the
predictions; this script checks the metric agrees.

WHAT THESE NUMBERS LICENSE. They are interventions on a TRAINED model, so they
establish what that model DEPENDS on. They do not show how a model trained
without the component would perform.

    python summarize_aca_interventions.py [--split val]
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3 = os.path.dirname(_HERE)
RESULTS = os.path.join(_P3, 'results', 'aca_interventions')
ARM = 'Holo_aca_render_fixed128_L6_latent'
CONDITIONS = ['none', 'no_cross', 'uniform_cross']


def recorded(protocol, split):
    """The arm's own evaluation mean, read from its metrics CSV, not retyped."""
    p = os.path.join(_P3, 'results', ARM, 'metrics',
                     f'{protocol}_{split}_per_image.csv')
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        v = [float(r['psnr_full']) for r in csv.DictReader(f)]
    return float(np.mean(v)) if v else None


def read_csv(path):
    """{image_id: psnr_full} from a masked_metrics per-image CSV."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = next((k for k in row if k.lower() in
                        ('image', 'image_id', 'filename', 'name')), None)
            psnr = next((k for k in row if k.lower() in
                         ('psnr_full', 'psnr')), None)
            if key and psnr and row[psnr] != '':
                out[os.path.splitext(row[key])[0]] = float(row[psnr])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    report = {'arm': ARM, 'split': args.split, 'checkpoint': 236000,
              'caveat': 'interventions on a TRAINED model establish dependence, '
                        'not the performance of a model trained without the '
                        'component',
              'protocols': {}}

    for protocol in ('full256', 'crop128'):
        data = {}
        for cond in CONDITIONS:
            p = os.path.join(RESULTS,
                             f'per_image_{cond}_{protocol}_{args.split}.csv')
            if os.path.isfile(p):
                data[cond] = read_csv(p)
        if 'none' not in data:
            print(f'{protocol}: no `none` condition on disk -- skipping')
            continue

        base = data['none']
        ref = recorded(protocol, args.split)
        mean_none = float(np.mean(list(base.values())))
        ok = ref is None or abs(mean_none - ref) < 5e-3
        print(f'\n=== {protocol} / {args.split}  (n={len(base)}) ===')
        print(f'  SELF-CHECK  none {mean_none:.4f}  vs recorded '
              f'{"n/a" if ref is None else f"{ref:.4f}"}  '
              f'{"OK" if ok else "*** MISMATCH -- DO NOT USE THIS TABLE ***"}')
        rows = []
        print(f'  {"condition":<16}{"PSNR":>9}{"delta":>9}{"95% CI":>20}'
              f'{"worse on":>11}{"wilcoxon p":>13}')
        for cond in CONDITIONS:
            if cond not in data:
                continue
            cur = data[cond]
            keys = sorted(set(base) & set(cur))
            m = float(np.mean([cur[k] for k in keys]))
            if cond == 'none':
                print(f'  {cond:<16}{m:>9.4f}{"—":>9}{"—":>20}{"—":>11}{"—":>13}')
                rows.append({'condition': cond, 'psnr': m, 'delta': None})
                continue
            d = np.array([cur[k] - base[k] for k in keys])
            rng = np.random.default_rng(args.seed)
            boots = np.array([rng.choice(d, len(d)).mean()
                              for _ in range(args.n_boot)])
            lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            p = wilcoxon(d).pvalue
            worse = int((d < 0).sum())
            print(f'  {cond:<16}{m:>9.4f}{d.mean():>+9.4f}'
                  f'{f"[{lo:+.3f}, {hi:+.3f}]":>20}{f"{worse}/{len(d)}":>11}'
                  f'{p:>13.2e}')
            rows.append({'condition': cond, 'psnr': m, 'delta': float(d.mean()),
                         'ci95': [float(lo), float(hi)], 'worse_on': worse,
                         'n': len(d), 'wilcoxon_p': float(p)})
        report['protocols'][protocol] = {
            'n': len(base), 'self_check_ok': bool(ok),
            'recorded_mean': ref, 'rows': rows}

    out = os.path.join(RESULTS, f'aca_interventions_{args.split}.json')
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
