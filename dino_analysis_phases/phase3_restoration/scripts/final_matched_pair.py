"""The final matched-pair analysis — the four-cell comparison and the paired
tests pre-registered for the two final experiments (DEVLOG Step 48 and the
devlogs `multilevel_addition_render_fixed128_B6.md` /
`multilevel_aca_render_fixed128_B6.md`).

The 2x2, fusion operator by injection layout, B6 throughout:

                    post-latent only      post-latent + dec3 + dec2
    addition        --pl-add              --ml-add
    ACA             --pl-aca              --ml-aca

Comparisons, each a per-image paired difference on psnr_full:

    A  ml_add - pl_add                       does repeated delivery help addition?
    B  ml_aca - pl_aca                       does repeated delivery help ACA?
    C  ml_aca - ml_add                       operator at matched sites
    D  (ml_aca - pl_aca) - (ml_add - pl_add) does ACA gain MORE from the extra
                                             sites than addition does?

For each, on every split x protocol: paired mean, 95% bootstrap interval,
Wilcoxon p, images better / worse. Then the PRE-REGISTERED verdict fields, kept
SEPARATE, never merged into one word:

    observed           the test full256 paired mean (the primary outcome)
    reliable           95% interval excludes 0 AND validation full256 has the
                       same sign — cross-split replication is this project's
                       standing substitute for seed repeats
    exceeds_threshold  observed >= +0.10 dB (the declared practical threshold,
                       distinct from statistical evidence)
    crop_tradeoff      test crop128 paired mean <= -0.10 dB, reported as a
                       trade-off, never hidden by a full256 gain

Also: the four cells' whole-image and foreground PSNR/SSIM means, and the three
best and worst images for A and C (representative improvements and failures).

SINGLE SEED. Every interval here measures image-to-image variation of fixed
trained models, NOT training-seed variability. A non-significant result is not
evidence of equivalence.
"""

import argparse
import csv
import datetime
import json
import os

import numpy as np
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3 = os.path.dirname(_HERE)
DEFAULT = {'pl_add': 'Holo_postlatent_render_fixed128_B6',
           'ml_add': 'Holo_multilevel_addition_render_fixed128_B6',
           'pl_aca': 'Holo_aca_postlatent_render_fixed128_B6',
           'ml_aca': 'Holo_multilevel_aca_render_fixed128_B6'}
THRESHOLD = 0.10
CELLS = [(s, p) for s in ('val', 'test') for p in ('full256', 'crop128')]


def load(exp, protocol, split, col='psnr_full'):
    p = os.path.join(_P3, 'results', exp, 'metrics',
                     f'{protocol}_{split}_per_image.csv')
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return {r['filename']: float(r[col]) for r in csv.DictReader(f)
                if r.get(col, '') != ''}


def summarise(d, n_boot, seed):
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {'n': int(len(d)), 'mean': float(d.mean()),
            'ci95': [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
            'wilcoxon_p': p, 'better': int((d > 0).sum()),
            'worse': int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULT.items():
        ap.add_argument('--' + k.replace('_', '-'), default=v)
    ap.add_argument('--n-boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(
        _P3, 'results', 'comparisons', 'final_matched_pair.json'))
    ap.add_argument('--label', default='final matched pair')
    args = ap.parse_args()
    arms = {k: getattr(args, k) for k in DEFAULT}

    report = {'label': args.label, 'arms': arms, 'threshold_db': THRESHOLD,
              'caveat': 'single seed: intervals measure image variation, not '
                        'training-seed variability; non-significance is not '
                        'equivalence',
              'created': datetime.datetime.now().astimezone().isoformat(),
              'cells': {}, 'comparisons': {}, 'verdict': {},
              'representative': {}, 'missing': []}

    print(f'=== {args.label} ===')
    for k, e in arms.items():
        print(f'  {k:7s} {e}')

    # ---- the four cells ------------------------------------------------------
    print('\n--- cells: mean psnr_full (psnr_mask, ssim_full, ssim_mask) ---')
    for split, proto in CELLS:
        row = {}
        for k, e in arms.items():
            vals = {c: load(e, proto, split, c) for c in
                    ('psnr_full', 'psnr_mask', 'ssim_full', 'ssim_mask')}
            if vals['psnr_full'] is None:
                report['missing'].append(f'{e} {proto}/{split}')
                row[k] = None
                continue
            row[k] = {c: float(np.mean(list(v.values()))) for c, v in vals.items()
                      if v is not None}
        report['cells'][f'{proto}_{split}'] = row
        print(f'  {proto}/{split}: ' + '   '.join(
            f'{k} {v["psnr_full"]:.3f}' if v else f'{k} MISSING'
            for k, v in row.items()))

    # ---- comparisons A-D -----------------------------------------------------
    defs = {'A': ('ml_add', 'pl_add', None, None),
            'B': ('ml_aca', 'pl_aca', None, None),
            'C': ('ml_aca', 'ml_add', None, None),
            'D': ('ml_aca', 'pl_aca', 'ml_add', 'pl_add')}
    print('\n--- comparisons (paired per image, psnr_full) ---')
    for name, (b, a, b2, a2) in defs.items():
        report['comparisons'][name] = {}
        for split, proto in CELLS:
            data = {k: load(arms[k], proto, split) for k in {b, a, b2, a2} - {None}}
            if any(v is None for v in data.values()):
                report['comparisons'][name][f'{proto}_{split}'] = 'MISSING'
                print(f'  {name} {proto}/{split}: MISSING')
                continue
            keys = sorted(set.intersection(*(set(v) for v in data.values())))
            d = np.array([data[b][k] - data[a][k] for k in keys])
            if b2:
                d = d - np.array([data[b2][k] - data[a2][k] for k in keys])
            s = summarise(d, args.n_boot, args.seed)
            report['comparisons'][name][f'{proto}_{split}'] = s
            print(f'  {name} {proto}/{split}: {s["mean"]:+.3f} dB  '
                  f'[{s["ci95"][0]:+.3f}, {s["ci95"][1]:+.3f}]  p={s["wilcoxon_p"]:.2e}  '
                  f'better {s["better"]}/{s["n"]}')
            if name in ('A', 'C') and (split, proto) == ('test', 'full256'):
                order = np.argsort(d)
                report['representative'][name] = {
                    'best': [(keys[i], float(d[i])) for i in order[::-1][:3]],
                    'worst': [(keys[i], float(d[i])) for i in order[:3]]}

    # ---- pre-registered verdict fields, kept separate -------------------------
    print('\n--- verdict fields (primary: test full256) ---')
    for name in defs:
        c = report['comparisons'][name]
        t, v, cr = c.get('full256_test'), c.get('full256_val'), c.get('crop128_test')
        if not isinstance(t, dict):
            report['verdict'][name] = 'MISSING'
            print(f'  {name}: MISSING')
            continue
        excl0 = t['ci95'][0] > 0 or t['ci95'][1] < 0
        same = isinstance(v, dict) and np.sign(v['mean']) == np.sign(t['mean'])
        fld = {'observed_db': t['mean'],
               'reliable': bool(excl0 and same),
               'ci_excludes_zero': bool(excl0),
               'val_same_sign': bool(same),
               'exceeds_threshold': bool(t['mean'] >= THRESHOLD),
               'crop_tradeoff': bool(isinstance(cr, dict) and cr['mean'] <= -THRESHOLD),
               'crop128_test_db': cr['mean'] if isinstance(cr, dict) else None}
        report['verdict'][name] = fld
        print(f'  {name}: observed {fld["observed_db"]:+.3f}  reliable={fld["reliable"]} '
              f'(CI excl 0 {fld["ci_excludes_zero"]}, val same sign {fld["val_same_sign"]})  '
              f'exceeds +{THRESHOLD}={fld["exceeds_threshold"]}  '
              f'crop trade-off={fld["crop_tradeoff"]}')

    if report['missing']:
        print('\nMISSING cells: ' + '; '.join(report['missing']))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
