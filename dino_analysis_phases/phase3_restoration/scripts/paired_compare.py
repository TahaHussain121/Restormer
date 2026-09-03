"""Paired per-image comparison of two arms: mean delta, bootstrap CI, Wilcoxon.

This is the procedure behind every "vs addition-render" row in DEVLOG Steps
32-33 and PHASE3_CHAPTER.md section 7.2. Until 2026-09-03 it existed only as
ad-hoc code in session scratch space; the numbers were reproduced exactly from
the tracked per-image CSVs before this file was written (aca-L6 full256 test:
+0.030, p=0.224; affm +0.230, p=4.3e-07; concat -0.016, p=0.978), so nothing
reported changes -- the script is now simply on record.

    python paired_compare.py --a <reference arm> --b <arm> \
        --protocol full256|crop128 --split val|test [--metric psnr_full]

Reads  results/<arm>/metrics/<protocol>_<split>_per_image.csv  (written by
summarize_metrics.py), pairs rows by filename, and reports for b - a:

  mean delta, 95% bootstrap CI of the mean (5000 resamples, seed 0),
  number of images where b > a, two-sided Wilcoxon signed-rank p-value.

The test split is READ here, never used to select anything. Output goes to
results/comparisons/paired/<b>_vs_<a>_<protocol>_<split>.json.
"""

import argparse
import csv
import datetime
import json
import os

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)


def load(exp, protocol, split, metric):
    p = os.path.join(_PHASE3, 'results', exp, 'metrics',
                     f'{protocol}_{split}_per_image.csv')
    if not os.path.isfile(p):
        raise SystemExit(f'missing {p}')
    with open(p) as f:
        return {r['filename']: float(r[metric]) for r in csv.DictReader(f)
                if r.get(metric, '') != ''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='reference arm (experiment name)')
    ap.add_argument('--b', required=True, help='arm compared against it')
    ap.add_argument('--protocol', required=True, choices=['full256', 'crop128'])
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--metric', default='psnr_full')
    ap.add_argument('--n-boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    a = load(args.a, args.protocol, args.split, args.metric)
    b = load(args.b, args.protocol, args.split, args.metric)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit('no shared images')
    if len(keys) != len(a) or len(keys) != len(b):
        print(f'WARNING: {len(keys)} shared images (a {len(a)}, b {len(b)})')
    d = np.array([b[k] - a[k] for k in keys])

    rng = np.random.default_rng(args.seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(args.n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    w = stats.wilcoxon(d)

    rec = {
        'a': args.a, 'b': args.b, 'delta': 'b - a', 'metric': args.metric,
        'protocol': args.protocol, 'split': args.split, 'n': int(len(d)),
        'mean_a': float(np.mean([a[k] for k in keys])),
        'mean_b': float(np.mean([b[k] for k in keys])),
        'mean_delta': float(d.mean()), 'ci95': [lo, hi],
        'b_better_on': int((d > 0).sum()), 'a_better_on': int((d < 0).sum()),
        'wilcoxon_p': float(w.pvalue), 'wilcoxon_stat': float(w.statistic),
        'n_boot': args.n_boot, 'seed': args.seed,
        'created': datetime.datetime.now().astimezone().isoformat(),
    }
    print(f'{args.b} vs {args.a}  [{args.protocol}/{args.split}, {args.metric}]')
    print(f'  n={rec["n"]}  mean delta {rec["mean_delta"]:+.3f} dB  '
          f'95% CI [{lo:+.3f}, {hi:+.3f}]  b better on {rec["b_better_on"]}/'
          f'{rec["n"]}  Wilcoxon p={rec["wilcoxon_p"]:.2e}')

    out = args.out or os.path.join(
        _PHASE3, 'results', 'comparisons', 'paired',
        f'{args.b}_vs_{args.a}_{args.protocol}_{args.split}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
