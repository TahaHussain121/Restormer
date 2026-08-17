"""Aggregate the EXISTING metric scripts' per-image CSVs into the Phase-3
summary files. Aggregation only -- no metric is defined or redefined here.

Inputs (produced by the repository's own, unmodified scripts):
  Deraining_Holo/masked_metrics.py --csv     -> filename, mask_frac, psnr_full,
                                                psnr_mask, ssim_full, ssim_mask
  Deraining_Holo/analyze_sharpness.py --csv  -> lap/grad/hf per image + ratios

Outputs, per the required layout and NEVER mixing protocols:
  results/<experiment>/metrics/<protocol>_<split>_summary.json
  results/<experiment>/metrics/<protocol>_<split>_per_image.csv
      (the merged per-image table, with an explicit `protocol` column)

Validation numbers come from the uint8 training-time path and test numbers from
the uint16 path; they are stored in separate files and must only ever be
compared val-to-val and test-to-test.
"""

import argparse
import csv
import datetime
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)


def read_csv(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def stats(values):
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return None
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)) if a.size > 1 else 0.0,
            'median': float(np.median(a)), 'min': float(a.min()),
            'max': float(a.max()), 'n': int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True)
    ap.add_argument('--protocol', required=True, choices=['full256', 'crop128'])
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--masked-csv', required=True)
    ap.add_argument('--sharpness-csv', default=None)
    ap.add_argument('--predict-metadata', default=None)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(_PHASE3, 'results', args.experiment,
                                         'metrics')
    os.makedirs(outdir, exist_ok=True)

    masked = read_csv(args.masked_csv)
    sharp = read_csv(args.sharpness_csv)
    if not masked:
        raise SystemExit(f'no rows in {args.masked_csv}')

    rows = []
    for fn, m in sorted(masked.items()):
        row = {'protocol': args.protocol, 'split': args.split, 'filename': fn}
        row.update({k: v for k, v in m.items() if k != 'filename'})
        s = sharp.get(fn, {})
        row.update({k: v for k, v in s.items() if k != 'filename'})
        rows.append(row)

    per_image = os.path.join(outdir, f'{args.protocol}_{args.split}_per_image.csv')
    with open(per_image, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def col(name):
        return [float(r[name]) for r in rows if name in r and r[name] != '']

    summary = {
        'experiment': args.experiment,
        'protocol': args.protocol,
        'split': args.split,
        'n_images': len(rows),
        'psnr_scale': ('uint16 / data_range 1.0 (skimage) — the test-side path; '
                       'NEVER compare this to the 8-bit training-time val PSNR'),
        'metrics': {k: stats(col(k)) for k in
                    ('psnr_full', 'psnr_mask', 'ssim_full', 'ssim_mask',
                     'mask_frac', 'lap_ratio', 'grad_ratio', 'hf_ratio')
                    if col(k)},
        'per_image_csv': os.path.abspath(per_image),
        'sources': {'masked_metrics_csv': os.path.abspath(args.masked_csv),
                    'sharpness_csv': (os.path.abspath(args.sharpness_csv)
                                      if args.sharpness_csv else None),
                    'metric_definitions': 'Deraining_Holo/masked_metrics.py and '
                                          'Deraining_Holo/analyze_sharpness.py, '
                                          'unmodified'},
        'created': datetime.datetime.now().astimezone().isoformat(),
    }
    if args.predict_metadata and os.path.isfile(args.predict_metadata):
        with open(args.predict_metadata) as f:
            summary['prediction'] = json.load(f)

    out = os.path.join(outdir, f'{args.protocol}_{args.split}_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'{args.experiment}  {args.protocol}/{args.split}  n={len(rows)}')
    for k, v in summary['metrics'].items():
        print(f'  {k:<12} {v["mean"]:+.4f} +/- {v["std"]:.4f}  '
              f'(median {v["median"]:+.4f})')
    print(f'wrote {out}\nwrote {per_image}')


if __name__ == '__main__':
    main()
