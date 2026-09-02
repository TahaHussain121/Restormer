"""Summarise the render-misalignment dose-response sweep.

Reads the per-image CSVs written by run_render_misalignment.sh and produces the
dose-response table: mean PSNR at each displacement, the loss against the
aligned k=0 condition, and a paired Wilcoxon test of that loss.

The k=0 row is a SELF-CHECK, not a result: it must reproduce the arm's recorded
validation figure. If it does not, the sweep is measuring something other than
the arm and the table must not be used.

    python summarize_misalignment.py [--split val]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3 = os.path.dirname(_HERE)
RESULTS = os.path.join(_P3, 'results', 'render_misalignment')

# The arm's own recorded validation numbers, for the k=0 self-check. Taken from
# the arm's evaluation CSVs (`metrics/_raw_masked_<protocol>_val.csv`, n=339),
# not retyped from a table.
RECORDED = {'full256': 24.1200, 'crop128': 22.1871}


def read_csv(path):
    """Return {image_id: psnr} from a masked_metrics per-image CSV."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = next((k for k in row if k.lower() in
                        ('image', 'image_id', 'filename', 'name')), None)
            psnr = next((k for k in row if k.lower() in
                         ('psnr', 'psnr_full', 'psnr_whole')), None)
            if key is None or psnr is None:
                raise SystemExit(f'unexpected columns in {path}: '
                                 f'{list(row)[:8]}')
            out[os.path.splitext(row[key])[0]] = float(row[psnr])
    if not out:
        raise SystemExit(f'empty CSV {path}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    args = ap.parse_args()

    shifts = sorted({int(os.path.basename(p).split('_')[2].replace('shift', ''))
                     for p in glob.glob(f'{RESULTS}/per_image_shift*.csv')})
    if not shifts:
        raise SystemExit(f'no per-image CSVs under {RESULTS} -- has the sweep '
                         f'finished?')

    report = {}
    for protocol in ('full256', 'crop128'):
        cells = {}
        for k in shifts:
            p = f'{RESULTS}/per_image_shift{k}_{protocol}_{args.split}.csv'
            if os.path.isfile(p):
                cells[k] = read_csv(p)
        if not cells or 0 not in cells:
            print(f'\n{protocol}: incomplete (need k=0), skipping')
            continue

        base = cells[0]
        ids = sorted(set(base))
        print(f'\n=== {protocol}  {args.split}  n={len(ids)} ===')
        print(f'{"shift px":>9} {"PSNR":>9} {"vs k=0":>9} {"median":>9} '
              f'{"worse on":>10} {"wilcoxon p":>12}')

        rows = []
        for k in sorted(cells):
            common = [i for i in ids if i in cells[k]]
            a = np.array([base[i] for i in common])
            b = np.array([cells[k][i] for i in common])
            d = b - a
            if k == 0:
                p = float('nan')
            else:
                p = wilcoxon(b, a).pvalue
            worse = int((d < 0).sum())
            print(f'{k:>9} {b.mean():>9.3f} {d.mean():>+9.3f} '
                  f'{np.median(d):>+9.3f} {worse:>6}/{len(common)} {p:>12.2e}')
            rows.append({'shift_px': k, 'n': len(common),
                         'psnr': float(b.mean()),
                         'delta_vs_aligned': float(d.mean()),
                         'median_delta': float(np.median(d)),
                         'worse_than_aligned': worse,
                         'wilcoxon_p': None if k == 0 else float(p)})
        report[protocol] = rows

        if protocol in RECORDED:
            got, want = rows[0]['psnr'], RECORDED[protocol]
            ok = abs(got - want) < 0.01
            print(f'\n  SELF-CHECK k=0: {got:.3f} vs recorded {want:.3f}  '
                  f'-> {"OK" if ok else "MISMATCH -- DO NOT USE THIS TABLE"}')
            if not ok:
                report.setdefault('_warnings', []).append(
                    f'{protocol} k=0 {got:.4f} != recorded {want:.4f}')

    out = f'{RESULTS}/misalignment_summary.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {out}')
    return 1 if report.get('_warnings') else 0


if __name__ == '__main__':
    sys.exit(main())
