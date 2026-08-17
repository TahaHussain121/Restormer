"""Six-panel E0-vs-E1 comparison PNGs, chosen so successes cannot be cherry-picked.

Panels, left to right:
    1e5 input · E0 prediction · E1 prediction · 1e7 target ·
    E0 absolute error · E1 absolute error

Cases, taken from the per-image delta CSV written by compare_e0_e1.py:
    representative  the first images of the split (fixed, not chosen by score)
    best            largest E1-minus-E0 PSNR gain
    median          the median case
    worst           the largest REGRESSION -- always included

PNG only. The same sample IDs are used for both models. Written under
results/comparisons/E0_vs_E1/figures/<protocol>_<split>/.
"""

import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'


def load(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'failed to read {p}')
    return img.squeeze().astype(np.float64) / 65535.


def panel(fig_path, image_id, noisy, e0, e1, gt, title):
    err0, err1 = np.abs(gt - e0), np.abs(gt - e1)
    emax = max(err0.max(), err1.max()) or 1.0
    fig, ax = plt.subplots(1, 6, figsize=(26, 4.6))
    for a, (name, img, kw) in zip(ax, [
            ('1e5 input', noisy, dict(cmap='inferno', vmin=0, vmax=1)),
            ('E0 prediction', e0, dict(cmap='inferno', vmin=0, vmax=1)),
            ('E1 prediction', e1, dict(cmap='inferno', vmin=0, vmax=1)),
            ('1e7 target', gt, dict(cmap='inferno', vmin=0, vmax=1)),
            ('E0 |error|', err0, dict(cmap='viridis', vmin=0, vmax=emax)),
            ('E1 |error|', err1, dict(cmap='viridis', vmin=0, vmax=emax))]):
        im = a.imshow(img, **kw)
        a.set_title(name, fontsize=12)
        a.set_xticks([]); a.set_yticks([])
        if name.endswith('|error|'):
            fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0', default='Holo_E0_fixed128_baseline')
    ap.add_argument('--e1', default='Holo_E1_addition_noisy_fixed128_spatial_B6_latent')
    ap.add_argument('--protocol', required=True, choices=['full256', 'crop128'])
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--n-representative', type=int, default=3)
    args = ap.parse_args()

    delta_csv = os.path.join(_PHASE3, 'results', 'comparisons', 'E0_vs_E1',
                             f'{args.protocol}_{args.split}_per_image_delta.csv')
    with open(delta_csv) as f:
        rows = [r for r in csv.DictReader(f)]
    rows.sort(key=lambda r: float(r['delta_psnr_full']))

    picks = []
    ordered_ids = sorted(r['filename'] for r in rows)
    by_id = {r['filename']: r for r in rows}
    for fn in ordered_ids[:args.n_representative]:
        picks.append((by_id[fn], 'representative'))
    picks.append((rows[-1], 'best_E1_improvement'))
    picks.append((rows[len(rows) // 2], 'median'))
    picks.append((rows[0], 'worst_regression'))

    def d(exp, sub):
        return os.path.join(_PHASE3, 'results', exp, 'predictions',
                            f'{args.protocol}_{args.split}', sub)

    if args.protocol == 'crop128':
        gt_dir = d(args.e0, 'gt')
        in_dir = d(args.e0, 'input')
    else:
        gt_dir = os.path.join(DATASET, f'{args.split}_clean')
        in_dir = os.path.join(DATASET, f'{args.split}_verynoisy')

    out_dir = os.path.join(_PHASE3, 'results', 'comparisons', 'E0_vs_E1',
                           'figures', f'{args.protocol}_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    for row, kind in picks:
        fn = row['filename']
        image_id = os.path.splitext(fn)[0]
        noisy, gt = load(os.path.join(in_dir, fn)), load(os.path.join(gt_dir, fn))
        e0 = load(os.path.join(d(args.e0, 'raw'), fn))
        e1 = load(os.path.join(d(args.e1, 'raw'), fn))
        title = (f'{kind}  —  {image_id}  [{args.protocol} / {args.split}]   '
                 f'E0 {float(row["e0_psnr_full"]):.3f} dB   '
                 f'E1 {float(row["e1_psnr_full"]):.3f} dB   '
                 f'delta {float(row["delta_psnr_full"]):+.3f} dB')
        out = os.path.join(out_dir, f'{kind}_{image_id}.png')
        panel(out, image_id, noisy, e0, e1, gt, title)
        print(f'  wrote {out}')

    print(f'\n{len(picks)} figures in {out_dir}\n'
          f'  worst regression is ALWAYS included -- do not delete it from the '
          f'report.')


if __name__ == '__main__':
    main()
