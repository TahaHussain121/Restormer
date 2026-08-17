"""Four-panel visual results for ONE arm (E0-Fixed has no partner arm yet).

Panels, left to right:
    1e5 input · prediction · 1e7 target · absolute error

Cases are chosen from the per-image CSV so the picture cannot be cherry-picked:
    best    highest psnr_full
    median  the median case
    worst   the lowest psnr_full  -- always included
    repr.   the first images of the split, fixed, not chosen by score

`make_comparison_figures.py` is the E0-vs-E1 six-panel version and stays the
tool of record once a second arm finishes. This exists because E0-Fixed
completed first and its re-characterisation should be visible now.

PNG only. Nothing here computes a metric; it reads the CSV the evaluation
chain already wrote.
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


def load16(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'failed to read {p}')
    return img.squeeze().astype(np.float64) / 65535.


def panel(ax, img, title, vmax=None, cmap='inferno'):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax if vmax else img.max())
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def make_figure(out_path, rows, suptitle):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (label, image_id, noisy, pred, gt, psnr) in enumerate(rows):
        vmax = max(gt.max(), pred.max(), noisy.max())
        panel(axes[r, 0], noisy, f'1e5 input — {image_id}', vmax)
        panel(axes[r, 1], pred, f'E0-Fixed prediction ({psnr:.2f} dB)', vmax)
        panel(axes[r, 2], gt, '1e7 target', vmax)
        err = np.abs(gt - pred)
        im = axes[r, 3].imshow(err, cmap='magma')
        axes[r, 3].set_title(f'|error|  max {err.max():.3f}', fontsize=9)
        axes[r, 3].axis('off')
        fig.colorbar(im, ax=axes[r, 3], fraction=0.046)
        axes[r, 0].set_ylabel(label)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center',
                        fontsize=11, fontweight='bold')
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='Holo_E0_fixed128_baseline')
    ap.add_argument('--protocol', default='full256')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-repr', type=int, default=2)
    args = ap.parse_args()

    root = os.path.join(_PHASE3, 'results', args.experiment)
    csv_path = os.path.join(root, 'metrics',
                            f'{args.protocol}_{args.split}_per_image.csv')
    pred_dir = os.path.join(root, 'predictions',
                            f'{args.protocol}_{args.split}', 'raw')

    with open(csv_path) as f:
        recs = [r for r in csv.DictReader(f)]
    recs.sort(key=lambda r: float(r['psnr_full']))
    worst, median, best = recs[0], recs[len(recs) // 2], recs[-1]

    if args.protocol == 'crop128':
        base = os.path.join(root, 'predictions', f'{args.protocol}_{args.split}')
        gt_dir, in_dir = os.path.join(base, 'gt'), os.path.join(base, 'input')
    else:
        gt_dir = os.path.join(DATASET, f'{args.split}_clean')
        in_dir = os.path.join(DATASET, f'{args.split}_verynoisy')

    def row(label, rec):
        fn = rec['filename']
        return (label, fn,
                load16(os.path.join(in_dir, fn)),
                load16(os.path.join(pred_dir, fn)),
                load16(os.path.join(gt_dir, fn)),
                float(rec['psnr_full']))

    sel = [row('BEST', best), row('MEDIAN', median), row('WORST', worst)]
    for rec in recs[:0]:
        pass
    by_name = sorted(recs, key=lambda r: r['filename'])[:args.n_repr]
    sel += [row(f'REPR {i + 1}', r) for i, r in enumerate(by_name)]

    out = os.path.join(root, 'visuals',
                       f'cases_{args.protocol}_{args.split}.png')
    make_figure(out, sel,
                f'{args.experiment} — {args.protocol}/{args.split} '
                f'(best / median / worst by PSNR, plus fixed representatives)')


if __name__ == '__main__':
    main()
