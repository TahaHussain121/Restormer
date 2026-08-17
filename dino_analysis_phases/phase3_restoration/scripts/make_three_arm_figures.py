"""Side-by-side predictions for E0 / E1-noisy / E1-render on the SAME images.

Cases are selected by where **E0** struggles, not by where any DINO arm wins:
the N lowest-PSNR images in E0's own per-image CSV. That is the honest way to
ask "does the prior rescue the cases the baseline cannot do?", because the
selection never sees the E1 numbers. A run keyed on E1's best images would be
cherry-picking by construction.

Panels per row:
    1e5 input · E0 · E1-noisy · E1-render · 1e7 target

with each arm's PSNR on that image in its own panel title, so a row can be read
without going back to the CSV.

`make_comparison_figures.py` is the two-arm six-panel tool with error maps;
this is the three-arm view, which needs the width for the predictions instead.
Nothing here computes a metric — every number is read from the CSVs the
evaluation chain already wrote.
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

ARMS = [
    ('E0', 'Holo_E0_fixed128_baseline', 'no DINO'),
    ('E1-noisy', 'Holo_E1_addition_noisy_fixed128_spatial_B6_latent', 'DINO<-1e5'),
    ('E1-render', 'Holo_E1_addition_render_fixed128_spatial_B6_latent', 'DINO<-render'),
]


def load16(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'failed to read {p}')
    return img.squeeze().astype(np.float64) / 65535.


def read_csv(exp, protocol, split):
    p = os.path.join(_PHASE3, 'results', exp, 'metrics',
                     f'{protocol}_{split}_per_image.csv')
    with open(p) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='full256')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-cases', type=int, default=5)
    ap.add_argument('--mode', default='harsh', choices=['harsh', 'median'],
                    help='harsh = the images E0 does WORST on')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tables = {name: read_csv(exp, args.protocol, args.split)
              for name, exp, _ in ARMS}
    e0 = tables['E0']

    common = set.intersection(*(set(t) for t in tables.values()))
    ranked = sorted(common, key=lambda fn: float(e0[fn]['psnr_full']))
    if args.mode == 'harsh':
        chosen = ranked[:args.n_cases]
        title_mode = f"the {args.n_cases} images E0 does WORST on"
    else:
        mid = len(ranked) // 2
        chosen = ranked[mid:mid + args.n_cases]
        title_mode = f"{args.n_cases} median-difficulty images for E0"

    gt_dir = os.path.join(DATASET, f'{args.split}_clean')
    in_dir = os.path.join(DATASET, f'{args.split}_verynoisy')
    if args.protocol == 'crop128':
        base = os.path.join(_PHASE3, 'results', ARMS[0][1], 'predictions',
                            f'{args.protocol}_{args.split}')
        gt_dir, in_dir = os.path.join(base, 'gt'), os.path.join(base, 'input')

    ncol = 2 + len(ARMS)
    fig, axes = plt.subplots(len(chosen), ncol,
                             figsize=(3.05 * ncol, 3.35 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for r, fn in enumerate(chosen):
        noisy = load16(os.path.join(in_dir, fn))
        gt = load16(os.path.join(gt_dir, fn))
        preds = []
        for name, exp, _ in ARMS:
            p = os.path.join(_PHASE3, 'results', exp, 'predictions',
                             f'{args.protocol}_{args.split}', 'raw', fn)
            preds.append((name, load16(p), float(tables[name][fn]['psnr_full'])))
        vmax = max([gt.max(), noisy.max()] + [p.max() for _, p, _ in preds])

        axes[r, 0].imshow(noisy, cmap='inferno', vmin=0, vmax=vmax)
        axes[r, 0].set_title(f'1e5 input — {fn}', fontsize=9)
        best = max(p[2] for p in preds)
        for c, (name, img, psnr) in enumerate(preds, start=1):
            axes[r, c].imshow(img, cmap='inferno', vmin=0, vmax=vmax)
            win = ' ★' if psnr == best else ''
            axes[r, c].set_title(f'{name}  {psnr:.2f} dB{win}', fontsize=9,
                                 fontweight='bold' if win else 'normal')
        axes[r, ncol - 1].imshow(gt, cmap='inferno', vmin=0, vmax=vmax)
        axes[r, ncol - 1].set_title('1e7 target', fontsize=9)
        for c in range(ncol):
            axes[r, c].axis('off')

    fig.suptitle(f'{title_mode} — {args.protocol}/{args.split}, '
                 f'best-validation checkpoint per arm  (★ = best on that image)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = args.out or os.path.join(
        _PHASE3, 'results', 'comparisons',
        f'three_arm_{args.mode}_{args.protocol}_{args.split}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')

    print(f'\n  {"image":<12} ' + ' '.join(f'{n:>11}' for n, _, _ in ARMS))
    for fn in chosen:
        print(f'  {fn:<12} ' + ' '.join(
            f'{float(tables[n][fn]["psnr_full"]):>11.2f}' for n, _, _ in ARMS))


if __name__ == '__main__':
    main()
