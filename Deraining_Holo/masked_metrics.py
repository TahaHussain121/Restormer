"""
Full-image vs foreground-masked PSNR / SSIM between predictions and GT.

The radar heatmaps are mostly black (~0) background with a smooth object that
fades into it. That large trivial background inflates full-image PSNR, so the
masked variant -- restricted to the foreground object -- better reflects object
reconstruction quality and makes DINO-variant comparisons more meaningful.

The foreground mask is built from the GT (gt > threshold, optionally dilated to
include the soft fade region), so it is identical across model variants.

Usage:
    python masked_metrics.py \
        --pred_dir  results/Holo_test_224k/raw \
        --gt_dir    $DATASET/test_clean \
        --threshold 0.01 \
        --dilate    3 \
        --csv       results/Holo_test_224k/masked_metrics_per_image.csv
"""
import os
import csv
import math
import argparse
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', required=True)
parser.add_argument('--gt_dir', required=True)
parser.add_argument('--threshold', type=float, default=0.01,
                    help='Foreground threshold on the normalized GT in [0,1] (default 0.01)')
parser.add_argument('--dilate', type=int, default=0,
                    help='Morphological dilation iterations to include the soft fade (default 0)')
parser.add_argument('--close', type=int, default=0,
                    help='Morphological closing iterations to fill interior holes (default 0)')
parser.add_argument('--csv', default=None, help='Optional path for per-image CSV export')
parser.add_argument('--save_mask_viz', default=None,
                    help='Optional path to save a GT/mask/overlay figure for a few sample images')
args = parser.parse_args()

def load(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.squeeze().astype(np.float64) / 65535.

KERNEL = np.ones((3, 3), np.uint8)

def build_mask(gt):
    """Foreground mask from GT: threshold, optional close (fill holes), optional dilate."""
    m = (gt > args.threshold).astype(np.uint8)
    if args.close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL, iterations=args.close)
    if args.dilate > 0:
        m = cv2.dilate(m, KERNEL, iterations=args.dilate)
    return m.astype(bool)

def psnr_masked(gt, pred, mask):
    mse = np.mean((gt[mask] - pred[mask]) ** 2)
    return float('inf') if mse == 0 else 10 * math.log10(1.0 / mse)

def ssim_masked(gt, pred, mask):
    _, smap = ssim_fn(gt, pred, data_range=1.0, full=True)
    return float(smap[mask].mean())

files = natsorted(glob(os.path.join(args.pred_dir, '*.png')))
records = []
viz_samples = []        # (filename, gt, mask) kept for the optional figure
for f in files:
    b = os.path.basename(f)
    gp = os.path.join(args.gt_dir, b)
    if not os.path.exists(gp):
        continue
    pr, gt = load(f), load(gp)
    if pr is None or gt is None:
        print(f'WARNING: skipping unreadable/corrupt file: {b}')
        continue

    mask = build_mask(gt)
    if not mask.any():
        print(f'WARNING: empty foreground mask for {b} (threshold too high?), skipping')
        continue
    if args.save_mask_viz:
        viz_samples.append((b, gt, mask))

    records.append({
        'filename': b,
        'mask_frac': float(mask.mean()),
        'psnr_full': psnr_fn(gt, pr, data_range=1.0),
        'psnr_mask': psnr_masked(gt, pr, mask),
        'ssim_full': ssim_fn(gt, pr, data_range=1.0),
        'ssim_mask': ssim_masked(gt, pr, mask),
    })

if not records:
    raise RuntimeError('No matching readable image pairs found.')

def ms(key):
    v = np.array([r[key] for r in records])
    return v.mean(), v.std()

n = len(records)
mf = np.mean([r['mask_frac'] for r in records])
print(f'{n} images | mean foreground coverage: {mf * 100:.1f}% '
      f'(threshold={args.threshold}, close={args.close}, dilate={args.dilate})\n')
print(f'{"Metric":<8}{"Full image":>22}{"Masked (foreground)":>24}')
for label, full_key, mask_key, unit in (
        ('PSNR', 'psnr_full', 'psnr_mask', ' dB'),
        ('SSIM', 'ssim_full', 'ssim_mask', '')):
    fm, fs = ms(full_key)
    mm, msd = ms(mask_key)
    print(f'{label:<8}{f"{fm:.4f} +/- {fs:.4f}{unit}":>22}{f"{mm:.4f} +/- {msd:.4f}{unit}":>24}')

if args.csv:
    fields = ['filename', 'mask_frac', 'psnr_full', 'psnr_mask', 'ssim_full', 'ssim_mask']
    with open(args.csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f'\nPer-image CSV -> {os.path.abspath(args.csv)}')

if args.save_mask_viz and viz_samples:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # three samples spread across the set: first, middle, last
    picks = [viz_samples[0], viz_samples[len(viz_samples) // 2], viz_samples[-1]]
    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 4 * len(picks)))
    for row, (b, gt, mask) in enumerate(picks):
        axes[row, 0].imshow(gt, cmap='inferno', vmin=0, vmax=1, aspect='auto')
        axes[row, 0].set_title(f'GT  ({b})', fontsize=11)
        axes[row, 1].imshow(mask, cmap='gray', aspect='auto')
        axes[row, 1].set_title(f'Foreground mask  ({mask.mean() * 100:.1f}% pixels)', fontsize=11)
        over = plt.get_cmap('inferno')(gt)[..., :3].copy()
        over[~mask] *= 0.25                                   # dim excluded background
        axes[row, 2].imshow(over, aspect='auto')
        axes[row, 2].contour(mask, levels=[0.5], colors='cyan', linewidths=1.2)
        axes[row, 2].set_title('Overlay (masked bright, bg dimmed)', fontsize=11)
        for c in range(3):
            axes[row, c].axis('off')
    plt.tight_layout()
    plt.savefig(args.save_mask_viz, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Mask visualization -> {os.path.abspath(args.save_mask_viz)}')
