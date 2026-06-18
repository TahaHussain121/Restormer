"""
Standalone blur/sharpness analysis for already-saved predictions.

Compares the high-frequency content of predictions vs ground truth (and the
noisy input) without re-running the model. Reports Laplacian variance, Sobel
gradient magnitude and FFT high-frequency energy (Pred/GT < 1 => blurrier),
and saves an azimuthally-averaged radial power-spectrum plot.

Usage:
    python analyze_sharpness.py \
        --pred_dir  results/Holo_test_224k/raw \
        --gt_dir    $DATASET/test_clean \
        --noisy_dir $DATASET/test_noisy \
        --out       results/Holo_test_224k/radial_power_spectrum.png
"""
import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', required=True)
parser.add_argument('--gt_dir', required=True)
parser.add_argument('--noisy_dir', required=True)
parser.add_argument('--out', default=None, help='Path for the power-spectrum PNG')
args = parser.parse_args()

load = lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED).squeeze().astype(np.float64) / 65535.

def lap_var(x):
    return cv2.Laplacian(x, cv2.CV_64F).var()

def grad_mag(x):
    gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

def hf_energy(x):
    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = x.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    return float(P[r > 0.25 * r.max()].sum() / P.sum())

NBINS = 128
def radial_power(x):
    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = x.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    idx = np.clip((r / r.max() * (NBINS - 1)).astype(int), 0, NBINS - 1)
    radial = np.bincount(idx.ravel(), weights=P.ravel(), minlength=NBINS)
    counts = np.bincount(idx.ravel(), minlength=NBINS)
    return radial / np.maximum(counts, 1)

files = natsorted(glob(os.path.join(args.pred_dir, '*.png')))
sharp = {k: {'noisy': [], 'pred': [], 'gt': []} for k in ('lap', 'grad', 'hf')}
radial_acc = {'noisy': np.zeros(NBINS), 'pred': np.zeros(NBINS), 'gt': np.zeros(NBINS)}
n = 0
for f in files:
    b = os.path.basename(f)
    gp, npth = os.path.join(args.gt_dir, b), os.path.join(args.noisy_dir, b)
    if not (os.path.exists(gp) and os.path.exists(npth)):
        continue
    pr, gt, no = load(f), load(gp), load(npth)
    for name, fn in (('lap', lap_var), ('grad', grad_mag), ('hf', hf_energy)):
        sharp[name]['noisy'].append(fn(no))
        sharp[name]['pred'].append(fn(pr))
        sharp[name]['gt'].append(fn(gt))
    for key, arr in (('noisy', no), ('pred', pr), ('gt', gt)):
        radial_acc[key] += radial_power(arr)
    n += 1

names = {'lap': 'Laplacian var', 'grad': 'Sobel grad', 'hf': 'HF energy frac'}
print(f'{n} images\n')
print(f'{"Sharpness":<16}{"Noisy":>12}{"Pred":>12}{"GT":>12}{"Pred/GT":>10}')
for k in ('lap', 'grad', 'hf'):
    pn, pp, pg = np.mean(sharp[k]['noisy']), np.mean(sharp[k]['pred']), np.mean(sharp[k]['gt'])
    print(f'{names[k]:<16}{pn:>12.5f}{pp:>12.5f}{pg:>12.5f}{pp / pg:>10.3f}')
print('(Pred/GT < 1.0 => prediction is blurrier than GT)')

freq = np.linspace(0, 1, NBINS)
fig, ax = plt.subplots(figsize=(7, 5))
for key, color in (('noisy', 'tab:orange'), ('pred', 'tab:blue'), ('gt', 'tab:green')):
    ax.plot(freq, radial_acc[key] / n + 1e-20, label=key.capitalize(), color=color, lw=2)
ax.set_yscale('log')
ax.set_xlabel('Spatial frequency (fraction of Nyquist)', fontsize=12)
ax.set_ylabel('Mean radial power (log scale)', fontsize=12)
ax.set_title('Azimuthally-averaged power spectrum', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
out = args.out or os.path.join(args.pred_dir, '..', 'radial_power_spectrum.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nRadial power spectrum -> {os.path.abspath(out)}')
