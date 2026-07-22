"""
Standalone blur/sharpness analysis for already-saved predictions.

Compares the high-frequency content of predictions vs ground truth (and the
noisy input) without re-running the model. Reports Laplacian variance, Sobel
gradient magnitude and FFT high-frequency energy (Pred/GT < 1 => blurrier),
with mean/std/median of the per-image Pred/GT ratios, and saves an
azimuthally-averaged radial power-spectrum plot.

Usage:
    python analyze_sharpness.py \
        --pred_dir  results/Holo_test_224k/raw \
        --gt_dir    $DATASET/test_clean \
        --noisy_dir $DATASET/test_noisy \
        --out       results/Holo_test_224k/radial_power_spectrum.png \
        --csv       results/Holo_test_224k/sharpness_per_image.csv \
        --window    # optional: Hann-window before FFT to reduce leakage
"""
import os
import csv
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
parser.add_argument('--csv', default=None, help='Optional path for per-image CSV export')
parser.add_argument('--window', action='store_true',
                    help='Apply a 2D Hann window before the FFT (reduces spectral leakage)')
args = parser.parse_args()

def load(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.squeeze().astype(np.float64) / 65535.

def lap_var(x):
    return cv2.Laplacian(x, cv2.CV_64F).var()

def grad_mag(x):
    gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

def hf_energy(x, win=None):
    if win is not None:
        x = x * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = x.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    return float(P[r > 0.25 * r.max()].sum() / P.sum())

NBINS = 128
def radial_power(x, win=None):
    if win is not None:
        x = x * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = x.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    idx = np.clip((r / r.max() * (NBINS - 1)).astype(int), 0, NBINS - 1)
    radial = np.bincount(idx.ravel(), weights=P.ravel(), minlength=NBINS)
    counts = np.bincount(idx.ravel(), minlength=NBINS)
    return radial / np.maximum(counts, 1)

files = natsorted(glob(os.path.join(args.pred_dir, '*.png')))
records = []                                                   # one dict per image
radial_acc = {'noisy': np.zeros(NBINS), 'pred': np.zeros(NBINS), 'gt': np.zeros(NBINS)}
shape_hw = None
n = 0
for f in files:
    b = os.path.basename(f)
    gp, npth = os.path.join(args.gt_dir, b), os.path.join(args.noisy_dir, b)
    if not (os.path.exists(gp) and os.path.exists(npth)):
        continue
    pr, gt, no = load(f), load(gp), load(npth)
    if pr is None or gt is None or no is None:
        print(f'WARNING: skipping unreadable/corrupt file: {b}')
        continue

    win = np.outer(np.hanning(pr.shape[0]), np.hanning(pr.shape[1])) if args.window else None

    rec = {'filename': b}
    for key, arr in (('noisy', no), ('pred', pr), ('gt', gt)):
        rec[f'lap_{key}'] = lap_var(arr)
        rec[f'grad_{key}'] = grad_mag(arr)
        rec[f'hf_{key}'] = hf_energy(arr, win)
        radial_acc[key] += radial_power(arr, win)
    for m in ('lap', 'grad', 'hf'):
        rec[f'{m}_ratio'] = rec[f'{m}_pred'] / rec[f'{m}_gt']
    records.append(rec)
    shape_hw = pr.shape
    n += 1

if n == 0:
    raise RuntimeError('No matching readable image triples found.')

# ---- spread reporting: mean / std / median of the per-image Pred/GT ratios ----
names = {'lap': 'Laplacian var', 'grad': 'Sobel grad', 'hf': 'HF energy frac'}
print(f'{n} images ({"Hann-windowed FFT" if args.window else "no FFT window"})\n')
print(f'{"Sharpness":<16}{"Noisy":>12}{"Pred":>12}{"GT":>12}{"Pred/GT (mean)":>16}{"std":>8}{"median":>9}')
for k in ('lap', 'grad', 'hf'):
    pn = np.mean([r[f'{k}_noisy'] for r in records])
    pp = np.mean([r[f'{k}_pred'] for r in records])
    pg = np.mean([r[f'{k}_gt'] for r in records])
    ratios = np.array([r[f'{k}_ratio'] for r in records])
    print(f'{names[k]:<16}{pn:>12.5f}{pp:>12.5f}{pg:>12.5f}'
          f'{ratios.mean():>16.3f}{ratios.std():>8.3f}{np.median(ratios):>9.3f}')
print('(Pred/GT < 1.0 => prediction is blurrier than GT)')

# ---- high-frequency noise-floor sanity check ---------------------------------
freq = np.linspace(0, 1, NBINS)               # = r / r.max() (corner radius)
hf_mask = freq > 0.25
hf_pred = (radial_acc['pred'] / n)[hf_mask].mean()
hf_gt = (radial_acc['gt'] / n)[hf_mask].mean()
hf_ratio = hf_pred / hf_gt
print(f'\nHF band (r > 0.25*r_max) mean radial power: '
      f'pred={hf_pred:.4g}  gt={hf_gt:.4g}  pred/gt={hf_ratio:.3f}')
if abs(hf_ratio - 1.0) < 0.05:
    print('WARNING: pred and gt agree within ~5% in this band — it may be sitting '
          'on the noise floor, so the HF ratio may be unstable.')

# ---- per-image CSV export ----------------------------------------------------
if args.csv:
    fields = ['filename']
    for m in ('lap', 'grad', 'hf'):
        fields += [f'{m}_noisy', f'{m}_pred', f'{m}_gt']
    fields += ['lap_ratio', 'grad_ratio', 'hf_ratio']
    with open(args.csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f'Per-image CSV -> {os.path.abspath(args.csv)}')

# ---- radial power spectrum plot ----------------------------------------------
# r.max() is the corner radius (~sqrt(2)*Nyquist), so the true Nyquist (h/2)
# lands below 1.0 on the x-axis -> mark it explicitly.
h, w = shape_hw
r_max = np.sqrt((h // 2) ** 2 + (w // 2) ** 2)
nyquist_frac = (h // 2) / r_max

fig, ax = plt.subplots(figsize=(7, 5))
for key, color in (('noisy', 'tab:orange'), ('pred', 'tab:blue'), ('gt', 'tab:green')):
    ax.plot(freq, radial_acc[key] / n + 1e-20, label=key.capitalize(), color=color, lw=2)
ax.axvline(nyquist_frac, ls='--', color='gray', lw=1.2)
ax.annotate('Nyquist', xy=(nyquist_frac, ax.get_ylim()[1]),
            xytext=(nyquist_frac + 0.01, ax.get_ylim()[1]),
            va='top', ha='left', fontsize=9, color='gray')
ax.set_yscale('log')
ax.set_xlabel('Spatial frequency (fraction of max radius)', fontsize=12)
ax.set_ylabel('Mean radial power (log scale)', fontsize=12)
ax.set_title('Azimuthally-averaged power spectrum', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
out = args.out or os.path.join(args.pred_dir, '..', 'radial_power_spectrum.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Radial power spectrum -> {os.path.abspath(out)}')
