import numpy as np
import os
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer

parser = argparse.ArgumentParser(description='Holo radar heatmap denoising with Restormer')
parser.add_argument('--input_dir', required=True, type=str, help='Directory of noisy input images (uint16 PNG)')
parser.add_argument('--result_dir', default='./results/Holo/', type=str, help='Directory to save restored images')
parser.add_argument('--weights', default='../experiments/Holo_Baseline_Restormer/models/net_g_latest.pth', type=str, help='Path to model weights')
parser.add_argument('--gt_dir', default=None, type=str, help='(Optional) Directory of clean GT images for PSNR/SSIM evaluation')
parser.add_argument('--cmap', default='inferno', type=str, help='Matplotlib colormap for heatmap (default: inferno)')
args = parser.parse_args()

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

yaml_file = 'Options/Holo_Baseline_Restormer.yml'
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
net_cfg = {k: v for k, v in x['network_g'].items() if k != 'type'}

model = Restormer(**net_cfg)
checkpoint = torch.load(args.weights, map_location='cpu')
state_dict = checkpoint.get('params', checkpoint)
model.load_state_dict(state_dict)
print(f"Loaded weights: {args.weights}")
model.cuda()
model = nn.DataParallel(model)
model.eval()

raw_dir  = os.path.join(args.result_dir, 'raw')        # uint16 PNGs
viz_dir  = os.path.join(args.result_dir, 'viz')        # heatmap grids
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

def load_uint16(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Failed to read {path}')
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img  # (H, W, 1) uint16

def psnr_uint16(gt, pred):
    gt_f = gt.astype(np.float64)
    pred_f = pred.astype(np.float64)
    mse = np.mean((gt_f - pred_f) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(65535.0 / math.sqrt(mse))

# ---- sharpness / blur metrics (operate on 2D float arrays in [0, 1]) ---------
def lap_var(x):
    """Variance of the Laplacian — fine-detail sharpness. Lower = blurrier."""
    return cv2.Laplacian(x.astype(np.float64), cv2.CV_64F).var()

def grad_mag(x):
    """Mean Sobel gradient magnitude — edge strength. Lower = blurrier."""
    xd = x.astype(np.float64)
    gx = cv2.Sobel(xd, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(xd, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

def hf_energy(x):
    """Fraction of FFT power beyond 1/4 of the Nyquist radius. Lower = blurrier."""
    F2 = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
    P = np.abs(F2) ** 2
    h, w = x.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    return float(P[r > 0.25 * r.max()].sum() / P.sum())

def radial_power(x, nbins=128):
    """Azimuthally-averaged power spectrum: returns (freq_fraction, power[nbins])."""
    F2 = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
    P = np.abs(F2) ** 2
    h, w = x.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    rmax = r.max()
    bin_idx = np.clip((r / rmax * (nbins - 1)).astype(int), 0, nbins - 1)
    radial = np.bincount(bin_idx.ravel(), weights=P.ravel(), minlength=nbins)
    counts = np.bincount(bin_idx.ravel(), minlength=nbins)
    radial = radial / np.maximum(counts, 1)
    freq = np.linspace(0, 1, nbins)  # fraction of Nyquist
    return freq, radial

def save_heatmap_grid(noisy, prediction, gt, save_path, cmap, psnr=None, psnr_noisy=None):
    """
    noisy, prediction, gt: 2D float32 arrays normalised to [0, 1].
    gt may be None if no GT is provided.
    """
    has_gt = gt is not None
    ncols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    panels = [('Noisy Input', noisy), ('Prediction', prediction)]
    if has_gt:
        panels.append(('Ground Truth', gt))

    # use a shared colour scale across all panels so they are comparable
    vmin = 0.0
    vmax = 1.0

    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        if title == 'Prediction' and psnr is not None:
            ax.set_title(f'{title}\nPSNR: {psnr:.2f} dB', fontsize=13, fontweight='bold')
        elif title == 'Noisy Input' and psnr_noisy is not None:
            ax.set_title(f'{title}\nPSNR: {psnr_noisy:.2f} dB', fontsize=13, fontweight='bold')
        else:
            ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

factor = 8
files = natsorted(glob(os.path.join(args.input_dir, '*.png')))
if not files:
    raise RuntimeError(f'No PNG files found in {args.input_dir}')

psnr_list = []
ssim_list = []
psnr_noisy_list = []
ssim_noisy_list = []

# sharpness accumulators (only populated when GT is available)
sharp = {k: {'noisy': [], 'pred': [], 'gt': []}
         for k in ('lap', 'grad', 'hf')}
NBINS = 128
radial_acc = {'noisy': np.zeros(NBINS), 'pred': np.zeros(NBINS), 'gt': np.zeros(NBINS)}
radial_freq = np.linspace(0, 1, NBINS)
radial_n = 0

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.empty_cache()

        img_raw = load_uint16(file_)                        # (H, W, 1) uint16
        img_f   = img_raw.astype(np.float32) / 65535.      # [0, 1]
        inp = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).cuda()

        h, w = inp.shape[2], inp.shape[3]
        H = ((h + factor - 1) // factor) * factor
        W = ((w + factor - 1) // factor) * factor
        inp_pad = F.pad(inp, (0, W - w, 0, H - h), 'reflect')

        out = model(inp_pad)
        out = out[:, :, :h, :w]
        out_f = torch.clamp(out, 0, 1).cpu().squeeze().numpy()  # (H, W) float32

        out_uint16 = (out_f * 65535.0).round().astype(np.uint16)
        basename   = os.path.splitext(os.path.basename(file_))[0]

        # save raw uint16
        cv2.imwrite(os.path.join(raw_dir, basename + '.png'), out_uint16)

        # load GT if available
        gt_f = None
        psnr = None
        psnr_noisy = None
        if args.gt_dir is not None:
            gt_path = os.path.join(args.gt_dir, os.path.basename(file_))
            if os.path.exists(gt_path):
                gt_raw = load_uint16(gt_path).squeeze()
                gt_f   = gt_raw.astype(np.float32) / 65535.

                # prediction vs GT
                psnr   = psnr_uint16(gt_raw, out_uint16)
                psnr_list.append(psnr)
                s = ssim_fn(gt_f, out_f, data_range=1.0)
                ssim_list.append(s)

                # noisy input vs GT (baseline)
                psnr_noisy = psnr_uint16(gt_raw, img_raw.squeeze())
                psnr_noisy_list.append(psnr_noisy)
                s_noisy = ssim_fn(gt_f, img_f.squeeze(), data_range=1.0)
                ssim_noisy_list.append(s_noisy)

                # sharpness / blur metrics
                noisy_2d = img_f.squeeze()
                for name, fn in (('lap', lap_var), ('grad', grad_mag), ('hf', hf_energy)):
                    sharp[name]['noisy'].append(fn(noisy_2d))
                    sharp[name]['pred'].append(fn(out_f))
                    sharp[name]['gt'].append(fn(gt_f))

                # radial power spectrum (averaged across images)
                for key, arr in (('noisy', noisy_2d), ('pred', out_f), ('gt', gt_f)):
                    _, rp = radial_power(arr, NBINS)
                    radial_acc[key] += rp
                radial_n += 1

        # save heatmap grid
        save_heatmap_grid(
            noisy      = img_f.squeeze(),
            prediction = out_f,
            gt         = gt_f,
            save_path  = os.path.join(viz_dir, basename + '_viz.png'),
            cmap       = args.cmap,
            psnr       = psnr,
            psnr_noisy = psnr_noisy,
        )

if psnr_list:
    mean_psnr_noisy = np.mean(psnr_noisy_list)
    mean_psnr       = np.mean(psnr_list)
    mean_ssim_noisy = np.mean(ssim_noisy_list)
    mean_ssim       = np.mean(ssim_list)
    print(f'\n{"Metric":<8}{"Noisy":>12}{"Denoised":>12}{"Improvement":>14}')
    print(f'{"PSNR":<8}{mean_psnr_noisy:>9.4f} dB{mean_psnr:>9.4f} dB{mean_psnr - mean_psnr_noisy:>+11.4f} dB')
    print(f'{"SSIM":<8}{mean_ssim_noisy:>12.4f}{mean_ssim:>12.4f}{mean_ssim - mean_ssim_noisy:>+14.4f}')

    # ---- sharpness / blur report (Pred/GT < 1 => prediction is blurrier) -----
    sharp_names = {'lap': 'Laplacian var', 'grad': 'Sobel grad', 'hf': 'HF energy frac'}
    print(f'\n{"Sharpness":<16}{"Noisy":>12}{"Pred":>12}{"GT":>12}{"Pred/GT":>10}')
    for k in ('lap', 'grad', 'hf'):
        pn, pp, pg = np.mean(sharp[k]['noisy']), np.mean(sharp[k]['pred']), np.mean(sharp[k]['gt'])
        print(f'{sharp_names[k]:<16}{pn:>12.5f}{pp:>12.5f}{pg:>12.5f}{pp / pg:>10.3f}')
    print('(Pred/GT < 1.0 => prediction has less high-frequency detail than GT, i.e. blurrier)')

    # ---- radial power spectrum plot ------------------------------------------
    if radial_n > 0:
        for key in radial_acc:
            radial_acc[key] /= radial_n
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, color in (('noisy', 'tab:orange'), ('pred', 'tab:blue'), ('gt', 'tab:green')):
            ax.plot(radial_freq, radial_acc[key] + 1e-20, label=key.capitalize(), color=color, lw=2)
        ax.set_yscale('log')
        ax.set_xlabel('Spatial frequency (fraction of Nyquist)', fontsize=12)
        ax.set_ylabel('Mean radial power (log scale)', fontsize=12)
        ax.set_title('Azimuthally-averaged power spectrum', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        spec_path = os.path.join(args.result_dir, 'radial_power_spectrum.png')
        plt.savefig(spec_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Radial power spectrum -> {spec_path}')

print(f'Raw uint16 results  -> {raw_dir}')
print(f'Heatmap grids       -> {viz_dir}')
