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

def save_heatmap_grid(noisy, prediction, gt, save_path, cmap, psnr=None):
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
        if args.gt_dir is not None:
            gt_path = os.path.join(args.gt_dir, os.path.basename(file_))
            if os.path.exists(gt_path):
                gt_raw = load_uint16(gt_path).squeeze()
                gt_f   = gt_raw.astype(np.float32) / 65535.
                psnr   = psnr_uint16(gt_raw, out_uint16)
                psnr_list.append(psnr)
                s = ssim_fn(gt_raw.astype(np.float32) / 65535.,
                            out_f, data_range=1.0)
                ssim_list.append(s)

        # save heatmap grid
        save_heatmap_grid(
            noisy      = img_f.squeeze(),
            prediction = out_f,
            gt         = gt_f,
            save_path  = os.path.join(viz_dir, basename + '_viz.png'),
            cmap       = args.cmap,
            psnr       = psnr,
        )

if psnr_list:
    print(f'\nMean PSNR : {np.mean(psnr_list):.4f} dB')
    print(f'Mean SSIM : {np.mean(ssim_list):.4f}')
print(f'Raw uint16 results  -> {raw_dir}')
print(f'Heatmap grids       -> {viz_dir}')
