"""Dump the AFFM per-position softmax weight MAPS for a fixed val batch.

The training log carries only the position-AVERAGED weights. That answers "which
layer does the network prefer" but not "is the preference spatially structured
or flat", and those are different findings: a w_b6 of 0.6 could be 0.6
everywhere, or 0.95 on the object and 0.2 on the background. This script
recovers the second.

RUN IT AFTER TRAINING, on the checkpoints named in the devlog (5k, 100k, 300k).
It is inference-only: checkpoints are opened read-only, nothing is written
outside the chosen --out directory, and no experiment directory is touched.

The batch is FIXED and read from the matched-128 crop manifest, so the maps from
different checkpoints are directly comparable position by position.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('TORCH_HOME', os.path.join(_REPO, 'torch_hub'))

from basicsr.models.archs import define_network                  # noqa: E402

DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
DEFAULT_CFG = os.path.join(_PHASE3, 'configs',
                           'affm_render_fixed128_spatial_L3691_latent.yml')
DEFAULT_MANIFEST = os.path.join(_PHASE3, 'results', 'crop_manifests',
                                'matched128_val.csv')


def read_manifest(path, n):
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    return [(r['image_id'], int(r['x']), int(r['y']), int(r['size']))
            for r in csv.DictReader(lines)][:n]


def load_batch(split, rows):
    radar, render = [], []
    for image_id, x, y, size in rows:
        lq = cv2.imread(os.path.join(DATASET, f'{split}_verynoisy',
                                     f'{image_id}.png'), cv2.IMREAD_UNCHANGED)
        rn = cv2.imread(os.path.join(DATASET, f'{split}_renders_blackbg',
                                     f'{image_id}.png'), cv2.IMREAD_UNCHANGED)
        rn = rn[:, :, 0] if rn.ndim == 3 else rn
        radar.append(lq[y:y + size, x:x + size].astype(np.float32) / 65535.)
        render.append(rn[y:y + size, x:x + size].astype(np.float32) / 255.)
    return torch.from_numpy(np.stack([np.stack(radar), np.stack(render)], 1))


@torch.no_grad()
def weights_for(cfg_path, weights_path, batch, device):
    cfg = yaml.safe_load(open(cfg_path))
    net = define_network(dict(cfg['network_g']))
    ck = torch.load(weights_path, map_location='cpu', weights_only=False)
    net.load_state_dict(ck.get('params', ck), strict=True)
    net = net.to(device).eval()
    net.set_dino_mode('train128')
    net._capture_dino_io = True                 # forces the weight capture
    net(batch.to(device))
    w = net._dino_capture['affm_weights'].detach().cpu()
    layers = list(net.dino_layers_1indexed)
    del net
    return w, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CFG)
    ap.add_argument('--experiment',
                    default='Holo_affm_render_fixed128_spatial_L3691_latent')
    ap.add_argument('--iters', type=int, nargs='+', default=[5000, 100000, 300000])
    ap.add_argument('--n-images', type=int, default=4)
    ap.add_argument('--split', default='val')
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(_PHASE3, 'results', args.experiment,
                                   'affm_weight_maps')
    os.makedirs(out, exist_ok=True)
    rows = read_manifest(args.manifest, args.n_images)
    batch = load_batch(args.split, rows)
    print(f'fixed batch: {", ".join(r[0] for r in rows)}  {tuple(batch.shape)}')

    have, record = [], {}
    for it in args.iters:
        wpath = os.path.join(_REPO, 'experiments', args.experiment, 'models',
                             f'net_g_{it}.pth')
        if not os.path.isfile(wpath):
            print(f'  (no checkpoint at {it}: {wpath}) -- skipped')
            continue
        w, layers = weights_for(args.config, wpath, batch, args.device)
        err = float((w.sum(1) - 1.0).abs().max())
        if err > 1e-4:
            raise SystemExit(f'iter {it}: weights do not sum to 1 ({err:.3e})')
        have.append((it, w, layers))
        record[str(it)] = {
            'mean_per_layer': {f'B{b}': float(w[:, i].mean())
                               for i, b in enumerate(layers)},
            'std_over_positions_per_layer': {f'B{b}': float(w[:, i].std())
                                             for i, b in enumerate(layers)},
            'max_abs_sum_error': err}
        print(f'  iter {it:>7}: ' + '  '.join(
            f'B{b} {float(w[:, i].mean()):.4f}+-{float(w[:, i].std()):.4f}'
            for i, b in enumerate(layers)))

    if not have:
        raise SystemExit('no checkpoints found -- run this after training')

    layers = have[0][2]
    fig, axes = plt.subplots(len(have), len(layers) + 1,
                             figsize=(3.5 * (len(layers) + 1), 3.4 * len(have)),
                             squeeze=False)
    for r, (it, w, _) in enumerate(have):
        axes[r][0].imshow(batch[0, 1].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[r][0].set_ylabel(f'iter {it:,}', fontsize=13, fontweight='bold')
        axes[r][0].set_title('render (image 0)' if r == 0 else '', fontsize=12)
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for c, b in enumerate(layers, start=1):
            im = axes[r][c].imshow(w[0, c - 1].numpy(), cmap='viridis',
                                   vmin=0, vmax=1)
            axes[r][c].set_title(
                (f'w_B{b}\n' if r == 0 else '')
                + f'mean {float(w[:, c - 1].mean()):.3f}', fontsize=12)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            plt.colorbar(im, ax=axes[r][c], fraction=0.046)
    fig.suptitle('AFFM per-position layer weights over training\n'
                 'flat colour = the choice is global; structure = the network '
                 'picks different depths in different places',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, 'affm_weight_maps.png')
    fig.savefig(p, dpi=125)
    plt.close(fig)

    with open(os.path.join(out, 'affm_weight_maps.json'), 'w') as f:
        json.dump({'experiment': args.experiment, 'layers_1indexed': layers,
                   'images': [r[0] for r in rows], 'split': args.split,
                   'manifest': args.manifest, 'per_iteration': record}, f,
                  indent=2)
    print('wrote', p)
    print('wrote', os.path.join(out, 'affm_weight_maps.json'))


if __name__ == '__main__':
    main()
