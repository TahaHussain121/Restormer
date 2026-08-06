"""Precompute the per-arm pooled-DINO mean vector used to CENTER the FiLM input.

Why (see design.md "Feature-separation analysis"): the pooled DINO vector is
~95% a shared constant offset, and the measured object signal lives entirely in
the residual -- raw features are object-blind (render<->radar d~0.03, n.s.),
centered ones are not (d=+0.25, +8.6 sigma). So the FiLM head is fed
`pooled - mean` instead of `pooled`.

A FIXED precomputed mean, not BatchNorm: the progressive schedule drops the
batch to 2 at 256px, and a two-sample mean is noise, not a mean.

The mean is computed on TRAINING crops drawn through the arm's OWN dataset
class, so the crops are produced by the exact code path training uses (same
random crop, same geometric augs, same loader / value range). Crop sizes are
sampled in proportion to the progressive schedule's iteration counts
(92k/64k/48k/96k at 128/160/192/256 -> 92/64/48/96 crops of 300), so the mean is
matched to the crop-size distribution the run will actually see rather than to
any single stage.

Which image the mean is taken over is decided by the arm, read from the config:
  renderDINO (Dataset_PairedImage_uint16_Render) -> the 'dino' render crop
  lqDINO     (Dataset_PairedImage_uint16)        -> the noisy 'lq' crop

Usage (offline, CPU is fine; a few minutes):
    python Deraining_Holo/compute_dino_feat_mean.py \
        --opt Deraining_Holo/Options/Holo_DINOv2_renderDINO_Restormer.yml \
        --out Deraining_Holo/experiment_results/exp3_dino_film/dino_feat_mean_renderDINO.pt

NOTE the config it reads must NOT yet be centered for this to be meaningful --
`dino_feat_mean` in the yml is ignored here on purpose (the extractor is always
built uncentered), so re-running after the yml is wired up still recomputes the
raw mean.
"""
import argparse
import copy
import datetime
import os
import random

import numpy as np
import torch
import yaml

from basicsr.data.paired_image_uint16_dataset import Dataset_PairedImage_uint16
from basicsr.data.paired_image_uint16_render_dataset import (
    Dataset_PairedImage_uint16_Render)
from basicsr.models.archs.restormer_dino_arch import DINOv2Extractor

DATASETS = {
    'Dataset_PairedImage_uint16': (Dataset_PairedImage_uint16, 'lq'),
    'Dataset_PairedImage_uint16_Render': (Dataset_PairedImage_uint16_Render, 'dino'),
}


def build_dataset(train_opt, cls, gt_size):
    """One dataset instance pinned to a single crop size."""
    opt = copy.deepcopy(train_opt)
    opt['phase'] = 'train'
    opt['scale'] = 1
    opt['gt_size'] = int(gt_size)
    return cls(opt)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True, help='arm config yml')
    ap.add_argument('--out', required=True, help='destination .pt')
    ap.add_argument('--n_crops', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    with open(args.opt) as f:
        cfg = yaml.safe_load(f)
    train_opt = cfg['datasets']['train']
    net = cfg['network_g']

    ds_type = train_opt['type']
    if ds_type not in DATASETS:
        raise ValueError(f'unsupported dataset type {ds_type!r}')
    cls, key = DATASETS[ds_type]
    arm = 'renderDINO' if key == 'dino' else 'lqDINO'
    print(f'arm            : {arm}  (config {os.path.basename(args.opt)})')
    print(f'DINO sees      : batch[{key!r}]  via {ds_type}')

    # crop sizes in proportion to the progressive schedule's iteration counts
    sizes = list(train_opt['gt_sizes'])
    iters = [int(i) for i in train_opt['iters']]
    total_it = sum(iters)
    counts = [max(1, round(args.n_crops * it / total_it)) for it in iters]
    print(f'crop schedule  : sizes {sizes} iters {iters}')
    print(f'crops per size : {counts}  (total {sum(counts)})')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # UNCENTERED extractor on purpose -- we are measuring the mean it needs.
    ext = DINOv2Extractor(
        layers=tuple(net['dino_layers']), img_size=net['dino_img_size'],
        model_name=net['dino_model_name'], hub_source=net['dino_hub_source'],
        hub_dir=net['dino_hub_dir'], weights=net['dino_weights'],
        feat_mean=None).to(args.device).eval()
    assert not ext.centered
    print(f'extractor      : {net["dino_model_name"]}  feat_dim {ext.feat_dim}'
          f'  layers {ext.layers}  device {args.device}')

    feats = []
    for size, n in zip(sizes, counts):
        ds = build_dataset(train_opt, cls, size)
        idx = random.sample(range(len(ds)), n) if n <= len(ds) else \
            [random.randrange(len(ds)) for _ in range(n)]
        for s in range(0, n, args.batch):
            imgs = torch.stack([ds[i][key] for i in idx[s:s + args.batch]])
            feats.append(ext(imgs.to(args.device)).cpu())
        print(f'  size {size:3d}: {n:3d} crops done')

    F = torch.cat(feats, 0)                      # [N, feat_dim]
    mu = F.mean(0)                               # [feat_dim]
    resid = F - mu
    frac = (mu.norm() ** 2 / (F.norm(dim=1) ** 2).mean()).item()
    print(f'\nN              : {F.shape[0]} crops, dim {F.shape[1]}')
    print(f'||mean||       : {mu.norm().item():.3f}')
    print(f'mean||residual||: {resid.norm(dim=1).mean().item():.3f}')
    print(f'offset energy  : {100 * frac:.1f}% of mean ||feature||^2')

    meta = dict(arm=arm, config=os.path.abspath(args.opt), dataset_type=ds_type,
                image_key=key, n_crops=int(F.shape[0]), sizes=sizes,
                counts=counts, seed=args.seed,
                dino_model=net['dino_model_name'],
                dino_layers=list(net['dino_layers']),
                dino_img_size=net['dino_img_size'],
                dino_weights=net['dino_weights'],
                geometric_augs=bool(train_opt.get('geometric_augs', False)),
                dataroot=train_opt.get('dataroot_render' if key == 'dino' else 'dataroot_lq'),
                mean_norm=float(mu.norm()),
                resid_norm=float(resid.norm(dim=1).mean()),
                offset_energy_frac=float(frac),
                created=datetime.datetime.now().isoformat(timespec='seconds'),
                script='Deraining_Holo/compute_dino_feat_mean.py')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'mean': mu, 'meta': meta}, args.out)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
