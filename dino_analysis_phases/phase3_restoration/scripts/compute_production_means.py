"""Phase-3 PRODUCTION centering means for the 1e5 domain, DINOv2 B6.

Two files, two regimes, both measured on the TRAIN split only:

  1e5_B6_train128_dino224_mean.pt   1e5 train, random fixed-128 crop -> DINO 224
                                    -> used by E1 TRAINING and matched-128 eval
  1e5_B6_eval256_dino448_mean.pt    1e5 train, full 256 image      -> DINO 448
                                    -> used by full-256 val/test

mu = average of the raw B6 patch tokens over sampled images AND over all token
positions -> one position-independent [768] vector. Accumulated in float64,
saved float32. No per-position mean, no L2 normalisation, no extra LayerNorm --
DINO's own norm=True is the only normalisation, exactly as in Phase 1/2.

WHY 1000 SAMPLES. Work Order 1's re-check used Phase 1's 150 images (38,400
tokens). The production means use >=1000, i.e. ~256,000 tokens at 128 and
~1,024,000 at 256 -- a ~6.7x larger image sample and, because mu is what every
token is measured against for 300k iterations, the cheapest possible place to
buy precision. 1000 of the 6101 training images is ~16% of the split, drawn
once with a fixed seed and recorded by ID in the metadata.

AUGMENTATION. None. mu averages over all token positions, so it is invariant to
the dataset's dihedral augmentations (a flip/rotation permutes token positions
without changing the position-averaged mean). What does matter is the CROP
distribution, and that is matched exactly: x0, y0 are drawn uniformly the way
basicsr/train.py:264-265 draws them.

NEVER computed from val or test data. The train split is asserted, not assumed.
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, os.path.join(_REPO, 'dino_analysis_phases'), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_dino_spatial_pca as base            # noqa: E402
import dino_shared                                   # noqa: E402

BLOCK_1IDX = 6
BLOCK_0IDX = dino_shared.b1_to_b0(BLOCK_1IDX)

# Which stream of the triplet dataset each domain reads. The render has its own
# means because its DINO statistics differ from the noisy heatmap's (clean
# geometry on a black background vs a noisy radar field) -- reusing the 1e5 mean
# for the render arm would centre it against the wrong distribution.
DOMAINS = {'1e5': 'lq', 'render': 'dino'}

REGIMES = {
    'train128': dict(radar=128, dino=dino_shared.DINO_SIZE_TRAIN128,
                     crop='random uniform 128x128 window, drawn as '
                          'basicsr/train.py:264-265 draws it',
                     suffix='train128_dino224'),
    'eval256': dict(radar=256, dino=dino_shared.DINO_SIZE_EVAL256,
                    crop='none -- the full 256x256 image',
                    suffix='eval256_dino448'),
}


def mean_filename(domain, regime):
    return f'{domain}_B{BLOCK_1IDX}_{REGIMES[regime]["suffix"]}_mean.pt'


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=_REPO).decode().strip()
    except Exception:                                 # noqa: BLE001
        return 'unknown'


@torch.no_grad()
def compute(ext, ds, stems, idxs, regime, seed, device, domain_key):
    spec = REGIMES[regime]
    radar, dino_size = spec['radar'], spec['dino']
    acc = torch.zeros(dino_shared.EMBED_DIM, dtype=torch.float64)
    n_tokens = 0
    for n, i in enumerate(idxs, 1):
        img = ds[i][domain_key].unsqueeze(0)      # [1,C,256,256] in [0,1]
        if radar != img.shape[-1]:
            rng = random.Random(f'{seed}:{stems[i]}:{regime}')
            h, w = img.shape[-2:]
            top = int((h - radar) * rng.random())
            left = int((w - radar) * rng.random())
            img = img[..., top:top + radar, left:left + radar]
        tok = dino_shared.extract_block(ext, img.to(device), BLOCK_0IDX, dino_size)
        acc += tok[0].double().sum(0).cpu()
        n_tokens += tok.shape[1]
        if n % 100 == 0 or n == len(idxs):
            print(f'    {regime}: {n}/{len(idxs)} images, {n_tokens} tokens',
                  flush=True)
    return (acc / n_tokens).float(), n_tokens


@torch.no_grad()
def residual_report(ext, ds, stems, idxs, regime, mu, seed, device, domain_key,
                    k=5):
    """Sanity: how big is what is LEFT after centering, on training samples."""
    spec = REGIMES[regime]
    out = []
    for i in idxs[:k]:
        img = ds[i][domain_key].unsqueeze(0)
        if spec['radar'] != img.shape[-1]:
            rng = random.Random(f'{seed}:{stems[i]}:{regime}')
            h, w = img.shape[-2:]
            top = int((h - spec['radar']) * rng.random())
            left = int((w - spec['radar']) * rng.random())
            img = img[..., top:top + spec['radar'], left:left + spec['radar']]
        tok = dino_shared.extract_block(ext, img.to(device), BLOCK_0IDX,
                                        spec['dino'])[0].float().cpu()
        cen = tok - mu
        out.append({
            'sample_id': stems[i],
            'raw_token_norm_mean': float(tok.norm(dim=1).mean()),
            'centered_token_norm_mean': float(cen.norm(dim=1).mean()),
            'residual_fraction': float(cen.norm() / tok.norm()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', default=os.path.join(
        _REPO, 'Deraining_Holo', 'Options', 'DINO_analysis_data.yml'))
    ap.add_argument('--regime', required=True, choices=sorted(REGIMES))
    ap.add_argument('--domain', default='1e5', choices=sorted(DOMAINS),
                    help='which stream DINO reads: 1e5 (E1-addition) or render '
                         '(E1-render). Never the 1e7 target.')
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default=os.path.join(_PHASE3, 'means'))
    ap.add_argument('--force', action='store_true',
                    help='allow overwriting an existing production mean')
    args = ap.parse_args()

    spec = REGIMES[args.regime]
    domain_key = DOMAINS[args.domain]
    out_path = os.path.join(args.outdir, mean_filename(args.domain, args.regime))
    os.makedirs(args.outdir, exist_ok=True)
    if os.path.isfile(out_path) and not args.force:
        raise SystemExit(f'REFUSING to overwrite {out_path} (pass --force). '
                         f'A production mean is part of an experiment identity.')

    with open(args.opt) as f:
        cfg = yaml.safe_load(f)

    ext = dino_shared.build_dino(device=args.device)
    ds = base.build_dataset(cfg, 'train')
    # train-only provenance, asserted rather than assumed
    for key in ('dataroot_lq', 'dataroot_gt', 'dataroot_render'):
        root = ds.opt[key]
        if '/train_' not in root:
            raise SystemExit(f'{key} is {root!r} -- not the train split')
        if '/val_' in root or '/test_' in root:
            raise SystemExit(f'{key} leaks a non-train split: {root!r}')
    root = ds.opt['dataroot_lq'] if args.domain == '1e5' else ds.opt['dataroot_render']
    print(f'  train pool: {len(ds)} images, domain {args.domain} from {root}')

    stems = [os.path.splitext(os.path.basename(p['gt_path']))[0]
             for p in ds.paths]
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.n_samples, len(ds))))
    print(f'  sampling {len(idxs)} images (seed {args.seed}), regime '
          f'{args.regime} [{args.domain}]: input {spec["radar"]} -> DINO {spec["dino"]} -> '
          f'{(spec["dino"] // dino_shared.PATCH_SIZE) ** 2} tokens/image')

    mu, n_tokens = compute(ext, ds, stems, idxs, args.regime, args.seed,
                           args.device, domain_key)
    if mu.shape != (dino_shared.EMBED_DIM,):
        raise SystemExit(f'mean has shape {tuple(mu.shape)}')
    if not torch.isfinite(mu).all():
        raise SystemExit('mean contains non-finite entries')

    resid = residual_report(ext, ds, stems, idxs, args.regime, mu, args.seed,
                            args.device, domain_key)

    meta = {
        'domain': args.domain,
        'domain_stream': ('LQ (verynoisy / 1e5) -- the network input'
                          if args.domain == '1e5' else
                          'render (renders_blackbg) -- DINO input only, never a '
                          'Restormer input and never a target') +
                         '; the 1e7 target is NEVER used for centering statistics',
        'source_split': 'train',
        'train_only': True,
        'dataroot': ds.opt['dataroot_lq'] if args.domain == '1e5'
                    else ds.opt['dataroot_render'],
        'dino_model': 'dinov2_vitb14',
        'block_1indexed': BLOCK_1IDX, 'block_0indexed': BLOCK_0IDX,
        'radar_input_size': spec['radar'],
        'dino_input_size': spec['dino'],
        'grid': spec['dino'] // dino_shared.PATCH_SIZE,
        'radar_px_per_token': spec['radar'] / (spec['dino'] // dino_shared.PATCH_SIZE),
        'n_images': len(idxs),
        'n_tokens': n_tokens,
        'tokens_per_image': n_tokens // len(idxs),
        'seed': args.seed,
        'crop_method': spec['crop'],
        'augmentation': 'none -- mu averages over all token positions and is '
                        'therefore invariant to the dataset dihedral augs',
        'preprocessing': 'grey->3ch repeat; bilinear resize align_corners=False; '
                         'ImageNet mean (0.485,0.456,0.406) std (0.229,0.224,0.225); '
                         'get_intermediate_layers(norm=True, '
                         'return_class_token=False), patch tokens only',
        'extractor': 'dino_analysis_phases/phase3_restoration/scripts/dino_shared.py',
        'accumulation_dtype': 'float64', 'stored_dtype': 'float32',
        'shape': list(mu.shape),
        'norm': float(mu.norm()),
        'created': datetime.datetime.now().astimezone().isoformat(),
        'git_commit': git_commit(),
        'hostname': os.uname().nodename,
        'device': args.device,
        'torch': torch.__version__,
        'sample_ids': [stems[i] for i in idxs],
        'residual_report': resid,
    }
    torch.save({'mean': mu, 'meta': meta}, out_path)

    print(f'\n  SANITY  {mean_filename(args.domain, args.regime)}')
    print(f'    shape            {tuple(mu.shape)}')
    print(f'    all finite       {bool(torch.isfinite(mu).all())}')
    print(f'    norm             {float(mu.norm()):.4f}')
    print(f'    images / tokens  {len(idxs)} / {n_tokens}')
    print(f'    block            B{BLOCK_1IDX} (index {BLOCK_0IDX})')
    print(f'    split            train only  ({root})')
    print(f'    residual norms on {len(resid)} training samples:')
    for r in resid:
        print(f'      {r["sample_id"]}: raw {r["raw_token_norm_mean"]:.3f}  '
              f'centered {r["centered_token_norm_mean"]:.3f}  '
              f'residual fraction {r["residual_fraction"]:.4f}')
    print(f'  wrote {out_path}')

    side = os.path.splitext(out_path)[0] + '_meta.json'
    with open(side, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  wrote {side}')


if __name__ == '__main__':
    main()
