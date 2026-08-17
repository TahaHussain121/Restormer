"""WORK ORDER 1, Task 1.3 -- re-check the DINO block choice at the ACTUAL
training scale (128 radar crop -> 224 DINO, ~8 radar px per token).

WHY. Phase 1 selected B6 at 256 radar -> 224 DINO (~16 radar px per token, the
whole scene in view). Phase-3 training conditions DINO on a 128x128 crop resized
to 224 -- a different magnification AND a different field of view (a crop can be
mostly background). DINOv2 patch statistics are strongly scale-dependent, so the
Phase-1 numbers are not evidence about this regime. This script measures the
same criterion in the regime that will actually be used.

CRITERION (identical to Phase 1): centered same-scene 1e5 <-> 1e7 patch-wise
cosine, `F.cosine_similarity(a, b, dim=1).mean()` over the 256 corresponding
tokens, per block, averaged over the split. The different-scene control is the
Phase-1 one: the PREVIOUS valid sample (a different object), so
same-vs-different advantage = same - different.

WHAT IS HELD FIXED
  * crop 128x128, taken from the full 256x256 image, IDENTICAL coordinates for
    1e5 and 1e7 (spatial correspondence is only meaningful that way)
  * preprocessing = dino_shared.preprocess(crop, 224): repeat to 3ch, bilinear
    align_corners=False, ImageNet norm -- the exact transform E1 will use
  * centering = one [768] vector per (domain, block), measured on the TRAIN
    split at the SAME 128->224 scale (never on val/test). Using the existing
    256-scale means here would reintroduce exactly the distribution mismatch the
    re-check exists to avoid.

TWO CROP PROTOCOLS, reported separately (never mixed):
  random  -- one seeded random 128 window per sample; matches the training
             crop distribution (train.py draws x0,y0 uniformly). PRIMARY.
  center  -- the central 128 window; a fixed-content control that removes
             crop-position variance.

Scope: inference only. No training, no config, no architecture. Writes only
under phase3_restoration/results/wo1_verification/.
"""

import argparse
import csv
import datetime
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, os.path.join(_REPO, 'dino_analysis_phases')):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_dino_spatial_pca as base        # noqa: E402
import dino_shared                                # noqa: E402

BLOCKS1 = [3, 6, 9, 12]
BLOCKS0 = [b - 1 for b in BLOCKS1]
DOMAINS = ['1e5', '1e7']
CROP = 128
DINO_SIZE = dino_shared.DINO_SIZE_TRAIN128        # 224 -> 16x16 tokens


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO).decode().strip()
    except Exception:                              # noqa: BLE001
        return 'unknown'


def crop_coords(protocol, sample_id, img_hw, crop, seed):
    """Deterministic, reproducible, and IDENTICAL for both domains.

    Keyed by sample id (not dataset index) so filesystem ordering cannot change
    the result -- `paired_paths_from_folder` does not sort.
    """
    h, w = img_hw
    if protocol == 'center':
        return (h - crop) // 2, (w - crop) // 2
    if protocol == 'random':
        rng = random.Random(f'{seed}:{sample_id}')
        # mirrors basicsr/train.py:264-265  int((gt_size - mini_gt_size) * rand)
        return (int((h - crop) * rng.random()),
                int((w - crop) * rng.random()))
    raise ValueError(protocol)


def take_crop(img, top, left, crop):
    return img[..., top:top + crop, left:left + crop]


@torch.no_grad()
def tokens_for_sample(ext, batch, top, left, device):
    """-> {domain: {block0: [256,768] float32 cpu}}  for one 128 crop."""
    imgs = []
    for d in DOMAINS:
        x = batch[base.DOMAIN_BATCH_KEY[d]].unsqueeze(0)          # [1,1,256,256]
        imgs.append(take_crop(x, top, left, CROP))
    stack = torch.cat(imgs, dim=0).to(device)                     # [2,1,128,128]
    got = dino_shared.extract_blocks(ext, stack, BLOCKS0, DINO_SIZE)
    return {d: {b0: got[b0][k].float().cpu() for b0 in BLOCKS0}
            for k, d in enumerate(DOMAINS)}


def patchwise_cosine(a, b):
    """Mean cosine over CORRESPONDING tokens -- same definition as Phase 1
    (visualize_dino_spatial_pca.patchwise_cosine), tensors instead of numpy."""
    return float(F.cosine_similarity(a.float(), b.float(), dim=1).mean())


@torch.no_grad()
def compute_train128_means(ext, cfg, n_samples, seed, protocol, device):
    """Per-(domain, block) [768] mean over TRAIN-split 128 crops and over all
    token positions. Accumulated in float64, stored float32. Train only."""
    ds = base.build_dataset(cfg, 'train')
    stems = [os.path.splitext(os.path.basename(p['gt_path']))[0]
             for p in ds.paths]
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(ds)), min(n_samples, len(ds))))
    print(f'  centering means: train pool {len(ds)} images, using {len(idxs)} '
          f'(seed {seed}), crop {CROP} protocol={protocol}, dino {DINO_SIZE}')

    sums = {d: {b: torch.zeros(dino_shared.EMBED_DIM, dtype=torch.float64)
                for b in BLOCKS0} for d in DOMAINS}
    counts = {d: 0 for d in DOMAINS}
    for n, i in enumerate(idxs, 1):
        batch = ds[i]
        sid = stems[i]
        top, left = crop_coords(protocol, sid, batch['lq'].shape[-2:], CROP, seed)
        toks = tokens_for_sample(ext, batch, top, left, device)
        for d in DOMAINS:
            for b0 in BLOCKS0:
                sums[d][b0] += toks[d][b0].double().sum(0)
            counts[d] += toks[d][BLOCKS0[0]].shape[0]
        if n % 25 == 0 or n == len(idxs):
            print(f'    {n}/{len(idxs)} train images')
    means = {d: {b0: (sums[d][b0] / counts[d]).float() for b0 in BLOCKS0}
             for d in DOMAINS}
    meta = {
        'source_split': 'train', 'n_images': len(idxs),
        'tokens_per_image': int(counts[DOMAINS[0]] / len(idxs)),
        'total_tokens_per_domain': {d: counts[d] for d in DOMAINS},
        'crop': CROP, 'crop_protocol': protocol, 'dino_input_size': DINO_SIZE,
        'seed': seed, 'augmentation': 'none (phase=val; the position-averaged '
                                      'mean is invariant to the dihedral augs)',
        'sample_ids': [stems[i] for i in idxs],
        'norms': {d: {f'B{b0 + 1}': float(means[d][b0].norm()) for b0 in BLOCKS0}
                  for d in DOMAINS},
    }
    return means, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', default=os.path.join(
        _REPO, 'Deraining_Holo', 'Options', 'DINO_analysis_data.yml'))
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-mean-samples', type=int, default=150)
    ap.add_argument('--protocols', nargs='+', default=['random', 'center'])
    ap.add_argument('--limit', type=int, default=0, help='debug: first N samples')
    ap.add_argument('--outdir', default=os.path.join(
        _PHASE3, 'results', 'wo1_verification'))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.opt) as f:
        cfg = yaml.safe_load(f)

    ext = dino_shared.build_dino(device=args.device)
    ds = base.build_dataset(cfg, args.split)
    stems = [os.path.splitext(os.path.basename(p['gt_path']))[0]
             for p in ds.paths]
    order = list(range(len(ds)))
    if args.limit:
        order = order[:args.limit]
    print(f'  {args.split} split: {len(ds)} samples, measuring {len(order)}')

    all_rows, summaries, mean_metas = [], [], {}
    for protocol in args.protocols:
        print(f'\n=== crop protocol: {protocol} ===')
        mean_cache = os.path.join(
            args.outdir, f'wo1_means_train128_dino224_{protocol}.pt')
        if os.path.isfile(mean_cache):
            obj = torch.load(mean_cache, map_location='cpu', weights_only=False)
            means, mmeta = obj['means'], obj['meta']
            print(f'  reusing cached means {mean_cache}')
        else:
            means, mmeta = compute_train128_means(
                ext, cfg, args.num_mean_samples, args.seed, protocol, args.device)
            torch.save({'means': means, 'meta': mmeta,
                        'note': 'WO1 VERIFICATION ONLY -- 4-block re-check '
                                'means. NOT the Work-Order-2 production '
                                'centering mean.'}, mean_cache)
            print(f'  wrote {mean_cache}')
        mean_metas[protocol] = mmeta
        for d in DOMAINS:
            print(f'    mean norms {d}: '
                  + '  '.join(f'B{b0 + 1} {float(means[d][b0].norm()):.2f}'
                              for b0 in BLOCKS0))

        prev = None                        # (sample_id, centered/raw 1e7 tokens)
        for n, i in enumerate(order, 1):
            sid = stems[i]
            batch = ds[i]
            top, left = crop_coords(protocol, sid, batch['lq'].shape[-2:],
                                    CROP, args.seed)
            toks = tokens_for_sample(ext, batch, top, left, args.device)
            variants = {
                'raw': toks,
                'centered': {d: {b0: dino_shared.center_tokens(
                    toks[d][b0], means[d][b0]) for b0 in BLOCKS0}
                    for d in DOMAINS},
            }
            for ft in ('raw', 'centered'):
                for b0 in BLOCKS0:
                    same = patchwise_cosine(variants[ft]['1e5'][b0],
                                            variants[ft]['1e7'][b0])
                    diff = ('' if prev is None else
                            patchwise_cosine(variants[ft]['1e5'][b0],
                                             prev[1][ft][b0]))
                    all_rows.append({
                        'crop_protocol': protocol, 'sample_id': sid,
                        'crop_top': top, 'crop_left': left,
                        'feature_type': ft, 'block_1indexed': b0 + 1,
                        'similarity_1e5_vs_1e7': f'{same:.6f}',
                        'control_partner_id': '' if prev is None else prev[0],
                        'control_1e5_vs_1e7': ('' if diff == '' else f'{diff:.6f}'),
                    })
            prev = (sid, {ft: {b0: variants[ft]['1e7'][b0] for b0 in BLOCKS0}
                          for ft in ('raw', 'centered')})
            if n % 25 == 0 or n == len(order):
                print(f'    {n}/{len(order)} {args.split} samples')

    # ---- aggregate straight from the rows we are about to save -------------
    per_sample_csv = os.path.join(args.outdir, 'wo1_layer_recheck_per_sample.csv')
    with open(per_sample_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nwrote {per_sample_csv}')

    rows = list(csv.DictReader(open(per_sample_csv)))
    for protocol in args.protocols:
        for ft in ('raw', 'centered'):
            for b1 in BLOCKS1:
                sel = [r for r in rows if r['crop_protocol'] == protocol
                       and r['feature_type'] == ft
                       and int(r['block_1indexed']) == b1]
                same = np.array([float(r['similarity_1e5_vs_1e7']) for r in sel])
                diff = np.array([float(r['control_1e5_vs_1e7']) for r in sel
                                 if r['control_1e5_vs_1e7'] != ''])
                summaries.append({
                    'crop_protocol': protocol, 'feature_type': ft,
                    'block_1indexed': b1, 'n_same': len(same), 'n_diff': len(diff),
                    'same_mean': float(same.mean()),
                    'same_sem': float(same.std(ddof=1) / np.sqrt(len(same))),
                    'same_std': float(same.std(ddof=1)),
                    'same_median': float(np.median(same)),
                    'diff_mean': float(diff.mean()),
                    'diff_sem': float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    'advantage_mean': float(same.mean() - diff.mean()),
                })
    summary_csv = os.path.join(args.outdir, 'wo1_layer_recheck_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f'wrote {summary_csv}')

    def get(protocol, ft, b1, field):
        for s in summaries:
            if (s['crop_protocol'] == protocol and s['feature_type'] == ft
                    and s['block_1indexed'] == b1):
                return s[field]
        return float('nan')

    for protocol in args.protocols:
        for ft in ('raw', 'centered'):
            print(f'\n{protocol.upper()} / {ft.upper()}  '
                  f'(128 crop -> DINO {DINO_SIZE}, n={len(order)})')
            print(f'  {"":<22}' + ''.join(f'{f"B{b}":>14}' for b in BLOCKS1))
            for label, field in (('same-scene 1e5<->1e7', 'same_mean'),
                                 ('different-scene ctrl', 'diff_mean'),
                                 ('advantage', 'advantage_mean')):
                print(f'  {label:<22}'
                      + ''.join(f'{get(protocol, ft, b, field):>+14.4f}'
                                for b in BLOCKS1))
            best_abs = max(BLOCKS1, key=lambda b: get(protocol, ft, b, 'same_mean'))
            best_adv = max(BLOCKS1,
                           key=lambda b: get(protocol, ft, b, 'advantage_mean'))
            print(f'  winner on absolute correspondence : B{best_abs}')
            print(f'  winner on scene advantage         : B{best_adv}')

    meta_path = os.path.join(args.outdir, 'wo1_layer_recheck_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'task': 'WO1 Task 1.3 -- fixed-128 layer re-check',
            'created': datetime.datetime.now().astimezone().isoformat(),
            'git_commit': git_commit(), 'hostname': os.uname().nodename,
            'device': args.device, 'torch': torch.__version__,
            'split': args.split, 'n_samples': len(order),
            'blocks_1indexed': BLOCKS1, 'blocks_0indexed': BLOCKS0,
            'radar_crop': CROP, 'dino_input_size': DINO_SIZE,
            'tokens': (DINO_SIZE // dino_shared.PATCH_SIZE) ** 2,
            'radar_px_per_token': CROP / (DINO_SIZE // dino_shared.PATCH_SIZE),
            'metric': 'patchwise cosine over corresponding tokens, mean',
            'control': 'previous valid sample (different object), Phase-1 rule',
            'centering': 'per-domain, per-block [768], train split, 128->224',
            'centering_means': mean_metas,
            'seed': args.seed,
            'summary': summaries,
        }, f, indent=2)
    print(f'wrote {meta_path}')


if __name__ == '__main__':
    main()
