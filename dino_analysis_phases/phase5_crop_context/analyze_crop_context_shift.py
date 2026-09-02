"""PHASE 5 — does DINO see the same thing in a CROP as in the FULL image?

THE QUESTION, AND WHY IT IS NOT ACADEMIC

Phase 3 trains on 128 crops (DINO at 224 -> 16x16 tokens) and evaluates on full
256 frames (DINO at 448 -> 32x32 tokens). Every arm therefore receives a prior
computed in ONE context during training and a DIFFERENT context at evaluation.
Nobody has checked whether DINO produces the same features for the same physical
patch under those two conditions.

This matters because Phase 3 found that every attention arm GAINS on full256 and
LOSES on crop128 (Finding 7), and the mechanism recorded for that is explicitly
labelled interpretation rather than result. There are two candidate causes:

  (a) the FUSION differs across protocols (channel statistics pooled over 4x the
      tokens), or
  (b) the PRIOR ITSELF differs across protocols.

(b) has never been measured. This script measures it.

WHAT DSGIR CLAIMS (Neurocomputing 696; PAYWALLED, abstract only, NOT verified)

    "under the local-crop setting used in restoration training, degraded patches
     undergo a noticeable semantic shift relative to their clean counterparts"
    "the semantic shift is particularly pronounced at deep semantic levels, where
     the absence of global contextual support makes degraded local patches more
     vulnerable to degradation interference"

That is a claim about an INTERACTION, not about cropping alone: degradation is
supposed to hurt MORE inside a crop than inside a full frame, and worst at deep
layers. So one axis is not enough. This script measures both.

  CONTEXT axis      full frame  vs  128 crop        (new)
  DEGRADATION axis  1e5 / render vs 1e7            (Phase 1 measured this at one
                                                    scale; here at both)
  INTERACTION       deg_crop - deg_full            <- DSGIR's actual claim

DSGIR is unverified. Nothing here depends on it being right; it supplies a
prediction to test, and either outcome is reportable.

ALIGNMENT

A 256 frame gives a 32x32 token grid, so ONE TOKEN COVERS 8x8 IMAGE PIXELS. A
128 crop at pixel (x, y) therefore corresponds EXACTLY to the 16x16 sub-block of
the full grid at token (x/8, y/8) -- but only if x and y are multiples of 8.

The Phase-3 matched-128 manifest does NOT satisfy this: only 5 of its 339 crops
have x and y both divisible by 8. This script therefore draws its OWN manifest
with snapped coordinates. It is an ANALYSIS manifest and is deliberately
separate from the evaluation manifest, which is not touched.

METRIC

Mean corresponding-position cosine similarity on the FULL 768-d features, never
on a reduced representation -- the same primary metric Phase 1 used, so the two
phases stay comparable. For each of the 256 positions p, cos(A[p], B[p]),
averaged over p.

A FLOOR IS INCLUDED. Cosine between high-dimensional DINO features is not
centred at zero, so a "high" number means nothing without a reference. The floor
is the same comparison against a DIFFERENT, randomly chosen sub-block of the
same full grid: the score for the right layer at the wrong place.

CENTRING IS REPORTED BOTH WAYS. The arms subtract a train-only mean, and the two
regimes use DIFFERENT means (train128 vs eval256). So centring could HIDE a raw
inconsistency, or could be what already fixes it. Raw is the primary number --
it is a property of DINO. Centred is reported alongside for the render domain,
where means exist for all four depths, because that is what the model actually
sees.

Inference only. No training, no checkpoint, no experiment directory touched.
"""

import argparse
import csv
import datetime
import json
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASES = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASES))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_PHASES, 'phase3_restoration', 'scripts'))

import dino_shared                                            # noqa: E402

DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
MEANS = os.path.join(_PHASES, 'phase3_restoration', 'means')
LAYERS_1IDX = (3, 6, 9, 12)
DOMAINS = {'render': 'val_renders_blackbg',
           'clean': 'val_clean',
           'noisy': 'val_verynoisy'}
TOKEN_PX = 8            # 256 image pixels / 32 tokens
CROP = 128
CROP_TOKENS = CROP // TOKEN_PX          # 16


# ----------------------------------------------------------------- loading
def load01(path):
    """-> [H,W] float32 in [0,1]. uint16 radar / 8-bit render, as the dataset."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)


def as_batch(arr, device):
    return torch.from_numpy(arr)[None, None].to(device)


# ----------------------------------------------------------------- metric
def patchwise_cosine(a, b):
    """a, b: [768, g, g] -> mean cosine over the g*g positions."""
    a = a.reshape(a.shape[0], -1).T.double()          # [P, 768]
    b = b.reshape(b.shape[0], -1).T.double()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())


def sub_block(grid, tx, ty, n=CROP_TOKENS):
    """[768,32,32] -> the [768,n,n] block whose top-left token is (tx, ty)."""
    return grid[:, ty:ty + n, tx:tx + n]


# ----------------------------------------------------------------- manifest
def build_manifest(ids, size, image_size, seed):
    """Crops snapped to the token grid, so full/crop alignment is exact."""
    rng = np.random.RandomState(seed)
    hi = (image_size - size) // TOKEN_PX               # inclusive max token index
    rows = []
    for i in ids:
        tx, ty = int(rng.randint(0, hi + 1)), int(rng.randint(0, hi + 1))
        rows.append({'image_id': i, 'x': tx * TOKEN_PX, 'y': ty * TOKEN_PX,
                     'size': size, 'tx': tx, 'ty': ty})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=0, help='0 = all')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids = sorted(f for f in os.listdir(os.path.join(DATASET, 'val_clean'))
                 if f.endswith('.png'))
    if args.n_images:
        ids = ids[:args.n_images]
    manifest = build_manifest(ids, CROP, 256, args.seed)
    with open(os.path.join(args.out, 'aligned_crop_manifest.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'x', 'y', 'size', 'tx', 'ty'])
        w.writeheader(); w.writerows(manifest)

    blocks0 = [dino_shared.b1_to_b0(b) for b in LAYERS_1IDX]
    ext = dino_shared.build_dino(device=args.device, verbose=False)

    mu = {}
    for b1 in LAYERS_1IDX:
        for regime, tag, size in (('crop', 'train128_dino224', 224),
                                  ('full', 'eval256_dino448', 448)):
            p = os.path.join(MEANS, f'render_B{b1}_{tag}_mean.pt')
            obj = torch.load(p, map_location='cpu', weights_only=False)
            v = torch.as_tensor(obj['mean'] if isinstance(obj, dict) else obj,
                                dtype=torch.float32).reshape(-1)
            if v.numel() != dino_shared.EMBED_DIM or not torch.isfinite(v).all():
                raise ValueError(f'{p}: bad mean, width {v.numel()}')
            # the same guard the arch applies: a mean built for the wrong block
            # or the wrong scale would bias every token by a constant.
            meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
            if meta:
                if int(meta.get('block_1indexed', b1)) != b1:
                    raise ValueError(f'{p}: block {meta["block_1indexed"]} != {b1}')
                got = int(meta.get('dino_size', size))
                if got != size:
                    raise ValueError(f'{p}: dino_size {got} != {size}')
            mu[(b1, regime)] = v.to(args.device)

    acc = {}    # (layer, metric) -> list

    def add(b1, key, val):
        acc.setdefault((b1, key), []).append(val)

    rng = np.random.RandomState(args.seed + 1)
    hi = (256 - CROP) // TOKEN_PX

    print(f'{len(manifest)} images | layers {LAYERS_1IDX} | '
          f'token = {TOKEN_PX}px | crop = {CROP_TOKENS} tokens')
    for n, row in enumerate(manifest, 1):
        fid, x, y, tx, ty = (row['image_id'], row['x'], row['y'],
                             row['tx'], row['ty'])
        full_g, crop_g = {}, {}
        for dom, sub in DOMAINS.items():
            img = load01(os.path.join(DATASET, sub, fid))
            if img.shape != (256, 256):
                raise ValueError(f'{fid}: {img.shape}')
            crop = img[y:y + CROP, x:x + CROP]
            with torch.no_grad():
                tf = dino_shared.extract_blocks(
                    ext, as_batch(img, args.device), blocks0,
                    dino_shared.DINO_SIZE_EVAL256)
                tc = dino_shared.extract_blocks(
                    ext, as_batch(crop, args.device), blocks0,
                    dino_shared.DINO_SIZE_TRAIN128)
            for b1, b0 in zip(LAYERS_1IDX, blocks0):
                full_g[(dom, b1)] = dino_shared.tokens_to_grid(tf[b0])[0]
                crop_g[(dom, b1)] = dino_shared.tokens_to_grid(tc[b0])[0]

        # a wrong-place block of the SAME layer and domain: the chance floor
        while True:
            rx, ry = int(rng.randint(0, hi + 1)), int(rng.randint(0, hi + 1))
            if (rx, ry) != (tx, ty):
                break

        for b1 in LAYERS_1IDX:
            fr = sub_block(full_g[('render', b1)], tx, ty)
            fc = sub_block(full_g[('clean', b1)], tx, ty)
            fn = sub_block(full_g[('noisy', b1)], tx, ty)
            cr, cc, cn = (crop_g[('render', b1)], crop_g[('clean', b1)],
                          crop_g[('noisy', b1)])

            # --- CONTEXT axis: same domain, same place, different context ----
            add(b1, 'context_render', patchwise_cosine(cr, fr))
            add(b1, 'context_clean', patchwise_cosine(cc, fc))
            add(b1, 'context_noisy', patchwise_cosine(cn, fn))
            add(b1, 'context_render_FLOOR',
                patchwise_cosine(cr, sub_block(full_g[('render', b1)], rx, ry)))

            # --- DEGRADATION axis, measured in EACH context ------------------
            add(b1, 'deg_full_render_vs_clean', patchwise_cosine(fr, fc))
            add(b1, 'deg_crop_render_vs_clean', patchwise_cosine(cr, cc))
            add(b1, 'deg_full_noisy_vs_clean', patchwise_cosine(fn, fc))
            add(b1, 'deg_crop_noisy_vs_clean', patchwise_cosine(cn, cc))

            # --- CENTRED, render only (means exist for all four depths) ------
            crc = cr - mu[(b1, 'crop')].reshape(-1, 1, 1)
            frc = fr - mu[(b1, 'full')].reshape(-1, 1, 1)
            add(b1, 'context_render_centred', patchwise_cosine(crc, frc))

        if n % 25 == 0 or n == len(manifest):
            print(f'  {n}/{len(manifest)}')

    # ------------------------------------------------------------- report
    summary = {}
    for (b1, k), v in acc.items():
        a = np.array(v)
        summary.setdefault(str(b1), {})[k] = {
            'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
            'median': float(np.median(a)), 'n': int(a.size)}
    for b1 in LAYERS_1IDX:
        d = summary[str(b1)]
        dc = np.array(acc[(b1, 'deg_crop_render_vs_clean')])
        df = np.array(acc[(b1, 'deg_full_render_vs_clean')])
        delta = dc - df
        d['INTERACTION_render_crop_minus_full'] = {
            'mean': float(delta.mean()), 'std': float(delta.std(ddof=1)),
            'n': int(delta.size),
            'note': 'NEGATIVE means degradation agreement is WORSE inside a '
                    'crop than inside a full frame -- the DSGIR prediction'}

    # per-image values, so the INTERACTION can be significance-tested rather
    # than eyeballed from a mean and a standard deviation.
    keys = sorted({k for _, k in acc})
    with open(os.path.join(args.out, 'crop_context_per_image.csv'), 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'layer_1indexed'] + keys)
        for i, row in enumerate(manifest):
            for b1 in LAYERS_1IDX:
                w.writerow([row['image_id'], b1] +
                           [f"{acc[(b1, k)][i]:.6f}" if (b1, k) in acc else ''
                            for k in keys])

    rec = {'created': datetime.datetime.now().astimezone().isoformat(),
           'n_images': len(manifest), 'layers_1indexed': list(LAYERS_1IDX),
           'crop': CROP, 'token_px': TOKEN_PX, 'seed': args.seed,
           'metric': 'mean corresponding-position cosine on full 768-d features',
           'per_layer': summary}
    out = os.path.join(args.out, 'crop_context_shift.json')
    with open(out, 'w') as f:
        json.dump(rec, f, indent=2)

    print('\n=== CONTEXT AXIS: does a crop match the full frame? ===')
    print(f"{'layer':>6}{'render':>10}{'clean':>10}{'noisy':>10}"
          f"{'FLOOR':>10}{'centred':>10}")
    for b1 in LAYERS_1IDX:
        d = summary[str(b1)]
        print(f"  B{b1:<4}{d['context_render']['mean']:>10.4f}"
              f"{d['context_clean']['mean']:>10.4f}"
              f"{d['context_noisy']['mean']:>10.4f}"
              f"{d['context_render_FLOOR']['mean']:>10.4f}"
              f"{d['context_render_centred']['mean']:>10.4f}")

    print('\n=== DEGRADATION AXIS, and the INTERACTION (DSGIR) ===')
    print(f"{'layer':>6}{'render~clean full':>19}{'render~clean crop':>19}"
          f"{'crop - full':>13}")
    for b1 in LAYERS_1IDX:
        d = summary[str(b1)]
        print(f"  B{b1:<4}{d['deg_full_render_vs_clean']['mean']:>19.4f}"
              f"{d['deg_crop_render_vs_clean']['mean']:>19.4f}"
              f"{d['INTERACTION_render_crop_minus_full']['mean']:>+13.4f}")
    print(f"\n  wrote {out}")


if __name__ == '__main__':
    main()
