"""PHASE 5b — t-SNE of DINO features: full-frame context vs 128-crop context.

REGENERATED 2026-09-02. An earlier copy of this script produced
`tsne_{patch,image}_{raw,centred}.png` and `tsne_separability.json` in job
1799766, then did not survive a session interrupt. This is a faithful rewrite;
re-running it reproduces those four figures and the JSON.

WHY THIS EXISTS ALONGSIDE `visualize_crop_drift.py`

`visualize_crop_drift.py` already draws ONE t-SNE. This script draws FOUR,
crossing two choices that change what the picture means:

  patch vs image   is a point one 768-d patch token, or one whole image with its
                   positions averaged? Fine structure vs coarse structure.
  raw vs centred   the two regimes use DIFFERENT centring means (train128 vs
                   eval256). A raw plot can separate the contexts merely because
                   their means differ, which is real but trivial. Centred
                   removes that, so any separation left is NOT a mean offset.

THE NUMBERS ARE THE RESULT; THE PICTURE IS THE ILLUSTRATION. In a t-SNE map,
between-cluster distance is not meaningful, cluster size is not meaningful, and
the layout changes with perplexity and seed. So every panel is titled with a
LINEAR SEPARABILITY score instead: 5-fold cross-validated accuracy of a logistic
regression predicting CONTEXT (full vs crop) from the 768-d features, computed
on the 50-d PCA and NOT on the 2-D map. 0.50 is chance.

Inference only. No training, no checkpoint, no experiment directory touched.
"""

import argparse
import datetime
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASES = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PHASES, 'phase3_restoration', 'scripts'))
sys.path.insert(0, _HERE)

import dino_shared                                             # noqa: E402
from analyze_crop_context_shift import (                       # noqa: E402
    DATASET, DOMAINS, LAYERS_1IDX, MEANS, CROP, CROP_TOKENS,
    load01, as_batch, build_manifest, sub_block)

COLOURS = {'render': '#C4622D', 'clean': '#2E6F9E', 'noisy': '#6B7183'}
MARKERS = {'full': 'o', 'crop': '^'}


def load_mean(b1, regime):
    """The same mean file, with the same guard, the architecture uses."""
    tag, size = (('train128_dino224', 224) if regime == 'crop'
                 else ('eval256_dino448', 448))
    p = os.path.join(MEANS, f'render_B{b1}_{tag}_mean.pt')
    obj = torch.load(p, map_location='cpu', weights_only=False)
    v = torch.as_tensor(obj['mean'] if isinstance(obj, dict) else obj,
                        dtype=torch.float32).reshape(-1)
    if v.numel() != dino_shared.EMBED_DIM or not torch.isfinite(v).all():
        raise ValueError(f'{p}: bad mean, width {v.numel()}')
    meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
    if meta:
        if int(meta.get('block_1indexed', b1)) != b1:
            raise ValueError(f'{p}: block {meta["block_1indexed"]} != {b1}')
        if int(meta.get('dino_size', size)) != size:
            raise ValueError(f'{p}: mean built for the wrong scale')
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=80)
    ap.add_argument('--positions', type=int, default=25,
                    help='patch tokens sampled per image per condition')
    ap.add_argument('--unit', default='both', choices=['patch', 'image', 'both'])
    ap.add_argument('--perplexity', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids = sorted(f for f in os.listdir(os.path.join(DATASET, 'val_clean'))
                 if f.endswith('.png'))[:args.n_images]
    manifest = build_manifest(ids, CROP, 256, args.seed)
    blocks0 = [dino_shared.b1_to_b0(b) for b in LAYERS_1IDX]
    ext = dino_shared.build_dino(device=args.device, verbose=False)
    mu = {(b1, r): load_mean(b1, r) for b1 in LAYERS_1IDX for r in ('crop', 'full')}
    rng = np.random.RandomState(args.seed + 7)

    feats = {}                       # (unit, layer) -> [(vector, mean, dom, ctx)]
    print(f'{len(manifest)} images | layers {LAYERS_1IDX}')
    for n, row in enumerate(manifest, 1):
        fid, x, y, tx, ty = (row['image_id'], row['x'], row['y'],
                             row['tx'], row['ty'])
        pos = rng.choice(CROP_TOKENS ** 2,
                         size=min(args.positions, CROP_TOKENS ** 2),
                         replace=False)
        for dom, sub in DOMAINS.items():
            img = load01(os.path.join(DATASET, sub, fid))
            crop = img[y:y + CROP, x:x + CROP]
            with torch.no_grad():
                tf = dino_shared.extract_blocks(
                    ext, as_batch(img, args.device), blocks0,
                    dino_shared.DINO_SIZE_EVAL256)
                tc = dino_shared.extract_blocks(
                    ext, as_batch(crop, args.device), blocks0,
                    dino_shared.DINO_SIZE_TRAIN128)
            for b1, b0 in zip(LAYERS_1IDX, blocks0):
                g_full = sub_block(dino_shared.tokens_to_grid(tf[b0])[0], tx, ty)
                g_crop = dino_shared.tokens_to_grid(tc[b0])[0]
                for ctx, g in (('full', g_full), ('crop', g_crop)):
                    flat = g.reshape(g.shape[0], -1).T.cpu().numpy()   # [P,768]
                    m = mu[(b1, ctx)].numpy()
                    feats.setdefault(('patch', b1), []).extend(
                        (flat[p], m, dom, ctx) for p in pos)
                    feats.setdefault(('image', b1), []).append(
                        (flat.mean(0), m, dom, ctx))
        if n % 20 == 0 or n == len(manifest):
            print(f'  {n}/{len(manifest)}')

    units = ['patch', 'image'] if args.unit == 'both' else [args.unit]
    report = {}
    for unit in units:
        for centred in (False, True):
            tag = f'{unit}_{"centred" if centred else "raw"}'
            fig, axes = plt.subplots(1, len(LAYERS_1IDX),
                                     figsize=(4.6 * len(LAYERS_1IDX), 4.9))
            for ax, b1 in zip(np.atleast_1d(axes), LAYERS_1IDX):
                rows = feats[(unit, b1)]
                X = np.stack([(v - m) if centred else v for v, m, _, _ in rows])
                dom = np.array([d for _, _, d, _ in rows])
                ctx = np.array([c for _, _, _, c in rows])

                Xs = StandardScaler().fit_transform(X)
                Xp = PCA(n_components=min(50, Xs.shape[1], Xs.shape[0] - 1),
                         random_state=args.seed).fit_transform(Xs)
                yctx = (ctx == 'crop').astype(int)
                acc = float(cross_val_score(LogisticRegression(max_iter=2000),
                                            Xp, yctx, cv=5,
                                            scoring='accuracy').mean())
                sil = float(silhouette_score(Xp, yctx))
                report.setdefault(tag, {})[f'B{b1}'] = {
                    'context_linear_separability_cv_acc': acc,
                    'context_silhouette': sil, 'n_points': int(len(rows))}

                emb = TSNE(n_components=2,
                           perplexity=min(args.perplexity, (len(rows) - 1) / 3.0),
                           init='pca', random_state=args.seed).fit_transform(Xp)
                for d in DOMAINS:
                    for c in ('full', 'crop'):
                        s = (dom == d) & (ctx == c)
                        ax.scatter(emb[s, 0], emb[s, 1],
                                   s=7 if unit == 'patch' else 22,
                                   c=COLOURS[d], marker=MARKERS[c], alpha=0.55,
                                   linewidths=0,
                                   label=f'{d} · {c}' if b1 == LAYERS_1IDX[0] else None)
                ax.set_title(f'B{b1}   context sep. {acc:.2f}', fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
            h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
            fig.legend(h, l, loc='lower center', ncol=6, frameon=False,
                       fontsize=9, bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(
                f't-SNE of DINO features — {unit}-level, '
                f'{"mean-centred" if centred else "raw"}   '
                f'(circle = full frame, triangle = 128 crop)\n'
                f'"context sep." = 5-fold CV accuracy of a logistic regression '
                f'predicting context from the 768-d feature; 0.50 = chance',
                fontsize=12)
            fig.tight_layout(rect=[0, 0.04, 1, 0.94])
            p = os.path.join(args.out, f'tsne_{tag}.png')
            fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f'  wrote {p}')

    rec = {'created': datetime.datetime.now().astimezone().isoformat(),
           'n_images': len(manifest), 'positions_per_image': args.positions,
           'perplexity': args.perplexity, 'seed': args.seed,
           'note': 'separability and silhouette are computed on the 50-d PCA of '
                   'the full features, NOT on the 2-D t-SNE map',
           'separability': report}
    with open(os.path.join(args.out, 'tsne_separability.json'), 'w') as f:
        json.dump(rec, f, indent=2)

    print('\n=== CONTEXT SEPARABILITY (0.50 = chance) ===')
    for tag, d in report.items():
        print(f'  {tag:<16}' + '  '.join(
            f'{k} {v["context_linear_separability_cv_acc"]:.3f}'
            for k, v in d.items()))


if __name__ == '__main__':
    main()
