"""PHASE 5c — six views of the crop-versus-full DINO drift.

TWO SETUPS, AND THE DIFFERENCE MATTERS. Do not mix their numbers.

  SWEEP   every crop ratio is resized to the SAME DINO input (224), so the only
          variable is FIELD OF VIEW. This is DSGIR's comparison (their Fig. 10)
          and it is what makes crop ratios comparable to one another.

  ALIGNED this project's ACTUAL pipeline -- full frame at 448 (32x32 tokens),
          128 crop at 224 (16x16 tokens). Field of view AND token resolution
          both differ, which is the real deployment condition. Token alignment
          is exact because 256/32 = 8 image pixels per token, so a crop at
          (x, y) with x, y multiples of 8 is the 16x16 sub-block at (x/8, y/8).

WHAT DSGIR ACTUALLY SHOWS, having now read the paper:

  Fig. 8   t-SNE of DEGRADATION-TYPE representations from their DSE module.
           NOT the crop analysis. Do not cite it as such.
  Fig. 10  layer-wise cosine between degraded and GT, at crop ratios
           Full / 0.8 / 0.5 / 0.2, layers {1,4,8,12}. "smaller crops lead to
           lower semantic similarity, and the gap becomes more evident in
           deeper layers." THIS is the crop analysis.
  Fig. 12  KDE of per-image similarity, local-crop panels vs full-image panels,
           mean printed in the legend.

Parts 1 and 2 replicate Figs. 10 and 12 on radar data. Part 4 (the spatial
heatmap) and Part 5 (nearest-neighbour position retrieval) go BEYOND the paper:
neither asks WHERE inside a crop the drift happens, nor whether a crop token can
still identify its own position.

Layers {1,3,4,6,8,9,12} are read: {1,4,8,12} are DSGIR's, {3,6,9,12} are the
ones this project's arms actually use.

Inference only. No training, no checkpoint, no experiment directory touched.
"""

import argparse
import datetime
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASES = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PHASES, 'phase3_restoration', 'scripts'))
sys.path.insert(0, _HERE)
import dino_shared                                             # noqa: E402
from analyze_crop_context_shift import (                       # noqa: E402
    DATASET, DOMAINS, TOKEN_PX, CROP, CROP_TOKENS, load01, as_batch,
    build_manifest, sub_block, patchwise_cosine)

LAYERS = (1, 3, 4, 6, 8, 9, 12)
DSGIR_LAYERS = (1, 4, 8, 12)
OURS_LAYERS = (3, 6, 9, 12)
RATIOS = (1.0, 0.8, 0.5, 0.2)
RATIO_NAME = {1.0: 'Full', 0.8: 'Large 0.8', 0.5: 'Medium 0.5', 0.2: 'Small 0.2'}
RATIO_COL = {1.0: '#2E6F9E', 0.8: '#4C9F70', 0.5: '#C4622D', 0.2: '#9E3B32'}
PAIRS = (('render', 'clean'), ('noisy', 'clean'))
DOM_COL = {'render': '#C4622D', 'clean': '#2E6F9E', 'noisy': '#6B7183'}


def pooled(tokens):
    """[1,N,768] -> [768] mean over patch tokens, as a global semantic vector."""
    return tokens[0].mean(0)


def cos1(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.double()[None], b.double()[None], dim=1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sweep', type=int, default=339)
    ap.add_argument('--n-aligned', type=int, default=150)
    ap.add_argument('--tsne-images', type=int, default=70)
    ap.add_argument('--tsne-positions', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids = sorted(f for f in os.listdir(os.path.join(DATASET, 'val_clean'))
                 if f.endswith('.png'))
    blocks0 = [dino_shared.b1_to_b0(b) for b in LAYERS]
    ext = dino_shared.build_dino(device=args.device, verbose=False)
    rng = np.random.RandomState(args.seed)
    report = {}

    # ================================================================ PART 1+2
    # SWEEP: every ratio resized to 224. Degraded vs GT, same crop.
    print(f'PART 1/2  crop-ratio sweep, n={args.n_sweep}')
    sweep = {}                       # (pair, ratio, layer) -> [cos per image]
    for n, fid in enumerate(ids[:args.n_sweep], 1):
        imgs = {d: load01(os.path.join(DATASET, s, fid))
                for d, s in DOMAINS.items()}
        for r in RATIOS:
            s = int(round(256 * r))
            x = 0 if r == 1.0 else int(rng.randint(0, 256 - s + 1))
            y = 0 if r == 1.0 else int(rng.randint(0, 256 - s + 1))
            vec = {}
            for d, im in imgs.items():
                patch = im[y:y + s, x:x + s]
                with torch.no_grad():
                    t = dino_shared.extract_blocks(
                        ext, as_batch(patch, args.device), blocks0,
                        dino_shared.DINO_SIZE_TRAIN128)   # 224 for EVERY ratio
                vec[d] = {b1: pooled(t[b0]) for b1, b0 in zip(LAYERS, blocks0)}
            for a, b in PAIRS:
                for b1 in LAYERS:
                    sweep.setdefault((f'{a}~{b}', r, b1), []).append(
                        cos1(vec[a][b1], vec[b][b1]))
        if n % 50 == 0 or n == args.n_sweep:
            print(f'  {n}/{args.n_sweep}')

    for pair in [f'{a}~{b}' for a, b in PAIRS]:
        for r in RATIOS:
            for b1 in LAYERS:
                v = np.array(sweep[(pair, r, b1)])
                report.setdefault('sweep', {}).setdefault(pair, {}) \
                      .setdefault(RATIO_NAME[r], {})[f'B{b1}'] = {
                          'mean': float(v.mean()), 'std': float(v.std(ddof=1))}

    # ---- Fig-10 replica -----------------------------------------------------
    for tag, lay in (('dsgir_layers', DSGIR_LAYERS), ('our_layers', OURS_LAYERS)):
        fig, axes = plt.subplots(1, len(PAIRS), figsize=(6.2 * len(PAIRS), 4.6))
        for ax, (a, b) in zip(np.atleast_1d(axes), PAIRS):
            for r in RATIOS:
                m = [np.mean(sweep[(f'{a}~{b}', r, b1)]) for b1 in lay]
                ax.plot(range(len(lay)), m, marker='o', color=RATIO_COL[r],
                        label=RATIO_NAME[r], linewidth=1.9, markersize=5)
            ax.set_xticks(range(len(lay)))
            ax.set_xticklabels([str(b1) for b1 in lay])
            ax.set_xlabel('DINOv2 layer'); ax.set_ylabel('cosine similarity')
            ax.set_title(f'{a} vs {b}')
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle('Layer-wise cosine between degraded and clean, by crop ratio '
                     '(DSGIR Fig. 10 replica; all ratios resized to 224)',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(args.out, f'fig10_replica_{tag}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # ---- Fig-12 replica: KDE, full vs the 0.5 training ratio ---------------
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(3.5 * len(LAYERS), 3.6))
    for ax, b1 in zip(np.atleast_1d(axes), LAYERS):
        for r, lab in ((1.0, 'full image'), (0.5, 'local crop 0.5')):
            v = np.array(sweep[('render~clean', r, b1)])
            xs = np.linspace(v.min() - 0.02, v.max() + 0.02, 240)
            ax.fill_between(xs, gaussian_kde(v)(xs), alpha=0.45,
                            color=RATIO_COL[r], label=f'{lab} ({v.mean():.3f})')
        ax.set_title(f'B{b1}', fontsize=11); ax.set_yticks([])
        ax.set_xlabel('cosine'); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Distribution of render-vs-clean semantic similarity: '
                 'full image vs local crop (DSGIR Fig. 12 replica)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(args.out, 'fig12_replica_kde.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ================================================================ PART 4+5
    # ALIGNED: the project's real pipeline. full 448 -> 32x32, crop 224 -> 16x16
    print(f'PART 4/5  aligned drift, n={args.n_aligned}')
    man = build_manifest(ids[:args.n_aligned], CROP, 256, args.seed)
    heat = {b1: np.zeros((CROP_TOKENS, CROP_TOKENS)) for b1 in LAYERS}
    nn_hit = {b1: 0 for b1 in LAYERS}
    nn_tot = 0
    for n, row in enumerate(man, 1):
        fid, x, y, tx, ty = (row['image_id'], row['x'], row['y'],
                             row['tx'], row['ty'])
        img = load01(os.path.join(DATASET, DOMAINS['render'], fid))
        with torch.no_grad():
            tf = dino_shared.extract_blocks(
                ext, as_batch(img, args.device), blocks0,
                dino_shared.DINO_SIZE_EVAL256)
            tc = dino_shared.extract_blocks(
                ext, as_batch(img[y:y + CROP, x:x + CROP], args.device),
                blocks0, dino_shared.DINO_SIZE_TRAIN128)
        for b1, b0 in zip(LAYERS, blocks0):
            gf = dino_shared.tokens_to_grid(tf[b0])[0]
            gc = dino_shared.tokens_to_grid(tc[b0])[0]
            blk = sub_block(gf, tx, ty)
            a = gc.reshape(768, -1).T.double()
            bb = blk.reshape(768, -1).T.double()
            heat[b1] += torch.nn.functional.cosine_similarity(
                a, bb, dim=1).reshape(CROP_TOKENS, CROP_TOKENS).cpu().numpy()
            # position retrieval: nearest full-frame token for each crop token
            full_all = gf.reshape(768, -1).T.double()
            sim = torch.nn.functional.normalize(a, dim=1) @ \
                torch.nn.functional.normalize(full_all, dim=1).T
            pred = sim.argmax(1).cpu().numpy()
            gy, gx = np.divmod(np.arange(CROP_TOKENS ** 2), CROP_TOKENS)
            truth = (ty + gy) * 32 + (tx + gx)
            nn_hit[b1] += int((pred == truth).sum())
        nn_tot += CROP_TOKENS ** 2
        if n % 50 == 0 or n == len(man):
            print(f'  {n}/{len(man)}')

    for b1 in LAYERS:
        heat[b1] /= len(man)
        report.setdefault('aligned', {})[f'B{b1}'] = {
            'mean_position_cosine': float(heat[b1].mean()),
            'border_minus_centre': float(
                np.concatenate([heat[b1][0], heat[b1][-1],
                                heat[b1][:, 0], heat[b1][:, -1]]).mean()
                - heat[b1][4:12, 4:12].mean()),
            'nn_position_top1': nn_hit[b1] / nn_tot}

    vmin = min(h.min() for h in heat.values())
    vmax = max(h.max() for h in heat.values())
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(2.9 * len(LAYERS), 3.5))
    for ax, b1 in zip(np.atleast_1d(axes), LAYERS):
        im = ax.imshow(heat[b1], cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_title(f'B{b1}   mean {heat[b1].mean():.3f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.01)
    fig.suptitle('WHERE the crop drifts: per-position cosine between the crop '
                 'and the same region of the full frame\n'
                 '(16x16 crop token grid, averaged over images — bright = agrees, '
                 'dark = drifts)', fontsize=12)
    fig.savefig(os.path.join(args.out, 'spatial_drift_heatmap.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ---- PCA maps, one representative image --------------------------------
    row = man[0]
    img = load01(os.path.join(DATASET, DOMAINS['render'], row['image_id']))
    with torch.no_grad():
        tf = dino_shared.extract_blocks(ext, as_batch(img, args.device), blocks0,
                                        dino_shared.DINO_SIZE_EVAL256)
        tc = dino_shared.extract_blocks(
            ext, as_batch(img[row['y']:row['y'] + CROP,
                              row['x']:row['x'] + CROP], args.device),
            blocks0, dino_shared.DINO_SIZE_TRAIN128)
    fig, axes = plt.subplots(2, len(LAYERS), figsize=(2.7 * len(LAYERS), 5.8))
    for j, (b1, b0) in enumerate(zip(LAYERS, blocks0)):
        gf = dino_shared.tokens_to_grid(tf[b0])[0]
        gc = dino_shared.tokens_to_grid(tc[b0])[0]
        blk = sub_block(gf, row['tx'], row['ty'])
        A = blk.reshape(768, -1).T.cpu().numpy()
        B = gc.reshape(768, -1).T.cpu().numpy()
        p = PCA(3, random_state=args.seed).fit(np.vstack([A, B]))   # JOINT fit
        for i, (M, lab) in enumerate(((A, 'from full frame'), (B, 'from crop'))):
            v = p.transform(M)
            v = (v - v.min(0)) / (np.ptp(v, axis=0) + 1e-9)
            axes[i, j].imshow(v.reshape(CROP_TOKENS, CROP_TOKENS, 3))
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(lab, fontsize=10)
            if i == 0:
                axes[i, j].set_title(f'B{b1}', fontsize=11)
    fig.suptitle(f'Joint-PCA maps of the SAME 16x16 region, {row["image_id"]}\n'
                 'top: computed inside the full frame · bottom: computed inside '
                 'the crop · one PCA fitted on both, so colours are comparable',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(args.out, 'pca_maps_full_vs_crop.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ================================================================== PART 3
    print(f'PART 3  t-SNE, n={args.tsne_images}')
    pts = {b1: [] for b1 in LAYERS}
    for row in build_manifest(ids[:args.tsne_images], CROP, 256, args.seed):
        fid, tx, ty = row['image_id'], row['tx'], row['ty']
        sel = rng.choice(CROP_TOKENS ** 2, args.tsne_positions, replace=False)
        for dom, sub in DOMAINS.items():
            im = load01(os.path.join(DATASET, sub, fid))
            with torch.no_grad():
                tf = dino_shared.extract_blocks(
                    ext, as_batch(im, args.device), blocks0,
                    dino_shared.DINO_SIZE_EVAL256)
                tc = dino_shared.extract_blocks(
                    ext, as_batch(im[row['y']:row['y'] + CROP,
                                     row['x']:row['x'] + CROP], args.device),
                    blocks0, dino_shared.DINO_SIZE_TRAIN128)
            for b1, b0 in zip(LAYERS, blocks0):
                gf = sub_block(dino_shared.tokens_to_grid(tf[b0])[0], tx, ty)
                gc = dino_shared.tokens_to_grid(tc[b0])[0]
                for ctx, g in (('full', gf), ('crop', gc)):
                    f = g.reshape(768, -1).T.cpu().numpy()
                    pts[b1].extend((f[p], dom, ctx) for p in sel)

    fig, axes = plt.subplots(1, len(LAYERS), figsize=(3.4 * len(LAYERS), 4.2))
    for ax, b1 in zip(np.atleast_1d(axes), LAYERS):
        X = np.stack([v for v, _, _ in pts[b1]])
        dom = np.array([d for _, d, _ in pts[b1]])
        ctx = np.array([c for _, _, c in pts[b1]])
        Xp = PCA(50, random_state=args.seed).fit_transform(
            StandardScaler().fit_transform(X))
        acc = float(cross_val_score(LogisticRegression(max_iter=2000), Xp,
                                    (ctx == 'crop').astype(int), cv=5).mean())
        report.setdefault('tsne_context_separability', {})[f'B{b1}'] = acc
        emb = TSNE(2, perplexity=30, init='pca',
                   random_state=args.seed).fit_transform(Xp)
        for d in DOMAINS:
            for c in ('full', 'crop'):
                s = (dom == d) & (ctx == c)
                ax.scatter(emb[s, 0], emb[s, 1], s=6, c=DOM_COL[d],
                           marker='o' if c == 'full' else '^', alpha=0.5,
                           linewidths=0,
                           label=f'{d} · {c}' if b1 == LAYERS[0] else None)
        ax.set_title(f'B{b1}   context sep. {acc:.2f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('t-SNE of patch features — circle = full frame, triangle = crop\n'
                 '"context sep." is 5-fold CV accuracy of a logistic regression '
                 'on the 768-d features (0.50 = chance); the MAP itself is not a '
                 'measurement', fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.9])
    fig.savefig(os.path.join(args.out, 'tsne_context.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    report['created'] = datetime.datetime.now().astimezone().isoformat()
    with open(os.path.join(args.out, 'crop_drift_visualisations.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('\n=== DSGIR Fig-10 replica: render~clean cosine by crop ratio ===')
    print(f"{'layer':>6}" + ''.join(f'{RATIO_NAME[r]:>13}' for r in RATIOS))
    for b1 in LAYERS:
        print(f'  B{b1:<4}' + ''.join(
            f"{np.mean(sweep[('render~clean', r, b1)]):>13.4f}" for r in RATIOS))
    print('\n=== aligned drift (this project\'s real pipeline) ===')
    print(f"{'layer':>6}{'pos cosine':>12}{'border-centre':>15}{'NN top-1':>11}")
    for b1 in LAYERS:
        d = report['aligned'][f'B{b1}']
        print(f"  B{b1:<4}{d['mean_position_cosine']:>12.4f}"
              f"{d['border_minus_centre']:>+15.4f}{d['nn_position_top1']:>11.3f}")
    print(f"\n  figures + json -> {args.out}")


if __name__ == '__main__':
    main()
