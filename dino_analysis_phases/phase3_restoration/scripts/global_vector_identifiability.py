"""How uniquely does a per-image GLOBAL DINO vector identify its training scene?

WHY. global-render (the pooled B6 prior, broadcast to every position) ends
training with a LOWER pixel loss than E0 (0.0444 vs 0.0587) and a validation
PSNR 2.3 dB WORSE, collapsing 0.86 dB from its 60k peak. That is the signature
of memorisation: a constant-per-image vector can act as a scene identifier, and
with 6,101 training scenes the network can learn "this scene looks like this"
instead of restoration. The question this script answers, in feature space and
without training anything, is whether the vectors on offer are in fact
identifiers -- and whether the literature's global descriptor (the last-layer
CLS token, which global-render did NOT use) would be any less of one.

WHAT. For every training scene, TWO independent random 128 crops of the render,
drawn the way basicsr/train.py draws them (x0 = int(128 * random()), same for
y0), each pushed through the frozen DINOv2 at 224 exactly as the training
pipeline does. From each crop:

    pooled_b6    mean over the 256 patch tokens of block 6, centred with the
                 project's train128 render mean  (== global-render's prior)
    pooled_b12   the same for block 12 (the depth the papers pool)
    cls_b12      the CLS token of block 12, the designed global descriptor
    pooled_b3    block 3, for the shallow end

Retrieval: crop A of scene i is the query; the gallery is crop B of EVERY scene.
Top-1 accuracy = fraction of scenes whose nearest gallery vector is their own
other crop. Chance is 1/N. Reported for cosine distance on the centred vectors
(the project's convention) and for Euclidean distance (centering-invariant).
Also reported: mean same-scene cosine against mean different-scene cosine.

READING IT. Top-1 near 1.0 means a random crop's global vector is a fingerprint
of the scene even though the two crops differ, so a broadcast prior hands the
network a label it can memorise. If cls_b12 is as identifying as pooled_b6, the
memorisation mechanism covers the untested CLS design too; if it is much less
identifying, a CLS arm is worth its 39 hours.

The render is the DINO source, as in every render arm. Train split only.
"""

import argparse
import datetime
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import dino_shared                                            # noqa: E402

DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
CROP = 128
BLOCKS1 = [3, 6, 12]


def load_render(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return img[:, :, 0].astype(np.float32) / 255.       # BGR ch0, as the dataset


def random_crop(img, rng):
    span = img.shape[0] - CROP
    y0 = int(span * rng.random())
    x0 = int(span * rng.random())
    return img[y0:y0 + CROP, x0:x0 + CROP]


@torch.no_grad()
def embed(ext, batch, device):
    """batch [B,1,128,128] in [0,1] -> dict of per-image vectors [B,768]."""
    x = dino_shared.preprocess(batch.to(device), dino_shared.DINO_SIZE_TRAIN128)
    blocks0 = tuple(dino_shared.b1_to_b0(b) for b in BLOCKS1)
    feats = ext.dino.get_intermediate_layers(
        x, n=blocks0, reshape=False, return_class_token=True, norm=True)
    out = {}
    for b1, (patch, cls) in zip(sorted(BLOCKS1), feats):
        out[f'pooled_b{b1}'] = patch.mean(dim=1).float().cpu()
        if b1 == 12:
            out['cls_b12'] = cls.float().cpu()
    return out


def retrieval(q, g):
    """q, g: [N, D]. Returns top-1 accuracy under cosine and Euclidean."""
    qn = torch.nn.functional.normalize(q, dim=1)
    gn = torch.nn.functional.normalize(g, dim=1)
    cos = qn @ gn.T                                     # [N, N]
    top1_cos = float((cos.argmax(dim=1) == torch.arange(len(q))).float().mean())
    d2 = torch.cdist(q, g)                              # Euclidean
    top1_l2 = float((d2.argmin(dim=1) == torch.arange(len(q))).float().mean())
    same = cos.diag()
    diff = cos[~torch.eye(len(q), dtype=torch.bool)]
    # rank of the true match, for a median-rank figure
    ranks = (cos > same[:, None]).sum(dim=1) + 1
    return {'top1_cosine': top1_cos, 'top1_euclidean': top1_l2,
            'top5_cosine': float((ranks <= 5).float().mean()),
            'median_rank_cosine': float(ranks.float().median()),
            'same_scene_cos_mean': float(same.mean()),
            'diff_scene_cos_mean': float(diff.mean()),
            'diff_scene_cos_max_mean': float(
                cos.masked_fill(torch.eye(len(q), dtype=torch.bool), -2).max(dim=1).values.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=0, help='0 = every scene')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'global_identifiability', 'identifiability.json'))
    args = ap.parse_args()

    rdir = os.path.join(DATASET, f'{args.split}_renders_blackbg')
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(rdir)
                 if f.endswith('.png'))
    if args.limit:
        ids = ids[:args.limit]
    N = len(ids)
    print(f'{N} scenes from {rdir}')

    rng = random.Random(args.seed)
    ext = dino_shared.build_dino(device=args.device, verbose=True)
    mu = torch.load(os.path.join(
        _PHASE3, 'means', 'render_B6_train128_dino224_mean.pt'),
        map_location='cpu', weights_only=False)['mean'].reshape(-1).float()

    vecs = {'A': {}, 'B': {}}
    for start in range(0, N, args.batch):
        chunk = ids[start:start + args.batch]
        crops = {'A': [], 'B': []}
        for i in chunk:
            img = load_render(os.path.join(rdir, f'{i}.png'))
            crops['A'].append(random_crop(img, rng))
            crops['B'].append(random_crop(img, rng))
        for key in ('A', 'B'):
            batch = torch.from_numpy(np.stack(crops[key]))[:, None]
            e = embed(ext, batch, args.device)
            for k, v in e.items():
                vecs[key].setdefault(k, []).append(v)
        if (start // args.batch) % 10 == 0:
            print(f'  {min(start + args.batch, N)}/{N}', flush=True)
    for key in ('A', 'B'):
        for k in vecs[key]:
            vecs[key][k] = torch.cat(vecs[key][k])

    results = {}
    for k in vecs['A']:
        q, g = vecs['A'][k], vecs['B'][k]
        if k == 'pooled_b6':
            centre = mu                       # the project's own train128 mean
        else:
            centre = torch.cat([q, g]).mean(dim=0)   # set mean, same role
        r = retrieval(q - centre, g - centre)
        r['centre'] = ('render_B6_train128_dino224_mean.pt' if k == 'pooled_b6'
                       else 'mean over this set')
        r['raw_uncentred_top1_cosine'] = retrieval(q, g)['top1_cosine']
        results[k] = r
        print(f'{k:12s} top1 cos {r["top1_cosine"]:.3f}  top1 L2 '
              f'{r["top1_euclidean"]:.3f}  top5 {r["top5_cosine"]:.3f}  '
              f'median rank {r["median_rank_cosine"]:.0f}  same {r["same_scene_cos_mean"]:.3f} '
              f'diff {r["diff_scene_cos_mean"]:.3f}  hardest-other {r["diff_scene_cos_max_mean"]:.3f}')
    print(f'chance top-1 = 1/{N} = {1 / N:.5f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'split': args.split, 'n_scenes': N, 'crop': CROP,
                   'dino_size': dino_shared.DINO_SIZE_TRAIN128,
                   'seed': args.seed, 'chance_top1': 1 / N,
                   'protocol': 'two independent random 128 crops per scene; '
                               'query crop A vs gallery crop B of all scenes',
                   'results': results}, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
