"""Matched-128 evaluation: build the crop COORDINATE MANIFEST.

The matched-128 protocol measures both models on 128x128 crops, i.e. on the
conditioning distribution E1 was actually trained under. For that comparison to
mean anything, one crop set must be shared by:

    the 1e5 input · the 1e7 target · the E0 prediction · the E1 prediction

A seed alone is too weak a guarantee -- it survives only as long as nobody
touches the drawing code. This writes the coordinates to disk once, as an
explicit manifest, and every evaluator reads the file. If the file changes, the
protocol changed, visibly, in git.

One crop per image (size 128, drawn uniformly like basicsr/train.py:264-265,
keyed by image id so filesystem order is irrelevant). Written to
phase3_restoration/results/crop_manifests/matched128_<split>.csv.
"""

import argparse
import csv
import datetime
import hashlib
import os
import random
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=_REPO).decode().strip()
    except Exception:                                  # noqa: BLE001
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--outdir', default=os.path.join(
        _PHASE3, 'results', 'crop_manifests'))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    lq_dir = os.path.join(DATASET, f'{args.split}_verynoisy')
    gt_dir = os.path.join(DATASET, f'{args.split}_clean')
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                 if f.endswith('.png'))
    for i in ids:
        if not os.path.isfile(os.path.join(lq_dir, f'{i}.png')):
            raise SystemExit(f'{i}.png missing from {lq_dir}')

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f'matched128_{args.split}.csv')
    if os.path.isfile(out) and not args.force:
        raise SystemExit(f'REFUSING to overwrite {out} (pass --force). '
                         f'The manifest defines the protocol; replacing it '
                         f'invalidates every metric already computed from it.')

    span = args.image_size - args.size
    rows = []
    for image_id in ids:
        rng = random.Random(f'{args.seed}:{image_id}:{args.size}')
        rows.append({'image_id': image_id,
                     'x': int(span * rng.random()),      # left
                     'y': int(span * rng.random()),      # top
                     'size': args.size})

    with open(out, 'w', newline='') as f:
        f.write(f'# matched-128 crop manifest — the SAME coordinates are used '
                f'for the 1e5 input, the 1e7 target, the E0 prediction and the '
                f'E1 prediction\n')
        f.write(f'# split={args.split} n={len(rows)} size={args.size} '
                f'image_size={args.image_size} seed={args.seed}\n')
        f.write(f'# x=left, y=top (numpy img[y:y+size, x:x+size])\n')
        f.write(f'# created={datetime.datetime.now().astimezone().isoformat()} '
                f'git={git_commit()}\n')
        w = csv.DictWriter(f, fieldnames=['image_id', 'x', 'y', 'size'])
        w.writeheader()
        w.writerows(rows)

    with open(out, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f'wrote {out}\n  {len(rows)} crops, size {args.size}, seed {args.seed}'
          f'\n  sha256 {digest}')
    print(f'  first rows: {rows[:3]}')


if __name__ == '__main__':
    main()
