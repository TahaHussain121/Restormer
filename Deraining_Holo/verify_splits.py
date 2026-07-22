"""Verify split files and their materialized dirs agree, and are disjoint.

    python verify_splits.py --root .../holographic_image_dataset \
        --variants clean noisy verynoisy

Checks, per split: txt entry count vs the file count in each <split>_<variant>/
dir, and that train/val/test are pairwise disjoint. Exits non-zero on any
mismatch or overlap so it can be used as a gate before training.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset')
parser.add_argument('--variants', nargs='+', default=['clean', 'noisy'])
parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
args = parser.parse_args()

ok = True
splits = {}
for name in args.splits:
    p = os.path.join(args.root, 'splits', name + '.txt')
    if os.path.exists(p):
        with open(p) as f:
            splits[name] = {l for l in f.read().splitlines() if l.strip()}
    else:
        splits[name] = set()

    counts = []
    for variant in args.variants:
        d = os.path.join(args.root, f'{name}_{variant}')
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        counts.append(f'{variant}={n}')
        if splits[name] and n != len(splits[name]):
            ok = False
            counts[-1] += ' MISMATCH'
    print(f'{name}: txt={len(splits[name])}  ' + '  '.join(counts))

for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
    if a in splits and b in splits:
        n = len(splits[a] & splits[b])
        print(f'{a}∩{b}={n}' + ('' if n == 0 else '  <-- OVERLAP'))
        if n:
            ok = False

print('\nOK' if ok else '\nFAILED')
raise SystemExit(0 if ok else 1)
