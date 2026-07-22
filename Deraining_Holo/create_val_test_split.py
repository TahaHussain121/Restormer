"""Carve a held-out test split out of an existing val split.

Halves val.txt into a new (smaller) val.txt plus a test.txt, using a fixed
seed so the partition is reproducible. The train split is never touched.

Originally written for holo_image_dataset (DEVLOG Step 15); generalized so
the same recipe can be applied to holographic_image_dataset.

    python create_val_test_split.py --root /path/to/holographic_image_dataset

The previous val.txt is backed up to val.txt.bak before being overwritten.
Re-running is refused if test.txt already exists (pass --force to override),
because a second run would halve the *already halved* val set.

NOTE: files are written WITH a trailing newline. The original version used
'\n'.join(...) with no trailing newline, which silently dropped the last
entry in naive `while read` shell loops -- see DEVLOG Step 19.
"""
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset',
                    help='dataset root containing splits/')
parser.add_argument('--seed', type=int, default=42,
                    help='RNG seed (42 matches the original holo split)')
parser.add_argument('--force', action='store_true',
                    help='re-split even if test.txt already exists')
args = parser.parse_args()

random.seed(args.seed)

splits = os.path.join(args.root, 'splits')
val_path = os.path.join(splits, 'val.txt')
test_path = os.path.join(splits, 'test.txt')
bak_path = val_path + '.bak'

if os.path.exists(test_path) and not args.force:
    raise SystemExit(
        f'REFUSING: {test_path} already exists.\n'
        'Re-running would halve an already-halved val set. Pass --force if '
        'you really mean it (restore val.txt from val.txt.bak first).')

with open(val_path) as f:
    val_files = [l for l in f.read().splitlines() if l.strip()]
print(f'Current val: {len(val_files)}')

# back up before the destructive rewrite
if not os.path.exists(bak_path):
    with open(bak_path, 'w') as f:
        f.write('\n'.join(val_files) + '\n')
    print(f'Backed up original val.txt -> {bak_path}')

random.shuffle(val_files)
n_test = len(val_files) // 2
test_files = sorted(val_files[:n_test])
new_val_files = sorted(val_files[n_test:])
print(f'New val: {len(new_val_files)}, new test: {len(test_files)}')

assert not (set(test_files) & set(new_val_files)), 'val/test overlap!'

with open(val_path, 'w') as f:
    f.write('\n'.join(new_val_files) + '\n')
with open(test_path, 'w') as f:
    f.write('\n'.join(test_files) + '\n')
print('Done. Now re-run create_symlinks.py to rebuild the split dirs.')
