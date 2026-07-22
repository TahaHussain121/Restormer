"""Materialize <split>_<variant>/ dirs as symlinks driven by splits/*.txt.

For every split (train/val/test) present in splits/ and every requested
variant (clean, noisy, verynoisy, ...), builds <split>_<variant>/ containing
one symlink per filename listed in that split's txt, pointing at the master
<variant>/ dir.

    # original holo dataset (clean + noisy)
    python create_symlinks.py --root .../holo_image_dataset

    # holographic dataset, including the verynoisy variant
    python create_symlinks.py --root .../holographic_image_dataset \
        --variants clean noisy verynoisy

Idempotent: existing symlinks in the output dirs are cleared and rebuilt.

Some dirs on the holographic dataset were originally populated with real file
COPIES rather than symlinks. Replacing those means deleting files, so this
script refuses to remove a real file unless an identically-named file exists
in the master <variant>/ dir (i.e. the content is recoverable). Pass
--allow-replace-files to opt in to that removal.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset')
parser.add_argument('--variants', nargs='+', default=['clean', 'noisy'],
                    help='master image dirs to build splits for')
parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
parser.add_argument('--allow-replace-files', action='store_true',
                    help='permit deleting real files (only ever done when the '
                         'same filename exists in the master variant dir)')
args = parser.parse_args()

splits_dir = os.path.join(args.root, 'splits')

for split in args.splits:
    txt_path = os.path.join(splits_dir, split + '.txt')
    if not os.path.exists(txt_path):
        print(f'[SKIP] {split}.txt not found')
        continue

    with open(txt_path) as f:
        files = [l for l in f.read().splitlines() if l.strip()]

    for variant in args.variants:
        src_dir = os.path.join(args.root, variant)
        out_dir = os.path.join(args.root, f'{split}_{variant}')
        if not os.path.isdir(src_dir):
            print(f'[SKIP] master dir missing: {src_dir}')
            continue
        os.makedirs(out_dir, exist_ok=True)

        # clear the output dir (symlinks freely; real files only when the
        # content is recoverable from the master dir and the user opted in)
        for existing in os.listdir(out_dir):
            p = os.path.join(out_dir, existing)
            if os.path.islink(p):
                os.unlink(p)
            elif os.path.isfile(p):
                if not os.path.exists(os.path.join(src_dir, existing)):
                    raise SystemExit(
                        f'REFUSING to delete {p}: no matching file in {src_dir}. '
                        'This file is not recoverable -- resolve by hand.')
                if not args.allow_replace_files:
                    raise SystemExit(
                        f'{out_dir} contains real files, not symlinks. Re-run with '
                        '--allow-replace-files to replace them with symlinks '
                        f'(verified recoverable from {src_dir}).')
                os.remove(p)

        n_ok, n_miss = 0, 0
        for fname in files:
            src = os.path.join(src_dir, fname)
            if not os.path.exists(src):
                print(f'  [WARN] missing source: {src}')
                n_miss += 1
                continue
            os.symlink(src, os.path.join(out_dir, fname))
            n_ok += 1

        print(f'{split}_{variant}: {n_ok} symlinks'
              + (f'  ({n_miss} missing)' if n_miss else '')
              + f'  ->  {out_dir}')
