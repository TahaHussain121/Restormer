import os

dataset_root = '/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset'
clean_dir = os.path.join(dataset_root, 'clean')
noisy_dir = os.path.join(dataset_root, 'noisy')
splits_dir = os.path.join(dataset_root, 'splits')

split_configs = {
    'train': ('train.txt', 'train_clean', 'train_noisy'),
    'val':   ('val.txt',   'val_clean',   'val_noisy'),
    'test':  ('test.txt',  'test_clean',  'test_noisy'),
}

for split, (txt, clean_out_name, noisy_out_name) in split_configs.items():
    txt_path = os.path.join(splits_dir, txt)
    if not os.path.exists(txt_path):
        print(f'[SKIP] {txt} not found, skipping {split} split')
        continue

    with open(txt_path) as f:
        files = f.read().splitlines()

    clean_out = os.path.join(dataset_root, clean_out_name)
    noisy_out = os.path.join(dataset_root, noisy_out_name)
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)

    # idempotent: remove stale symlinks before recreating
    for d in (clean_out, noisy_out):
        for existing in os.listdir(d):
            p = os.path.join(d, existing)
            if os.path.islink(p):
                os.unlink(p)

    n_ok = 0
    for fname in files:
        src_clean = os.path.join(clean_dir, fname)
        src_noisy = os.path.join(noisy_dir, fname)
        if not os.path.exists(src_clean):
            print(f'  [WARN] missing clean source: {src_clean}')
            continue
        if not os.path.exists(src_noisy):
            print(f'  [WARN] missing noisy source: {src_noisy}')
            continue
        os.symlink(src_clean, os.path.join(clean_out, fname))
        os.symlink(src_noisy, os.path.join(noisy_out, fname))
        n_ok += 1

    print(f'{split}_clean: {n_ok} symlinks  →  {clean_out}')
    print(f'{split}_noisy: {n_ok} symlinks  →  {noisy_out}')
