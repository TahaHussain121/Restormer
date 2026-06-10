import os

dataset_root = '/mnt/home/taha.hussain@ad.five-d.ai/holographic_image_dataset'
clean_dir = os.path.join(dataset_root, 'clean')
noisy_dir = os.path.join(dataset_root, 'noisy')
splits_dir = os.path.join(dataset_root, 'splits')

split_dirs = {
    'train': {
        'files': open(os.path.join(splits_dir, 'train.txt')).read().splitlines(),
        'clean_out': os.path.join(dataset_root, 'train_clean'),
        'noisy_out': os.path.join(dataset_root, 'train_noisy'),
    },
    'val': {
        'files': open(os.path.join(splits_dir, 'val.txt')).read().splitlines(),
        'clean_out': os.path.join(dataset_root, 'val_clean'),
        'noisy_out': os.path.join(dataset_root, 'val_noisy'),
    },
}

for split, cfg in split_dirs.items():
    os.makedirs(cfg['clean_out'], exist_ok=True)
    os.makedirs(cfg['noisy_out'], exist_ok=True)
    for fname in cfg['files']:
        src_clean = os.path.join(clean_dir, fname)
        src_noisy = os.path.join(noisy_dir, fname)
        dst_clean = os.path.join(cfg['clean_out'], fname)
        dst_noisy = os.path.join(cfg['noisy_out'], fname)
        if not os.path.exists(dst_clean):
            os.symlink(src_clean, dst_clean)
        if not os.path.exists(dst_noisy):
            os.symlink(src_noisy, dst_noisy)
    n_clean = len(os.listdir(cfg['clean_out']))
    n_noisy = len(os.listdir(cfg['noisy_out']))
    print(f'{split}_clean: {n_clean} symlinks')
    print(f'{split}_noisy: {n_noisy} symlinks')
