import os
import random

random.seed(42)

dataset_root = '/mnt/home/taha.hussain@ad.five-d.ai/holographic_image_dataset'
clean_dir = os.path.join(dataset_root, 'clean')
noisy_dir = os.path.join(dataset_root, 'noisy')

all_files = sorted(os.listdir(clean_dir))
random.shuffle(all_files)

n_val = int(len(all_files) * 0.1)
val_files = all_files[:n_val]
train_files = all_files[n_val:]

print(f'Total: {len(all_files)}')
print(f'Train: {len(train_files)}')
print(f'Val:   {len(val_files)}')

splits_dir = os.path.join(dataset_root, 'splits')
os.makedirs(splits_dir, exist_ok=True)

with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_files))

with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_files))

print(f'Split files written to {splits_dir}')
print(f'First 5 train: {train_files[:5]}')
print(f'First 5 val:   {val_files[:5]}')
