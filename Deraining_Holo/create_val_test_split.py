import os, random
random.seed(42)

root = '/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset'
splits = os.path.join(root, 'splits')

with open(os.path.join(splits, 'val.txt')) as f:
    val_files = f.read().splitlines()
print(f'Current val: {len(val_files)}')

random.shuffle(val_files)
n_test = len(val_files) // 2
test_files = sorted(val_files[:n_test])
new_val_files = sorted(val_files[n_test:])
print(f'New val: {len(new_val_files)}, new test: {len(test_files)}')

with open(os.path.join(splits, 'val.txt'), 'w') as f:
    f.write('\n'.join(new_val_files))
with open(os.path.join(splits, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_files))
print('Done')
