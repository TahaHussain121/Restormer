import os

root = '/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset'

splits = {}
for name in ('train', 'val', 'test'):
    p = os.path.join(root, 'splits', name + '.txt')
    splits[name] = set(open(p).read().splitlines()) if os.path.exists(p) else set()
    d_clean = os.path.join(root, name + '_clean')
    d_noisy = os.path.join(root, name + '_noisy')
    n_c = len(os.listdir(d_clean)) if os.path.isdir(d_clean) else 0
    n_n = len(os.listdir(d_noisy)) if os.path.isdir(d_noisy) else 0
    print(f'{name}: txt={len(splits[name])}  clean_symlinks={n_c}  noisy_symlinks={n_n}')

tv = splits['train'] & splits['val']
tt = splits['train'] & splits['test']
vt = splits['val']   & splits['test']
print(f'train∩val={len(tv)}  train∩test={len(tt)}  val∩test={len(vt)}  (all should be 0)')
