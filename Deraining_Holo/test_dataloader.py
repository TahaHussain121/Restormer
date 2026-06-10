from basicsr.data.paired_image_uint16_dataset import Dataset_PairedImage_uint16

opt = {
    'dataroot_gt': '/mnt/home/taha.hussain@ad.five-d.ai/holographic_image_dataset/clean',
    'dataroot_lq': '/mnt/home/taha.hussain@ad.five-d.ai/holographic_image_dataset/noisy',
    'io_backend': {'type': 'disk'},
    'phase': 'train',
    'gt_size': 256,
    'use_hflip': False,
    'use_rot': False,
    'scale': 1
}

dataset = Dataset_PairedImage_uint16(opt)
sample = dataset[0]
img_gt = sample['gt']
img_lq = sample['lq']

print(f'gt shape: {img_gt.shape}')   # expect [1, 256, 256]
print(f'lq shape: {img_lq.shape}')   # expect [1, 256, 256]
print(f'gt dtype: {img_gt.dtype}')   # expect float32
print(f'gt min: {img_gt.min():.4f}  max: {img_gt.max():.4f}')  # expect [0, 1]
print(f'lq min: {img_lq.min():.4f}  max: {img_lq.max():.4f}')  # expect [0, 1]
print('PASS' if img_gt.shape[0] == 1 else 'FAIL — wrong channels')
