from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, imfrombytes_uint16, img2tensor, padding

import os


class Dataset_PairedImage_uint16_Render(data.Dataset):
    """uint16 GT/LQ pairs PLUS an aligned 3-channel render for DINO.

    Same as Dataset_PairedImage_uint16, but additionally loads a per-image
    'render' (e.g. the black-background render) and applies the IDENTICAL crop
    and geometric augmentation as the LQ/GT, so the render DINO sees covers the
    exact same region Restormer processes. Returned under key 'dino'.

    Extra config key:
        dataroot_render (str): folder of render PNGs, filenames matching GT/LQ.

    The render is loaded as a normal 8-bit image (cv2 -> [0,1], HWC, 3ch) via
    imfrombytes; renders here are grayscale-on-black so channel order is
    irrelevant. GT/LQ still use the uint16 loader (/65535).
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.render_folder = opt['dataroot_render']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # folder mode only (matches how the holo splits are set up)
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def _render_path(self, gt_path):
        # render shares the filename with gt/lq (verified: splits are 1:1).
        return os.path.join(self.render_folder, os.path.basename(gt_path))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        img_gt = imfrombytes_uint16(self.file_client.get(gt_path, 'gt'))

        lq_path = self.paths[index]['lq_path']
        img_lq = imfrombytes_uint16(self.file_client.get(lq_path, 'lq'))

        render_path = self._render_path(gt_path)
        # 8-bit render -> [0,1] float32, HWC, 3 channels (bgr2rgb irrelevant: grayscale)
        img_render = imfrombytes(self.file_client.get(render_path, 'lq'),
                                 flag='color', float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_render, _ = padding(img_render, img_render, gt_size)

            # crop lq AND render with the SAME offset (scale=1 -> gt uses it too)
            img_gt, (img_lq, img_render) = paired_random_crop(
                img_gt, [img_lq, img_render], gt_size, scale, gt_path)

            if self.geometric_augs:
                # variadic -> identical flip/rotation applied to all three
                img_gt, img_lq, img_render = random_augmentation(
                    img_gt, img_lq, img_render)

        img_gt, img_lq, img_render = img2tensor(
            [img_gt, img_lq, img_render], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'dino': img_render,        # 3-channel render for DINO
            'lq_path': lq_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paths)
