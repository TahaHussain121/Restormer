"""1e5/1e7 pairs with the aligned render STACKED into the LQ tensor.

WHY STACKED, AND NOT A THIRD DICT KEY

`basicsr/train.py` sub-crops and mini-batch-subsamples only `train_data['lq']`
and `train_data['gt']` (lines 255-269), and then hands the model exactly
`{'lq': lq, 'gt': gt}`. A third stream returned under its own key would arrive
at the model at full 256x256 and full batch -- silently misaligned with the crop
the network actually sees. REPO_INVESTIGATION_REPORT section N.1 calls this "the
single most important pipeline fact" and "the highest-probability silent bug in
the whole design".

This dataset sidesteps it completely rather than guarding against it: the render
travels as CHANNEL 1 of the LQ tensor.

    lq[:, 0]  = 1e5 noisy radar   -> Restormer input
    lq[:, 1]  = render            -> DINO input

Every spatial crop and every batch subsample `train.py` performs slices that one
tensor, so both channels receive byte-identically the same window and the same
rows -- not "the same coordinates recomputed", but literally the same operation
on the same object. The architecture splits the channels inside `forward`.
No change to train.py is required, which also means the two experiments already
training against it are untouched.

CROP AND AUGMENTATION

Drawn here, once per sample, and applied to gt / lq / render together, using the
same primitives and the same RNG call order as the stock pipeline:

    top  = random.randint(0, h - gt_size)     as paired_random_crop draws it
    left = random.randint(0, w - gt_size)
    flag = random.randint(0, 7)               as random_augmentation draws it
    data_augmentation(x, flag)                the identical dihedral op

They are drawn explicitly (rather than inside the helpers) for one reason: so
`return_crop_meta` can hand the chosen values back and the smoke test can PROVE
the alignment by re-deriving the render crop from disk, instead of asserting it
in a comment.

RENDER CHANNELS. `renders_blackbg` PNGs are 8-bit RGB with all three channels
identical (verified in the investigation report, section D). Channel 0 is kept
and `dino_preprocess` repeats it back to 3 -- numerically the same tensor DINO
saw in Phase 1/2, at a third of the memory. The smoke test asserts the three
channels really are identical on real samples.
"""

import os
import random

import numpy as np
from torch.utils import data as data
import torch

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import data_augmentation
from basicsr.utils import (FileClient, imfrombytes, imfrombytes_uint16,
                           img2tensor, padding)


class Dataset_PairedImage_uint16_RenderStacked(data.Dataset):
    """Returns {'lq': [2,H,W], 'gt': [1,H,W], ...}, lq = (radar, render).

    Config keys: as Dataset_PairedImage_uint16, plus
        dataroot_render (str): folder of render PNGs, filenames matching GT.
        return_crop_meta (bool, optional): also return the drawn crop offsets
            and augmentation flag. Smoke tests only -- collating extra keys is
            harmless but pointless during training.
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
        self.return_crop_meta = opt.get('return_crop_meta', False)

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def _render_path(self, gt_path):
        # the render shares the filename with gt/lq (the whole pairing mechanism)
        return os.path.join(self.render_folder, os.path.basename(gt_path))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        render_path = self._render_path(gt_path)
        if not os.path.isfile(render_path):
            raise FileNotFoundError(f'render missing for {gt_path}: {render_path}')

        img_gt = imfrombytes_uint16(self.file_client.get(gt_path, 'gt'))
        img_lq = imfrombytes_uint16(self.file_client.get(lq_path, 'lq'))
        # 8-bit render -> [0,1] float32 HWC 3ch; keep one channel (all identical)
        img_render = imfrombytes(self.file_client.get(render_path, 'lq'),
                                 flag='color', float32=True)[:, :, :1]

        top = left = flag = -1
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_render, _ = padding(img_render, img_render, gt_size)

            # --- ONE crop, applied to all three (paired_random_crop's draw) ---
            h, w, _ = img_lq.shape
            if h < gt_size or w < gt_size:
                raise ValueError(f'{lq_path}: {h}x{w} smaller than {gt_size}')
            top = random.randint(0, h - gt_size)
            left = random.randint(0, w - gt_size)
            img_gt = img_gt[top:top + gt_size, left:left + gt_size, ...]
            img_lq = img_lq[top:top + gt_size, left:left + gt_size, ...]
            img_render = img_render[top:top + gt_size, left:left + gt_size, ...]

            # --- ONE dihedral op, applied to all three (random_augmentation) --
            if self.geometric_augs:
                flag = random.randint(0, 7)
                img_gt = data_augmentation(img_gt, flag).copy()
                img_lq = data_augmentation(img_lq, flag).copy()
                img_render = data_augmentation(img_render, flag).copy()

        img_gt, img_lq, img_render = img2tensor(
            [img_gt, img_lq, img_render], bgr2rgb=False, float32=True)

        # radar in channel 0, render in channel 1 -- one tensor, one geometry
        lq_stacked = torch.cat([img_lq, img_render], dim=0)

        out = {'lq': lq_stacked, 'gt': img_gt,
               'lq_path': lq_path, 'gt_path': gt_path}
        if self.return_crop_meta:
            out.update({'crop_top': top, 'crop_left': left, 'aug_flag': flag,
                        'render_path': render_path})
        return out

    def __len__(self):
        return len(self.paths)
