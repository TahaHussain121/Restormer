"""Cache the FROZEN E0-Fixed baseline's raw float32 full-frame outputs.

    python cache_e0_outputs.py --splits val train [test]

For every image of a split: the noisy 256x256 frame, normalised exactly as E0's
recorded evaluation normalises it (uint16 / 65535, float32, [1,1,256,256], no
padding, no crop, batch 1), is passed through E0 in eval mode under no_grad.
The output is stored BEFORE clamping and quantisation, as float32
[N,256,256], in the dataset's sorted-id order. Nothing else is duplicated: the
noisy frames and targets stay in the dataset directories.

Valid existing caches (complete, same E0 md5, same ids) are REUSED, never
rebuilt. A provenance JSON records the checkpoint, ids, preprocessing,
inference context, dtype, clipping statistics, and -- for val/test -- whether
quantising the cache reproduces E0's recorded reference predictions.
"""

import argparse
import csv
import hashlib
import os
import time

import numpy as np
import torch

import refiner_common as rc


def param_digest(net):
    h = hashlib.md5()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reference_check(split, ids, arr):
    """Quantise the cache exactly as predict_phase3 did and compare with E0's
    recorded full256 predictions and per-image PSNR."""
    ref_dir = rc.E0_REF_PRED.format(split=split)
    ref_csv = rc.E0_REF_CSV.format(split=split)
    if not (os.path.isdir(ref_dir) and os.path.isfile(ref_csv)):
        return {'available': False}
    with open(ref_csv) as f:
        ref_psnr = {r['filename']: float(r['psnr_full']) for r in csv.DictReader(f)}
    n_ident, max_lvl, n_pix, dps, mine = 0, 0, 0, [], []
    for k, i in enumerate(ids):
        q = rc.quantize_np(arr[k])
        ref = rc.load_uint16(os.path.join(ref_dir, f'{i}.png'))
        d = np.abs(q.astype(np.int64) - ref.astype(np.int64))
        n_ident += int(d.max() == 0)
        max_lvl = max(max_lvl, int(d.max()))
        n_pix += int((d > 0).sum())
        gt = rc.load_uint16(os.path.join(rc.DATASET, f'{split}_clean', f'{i}.png'))
        p = rc.psnr_u16(gt, q)
        mine.append(p)
        dps.append(p - ref_psnr[f'{i}.png'])
    return {'available': True, 'reference_pred_dir': ref_dir,
            'reference_csv': ref_csv, 'n_images': len(ids),
            'n_identical_images': n_ident,
            'n_differing_pixels': n_pix,
            'max_abs_level_diff_uint16': max_lvl,
            'mean_psnr_full_from_cache': float(np.mean(mine)),
            'mean_psnr_full_reference': float(np.mean(list(ref_psnr.values()))),
            'max_abs_per_image_psnr_diff_db': float(np.max(np.abs(dps)))}


def cache_is_valid(split, ids, md5):
    cp = rc.cache_paths(split)
    if not all(os.path.isfile(p) for p in cp.values()):
        return False
    try:
        rc.load_cache(split, ids, mmap=True)
    except SystemExit as e:
        print(f'  existing {split} cache rejected: {e}')
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', required=True,
                    choices=['train', 'val', 'test'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    rc.strict_fp32()
    net, info = rc.build_frozen_e0(args.device)
    assert not net.training
    assert not any(p.requires_grad for p in net.parameters())
    digest_before = param_digest(net)
    print(f'E0 {info["checkpoint"]}  iter {info["best_iter"]}  md5 {info["md5"]}')

    os.makedirs(rc.CACHE_DIR, exist_ok=True)
    for split in args.splits:
        ids = rc.split_ids(split)
        cp = rc.cache_paths(split)
        if cache_is_valid(split, ids, info['md5']):
            print(f'  {split}: valid cache exists, REUSED ({cp["npy"]})')
            continue

        n = len(ids)
        partial = cp['npy'] + '.partial'
        arr = None
        n_below = n_above = n_img_clip = 0
        gmin, gmax = np.inf, -np.inf
        t0 = time.time()
        with torch.no_grad():
            for k, image_id in enumerate(ids):
                lq = rc.load_uint16(os.path.join(rc.DATASET, f'{split}_verynoisy',
                                                 f'{image_id}.png'))
                if arr is None:
                    arr = np.lib.format.open_memmap(partial, mode='w+',
                                                    dtype=np.float32,
                                                    shape=(n,) + lq.shape)
                if lq.shape != arr.shape[1:]:
                    raise SystemExit(f'{image_id}: shape {lq.shape}')
                lq_f = lq.astype(np.float32) / 65535.
                t = torch.from_numpy(np.ascontiguousarray(lq_f))[None, None]
                o = net(t.to(args.device))
                raw = o[0, 0].float().cpu().numpy()
                if raw.shape != lq.shape:
                    raise SystemExit(f'{image_id}: output {raw.shape}')
                arr[k] = raw
                b, a = int((raw < 0).sum()), int((raw > 1).sum())
                n_below += b
                n_above += a
                n_img_clip += int(b + a > 0)
                gmin, gmax = min(gmin, float(raw.min())), max(gmax, float(raw.max()))
                if (k + 1) % 500 == 0 or k + 1 == n:
                    print(f'  {split} {k + 1}/{n}  {time.time() - t0:.0f}s', flush=True)
        arr.flush()
        del arr
        os.replace(partial, cp['npy'])
        with open(cp['ids'], 'w') as f:
            f.write('\n'.join(ids) + '\n')

        arr = np.load(cp['npy'], mmap_mode='r')
        ref = reference_check(split, ids, arr)
        npx = n * arr.shape[1] * arr.shape[2]
        prov = {
            'experiment': rc.EXP,
            'what': 'raw float32 output of the frozen E0-Fixed baseline, '
                    'full 256x256 frames, BEFORE clamping and quantisation',
            'e0': info,
            'split': split, 'n_images': n,
            'ids_file': cp['ids'], 'ids_sha256': rc.file_digest(cp['ids'], 'sha256'),
            'input_dir': os.path.join(rc.DATASET, f'{split}_verynoisy'),
            'preprocessing': 'cv2.IMREAD_UNCHANGED uint16 -> float32 / 65535 '
                             '(identical to predict_phase3 and basicsr '
                             'imfrombytes_uint16); tensor [1,1,256,256]; no '
                             'padding, no crop, no augmentation',
            'inference_context': {
                'protocol': 'full256 (whole frame in one pass, as E0\'s recorded '
                            'full256 evaluation)',
                'batch_size': 1, 'module_mode': 'eval', 'grad': 'torch.no_grad',
                'e0_requires_grad_any': False,
                'e0_param_digest_before': digest_before,
                **rc.device_record(args.device)},
            'output': {'file': cp['npy'], 'dtype': 'float32',
                       'shape': list(arr.shape), 'clamped': False,
                       'quantised': False,
                       'sha256': rc.file_digest(cp['npy'], 'sha256'),
                       'bytes': os.path.getsize(cp['npy'])},
            'clipping_if_clamped_to_0_1': {
                'frac_pixels_below_0': n_below / npx,
                'frac_pixels_above_1': n_above / npx,
                'n_images_with_any_out_of_range': n_img_clip,
                'global_min': gmin, 'global_max': gmax},
            'reference_check': ref,
            'seconds': time.time() - t0,
            'complete': True, 'created': rc.now(),
        }
        rc.write_json(cp['prov'], prov)
        print(f'  {split}: wrote {cp["npy"]}  ({prov["output"]["bytes"] / 1e9:.2f} GB)')
        print(f'  {split}: clipping {prov["clipping_if_clamped_to_0_1"]}')
        print(f'  {split}: reference check {ref}')

    digest_after = param_digest(net)
    if digest_after != digest_before:
        raise SystemExit('E0 parameters CHANGED during caching')
    print(f'E0 parameters unchanged (digest {digest_after})')


if __name__ == '__main__':
    main()
