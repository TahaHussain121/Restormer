"""Calibrate the synthetic corruption recipe using TRAINING data only.

Read-only: training clean targets and the EXISTING frozen-E0 train cache
(experiments/Holo_E0_frozen_noisy_output_residual_refiner/e0_cache, memory-
mapped). Writes one JSON inside the experiment root.

It measures, on a fixed subset of training images, the shape statistics of the
regions E0 actually removes -- the documented missing-structure rule
    target > 0.05 and E0 < 0.02
-- so that the synthetic holes are of a comparable size and elongation. No
validation or test image is read. Nothing here looks at a completion output.
"""
import json, os, sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                   # noqa: E402

N_IMAGES = 400          # fixed subset: the first 400 training ids, sorted
MISS_GT, MISS_E0 = 0.05, 0.02


def main():
    ids = mc.split_ids('train')[:N_IMAGES]
    all_ids = mc.split_ids('train')
    cache, cids, _ = mc.load_cache('train', all_ids, mmap=True)
    idx = {i: k for k, i in enumerate(cids)}

    areas, widths, heights, elong, frac_missing, fg_fracs, comp_per_img = [], [], [], [], [], [], []
    att_ratio = []
    for i in ids:
        g = mc.to_unit(mc.load_uint16(os.path.join(mc.DATASET, 'train_clean', f'{i}.png')))
        e0 = np.asarray(cache[idx[i]], dtype=np.float32)
        miss = (g > MISS_GT) & (e0 < MISS_E0)
        fg_fracs.append(float((g > 0.01).mean()))
        frac_missing.append(float(miss.mean()))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8), 8)
        comp_per_img.append(n - 1)
        for c in range(1, n):
            a = int(stats[c, cv2.CC_STAT_AREA])
            if a < 4:
                continue
            w, h = int(stats[c, cv2.CC_STAT_WIDTH]), int(stats[c, cv2.CC_STAT_HEIGHT])
            areas.append(a); widths.append(w); heights.append(h)
            elong.append(max(w, h) / max(1, min(w, h)))
        # how much intensity survives where E0 weakens (but does not erase) structure
        weak = (g > MISS_GT) & (e0 >= MISS_E0)
        if weak.any():
            att_ratio.append(float((e0[weak] / g[weak]).mean()))

    def q(v, name):
        v = np.asarray(v, dtype=np.float64)
        return {'n': int(v.size), 'mean': float(v.mean()), 'p10': float(np.percentile(v, 10)),
                'p25': float(np.percentile(v, 25)), 'median': float(np.median(v)),
                'p75': float(np.percentile(v, 75)), 'p90': float(np.percentile(v, 90)),
                'p99': float(np.percentile(v, 99)), 'max': float(v.max()), 'name': name}

    out = {'source': 'training split only, first %d ids' % N_IMAGES,
           'missing_rule': 'target > %.2f and E0 < %.2f' % (MISS_GT, MISS_E0),
           'e0_cache': mc.cache_paths('train')['npy'],
           'component_area_px': q(areas, 'area'),
           'component_width_px': q(widths, 'width'),
           'component_height_px': q(heights, 'height'),
           'component_elongation': q(elong, 'max/min bbox side'),
           'components_per_image': q(comp_per_img, 'n components (area>=1)'),
           'frac_pixels_missing_per_image': q(frac_missing, 'fraction'),
           'foreground_frac_per_image': q(fg_fracs, 'gt > 0.01'),
           'e0_over_gt_where_weakened_not_erased': q(att_ratio, 'mean ratio'),
           'created': mc.now(), 'git_commit': mc.git_commit()}
    mc.write_json_exclusive(os.path.join(mc.ROOT, 'results', 'recipe_calibration.json'), out)
    print(json.dumps({k: v for k, v in out.items() if isinstance(v, dict)}, indent=1))


if __name__ == '__main__':
    main()
