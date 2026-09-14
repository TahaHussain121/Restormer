"""Fix the qualitative validation cases BEFORE any refiner training.

Run once, before training. It reads only the validation split and E0's recorded
full256 validation predictions (the same checkpoint, uint16-quantised), never
the test split and never a refiner output. The output file is tracked and is
not edited after training starts.

A 48x48 window (stride 8, fully inside the frame) is a WEAK-STRUCTURE window if
    0.03 <= mean(gt) <= 0.25   and   >= 30% of its pixels are foreground (gt > 0.01).
Inside it, with m = gt > 0.01:
    E0 retention  r_E = sum(E0 * m) / sum(gt * m)          (1 = fully kept)
    noisy support c_N = Pearson corr(GaussianBlur(noisy, sigma 2), gt) over the window
Two groups, each keeping the single best window per image:
    RECOVERY  r_E <= 0.6 and c_N >= 0.5    score = mean(gt) * (1 - r_E) * c_N
              E0 weakened a weak target structure the measurement still shows
    CONTROL   r_E <= 0.6 and c_N <= 0.2    score = mean(gt) * (1 - r_E)
              E0 weakened a weak target structure the measurement does NOT
              clearly show; a refiner that "restores" these is not using the
              measurement
Illustrative panel: the top 4 RECOVERY images, the top 2 CONTROL images, and
the 2 images whose E0 full256 PSNR is nearest the validation median (TYPICAL,
crop centred on the foreground centroid). Display crop 96x96 around each window.

The FULL recovery and control populations are also written, for a descriptive
window-level comparison after evaluation. Caveat fixed now: these windows are
chosen where E0 did badly, so any correction towards the target will look good
there; the informative quantity is the CONTRAST between the recovery and
control groups, not either group alone.
"""

import csv
import os

import cv2
import numpy as np

import refiner_common as rc

W, STRIDE, CROP = 48, 8, 96
WEAK_LO, WEAK_HI, MIN_FG = 0.03, 0.25, 0.30
MAX_RET, MIN_SUPPORT, MAX_SUPPORT_CTL, BLUR = 0.6, 0.5, 0.2, 2.0
N_REC, N_CTL, N_TYP = 4, 2, 2


def box(a):
    return cv2.boxFilter(a, -1, (W, W), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def corr(a, b):
    ma, mb = box(a), box(b)
    cov = box(a * b) - ma * mb
    va, vb = box(a * a) - ma ** 2, box(b * b) - mb ** 2
    return cov / np.sqrt(np.maximum(va * vb, 1e-20))


def crop_at(cy, cx, size=CROP, full=256):
    y0 = int(np.clip(cy - size // 2, 0, full - size))
    x0 = int(np.clip(cx - size // 2, 0, full - size))
    return [y0, x0, size]


def main():
    ids = rc.split_ids('val')
    ref = rc.E0_REF_PRED.format(split='val')
    with open(rc.E0_REF_CSV.format(split='val')) as f:
        e0_psnr = {r['filename'][:-4]: float(r['psnr_full']) for r in csv.DictReader(f)}
    cent = np.arange(W // 2, 256 - W // 2 + 1, STRIDE)
    sub = np.ix_(cent, cent)
    rec, ctl = [], []
    for i in ids:
        g = rc.load_uint16(os.path.join(rc.DATASET, 'val_clean', f'{i}.png')).astype(np.float64) / 65535.
        n = rc.load_uint16(os.path.join(rc.DATASET, 'val_verynoisy', f'{i}.png')).astype(np.float64) / 65535.
        e = rc.load_uint16(os.path.join(ref, f'{i}.png')).astype(np.float64) / 65535.
        m = (g > rc.FG_THRESHOLD).astype(np.float64)
        ns = cv2.GaussianBlur(n, (0, 0), BLUR)
        gm, fgf = box(g)[sub], box(m)[sub]
        r_e = (box(e * m) / np.maximum(box(g * m), 1e-12))[sub]
        c_n = corr(ns, g)[sub]
        c_e = corr(e, g)[sub]
        weak = (gm >= WEAK_LO) & (gm <= WEAK_HI) & (fgf >= MIN_FG)
        for group, mask, score in (
                (rec, weak & (r_e <= MAX_RET) & (c_n >= MIN_SUPPORT), gm * (1 - r_e) * c_n),
                (ctl, weak & (r_e <= MAX_RET) & (c_n <= MAX_SUPPORT_CTL), gm * (1 - r_e))):
            if not mask.any():
                continue
            s = np.where(mask, score, -np.inf)
            a, b = np.unravel_index(np.argmax(s), s.shape)
            group.append({'id': i, 'cy': int(cent[a]), 'cx': int(cent[b]),
                          'score': float(s[a, b]), 'gt_mean': float(gm[a, b]),
                          'fg_frac': float(fgf[a, b]),
                          'e0_retention': float(r_e[a, b]),
                          'noisy_support_corr': float(c_n[a, b]),
                          'e0_corr': float(c_e[a, b])})
    rec.sort(key=lambda r: -r['score'])
    ctl.sort(key=lambda r: -r['score'])

    cases = []
    for role, pool, k in (('recovery', rec, N_REC), ('control', ctl, N_CTL)):
        taken = 0
        for r in pool:
            if taken == k:
                break
            if any(c['id'] == r['id'] for c in cases):
                continue
            cases.append({'role': role, **r,
                          'crop_yx_size': crop_at(r['cy'], r['cx']),
                          'window_yx_size': [r['cy'] - W // 2, r['cx'] - W // 2, W]})
            taken += 1
    med = float(np.median(list(e0_psnr.values())))
    for i in sorted(e0_psnr, key=lambda k: abs(e0_psnr[k] - med)):
        if sum(c['role'] == 'typical' for c in cases) == N_TYP:
            break
        if any(c['id'] == i for c in cases):
            continue
        g = rc.load_uint16(os.path.join(rc.DATASET, 'val_clean', f'{i}.png')).astype(np.float64) / 65535.
        yy, xx = np.nonzero(g > rc.FG_THRESHOLD)
        cy, cx = int(yy.mean()), int(xx.mean())
        cases.append({'role': 'typical', 'id': i, 'cy': cy, 'cx': cx,
                      'e0_psnr_full': e0_psnr[i], 'val_median_e0_psnr_full': med,
                      'crop_yx_size': crop_at(cy, cx),
                      'window_yx_size': [int(np.clip(cy - W // 2, 0, 256 - W)),
                                         int(np.clip(cx - W // 2, 0, 256 - W)), W]})

    out = {
        'experiment': rc.EXP,
        'fixed_before_training': True,
        'split': 'val',
        'source_of_e0_outputs': ref + ' (the same validation-selected E0 '
                                'checkpoint, uint16-quantised full256 predictions)',
        'rule': __doc__,
        'params': {'window': W, 'stride': STRIDE, 'crop': CROP,
                   'weak_gt_mean': [WEAK_LO, WEAK_HI], 'min_fg_frac': MIN_FG,
                   'max_e0_retention': MAX_RET, 'min_noisy_support': MIN_SUPPORT,
                   'max_noisy_support_control': MAX_SUPPORT_CTL,
                   'noisy_blur_sigma': BLUR, 'fg_threshold': rc.FG_THRESHOLD},
        'n_images_with_recovery_window': len(rec),
        'n_images_with_control_window': len(ctl),
        'cases': cases,
        'population': {'recovery': rec, 'control': ctl},
        'created': rc.now(), 'git_commit': rc.git_commit(),
    }
    if os.path.exists(rc.CASES_JSON):
        raise SystemExit(f'{rc.CASES_JSON} exists; the case set is fixed once and '
                         f'never regenerated')
    rc.write_json(rc.CASES_JSON, out)
    print(f'recovery windows on {len(rec)} images, control windows on {len(ctl)} images')
    for c in cases:
        print(f"  {c['role']:9s} {c['id']}  crop {c['crop_yx_size']}  "
              f"r_E {c.get('e0_retention', float('nan')):.2f}  "
              f"c_N {c.get('noisy_support_corr', float('nan')):.2f}")


if __name__ == '__main__':
    main()
