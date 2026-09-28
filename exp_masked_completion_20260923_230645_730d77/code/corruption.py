"""The FIXED synthetic corruption recipe. Chosen once, from TRAINING data only
(results/recipe_calibration.json), and not tuned afterwards.

What it does, per clean target G (unit float32, 256x256):

  n_regions ~ randint(1, 5)   regions per image
  each region is one of
      ellipse          compact hole                       p = 0.35
      rectangle        compact hole, axis aligned          p = 0.20
      band             oriented bar -> contiguous missing
                       section of a structure              p = 0.45
  size: area ~ loguniform(25, 900) px, elongation ~ U(1, 5), angle ~ U(0, pi)
        (calibration: real E0-missing components have median area 26 px,
         p75 68, p90 179, median bbox 6x8, elongation median 1.8 / p90 5.4,
         ~29 components per image covering a median 2.2% of pixels)
  placement:
      p = 0.75  centre drawn from the FOREGROUND pixels (G > 0.01)
      p = 0.25  centre drawn from deep BACKGROUND pixels (G <= 0.01 and at
                least 6 px from any foreground pixel) -> a BACKGROUND-MASK
                example: the mask is supplied but nothing was damaged, so a
                mask never implies that an object belongs there
  damage inside a foreground-placed region:
      p = 0.60  erase      Y_dmg = G * u,  u ~ U(0.00, 0.05)   (near-zero,
                           the E0 suppression signature: G > 0.05, E0 < 0.02)
      p = 0.40  attenuate  Y_dmg = G * u,  u ~ U(0.05, 0.40)
      one factor u per region, so the damage is a suppression, not only an
      ideal rectangular hole
  background-placed regions are NOT damaged (G there is already ~0).

The mask M is the union of all regions, damaged or not. Outside M the image is
exactly G. Ground truth is used for supervision and to build the damage; it is
NEVER an input channel.
"""

import numpy as np
import cv2

N_REGIONS_LO, N_REGIONS_HI = 1, 5           # inclusive
AREA_LO, AREA_HI = 25.0, 900.0
ELONG_LO, ELONG_HI = 1.0, 5.0
P_SHAPE = {'ellipse': 0.35, 'rect': 0.20, 'band': 0.45}
P_BACKGROUND_REGION = 0.25
P_ERASE = 0.60
ERASE_LO, ERASE_HI = 0.00, 0.05
ATTEN_LO, ATTEN_HI = 0.05, 0.40
FG_THRESHOLD = 0.01
BG_MARGIN = 6                                # px from any foreground pixel
RECIPE_VERSION = 'fixed-2026-09-23'


def _pick(rng, probs):
    keys = list(probs)
    return keys[int(rng.choice(len(keys), p=[probs[k] for k in keys]))]


def _region_mask(rng, shape_kind, cy, cx, H, W):
    """Rasterise one region into a uint8 mask of the full frame."""
    area = float(np.exp(rng.uniform(np.log(AREA_LO), np.log(AREA_HI))))
    elong = float(rng.uniform(ELONG_LO, ELONG_HI))
    ang = float(rng.uniform(0, np.pi))
    m = np.zeros((H, W), np.uint8)
    if shape_kind == 'band':
        # long thin bar: length = elongation * width, area preserved
        wid = max(2.0, np.sqrt(area / max(elong, 1e-6)))
        length = max(wid, area / wid)
        rect = ((cx, cy), (length, wid), np.degrees(ang))
        cv2.fillConvexPoly(m, np.int32(np.round(cv2.boxPoints(rect))), 1)
    elif shape_kind == 'rect':
        wid = max(2.0, np.sqrt(area / max(elong, 1e-6)))
        hgt = max(2.0, area / wid)
        y0, x0 = int(round(cy - hgt / 2)), int(round(cx - wid / 2))
        cv2.rectangle(m, (x0, y0), (int(round(x0 + wid)), int(round(y0 + hgt))), 1, -1)
    else:                                     # ellipse
        b = max(1.5, np.sqrt(area / (np.pi * max(elong, 1e-6))))
        a = max(b, area / (np.pi * b))
        cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                    (int(round(a)), int(round(b))), np.degrees(ang), 0, 360, 1, -1)
    return m


def corrupt_one(g, rng):
    """g: float32 [H,W] clean target in [0,1]. Returns (damaged, mask uint8, info)."""
    H, W = g.shape
    fg = g > FG_THRESHOLD
    fg_idx = np.flatnonzero(fg)
    far_bg = (cv2.dilate(fg.astype(np.uint8), np.ones((2 * BG_MARGIN + 1,) * 2, np.uint8)) == 0)
    bg_idx = np.flatnonzero(far_bg)
    dmg = g.copy()
    mask = np.zeros((H, W), np.uint8)
    regions = []
    n = int(rng.integers(N_REGIONS_LO, N_REGIONS_HI + 1))
    for _ in range(n):
        want_bg = (rng.random() < P_BACKGROUND_REGION) and bg_idx.size > 0
        pool = bg_idx if want_bg else fg_idx
        if pool.size == 0:
            pool = bg_idx if bg_idx.size else np.arange(H * W)
            want_bg = True
        p = int(rng.choice(pool))
        cy, cx = divmod(p, W)
        kind = _pick(rng, P_SHAPE)
        rm = _region_mask(rng, kind, cy, cx, H, W)
        if not rm.any():
            continue
        if want_bg:
            factor = None                      # background region: no damage
        else:
            if rng.random() < P_ERASE:
                factor = float(rng.uniform(ERASE_LO, ERASE_HI))
                mode = 'erase'
            else:
                factor = float(rng.uniform(ATTEN_LO, ATTEN_HI))
                mode = 'attenuate'
            sel = rm.astype(bool)
            dmg[sel] = dmg[sel] * factor
        mask |= rm
        regions.append({'kind': kind, 'background_region': bool(want_bg),
                        'cy': int(cy), 'cx': int(cx), 'area_px': int(rm.sum()),
                        'factor': factor,
                        'mode': 'none' if want_bg else mode})
    info = {'n_regions': len(regions), 'mask_frac': float(mask.mean()),
            'regions': regions}
    return dmg.astype(np.float32), mask, info


def corrupt_batch(gs, rng):
    d, m, infos = [], [], []
    for g in gs:
        a, b, i = corrupt_one(g, rng)
        d.append(a); m.append(b); infos.append(i)
    return (np.stack(d).astype(np.float32),
            np.stack(m).astype(np.float32), infos)


def recipe_record():
    return {'version': RECIPE_VERSION, 'n_regions': [N_REGIONS_LO, N_REGIONS_HI],
            'area_px_loguniform': [AREA_LO, AREA_HI],
            'elongation_uniform': [ELONG_LO, ELONG_HI],
            'shape_probs': P_SHAPE, 'p_background_region': P_BACKGROUND_REGION,
            'p_erase_given_foreground': P_ERASE,
            'erase_factor_uniform': [ERASE_LO, ERASE_HI],
            'attenuate_factor_uniform': [ATTEN_LO, ATTEN_HI],
            'fg_threshold': FG_THRESHOLD, 'background_margin_px': BG_MARGIN,
            'doc': __doc__}
