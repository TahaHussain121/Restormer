"""Metric definitions, fixed in docs/PROTOCOL.md section 6 BEFORE any output of
this experiment was inspected.

All metrics operate on quantised uint16 predictions expressed in [0,1] and are
restricted to a boolean mask. Intensity measures (fill ratio, fraction above
0.02, hit rate) are kept separate from structure measures (target correlation,
detail correlation, gradient NCC), because a uniform brightening raises the
former and cannot raise the latter.
"""
import numpy as np
import cv2


def box5(a):
    return cv2.blur(a.astype(np.float32), (5, 5), borderType=cv2.BORDER_REFLECT).astype(np.float64)


def grad_mag(a):
    a32 = a.astype(np.float32)
    gx = cv2.Sobel(a32, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(a32, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(gx * gx + gy * gy).astype(np.float64)


def pearson(a, b):
    """Centred correlation; None when either vector is constant (counted, never
    scored as 0)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.size < 2:
        return None
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else None


def ncc(a, b):
    """Uncentred normalised cross-correlation (for non-negative gradient
    magnitudes); None if either side is all zero."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else None


def window_metrics(pred, gt, mask, base=None, pre=None):
    """pred/gt/base: full-frame float64 in [0,1]; mask: full-frame bool.

    `pre` may carry cached high-pass / gradient maps of gt to save work.
    Returns a dict; None where a metric is undefined for this window.
    """
    m = mask
    n = int(m.sum())
    out = {'n_mask_px': n}
    if n == 0:
        return out
    p, g = pred[m], gt[m]
    d = p - g
    out['mae'] = float(np.abs(d).mean())
    mse = float((d * d).mean())
    out['psnr'] = float('inf') if mse == 0 else float(10. * np.log10(1. / mse))
    gs = float(g.sum())
    out['fill_ratio'] = float(p.sum() / gs) if gs > 0 else None
    out['frac_above_0p02'] = float((p > 0.02).mean())
    out['frac_above_0p05'] = float((p > 0.05).mean())
    out['overshoot_frac'] = float((d > 0.05).mean())
    out['mean_positive_excess'] = float(np.maximum(d, 0).mean())
    out['mean_signed_change_vs_base'] = (float((pred[m] - base[m]).mean())
                                         if base is not None else None)
    # structure / shape
    out['target_corr'] = pearson(p, g)
    hp_g = pre['hp_gt'] if pre else gt - box5(gt)
    out['detail_corr'] = pearson(pred[m] - box5(pred)[m], hp_g[m])
    gg = pre['grad_gt'] if pre else grad_mag(gt)
    out['grad_ncc'] = ncc(grad_mag(pred)[m], gg[m])
    tb = g > 0.05
    pb = p > 0.05
    inter = int((tb & pb).sum()); union = int((tb | pb).sum())
    out['iou_0p05'] = float(inter / union) if union > 0 else None
    out['hit_rate_0p05'] = float(pb[tb].mean()) if tb.any() else None
    if base is not None:
        outside = ~m
        out['max_abs_change_outside_mask'] = float(np.abs(pred - base)[outside].max())
    return out


HIGHER = {'psnr', 'fill_ratio', 'frac_above_0p02', 'frac_above_0p05', 'target_corr',
          'detail_corr', 'grad_ncc', 'iou_0p05', 'hit_rate_0p05'}
LOWER = {'mae', 'overshoot_frac', 'mean_positive_excess'}


def paired(a, b, higher_is_better, n_boot=5000, seed=0):
    """Paired difference b - a over windows/images with both values defined."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {'n': 0}
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, d.size).mean() for _ in range(n_boot)])
    better = (d > 0) if higher_is_better else (d < 0)
    return {'n': int(d.size), 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'mean_delta': float(d.mean()),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'n_better': int(better.sum()), 'n_worse': int((~better & (d != 0)).sum()),
            'n_equal': int((d == 0).sum()), 'higher_is_better': bool(higher_is_better)}
