"""Non-learned masked fills. They receive exactly the same mask as the learned
completion, so the comparison isolates how much of any improvement comes simply
from knowing WHERE to fill.

  ring   : constant fill with the median of the supplied image in a 3-px ring
           around the mask (the registered control).
  diff   : Laplace / diffusion fill -- solve the discrete Laplace equation
           inside the mask with the supplied image on the boundary (200 Jacobi
           sweeps). Added alongside the registered controls because it is a
           float-precision, structure-free interpolant; reported, not selected.
  telea  : cv2.inpaint (Telea, radius 3), the registered control. OpenCV's
           inpaint takes 8-bit input only, so the image is scaled to uint8 and
           back; that 1/255 quantisation is a property of this control and is
           stated wherever its numbers appear.

Each returns a full-frame float image that is EXACTLY the input outside the mask.
"""
import numpy as np
import cv2

K3 = np.ones((3, 3), np.uint8)


def _restore_outside(out, img, mask):
    out = np.asarray(out, np.float64).copy()
    out[~mask] = img[~mask]
    return out


def ring_fill(img, mask, ring_px=3):
    if not mask.any():
        return img.copy()
    ring = (cv2.dilate(mask.astype(np.uint8), K3, iterations=ring_px) > 0) & (~mask)
    val = float(np.median(img[ring])) if ring.any() else float(np.median(img))
    out = img.copy()
    out[mask] = val
    return _restore_outside(out, img, mask)


def diffusion_fill(img, mask, sweeps=200):
    if not mask.any():
        return img.copy()
    u = img.copy()
    u[mask] = float(img[~mask].mean()) if (~mask).any() else 0.0
    m = mask
    for _ in range(sweeps):
        s = np.zeros_like(u)
        s[1:, :] += u[:-1, :]; s[:-1, :] += u[1:, :]
        s[:, 1:] += u[:, :-1]; s[:, :-1] += u[:, 1:]
        c = np.zeros_like(u)
        c[1:, :] += 1; c[:-1, :] += 1; c[:, 1:] += 1; c[:, :-1] += 1
        u = np.where(m, s / c, u)
    return _restore_outside(u, img, mask)


def telea_fill(img, mask, radius=3):
    if not mask.any():
        return img.copy()
    u8 = np.clip(img * 255., 0, 255).astype(np.uint8)
    out8 = cv2.inpaint(u8, mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return _restore_outside(out8.astype(np.float64) / 255., img, mask)
