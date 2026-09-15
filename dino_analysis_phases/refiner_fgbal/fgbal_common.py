"""Shared pieces of ONE follow-up to refiner_e0: the foreground-balanced refiner.

    Holo_E0_frozen_noisy_output_fgbalanced_refiner

Identical to refiner_e0 (same frozen E0 checkpoint, same verified float32
caches, same refiner architecture and initialisation, same optimiser, schedule,
budget, batch, seed, clipping, validation frequency, no augmentation) except:

  1. LOSS      0.5 * mean_fg|Yhat - Y| + 0.5 * mean_bg|Yhat - Y|, region means
               per image, then averaged over the images that HAVE that region.
  2. SELECTION validation foreground PSNR, among checkpoints whose validation
               background MAE AND RMSE are no worse than frozen E0's; E0 itself
               (the zero-correction step-0 output) is the fallback candidate.
  3. derived   early stopping keeps its mechanics (every 500, >= 3000 updates,
               5 checks) but counts checks without a new ELIGIBLE best, because
               the stopping rule follows the selection rule in both runs.
  4. logging   a checkpoint at every validation (so both selection rules can be
               applied to this run); no effect on training.

refiner_e0's modules are imported read-only; nothing in refiner_e0 is changed.
"""

import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
E0DIR = os.path.join(os.path.dirname(HERE), 'refiner_e0')
for _p in (E0DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refiner_common as rc                                   # noqa: E402

EXP = 'Holo_E0_frozen_noisy_output_fgbalanced_refiner'
EXP_DIR = os.path.join(rc.REPO, 'experiments', EXP)
MODEL_DIR = os.path.join(EXP_DIR, 'models')
# gitignored. FGBAL_RESULTS redirects outputs for a dry run (never for a real one).
RESULTS = os.environ.get('FGBAL_RESULTS', os.path.join(HERE, 'results'))
CONFIG = os.path.join(HERE, 'configs', 'refiner_fgbal.yml')
SELECTION_JSON = os.path.join(EXP_DIR, 'selection.json')

OLD_EXP = rc.EXP
OLD_CONFIG = rc.CONFIG
OLD_EXP_DIR = rc.EXP_DIR
OLD_RESULTS = rc.RESULTS
OLD_SELECTED = os.path.join(OLD_EXP_DIR, 'models', 'refiner_best.pth')

# keys of the new config that may differ from refiner_e0's; everything else
# must be identical, checked mechanically before training
ALLOWED_CHANGED = {'name', 'loss', 'selection_metric'}
NEW_ONLY = {'based_on', 'fg_threshold', 'loss_region_weights', 'eligibility',
            'early_stopping_counts', 'save_every_validation'}


def load_configs():
    with open(CONFIG) as f:
        new = yaml.safe_load(f)
    with open(OLD_CONFIG) as f:
        old = yaml.safe_load(f)
    changed = sorted(k for k in old if k in new and new[k] != old[k])
    missing = sorted(k for k in old if k not in new)
    extra = sorted(k for k in new if k not in old)
    bad = [k for k in changed if k not in ALLOWED_CHANGED] + missing + \
          [k for k in extra if k not in NEW_ONLY]
    if bad:
        raise SystemExit(f'config differs from refiner_e0 beyond the allowed '
                         f'keys: {bad}')
    return new, old, {'changed': changed, 'new_only': extra}


def region_l1(yhat, gt, thr=rc.FG_THRESHOLD, w_fg=0.5, w_bg=0.5):
    """0.5 * mean_fg|e| + 0.5 * mean_bg|e|, fg = gt > thr (the study's mask).

    Region means are taken PER IMAGE over that image's own region pixels (no
    dilution by the other region's pixel count), then averaged over the images
    in which the region is non-empty. An image with an empty region contributes
    nothing to that term. If a region is empty in EVERY image of the batch, its
    term is omitted (weight not renormalised) -- never a NaN. The mask comes
    from the target and is used only here, never as a model input.
    """
    e = (yhat - gt).abs().flatten(1)
    m = (gt > thr).flatten(1).to(e.dtype)
    b = 1. - m
    nf, nb = m.sum(1), b.sum(1)
    vf, vb = nf > 0, nb > 0
    fg_i = (e * m).sum(1) / nf.clamp(min=1.)
    bg_i = (e * b).sum(1) / nb.clamp(min=1.)
    loss = e.new_zeros(())
    lf = fg_i[vf].mean() if vf.any() else None
    lb = bg_i[vb].mean() if vb.any() else None
    if lf is not None:
        loss = loss + w_fg * lf
    if lb is not None:
        loss = loss + w_bg * lb
    return loss, {'fg': None if lf is None else float(lf),
                  'bg': None if lb is None else float(lb),
                  'n_fg_images': int(vf.sum()), 'n_bg_images': int(vb.sum())}
