"""Shared pieces of the two FINAL multi-level arms (addition and ACA).

DELIBERATELY NOT A `*_arch.py` FILE, so the network registry does not pick it
up. Only the two multi-level arch files import it; no existing arm does.

THE LAYOUT, identical in both arms:

    site 'pl'   after the eight latent blocks      prior at the native grid
    site 'd3'   decoder level 3 input, AFTER skip concatenation and
                reduce_chan_level3                 prior expanded 2x
    site 'd2'   decoder level 2 input, AFTER skip concatenation and
                reduce_chan_level2                 prior expanded 4x

No pre-latent injection. No decoder-level-1 injection.

THE SPATIAL MAPPING IS NEW, AND IT IS NEAREST-NEIGHBOUR UPSAMPLING.

The DINO grid equals the latent grid (16x16 at train128, 32x32 at eval256), so
the post-latent site needs no resizing. Decoder level 3 is 2x finer and level 2
is 4x finer. Each stage's pointwise projection is applied at the native prior
resolution and the result is then expanded by pixel replication
(`repeat_interleave`), which IS nearest-neighbour upsampling. The project's
original "never resize the feature grid" condition therefore does NOT hold at
the two decoder sites; this is a new, documented mapping, not a continuation of
that rule. What it does preserve is exact token boundaries: every decoder
position inherits the single token whose 8x8 radar-pixel footprint it lies in.
The expansion factors are identical in both regimes (2x and 4x), but matching
scale ratios does NOT prove crop/full-image feature invariance.

`nn_expand` refuses to return anything whose shape does not match the stage's
feature map exactly, so a regime/size mismatch fails loudly rather than being
papered over by broadcasting.
"""

import torch

LAYOUT = 'post_latent+dec3+dec2'
SITES = ('pl', 'd3', 'd2')
EXPANSION = {'pl': 1, 'd3': 2, 'd2': 4}


def nn_expand(x, k, target_hw, site):
    """Nearest-neighbour upsampling by integer factor k, shape-checked."""
    out = x if k == 1 else x.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
    if tuple(out.shape[-2:]) != tuple(target_hw):
        raise RuntimeError(
            f'site {site}: prior {tuple(x.shape[-2:])} x{k} -> '
            f'{tuple(out.shape[-2:])} does not match the stage feature '
            f'{tuple(target_hw)}; refusing to broadcast or resize further')
    return out


@torch.no_grad()
def site_stats(site, feat, update, **extra):
    """Per-site monitoring, recorded SEPARATELY for every injection site.

    feat    the stage feature the prior is applied to
    update  what the injection changed at that site (guided - feat)
    extra   additional named tensors, recorded by their norm

    `finite` is 1.0 only if both tensors are finite. A non-finite value at ANY
    site also propagates to the loss, where the existing unwindowed NaN rule of
    the stability gate stops the run and archives a record; these keys ride in
    `last_dino_stats`, so that record names the site. No new ratio THRESHOLD is
    defined on them.
    """
    fn = float(feat.norm())
    un = float(update.norm())
    finite = bool(torch.isfinite(feat).all()) and bool(torch.isfinite(update).all())
    d = {f'site_{site}_feat_norm': fn,
         f'site_{site}_update_norm': un,
         f'site_{site}_ratio': un / (fn + 1e-8),
         f'site_{site}_finite': 1.0 if finite else 0.0}
    for name, t in extra.items():
        d[f'site_{site}_{name}'] = float(t.norm())
    return d
