"""global-render: the SPATIAL ablation of E1-render.

ONE thing differs from `RestormerDinoSpatialRender`: the centered DINO grid is
POOLED to a single vector and broadcast back before the projection.

    E1-render       D_centered [B,768,g,g] -----------------> P -> + at latent
    global-render   D_centered [B,768,g,g]
                      -> mean over the g*g spatial positions -> [B,768,1,1]
                      -> broadcast back to [B,768,g,g] ------> P -> + at latent

Every spatial position therefore receives the SAME vector. The only difference
from E1-render is the presence or absence of spatial variation, which is exactly
the question this arm asks: does the gain E1-render shows come from DINO's
*spatial layout*, or would a single global descriptor of the render do just as
well?

WHAT IS **NOT** DIFFERENT (all inherited, none of it re-implemented here):
  * DINO SOURCE IS STILL THE RENDER. `forward` is inherited unmodified from
    `RestormerDinoSpatialRender`, so the stacked [B,2,H,W] input is split the
    same way -- radar = channel 0 to Restormer and the global residual, render =
    channel 1 to DINO. This class overrides `dino_prior` only.
  * CROP + AUGMENTATION ALIGNMENT. Unchanged and untouchable from here: the
    render rides as channel 1 of the LQ tensor
    (`Dataset_PairedImage_uint16_RenderStacked`), so `train.py`'s sub-crop is one
    slice hitting both channels. Pooling happens AFTER the crop, inside the
    model, so a desynced crop would still be fatal -- it just cannot happen,
    because there is only one tensor to crop.
  * block B6 (index 5), the frozen ViT, the RENDER centering means and their
    explicit train128/eval256 switch, the zero-initialized 1x1 projection
    768->384, the residual addition at `inp_enc_level4`, the no-interpolation
    gate, the DINO-free checkpoint handling, the stability statistics.

PARAMETERS. Pooling is a mean and a broadcast: **no parameters**. This class adds
none, so the delta against E0 is still exactly 768*384 + 384 = 295,296, the same
as E1-render.

CENTERING AND POOLING COMMUTE, which is why no new means are needed. mu is one
position-independent [768] vector, so

    mean_p(D_p - mu) = mean_p(D_p) - mu

Centering first (as here, inherited) and pooling first are numerically the same
operation up to float associativity. The smoke test checks both orders on a real
batch rather than trusting this paragraph.

PATCH TOKENS, NOT CLS. The pooled vector is the mean of the *patch* tokens.
Substituting DINO's CLS token would be a different object and would change two
things at once; `dino_shared.extract_block` never returns CLS in the first place
(`return_class_token=False`).

WHY THE ASSERTION STAYS ACTIVE. `assert_no_interpolation_needed` runs inside the
inherited `dino_prior` on the grid BEFORE pooling, so the arm still refuses any
size pairing that would need feature-grid interpolation -- even though the
pooled output would happily broadcast to any grid. Pooling must never become a
way to silently paper over a scale mismatch.
"""

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)


class RestormerDinoSpatialGlobalRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        # surfaced in the YAML so the pooling is visible in the config, not only
        # in the class name. 'global_mean' is the only accepted value.
        pooling = kwargs.pop('dino_pooling', 'global_mean')
        if pooling != 'global_mean':
            raise ValueError(
                f'RestormerDinoSpatialGlobalRender requires '
                f'dino_pooling: global_mean, got {pooling!r} (the spatial arm '
                f'is RestormerDinoSpatialRender)')
        super().__init__(*args, **kwargs)
        self.dino_pooling = pooling

    def dino_prior(self, inp_img, mode, latent_hw):
        """Inherited centered render grid, pooled to one vector and broadcast.

        In plain words, three steps:
          1. `super().dino_prior(...)` does everything E1-render does -- DINO
             preprocessing, frozen B6 extraction, subtracting the render
             centering mean, reshaping tokens to a [B,768,g,g] grid, and
             asserting that grid already equals the latent grid. Centering
             happens HERE, before the pooling.
          2. average that grid over its two spatial axes, giving one [B,768,1,1]
             vector per image -- the mean centered patch token. All spatial
             variation is gone at this point; what survives is "what is in this
             render", not "where it is".
          3. copy that single vector back to every one of the g*g positions, so
             the tensor handed to the projection has the same shape E1-render
             hands it, and every position carries an identical vector.
        The projection, the residual addition and everything downstream are then
        byte-for-byte the E1-render code path.
        """
        grid = super().dino_prior(inp_img, mode, latent_hw)   # [B,768,g,g]

        pooled = grid.mean(dim=(2, 3), keepdim=True)          # [B,768,1,1]
        broadcast = pooled.expand_as(grid).contiguous()       # [B,768,g,g]

        if self._capture_dino_io:
            # `grid` (pre-pool) is already captured by the parent; these two are
            # what the smoke test needs to prove the pooling itself.
            self._dino_capture['pooled'] = pooled
            self._dino_capture['broadcast'] = broadcast
        return broadcast
