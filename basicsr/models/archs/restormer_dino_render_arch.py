"""E1-render: the source ablation of E1-addition.

ONE thing differs from `RestormerDinoSpatial`: which tensor DINO sees.

    E1-addition   DINO sees the 1e5 noisy crop   (dino_source: same_lq)
    E1-render     DINO sees the RENDER crop      (dino_source: render)

Restormer still receives the 1e5 crop. The target is still 1e7. The render is
DINO input and nothing else -- never a Restormer input, never a target, never a
centering statistic for the radar arm.

INPUT LAYOUT. `Dataset_PairedImage_uint16_RenderStacked` delivers the pair as
one tensor, radar in channel 0 and render in channel 1:

    inp_img [B, 2, H, W]  ->  radar = inp_img[:, 0:1]   Restormer + residual
                              render = inp_img[:, 1:2]  DINO

The two channels travel as a single tensor precisely so that every crop and
batch subsample `basicsr/train.py` applies hits both identically -- the same
slice of the same object, not two coordinate calculations that have to agree.

Everything else is inherited unchanged from `RestormerDinoSpatial`: B6 at index
5, the frozen ViT, the centering buffers and their explicit train128/eval256
switch, the zero-initialized 1x1 projection 768->384, the residual addition at
`inp_enc_level4` before the latent blocks, the no-interpolation gate, and the
DINO-free checkpoint handling. This file only re-routes one tensor.
"""

import torch

from basicsr.models.archs.restormer_dino_spatial_arch import RestormerDinoSpatial


class RestormerDinoSpatialRender(RestormerDinoSpatial):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dino_source', 'render')
        if kwargs['dino_source'] != 'render':
            raise ValueError('RestormerDinoSpatialRender requires '
                             'dino_source: render (use RestormerDinoSpatial '
                             'for the same_lq arm)')
        super().__init__(*args, **kwargs)
        self.inp_stack_channels = 2      # (radar, render)

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. E1-render needs '
                f'Dataset_PairedImage_uint16_RenderStacked.')

        radar = inp_img[:, 0:1]          # what Restormer restores
        render = inp_img[:, 1:2]         # what DINO looks at
        if self._capture_dino_io:
            self._dino_capture['radar'] = radar
            self._dino_capture['render'] = render
            self._dino_capture['stacked'] = inp_img

        inp_enc_level1 = self.patch_embed(radar)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        # ---- the ONLY difference from E1-addition: the prior's source -------
        d_centered = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        projected = self.P(d_centered.to(inp_enc_level4.dtype))
        guided = inp_enc_level4 + projected
        self._record_stats(inp_enc_level4, projected, mode)
        # ---------------------------------------------------------------------

        latent = self.latent(guided)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            # global residual against the RADAR, never the render
            out_dec_level1 = self.output(out_dec_level1) + radar

        return out_dec_level1
