"""multilevel-addition-render — FINAL experiment A.

The B6 render prior added at THREE sites instead of one:

    postlatent-render     latent_out = self.latent(F) ; guided = latent_out + P(D)
    multilevel-addition   same at the post-latent site, AND
                          dec3_in += NN2( P_dec3(D) )   after skip cat + channel reduction
                          dec2_in += NN4( P_dec2(D) )   after skip cat + channel reduction

Layout, spatial mapping and monitoring are defined once in `dino_multilevel.py`
and shared verbatim with final experiment B (multi-level ACA), so the two arms
differ ONLY in the fusion operator at each site.

REFERENCE. `postlatent-render` (the completed single-site additive arm). This
class subclasses it: the trunk, the frozen DINO, B6, the render source, the
centering means and the post-latent projection `P` are that arm's, unchanged.

NEW PARAMETERS. Two zero-initialised 1x1 projections with the existing bias
convention (weight and bias, zero): P_dec3 768->192 (147,648) and P_dec2 768->96
(73,824). Added over E0: 295,296 + 147,648 + 73,824 = 516,768. Built inside an
RNG fence so the trunk initialisation and every later random draw stay
byte-identical to E0's for the same seed.

ONE DINO EXTRACTION PER FORWARD. `dino_prior` is called once; the centred B6
grid is reused at all three sites. The native-grid assertion inherited from the
parent still applies to that single grid against the latent grid.

INITIALISATION AND GRADIENTS. Every projection is zero, so every injection is
zero and the step-0 output is E0's exactly. There is no staircase here: each
P_s receives gradient dL/dguided_s (x) D on the FIRST backward, because D is
non-zero. Verified on a real backward in the smoke test.

MONITORING. The stability gate keeps its three keys, measured at the
post-latent site exactly as `postlatent-render` measures them, so that arm and
this one remain comparable under the same rule. All three sites are recorded
separately — feature norm, update norm, their ratio, and a finiteness flag — as
observations and inside `last_dino_stats`, so any stability record names the
site. One monitored site does not certify the other two; no new threshold is
invented for the decoder sites.
"""

import torch
import torch.nn as nn

from basicsr.models.archs.restormer_postlatent_render_arch import (
    RestormerPostLatentRender)
from basicsr.models.archs.dino_multilevel import LAYOUT, nn_expand, site_stats


class RestormerMultiLevelAdditionRender(RestormerPostLatentRender):

    def __init__(self, *args, **kwargs):
        layout = kwargs.pop('dino_injection', LAYOUT)
        if layout != LAYOUT:
            raise ValueError(f'RestormerMultiLevelAdditionRender requires '
                             f'dino_injection: {LAYOUT!r}, got {layout!r}')
        if int(kwargs.get('dino_block', 6)) != 6:
            raise ValueError('the final multi-level arms are fixed at B6')
        if 'dino_layers' in kwargs:
            raise ValueError('the final multi-level arms are single-depth (B6)')
        kwargs['dino_injection'] = 'post_latent'   # what the parent's guard accepts
        super().__init__(*args, **kwargs)
        self.dino_injection = LAYOUT

        self.ch_dec3 = int(self.reduce_chan_level3.out_channels)   # 192
        self.ch_dec2 = int(self.reduce_chan_level2.out_channels)   # 96

        # ---- RNG FENCE around the two new projections -----------------------
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.P_dec3 = nn.Conv2d(self.dino_embed_dim, self.ch_dec3, kernel_size=1)
            self.P_dec2 = nn.Conv2d(self.dino_embed_dim, self.ch_dec2, kernel_size=1)
            for conv in (self.P_dec3, self.P_dec2):
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self.last_attn_stats = {}

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. multilevel-addition needs '
                f'Dataset_PairedImage_uint16_RenderStacked.')

        radar = inp_img[:, 0:1]
        render = inp_img[:, 1:2]
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

        latent_out = self.latent(inp_enc_level4)          # unguided, as post-latent

        # ---- ONE extraction, reused at all three sites ---------------------
        d = self.dino_prior(render, mode, latent_out.shape[-2:])
        d = d.to(latent_out.dtype)

        # site pl: native grid
        inj_pl = self.P(d)
        guided_pl = latent_out + inj_pl

        # site d3: decoder level 3 input, after skip cat + channel reduction
        x3 = self.up4_3(guided_pl)
        x3 = torch.cat([x3, out_enc_level3], 1)
        x3 = self.reduce_chan_level3(x3)
        native_d3 = self.P_dec3(d)
        inj_d3 = nn_expand(native_d3, 2, x3.shape[-2:], 'd3')
        out_dec_level3 = self.decoder_level3(x3 + inj_d3)

        # site d2: decoder level 2 input, after skip cat + channel reduction
        x2 = self.up3_2(out_dec_level3)
        x2 = torch.cat([x2, out_enc_level2], 1)
        x2 = self.reduce_chan_level2(x2)
        native_d2 = self.P_dec2(d)
        inj_d2 = nn_expand(native_d2, 4, x2.shape[-2:], 'd2')
        out_dec_level2 = self.decoder_level2(x2 + inj_d2)

        # decoder level 1: UNTOUCHED
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        # ---- monitoring: the gate at the post-latent site; all sites recorded
        self._record_stats(latent_out, inj_pl, mode)
        sites = {}
        sites.update(site_stats('pl', latent_out, inj_pl))
        sites.update(site_stats('d3', x3, inj_d3))
        sites.update(site_stats('d2', x2, inj_d2))
        self.last_dino_stats.update(sites)
        self.last_attn_stats = dict(sites)

        if self._capture_dino_io:
            self._dino_capture.update({
                'latent_out': latent_out,
                'site_pl_feat': latent_out, 'site_pl_prior': inj_pl,
                'site_d3_feat': x3, 'site_d3_prior': inj_d3, 'd3_native': native_d3,
                'site_d2_feat': x2, 'site_d2_prior': inj_d2, 'd2_native': native_d2})

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + radar
        return out_dec_level1
