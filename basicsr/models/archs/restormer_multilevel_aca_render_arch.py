"""multilevel-aca-render — FINAL experiment B.

Exactly final experiment A's layout, prior, sites and spatial mapping, with the
additive injection at each site replaced by the existing ACA block:

    projected_prior_s = expand_s( P_s(D) )
    guided_s          = ACA_s( feature_s, projected_prior_s )

at the post-latent site (384 ch), decoder level 3 input (192 ch) and decoder
level 2 input (96 ch), the two decoder sites taken AFTER skip concatenation and
channel reduction. Layout, mapping and monitoring come verbatim from
`dino_multilevel.py`, shared with experiment A.

REFERENCE. `aca-L6-postlatent` (the completed single-site ACA arm). This class
subclasses it: trunk, DINO, B6, render source, means, the post-latent `P` and
the post-latent `DinoAca` (384 ch) are that arm's, unchanged.

THE ACA STAGES are independent instances of the existing `DinoAca`, imported
not copied, adapted ONLY to the stage width: six heads (192/6 = 32 and 96/6 = 16
channels per head), the trunk's LayerNorm type and bias convention, the
self-attention path, the channel cross-attention path, the scalar sigmoid gate
initialised from logit -2.0, and the zero-initialised output projection. No
channel-wise gates, no AFFM, no new attention type, no direct additive bypass.

NEW PARAMETERS. P_dec3 768->192 (147,648), P_dec2 768->96 (73,824), DinoAca(192)
(269,197), DinoAca(96) (70,093). Added over E0: 1,349,773 + 560,762 =
1,910,535. Against multi-level addition that is 3.70x the ADDED parameters and
+5.2% of TOTAL trainable parameters (28,034,587 vs 26,640,820); runtime and
memory cannot be read from either ratio. Built inside an RNG fence.

THE GRADIENT STAIRCASE, AT EVERY STAGE. Each stage has two zero-inits in series,
`project_out` and `P_s`, so each stage has the three-step staircase dinolight
has: step 1 only `project_out` receives gradient; step 2 the feature path and
`P_s`; step 3 the cross path and alpha. It cannot deadlock the way
`aca-L6-nosa` did, because the self-attention path keeps `project_out`'s
gradient non-zero from step 1. This is verified on a real multi-step backward
PER STAGE in the smoke test; a step-0 identity check alone does not show it.

MONITORING. The gate keeps aca-L6's convention at the post-latent site
(`projected_norm` = ||alpha * F_ca||), so this arm and aca-L6-postlatent remain
comparable. Every site is recorded separately — feature norm, the actual update
||guided - feature||, their ratio, ||alpha * F_ca||, and finiteness — every
forward, plus each stage's ACA statistics (alpha, temperatures, entropies)
every `dino_aca_stats_freq` forwards, prefixed pl_/d3_/d2_.
"""

import torch
import torch.nn as nn

from basicsr.models.archs.restormer_aca_l6_postlatent_render_arch import (
    RestormerAcaL6PostLatentRender)
from basicsr.models.archs.dino_aca import DinoAca
from basicsr.models.archs.dino_multilevel import LAYOUT, nn_expand, site_stats


class RestormerMultiLevelAcaRender(RestormerAcaL6PostLatentRender):

    def __init__(self, *args, **kwargs):
        layout = kwargs.pop('dino_injection', LAYOUT)
        if layout != LAYOUT:
            raise ValueError(f'RestormerMultiLevelAcaRender requires '
                             f'dino_injection: {LAYOUT!r}, got {layout!r}')
        kwargs['dino_injection'] = 'post_latent'   # what the parent's guard accepts
        super().__init__(*args, **kwargs)          # builds P and the 384-ch ACA
        self.dino_injection = LAYOUT

        self.ch_dec3 = int(self.reduce_chan_level3.out_channels)   # 192
        self.ch_dec2 = int(self.reduce_chan_level2.out_channels)   # 96

        # ---- RNG FENCE around the new projections and ACA stages ------------
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.P_dec3 = nn.Conv2d(self.dino_embed_dim, self.ch_dec3, kernel_size=1)
            self.P_dec2 = nn.Conv2d(self.dino_embed_dim, self.ch_dec2, kernel_size=1)
            for conv in (self.P_dec3, self.P_dec2):
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
            self.aca_dec3 = DinoAca(dim=self.ch_dec3, heads=self.aca_heads,
                                    bias=self.aca_trunk_bias,
                                    alpha_init=self.aca_alpha_init,
                                    LayerNorm_type=self.aca_trunk_ln)
            self.aca_dec2 = DinoAca(dim=self.ch_dec2, heads=self.aca_heads,
                                    bias=self.aca_trunk_bias,
                                    alpha_init=self.aca_alpha_init,
                                    LayerNorm_type=self.aca_trunk_ln)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._aca_obs = {}          # per-stage ACA statistics, a step function
        self.last_attn_stats = {}

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. multilevel-aca needs '
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

        self._aca_forward_count += 1
        want = (self._capture_dino_io
                or (self._aca_forward_count - 1) % self.aca_stats_freq == 0)

        # site pl: native grid, the parent's 384-ch ACA
        pr_pl = self.P(d)
        g_pl, ca_pl, st_pl = self.aca(latent_out, pr_pl, collect_stats=want)

        # site d3: decoder level 3 input, after skip cat + channel reduction
        x3 = self.up4_3(g_pl)
        x3 = torch.cat([x3, out_enc_level3], 1)
        x3 = self.reduce_chan_level3(x3)
        native_d3 = self.P_dec3(d)
        pr_d3 = nn_expand(native_d3, 2, x3.shape[-2:], 'd3')
        g_d3, ca_d3, st_d3 = self.aca_dec3(x3, pr_d3, collect_stats=want)
        out_dec_level3 = self.decoder_level3(g_d3)

        # site d2: decoder level 2 input, after skip cat + channel reduction
        x2 = self.up3_2(out_dec_level3)
        x2 = torch.cat([x2, out_enc_level2], 1)
        x2 = self.reduce_chan_level2(x2)
        native_d2 = self.P_dec2(d)
        pr_d2 = nn_expand(native_d2, 4, x2.shape[-2:], 'd2')
        g_d2, ca_d2, st_d2 = self.aca_dec2(x2, pr_d2, collect_stats=want)
        out_dec_level2 = self.decoder_level2(g_d2)

        # decoder level 1: UNTOUCHED
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        # ---- monitoring --------------------------------------------------------
        # gate at the post-latent site, aca-L6's convention: ||alpha * F_ca||
        self._record_stats(latent_out, ca_pl, mode)
        with torch.no_grad():
            sites = {}
            sites.update(site_stats('pl', latent_out, g_pl - latent_out, ca=ca_pl))
            sites.update(site_stats('d3', x3, g_d3 - x3, ca=ca_d3))
            sites.update(site_stats('d2', x2, g_d2 - x2, ca=ca_d2))
        self.last_dino_stats.update(sites)
        if want:
            obs = {}
            for tag, st in (('pl', st_pl), ('d3', st_d3), ('d2', st_d2)):
                obs.update({f'{tag}_{k}': v for k, v in st.items()})
            self._aca_obs = obs
        self.last_attn_stats = {**sites, **self._aca_obs}

        if self._capture_dino_io:
            self._dino_capture.update({
                'latent_out': latent_out,
                'site_pl_feat': latent_out, 'site_pl_prior': pr_pl,
                'site_d3_feat': x3, 'site_d3_prior': pr_d3, 'd3_native': native_d3,
                'site_d2_feat': x2, 'site_d2_prior': pr_d2, 'd2_native': native_d2,
                'ca_term': ca_pl})

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + radar
        return out_dec_level1
