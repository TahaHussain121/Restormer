"""aca-L6: the FUSION-OPERATOR ablation of addition-render.

THIS IS THE ONE-FACTOR ARM THE LADDER HAS BEEN MISSING.

    addition-render   D = centered B6   guided = F + P(D)
    aca-L6            D = centered B6   guided = ACA(F, P(D))

Same layer (B6), same source (render), same centering mean, same injection
point (`inp_enc_level4`), same everything else. **Only the fusion operator
differs.** Nothing else in Phase 3 gives a clean ACA-vs-addition comparison:
dinolight-render changes the operator AND the layer count at once, and
crossattn-render changed the operator, the direction and the attention type
together.

NO AFFM HERE, DELIBERATELY. With a single layer the AFFM softmax runs over an
axis of length 1 and is therefore identically 1.0 -- a no-op that would add
769 dead parameters and one meaningless observation series. The centered B6
grid goes straight to `P`. **Do not read this arm as inconsistent with
aca-L36 / aca-L6912 / dinolight-render**: those fuse 2, 3 and 4 layers
respectively and need AFFM; this one has nothing to fuse.

`dino_prior` is INHERITED UNCHANGED from `RestormerDinoSpatialRender`, so the
prior this arm sees is byte-for-byte the prior addition-render sees. That is
what makes the comparison one-factor.

TWO COMPARISONS, TWO DIFFERENT CAVEATS -- DO NOT CONFLATE THEM

  vs addition-render   ONE factor (the operator), but CAPACITY CONFOUNDED:
                       ~4.6x the parameters. A win cannot be attributed to the
                       operator alone.
  vs the ACA ladder    aca-L6 / aca-L36 / aca-L6912 / dinolight-render differ
                       only by AFFM scoring convs, 769 parameters per layer,
                       i.e. well under 1% across the whole ladder. A difference
                       ACROSS THIS LADDER is NOT attributable to capacity.

The attention block is `DinoAca` imported from `dino_aca.py` -- the same object
dinolight-render runs, not a copy. It is CHANNEL-transposed: the attention
matrix is `C/heads x C/heads`, independent of token count, which is why these
arms are expected to survive the 128 -> 256 evaluation switch that broke
crossattn-render (+2.00 dB at crop128 vs -7.56 dB at full256 from identical
weights).

MONITORING follows dinolight-render exactly. `projected_norm` logs
`||alpha * F_ca||` measured BEFORE the zero-init output conv. **That makes this
arm's `injection_ratio` roughly an order of magnitude smaller than the addition
arms' and NOT directly comparable to their ~1.03.** The true injected delta is
logged separately as `aca_injected_norm`.

INITIALISATION: `project_out` zero -> step-0 output bit-identical to E0.
`alpha_logit = -2.0` -> alpha ~ 0.119. Temperatures 1.0 per head, Restormer's
multiplicative MDTA convention. `P` is zero-init as the parent builds it, so
there is a THREE-step gradient staircase (two zero-inits in series): step 1
only `project_out`; step 2 the feature path and `P`; step 3 everything.

RNG ORDER: the parent builds the whole Restormer trunk first, then fences the
ViT and `P`; the ACA is built inside a second fence added here. Every trunk
weight stays byte-identical to E0's for the same seed.
"""

import torch

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)
from basicsr.models.archs.dino_aca import DinoAca


class RestormerAcaL6Render(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        fusion = kwargs.pop('dino_fusion', 'aca')
        if fusion != 'aca':
            raise ValueError(
                f'RestormerAcaL6Render requires dino_fusion: aca, got '
                f'{fusion!r} (addition is RestormerDinoSpatialRender)')
        self.aca_heads = int(kwargs.pop('dino_aca_heads', 6))
        self.aca_alpha_init = float(kwargs.pop('dino_aca_alpha_init', -2.0))
        self.aca_stats_freq = int(kwargs.pop('dino_aca_stats_freq', 5000))
        # READ, never pop -- the parent needs them. Threading the trunk's own
        # bias / LayerNorm_type into the ACA is what stops this module drifting
        # from Restormer's convention.
        self.aca_trunk_bias = bool(kwargs.get('bias', False))
        self.aca_trunk_ln = str(kwargs.get('LayerNorm_type', 'WithBias'))
        # single layer, and it must be B6 -- this arm exists to be
        # addition-render with one thing changed.
        if int(kwargs.get('dino_block', 6)) != 6:
            raise ValueError('aca-L6 is fixed at B6; use aca-L36 / aca-L6912 / '
                             'dinolight-render for multi-layer sets')
        kwargs.setdefault('dino_block', 6)
        if 'dino_layers' in kwargs:
            raise ValueError('aca-L6 takes no dino_layers: it is single-layer '
                             'by construction and has no AFFM')

        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion
        self.dino_layers_1indexed = [6]          # for the manifest / reporting

        # ---- RNG FENCE around the ACA construction -------------------------
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.aca = DinoAca(dim=self.latent_channels, heads=self.aca_heads,
                               bias=self.aca_trunk_bias,
                               alpha_init=self.aca_alpha_init,
                               LayerNorm_type=self.aca_trunk_ln)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._aca_forward_count = 0
        self.last_attn_stats = {}

    # ------------------------------------------------------------------
    # dino_prior is INHERITED: single centered B6 grid, exactly
    # addition-render's prior. Not overridden on purpose.
    # ------------------------------------------------------------------

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. aca-L6 needs '
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

        # ---- the ONLY difference from addition-render: the fusion operator --
        d_centered = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        d_proj = self.P(d_centered.to(inp_enc_level4.dtype))
        self._aca_forward_count += 1
        want = (self._capture_dino_io
                or (self._aca_forward_count - 1) % self.aca_stats_freq == 0)
        guided, ca_term, aca_stats = self.aca(inp_enc_level4, d_proj,
                                              collect_stats=want)
        # projected_norm := ||alpha * F_ca||  (see the module docstring)
        self._record_stats(inp_enc_level4, ca_term, mode)
        if want:
            self.last_attn_stats = dict(aca_stats)
        if self._capture_dino_io:
            self._dino_capture['d_proj'] = d_proj
            self._dino_capture['ca_term'] = ca_term
            self._dino_capture['guided'] = guided
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
            out_dec_level1 = self.output(out_dec_level1) + radar
        return out_dec_level1
