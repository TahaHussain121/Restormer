"""gated-render: the "let the model decide" ablation of addition-render.

ONE thing differs from `RestormerDinoSpatialRender`: how MUCH of the projected
prior is added at each position.

    addition-render   guided = F + P(D)
    gated-render      guided = F + g * P(D),  g = sigmoid(G([F, P(D)])) in (0,1)

Same DINO, same block B6, same render source, same centering means, same
zero-initialised projection `P`, same injection point, same recipe, same seed.
The gate is the only new object; see `dino_gate.py` for why this degree of
freedom is the one no previous arm varied, and for the initialisation and the
one-step gradient staircase.

WHAT THIS ARM IS FOR. Three arms have varied the fusion OPERATOR and all were
nulls at matched depth. The recurring objection to plain addition is that it
hands the network the whole prior everywhere instead of letting the network
select. That objection is only partly right — `P` is a learned readout and can
silence whole channels — but `P` applies ONE rule at every position of every
image. This arm gives the network a per-position, per-image amount, including
zero, and asks whether it is worth anything.

PARAMETERS: +295,296 for the gate on top of addition-render's +295,296, so
+590,592 over E0. NOT parameter-matched to addition-render; report it that way.
"""

import torch

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)
from basicsr.models.archs.dino_gate import DinoGate


class RestormerGatedRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        fusion = kwargs.pop('dino_fusion', 'gate')
        if fusion != 'gate':
            raise ValueError(
                f'RestormerGatedRender requires dino_fusion: gate, got '
                f'{fusion!r} (plain addition is RestormerDinoSpatialRender)')
        self.gate_stats_freq = int(kwargs.pop('dino_gate_stats_freq', 5000))
        # READ, never pop: the parent needs it, and threading the trunk's own
        # bias convention into the gate stops it drifting from the trunk.
        gate_bias = bool(kwargs.get('bias', False))
        if int(kwargs.get('dino_block', 6)) != 6:
            raise ValueError('gated-render is fixed at B6: it exists to be '
                             'addition-render with exactly one thing changed')
        if 'dino_layers' in kwargs:
            raise ValueError('gated-render is single-layer by construction and '
                             'has no AFFM; use a gated AFFM arm for that')

        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion

        # ---- RNG FENCE around the gate construction ------------------------
        # nn.Conv2d draws from the generator even though the draw is overwritten
        # by the zero-init, so without this every later draw (data order, crop
        # offsets, augmentation flags) would shift away from E0's for this seed.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.gate = DinoGate(dim=self.latent_channels, bias=True)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._gate_forward_count = 0
        self.last_attn_stats = {}       # the wrapper's generic observation hook

    # ------------------------------------------------------------------
    # dino_prior is INHERITED unchanged: the centered B6 render grid.
    # ------------------------------------------------------------------

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. gated-render needs '
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

        # ---- the ONLY difference from addition-render: the gate ------------
        d_centered = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        projected = self.P(d_centered.to(inp_enc_level4.dtype))
        self._gate_forward_count += 1
        want = (self._capture_dino_io
                or (self._gate_forward_count - 1) % self.gate_stats_freq == 0)
        guided, injected, gstats = self.gate(inp_enc_level4, projected,
                                             collect_stats=want)
        # projected_norm := ||g * P(D)||, the quantity actually injected, so the
        # existing gate rules read the same shape of thing as in every arm.
        self._record_stats(inp_enc_level4, injected, mode)
        if want:
            self.last_attn_stats = dict(gstats)
        if self._capture_dino_io:
            self._dino_capture['projected'] = projected
            self._dino_capture['injected'] = injected
            self._dino_capture['guided'] = guided
        # --------------------------------------------------------------------

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
