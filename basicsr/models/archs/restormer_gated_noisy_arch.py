"""gated-noisy: the gate on the arm the prior HURTS. The sharp test.

ONE thing differs from `RestormerDinoSpatial` (E1-addition-noisy):

    addition-noisy   guided = F + P(D)
    gated-noisy      guided = F + g * P(D),  g = sigmoid(G([F, P(D)])) in (0,1)

DINO still reads the SAME 1e5 radar tensor the network is restoring, exactly as
addition-noisy does. Same B6, same 1e5 centering means, same zero-initialised
`P`, same injection point, same recipe, same seed.

WHY THIS ARM IS THE INTERESTING ONE, and why it is worth a slot even though
nobody expects it to top the table.

addition-noisy is the project's cleanest negative result: a prior computed on
the degraded input scores 0.577 dB BELOW the no-prior baseline on the test
split. A prior that is merely useless should cost nothing — `P` is
zero-initialised and could simply stay near zero — yet this one costs real
performance, and the training curves show it also overfits (train loss 0.0471
against E0's 0.0587, validation peaking at 128k and then declining).

The standing explanation is that the network cannot cheaply ignore a prior that
is added at every position of every image. This arm tests that explanation
directly, because a per-position gate CAN switch the prior off — AFFM cannot,
since its weights sum to one by construction. Two outcomes, both publishable:

  * The gate closes and the arm returns to roughly baseline. Then "the model can
    decide" is worth something precisely where the prior is bad, the mechanism
    behind the negative result is identified, and the gate has a job even though
    it may be inert on the render arm.
  * The arm still lands below baseline. Then selection does NOT rescue a bad
    prior, the SOURCE finding gets stronger rather than weaker, and the reading
    that the damage comes from the prior's content rather than from the
    network's inability to suppress it is supported.

The prediction is registered in the devlog BEFORE the run.

INPUT. Single-stream, like addition-noisy: the DINO input is derived from the
same `inp_img` tensor inside `forward`, so there is no second dataset stream and
nothing can be misaligned by the training-time sub-crop.

PARAMETERS: +590,592 over E0 (P 295,296 + gate 295,296). Not
parameter-matched to addition-noisy; report it that way.
"""

import torch

from basicsr.models.archs.restormer_dino_spatial_arch import RestormerDinoSpatial
from basicsr.models.archs.dino_gate import DinoGate


class RestormerGatedNoisy(RestormerDinoSpatial):

    def __init__(self, *args, **kwargs):
        fusion = kwargs.pop('dino_fusion', 'gate')
        if fusion != 'gate':
            raise ValueError(
                f'RestormerGatedNoisy requires dino_fusion: gate, got '
                f'{fusion!r} (plain addition is RestormerDinoSpatial)')
        self.gate_stats_freq = int(kwargs.pop('dino_gate_stats_freq', 5000))
        if kwargs.get('dino_source', 'same_lq') != 'same_lq':
            raise ValueError('gated-noisy requires dino_source: same_lq — it is '
                             'the gate applied to the arm whose prior HURTS; '
                             'the render version is RestormerGatedRender')
        if int(kwargs.get('dino_block', 6)) != 6:
            raise ValueError('gated-noisy is fixed at B6, to be addition-noisy '
                             'with exactly one thing changed')

        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion

        # ---- RNG FENCE around the gate construction ------------------------
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
        self.last_attn_stats = {}

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        # ---- the ONLY difference from addition-noisy: the gate -------------
        d_centered = self.dino_prior(inp_img, mode, inp_enc_level4.shape[-2:])
        projected = self.P(d_centered.to(inp_enc_level4.dtype))
        self._gate_forward_count += 1
        want = (self._capture_dino_io
                or (self._gate_forward_count - 1) % self.gate_stats_freq == 0)
        guided, injected, gstats = self.gate(inp_enc_level4, projected,
                                             collect_stats=want)
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
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1
