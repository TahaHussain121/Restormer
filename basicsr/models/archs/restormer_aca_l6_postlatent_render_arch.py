"""aca-L6-postlatent: the ACA block moved AFTER the latent stage.

ONE thing differs from `RestormerAcaL6Render`: where the fusion happens.

    aca-L6              guided = ACA(F, P(D))          ; latent = self.latent(guided)
    aca-L6-postlatent   latent_out = self.latent(F)    ; guided = ACA(latent_out, P(D))

Same `DinoAca` block, imported not copied, so the operator is literally the same
object. Same B6, same render source, same centering means, same zero-initialised
`P`, same alpha_init, same heads, same recipe, same seed. **Identical parameter
count to aca-L6**, because the tensor on either side of the latent stage has the
same shape [B, 384, g, g].

WHY THIS ARM, AND WHY THE COMPARISON THAT MATTERS IS NOT THE OBVIOUS ONE

Wave 2 found that moving PLAIN ADDITION after the latent stage is worth +0.306
dB on test. The fusion-operator question and the injection-location question
have so far been varied one at a time, which leaves one cell of a 2x2 empty:

    fusion \\ location    before latent      after latent
    addition, B6         24.081             24.387
    ACA, B6              24.111             THIS ARM

The decisive comparison is therefore **aca-L6-postlatent against
postlatent-render**, both at the same location — NOT against the weaker
before-latent addition reference. Three outcomes and what each licenses:

  * beats postlatent-render: evidence that the ACA operator has value at this
    location, which it did not have at the old one;
  * beats aca-L6 but only matches postlatent-render: the LOCATION helped, and an
    operator advantage remains unestablished;
  * still loses on crop128: moving the block does not resolve the protocol
    penalty that every attention arm has shown.

WHAT THIS ARM CANNOT SETTLE. `DinoAca` computes
`project_out(F_sa + alpha*F_ca) + F` — self-attention AND gated cross-attention
behind one shared output projection. So this arm, like every ACA arm before it,
compares that WHOLE BLOCK against addition; it does not isolate the cross path.
Isolating it needs a matched self-attention control at the same location, which
is a separate pair of runs and is deliberately not attempted here.

CONFIG. `dino_injection: post_latent`. The grandparent's guard accepts only
`latent`, so this class pops the key, validates it, hands the parent the string
it demands, and records the truth afterwards — the same four-line device
`RestormerPostLatentRender` uses, and confined to this class so the guard stays
strict everywhere else.

MONITORING. `latent_norm` measures the latent stage OUTPUT here, because that is
the tensor the block is applied to. `projected_norm` keeps aca-L6's convention
of logging `||alpha * F_ca||`. Both are comparable to aca-L6 in KIND but their
5,000-iteration references are not numerically comparable to any before-latent
arm; compare this arm with itself and with postlatent-render.
"""

import torch

from basicsr.models.archs.restormer_aca_l6_render_arch import RestormerAcaL6Render


class RestormerAcaL6PostLatentRender(RestormerAcaL6Render):

    def __init__(self, *args, **kwargs):
        injection = kwargs.pop('dino_injection', 'post_latent')
        if injection != 'post_latent':
            raise ValueError(
                f'RestormerAcaL6PostLatentRender requires dino_injection: '
                f'post_latent, got {injection!r} (the before-latent ACA arm is '
                f'RestormerAcaL6Render)')
        kwargs['dino_injection'] = 'latent'      # what the grandparent's guard accepts
        super().__init__(*args, **kwargs)
        self.dino_injection = 'post_latent'

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. aca-L6-postlatent needs '
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

        # ---- THE ONE DIFFERENCE from aca-L6: the latent stage runs UNGUIDED --
        latent_out = self.latent(inp_enc_level4)

        d_centered = self.dino_prior(render, mode, latent_out.shape[-2:])
        d_proj = self.P(d_centered.to(latent_out.dtype))
        self._aca_forward_count += 1
        want = (self._capture_dino_io
                or (self._aca_forward_count - 1) % self.aca_stats_freq == 0)
        guided, ca_term, aca_stats = self.aca(latent_out, d_proj,
                                              collect_stats=want)
        # latent_norm is the LATENT OUTPUT here -- the tensor the block acts on.
        self._record_stats(latent_out, ca_term, mode)
        if want:
            self.last_attn_stats = dict(aca_stats)
        if self._capture_dino_io:
            self._dino_capture['latent_out'] = latent_out
            self._dino_capture['d_proj'] = d_proj
            self._dino_capture['ca_term'] = ca_term
            self._dino_capture['guided'] = guided
        # --------------------------------------------------------------------

        inp_dec_level3 = self.up4_3(guided)
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
