"""postlatent-render: the INJECTION-POINT ablation of addition-render.

ONE thing differs from `RestormerDinoSpatialRender`: WHERE the prior is added.

    addition-render     guided = F + P(D)   ;  latent = self.latent(guided)
    postlatent-render   latent = self.latent(F) ;  guided = latent + P(D)

Same DINO, same block B6, same render, same centering means, same
zero-initialised `P = Conv2d(768, 384, 1)`, same recipe, same seed, and
EXACTLY the same parameter count as addition-render — the tensor on either side
of the latent stage has the same shape [B, 384, g, g], so nothing about the
projection changes. This is the cleanest one-factor arm in the whole study:
one line of the forward pass moves.

WHY IT MATTERS. Section 4.5 of the chapter argues that injecting BEFORE the
eight latent blocks strictly contains what injecting after would provide,
because each transformer block carries an identity path, so a signal added at
the input rides the residual stream through all eight and is still present at
the output *in addition to* having been processed. The argument is sound, and
it was never tested: the code hard-refuses any other injection point, and §9
lists "one injection point" as a limitation. This arm supplies the missing
measurement.

WHAT DIFFERS DOWNSTREAM. Injecting after the latent stage means the eight
latent blocks — 14,370,448 parameters, 55.0% of the whole network — never see
the prior. Only the decoder does. If the chapter's argument is right this arm
should be clearly worse; if it is a null, the argument is wrong in an
interesting way, because it would mean the prior's benefit is realised in the
decoder rather than in the latent processing.

CONFIG. `dino_injection: post_latent`. The parent validates that key and
accepts only `latent`, so this subclass pops it, checks it, and hands the parent
the string it demands before overwriting the attribute. That indirection is
deliberate and is confined to these four lines: the parent's guard stays strict
for every other arm, which is what stops a typo in a YAML silently relocating an
injection.

MONITORING. `latent_norm` now measures the LATENT STAGE OUTPUT rather than its
input, because that is the tensor the prior is added to and therefore the
quantity `injection_ratio` must be relative to. The gate rules are unchanged in
form, but the reference value they establish at iteration 5,000 is not
comparable with the other arms' — compare this arm only with itself.
"""

import torch

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)


class RestormerPostLatentRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        injection = kwargs.pop('dino_injection', 'post_latent')
        if injection != 'post_latent':
            raise ValueError(
                f'RestormerPostLatentRender requires dino_injection: '
                f'post_latent, got {injection!r} (the before-latent arm is '
                f'RestormerDinoSpatialRender)')
        # the parent's guard accepts only 'latent'; hand it that, then record
        # the truth. Confined to these lines so the guard stays strict elsewhere.
        kwargs['dino_injection'] = 'latent'
        super().__init__(*args, **kwargs)
        self.dino_injection = 'post_latent'

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. postlatent-render needs '
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

        # ---- THE ONE DIFFERENCE: the latent stage runs UNGUIDED -------------
        latent_out = self.latent(inp_enc_level4)

        d_centered = self.dino_prior(render, mode, latent_out.shape[-2:])
        projected = self.P(d_centered.to(latent_out.dtype))
        guided = latent_out + projected
        # latent_norm is the LATENT OUTPUT here, the tensor being added to.
        self._record_stats(latent_out, projected, mode)
        if self._capture_dino_io:
            self._dino_capture['latent_out'] = latent_out
            self._dino_capture['projected'] = projected
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
