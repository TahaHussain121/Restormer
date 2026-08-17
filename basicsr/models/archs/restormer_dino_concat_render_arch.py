"""concat-render: the FUSION ablation of E1-addition-render.

ONE thing differs from `RestormerDinoSpatialRender`: how the centered DINO grid
meets the latent tensor.

    addition-render   guided = F + P(D)                    P:  1x1, 768 -> 384
    concat-render     guided = fuse(cat([F, D], dim=1))    fuse: 1x1, 1152 -> 384

The residual addition is GONE, not kept alongside the concatenation: `self.P` is
removed from this module entirely, so a checkpoint of this arm has no `P.*` key
and no dead parameter can quietly contribute.

THE FEATURES ARE STILL SPATIAL. D is the same [B,768,g,g] B6 patch-token grid
addition-render uses -- g=16 at the 128 training crop, g=32 at 256 eval. There
is NO pooling and NO broadcast anywhere in this file. The pooled arm
(`restormer_dino_global_render_arch.py`) answers a different question and none
of its code or style is inherited here; this class extends
`RestormerDinoSpatialRender` directly.

WHY CONCAT AT ALL
    In addition, the radar path is FIXED at identity -- F enters the sum with an
    implicit weight of 1 that no gradient can change, and the only way the
    network can rebalance the two streams is to change ||P(D)||. Concat makes
    the radar half learnable: W_F is a real parameter, so the balance between F
    and D can move from either side. ACT (2203.07682) ablates direct concat
    against summation and concat wins; Feature-Fused SSD (1709.05054) is the
    learned-vs-fixed combination-weight comparison.

INITIALIZATION -- the arm starts functionally equal to E0
    W[:, :384]  identity  (torch.eye reshaped to [384,384,1,1])
    W[:, 384:]  zero
    bias        zero
    so at step 0, guided == F exactly, which is what E0 feeds its latent blocks.
    The DINO half starting at zero does NOT freeze it: dL/dW_D = dL/d(guided) (x)
    D and D is non-zero, so W_D gets a real gradient on the very first backward.
    The smoke test asserts BOTH -- step-0 equality with E0 and a non-zero first
    gradient on W[:, 384:] -- rather than trusting this paragraph.

RNG ORDER
    `nn.Conv2d.__init__` draws from the RNG before `zeros_`/`eye` overwrite the
    result, which would shift every subsequent draw (data order, crop offsets,
    augmentation flags) away from E0's for the same seed. `fuse` is therefore
    built inside an EXPLICIT SAVE/RESTORE FENCE, the same device that
    `RestormerDinoSpatial.__init__` already uses for the ViT and P. The parent
    fences its own construction, so RNG state on entry to this fence is already
    E0's; restoring it on exit leaves every trunk weight and every later draw
    byte-identical to E0.

PARAMETERS
    1152 * 384 + 384 = 442,752 over E0, versus addition-render's 295,296. The
    two arms are NOT parameter-matched, and concat's radar path is learnable
    where addition's is fixed at identity. That is inherent to the comparison
    between the two fusion operators and is recorded in the devlog, not
    corrected for.

MONITORING -- the SAME three numbers the addition arms log
    A 1x1 conv on a concatenation is exactly the sum of two 1x1 convs:

        fuse(cat([F, D])) = conv(F, W_F) + conv(D, W_D) + b

    so splitting the weight recovers quantities directly comparable to the
    addition arms' ||F||, ||P(D)|| and their ratio, and they go to the same
    TensorBoard tags and the same dino_stability.csv through the unchanged
    model wrapper. At initialization W_F is the identity, so `latent_norm`
    starts as literally ||F|| -- the same number addition-render logs at step 0.
    The bias is excluded from the split; it belongs to neither stream.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)


class RestormerDinoConcatRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        # surfaced in the YAML so the fusion operator is visible in the config,
        # not only in the class name. 'concat' is the only accepted value.
        fusion = kwargs.pop('dino_fusion', 'concat')
        if fusion != 'concat':
            raise ValueError(
                f'RestormerDinoConcatRender requires dino_fusion: concat, got '
                f'{fusion!r} (the residual-addition arm is '
                f'RestormerDinoSpatialRender)')
        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion

        # ---- the residual projection is REMOVED, not left dangling ---------
        # the parent built it inside its own RNG fence, so deleting it here
        # costs nothing and guarantees no `P.*` key reaches a checkpoint and no
        # unused parameter joins the optimizer.
        del self.P

        # ---- RNG FENCE around fuse's construction (see module docstring) ---
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.fuse = nn.Conv2d(self.latent_channels + self.dino_embed_dim,
                                  self.latent_channels, kernel_size=1,
                                  bias=True)
            with torch.no_grad():
                # radar half -> identity, DINO half -> zero, bias -> zero
                eye = torch.eye(self.latent_channels).reshape(
                    self.latent_channels, self.latent_channels, 1, 1)
                self.fuse.weight[:, :self.latent_channels].copy_(eye)
                self.fuse.weight[:, self.latent_channels:].zero_()
                self.fuse.bias.zero_()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    # ------------------------------------------------------------------
    # the two halves of the fused weight, named once so nothing re-slices
    # ------------------------------------------------------------------
    @property
    def W_F(self):
        """[384, 384, 1, 1] -- the RADAR half. Identity at init, then learned."""
        return self.fuse.weight[:, :self.latent_channels]

    @property
    def W_D(self):
        """[384, 768, 1, 1] -- the DINO half. Zero at init, then learned."""
        return self.fuse.weight[:, self.latent_channels:]

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. concat-render needs '
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

        # ---- the ONLY difference from addition-render: concat, not add ------
        # dino_prior is INHERITED unchanged: frozen B6 patch tokens from the
        # render, centered by the render mean, reshaped to the [B,768,g,g] grid,
        # with the no-interpolation gate already asserted inside it. The grid is
        # concatenated along channels -- never pooled, never broadcast.
        d_centered = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        d_centered = d_centered.to(inp_enc_level4.dtype)
        guided = self.fuse(torch.cat([inp_enc_level4, d_centered], dim=1))
        if self._capture_dino_io:
            self._dino_capture['concat'] = torch.cat(
                [inp_enc_level4, d_centered], dim=1)
            self._dino_capture['guided'] = guided
        self._record_stats_concat(inp_enc_level4, d_centered, mode)
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

    @torch.no_grad()
    def _record_stats_concat(self, latent_in, d_centered, mode):
        """The addition arms' three numbers, recovered from the split weight.

        fuse(cat([F, D])) = conv(F, W_F) + conv(D, W_D) + b, so each half's
        contribution is a real tensor that can be normed, and the ratio means
        the same thing it means in the addition arms. Written into the SAME
        `last_dino_stats` keys, so the model wrapper's TensorBoard tags,
        dino_stability.csv and the abort gate are reused unmodified.
        """
        eps = 1e-8
        contrib_F = F_nn.conv2d(latent_in, self.W_F)
        contrib_D = F_nn.conv2d(d_centered, self.W_D)
        ln = float(contrib_F.detach().norm())
        pn = float(contrib_D.detach().norm())
        self.last_dino_stats = {
            'latent_norm': ln,
            'projected_norm': pn,
            'injection_ratio': pn / (ln + eps),
        }
        self.last_dino_mode = mode
        self.last_mean_key = ('mu_train128' if mode == 'train128'
                              else 'mu_eval256')
