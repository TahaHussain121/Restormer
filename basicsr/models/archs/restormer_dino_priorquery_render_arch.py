"""priorquery-render: cross-attention with the DINO prior as QUERY.

The direction counterpart of crossattn-render, and the direction used by the
published DINO-prior restoration methods:

    crossattn-render   Q = radar latent F,  K = V = DINO D    (our direction)
    priorquery-render  Q = DINO D,          K = V = radar F   (Perceive-IR's PGCA,
                                                               inherited by DSGIR)

The pair exists to isolate DIRECTION as a factor. Nothing here is shared with
the crossattn-render module -- not the class, not a mode flag, not an argument
swap at the call site. The duplication below is DELIBERATE: a shared module
would mean a later edit for one direction could silently change the other, and
these two arms only mean something if each is frozen independently.

WHAT IS CONCEPTUALLY DIFFERENT, AND MUST NOT BE SMOOTHED OVER
    The attention output here is a re-mix of RADAR values. DINO content never
    reaches the output directly -- it enters ONLY through the softmax weights.
    DINO is a ROUTER over radar positions, not a content source.

    There is therefore NO second path injecting D directly: no concat, no
    residual P(D), nothing. Routing-only is the hypothesis under test, and a
    content path would quietly turn this into a different experiment.

SHAPES, verified by a real forward pass before this file was written:
    D : [B, 768, 16, 16] -> 256 QUERY tokens of dim 768
    F : [B, 384, 16, 16] -> 256 key/value tokens of dim 384

    The output token count is set by the QUERY count -- i.e. by the number of
    DINO tokens. The residual add back onto F is only well defined because the
    DINO grid is 16x16, exactly matching the latent grid. `dino_prior` already
    asserts that equality (assert_no_interpolation_needed), and the forward
    below re-checks it explicitly rather than interpolating, padding or cropping
    anything to make a mismatch fit.

MODULE
    LayerNorm(768) on the query tokens, LayerNorm(384) on the key/value tokens,
    both pre-norm with elementwise_affine=True, then

        W_q 768->384    W_k 384->384    W_v 384->384    W_o 384->384 (zero-init)
        heads = 6 (= 384/64),  attn = softmax(Q K^T / sqrt(64)) V
        guided = F + reshape(W_o(attn))

ZERO-INIT GRADIENT STAIRCASE (identical to crossattn-render)
    W_o is zero at the OUTPUT of the fusion, so on the FIRST backward
    dL/dW_q = dL/dW_k = dL/dW_v = 0 -- the gradient must travel through W_o.
    W_o learns at step 1, q/k/v from step 2. A step-1 assertion on q/k/v would
    fail, correctly. The smoke test asserts the staircase and fails only if
    q/k/v are still dead at step 10.

RNG ORDER
    The attention module is built inside an EXPLICIT SAVE/RESTORE FENCE, the
    same device the parent uses for the ViT and P, so every trunk weight and
    every later RNG draw stays byte-identical to E0 at the same seed.

PARAMETERS (+741,120 over E0)
    W_q 295,296 | W_k 147,840 | W_v 147,840 | W_o 147,840 | LN_q 1,536 | LN_kv 768

    NOT parameter-matched to crossattn-render (888,576). The asymmetry is
    structural: there both K and V came from the wide 768 side, here only Q
    does. Recorded, not corrected.

MONITORING -- AND ONE NUMBER THAT DOES NOT MEAN WHAT IT MEANS ELSEWHERE
    latent_norm    = ||F||
    projected_norm = ||W_o(attn)||
    injection_ratio = projected_norm / (latent_norm + eps)

    In every other arm the numerator is built from DINO features. HERE IT IS
    BUILT FROM RADAR VALUES. The ratio is logged under the same tags for
    continuity, but it is NOT comparable to the other arms' ratio and must not
    share a table column with them without that caveat. See the devlog.

    attn_entropy and attn_diag_mass are computed every `attn_stats_freq`
    forwards. attn_diag_mass is the key metric for this arm: near 1.0 means each
    DINO token routes to the radar token at its own position, i.e. the module
    has collapsed to a per-position reweighting that a simpler operator could
    do. That is a RESULT to report, not a failure.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)


class DinoPriorQueryAttention(nn.Module):
    """Multi-head cross-attention: DINO prior queries, radar keys/values.

    Deliberately a separate class from the radar-query module, and deliberately
    not nn.MultiheadAttention: q/k/v have different widths (768 and 384), the
    packed weight would hide W_q/W_k/W_v from assertion, and the per-head
    attention matrix is needed for attn_diag_mass.
    """

    def __init__(self, dim_prior=768, dim_latent=384, heads=6):
        super().__init__()
        if dim_latent % heads != 0:
            raise ValueError(f'dim_latent {dim_latent} not divisible by '
                             f'heads {heads}')
        self.dim_prior, self.dim_latent, self.heads = dim_prior, dim_latent, heads
        self.head_dim = dim_latent // heads              # 64
        self.scale = 1.0 / math.sqrt(self.head_dim)      # 1/sqrt(64)

        # PRE-NORM: query side is the 768-wide prior, key/value side the latent
        self.norm_q = nn.LayerNorm(dim_prior, elementwise_affine=True)
        self.norm_kv = nn.LayerNorm(dim_latent, elementwise_affine=True)
        self.W_q = nn.Linear(dim_prior, dim_latent, bias=True)    # 768 -> 384
        self.W_k = nn.Linear(dim_latent, dim_latent, bias=True)   # 384 -> 384
        self.W_v = nn.Linear(dim_latent, dim_latent, bias=True)   # 384 -> 384
        self.W_o = nn.Linear(dim_latent, dim_latent, bias=True)   # 384 -> 384
        nn.init.zeros_(self.W_o.weight)                  # exact no-op at step 0
        nn.init.zeros_(self.W_o.bias)

        self.last_attn_stats = {}

    def forward(self, D_grid, F_grid, collect_attn_stats=False):
        """D_grid [B,768,h,w] queries, F_grid [B,384,h,w] keys/values.

        Returns [B,384,h,w] -- a re-mix of RADAR values, routed by DINO.
        """
        b, _, h, w = F_grid.shape
        if D_grid.shape[-2:] != F_grid.shape[-2:]:
            # the residual add would be ill-defined; never reshape to fit
            raise RuntimeError(
                f'prior grid {tuple(D_grid.shape[-2:])} != latent grid '
                f'{tuple(F_grid.shape[-2:])}. In this direction the output '
                f'token count follows the QUERY (the prior), so a mismatch '
                f'cannot be residually added. Refusing to interpolate, pad or '
                f'crop.')
        n = h * w
        q_tok = D_grid.flatten(2).transpose(1, 2)        # [B,N,768] prior
        kv_tok = F_grid.flatten(2).transpose(1, 2)       # [B,N,384] radar

        q = self.W_q(self.norm_q(q_tok))
        kv = self.norm_kv(kv_tok)
        k = self.W_k(kv)
        v = self.W_v(kv)                                 # VALUES ARE RADAR

        def split(t):
            return t.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        qh, kh, vh = split(q), split(k), split(v)

        if collect_attn_stats:
            logits = (qh @ kh.transpose(-2, -1)) * self.scale
            attn = logits.softmax(dim=-1)                # [B,heads,Nq,Nk]
            ctx = attn @ vh
            with torch.no_grad():
                p = attn.clamp_min(1e-12)
                self.last_attn_stats = {
                    'attn_entropy': float(-(p * p.log()).sum(-1).mean()),
                    'attn_diag_mass': (
                        float(attn.diagonal(dim1=-2, dim2=-1).mean())
                        if attn.shape[-1] == attn.shape[-2] else float('nan')),
                    'attn_uniform_entropy': float(math.log(attn.shape[-1])),
                }
        else:
            ctx = F_nn.scaled_dot_product_attention(qh, kh, vh)

        ctx = ctx.transpose(1, 2).reshape(b, n, self.dim_latent)
        injected = self.W_o(ctx)                         # zero at step 0
        return injected.transpose(1, 2).view(b, self.dim_latent, h, w)


class RestormerDinoPriorQueryRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        fusion = kwargs.pop('dino_fusion', 'priorquery')
        if fusion != 'priorquery':
            raise ValueError(
                f'RestormerDinoPriorQueryRender requires dino_fusion: '
                f'priorquery, got {fusion!r}. The radar-query direction is '
                f'RestormerDinoCrossAttnRender and is a separate arm.')
        heads = int(kwargs.pop('dino_attn_heads', 6))
        self.attn_stats_freq = int(kwargs.pop('dino_attn_stats_freq', 5000))
        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion

        # the addition arm's residual projection is removed: no P.* key in any
        # checkpoint, and -- more importantly for THIS arm -- no second path by
        # which DINO content could reach the output. Routing only.
        del self.P

        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.pqattn = DinoPriorQueryAttention(
                dim_prior=self.dino_embed_dim,
                dim_latent=self.latent_channels, heads=heads)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._forward_count = 0
        self.last_attn_stats = {}

    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}. priorquery-render needs '
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

        # ---- the ONLY difference from crossattn-render: the DIRECTION --------
        # dino_prior is inherited unchanged: frozen B6 patch tokens from the
        # render, centered by the render mean, as a [B,768,g,g] grid. Here that
        # grid supplies the QUERIES; the radar latent supplies keys AND values.
        # No pooling, no broadcast, and no direct injection of D anywhere.
        d_centered = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        d_centered = d_centered.to(inp_enc_level4.dtype)
        self._forward_count += 1
        want_stats = (self._capture_dino_io
                      or self._forward_count % self.attn_stats_freq == 1)
        injected = self.pqattn(d_centered, inp_enc_level4,
                               collect_attn_stats=want_stats)
        guided = inp_enc_level4 + injected
        if want_stats:
            self.last_attn_stats = dict(self.pqattn.last_attn_stats)
        if self._capture_dino_io:
            self._dino_capture['injected'] = injected
            self._dino_capture['guided'] = guided
        self._record_stats(inp_enc_level4, injected, mode)
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
