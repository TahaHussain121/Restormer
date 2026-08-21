"""crossattn-render: the cross-attention fusion arm of addition-render.

ONE thing differs from `RestormerDinoSpatialRender`: how the centered DINO grid
meets the latent tensor.

    addition-render   guided = F + P(D)                      P: 1x1, 768 -> 384
    concat-render     guided = fuse(cat([F, D], dim=1))    fuse: 1x1, 1152 -> 384
    crossattn-render  guided = F + W_o(MHCA(Q=F, K=D, V=D))

Addition and concat both mix token i of DINO with token i of the radar latent
and nothing else -- the fusion is strictly positionwise. Cross-attention removes
that constraint: every radar position can read every DINO position.

DIRECTION, ON PURPOSE AND NOT NEGOTIABLE
    RADAR IS THE QUERY. DINO SUPPLIES KEYS AND VALUES.

    The published DINO-prior restoration methods run it the other way -- the
    prior is the query and the restoration feature is key and value (DINO-IR,
    arXiv 2312.01677; Perceive-IR's PGCA, TIP 2026, where Y_l is the query and
    X_l^d is both K and V; DSGIR, Neurocomputing 696, which reuses PGCA
    unchanged). We reverse it deliberately: the point here is for DINO to inform
    the radar features, not for the radar to reorganise DINO's. Do not "fix"
    this to match the papers.

SHAPES, all verified by a real forward pass before this file was written:
    F : [B, 384, 16, 16] -> 256 query tokens of dim 384
    D : [B, 768, 16, 16] -> 256 key/value tokens of dim 768
    256 == 256, so the attention matrix is square and `attn_diag_mass` below is
    well defined. The grids are equal because dino_prior already asserts it.

THE FEATURES ARE SPATIAL. D is the [B,768,g,g] B6 patch-token grid, flattened to
tokens in row-major order and never pooled, never broadcast, and never the CLS
token (`dino_shared.extract_block` does not return CLS at all). This class
extends `RestormerDinoSpatialRender`; nothing is inherited from the pooled
global arm, which answers a different question.

MODULE
    LayerNorm(384) on the query tokens, LayerNorm(768) on the key/value tokens,
    both pre-norm with elementwise_affine=True, then

        W_q 384->384    W_k 768->384    W_v 768->384    W_o 384->384 (zero-init)
        heads = 6 (= 384/64),  attn = softmax(Q K^T / sqrt(64)) V
        guided = F + reshape(W_o(attn))

WHY IT STARTS AS AN EXACT NO-OP, AND WHY THE GRADIENT IS DIFFERENT HERE
    W_o's weight and bias are zero, so the injected term is exactly zero at step
    0 and the arm's output is E0's.

    But unlike the addition and concat arms, the zero sits at the OUTPUT of the
    fusion, so on the FIRST backward dL/dW_q = dL/dW_k = dL/dW_v = 0 -- their
    gradient has to travel through W_o, which is still zero. Only W_o receives a
    gradient at step 1. From step 2, W_o is non-zero and q/k/v start learning.
    That is correct behaviour, not a bug; the smoke test asserts exactly this
    staircase (W_o at step 1, q/k/v by step 2) and fails if q/k/v are still
    dead at step 10.

RNG ORDER
    `nn.Linear`/`nn.LayerNorm` draw from the generator, which would shift every
    later draw (data order, crop offsets, augmentation flags) away from E0's for
    the same seed. The whole attention module is therefore built inside an
    EXPLICIT SAVE/RESTORE FENCE, the same device `RestormerDinoSpatial.__init__`
    already uses for the ViT and P. The parent fences its own construction, so
    RNG state on entry is already E0's and restoring it on exit leaves every
    trunk weight and every later draw byte-identical to E0.

PARAMETERS (+888,576 over E0)
    W_q 147,840 | W_k 295,296 | W_v 295,296 | W_o 147,840 | LN_q 768 | LN_kv 1,536
    This is the largest arm in the family: addition 295,296 < concat 442,752 <
    crossattn 888,576. A cross-attention win is therefore NOT by itself evidence
    that attention is the better operator. Recorded in the devlog, not corrected.

MONITORING
    The same three quantities every other arm logs, into the same
    `last_dino_stats` keys, so the model wrapper's TensorBoard tags, the
    dino_stability.csv and the abort gate are reused unmodified:

        latent_norm    = ||F||                 (exactly as the addition arms)
        projected_norm = ||W_o(attn)||         the injected part only
        injection_ratio = projected_norm / (latent_norm + eps)

    Plus two observations this arm alone can make, computed only every
    `attn_diag_freq` forwards because they need the full attention matrix:

        attn_entropy    mean entropy of the 256-way softmax rows, in nats
        attn_diag_mass  mean of the attention matrix diagonal

    attn_diag_mass near 1.0 would mean the attention learned to read only its
    own spatial position -- i.e. it reduced itself to the addition arm. That is
    a RESULT, not a failure. Neither quantity is a gate rule.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)


class DinoCrossAttention(nn.Module):
    """Multi-head cross-attention: radar latent queries, DINO keys/values.

    Deliberately written out rather than delegating to nn.MultiheadAttention:
    that module requires equal embed_dim for q/k/v (ours are 384 and 768), packs
    the projections into one weight so W_q/W_k/W_v could not be inspected or
    asserted on separately, and does not expose the attention matrix per head
    that attn_diag_mass needs.
    """

    def __init__(self, dim_q=384, dim_kv=768, heads=6):
        super().__init__()
        if dim_q % heads != 0:
            raise ValueError(f'dim_q {dim_q} not divisible by heads {heads}')
        self.dim_q, self.dim_kv, self.heads = dim_q, dim_kv, heads
        self.head_dim = dim_q // heads                  # 64
        self.scale = 1.0 / math.sqrt(self.head_dim)     # 1/sqrt(64)

        self.norm_q = nn.LayerNorm(dim_q, elementwise_affine=True)
        self.norm_kv = nn.LayerNorm(dim_kv, elementwise_affine=True)
        self.W_q = nn.Linear(dim_q, dim_q, bias=True)
        self.W_k = nn.Linear(dim_kv, dim_q, bias=True)
        self.W_v = nn.Linear(dim_kv, dim_q, bias=True)
        self.W_o = nn.Linear(dim_q, dim_q, bias=True)
        nn.init.zeros_(self.W_o.weight)                 # exact no-op at step 0
        nn.init.zeros_(self.W_o.bias)

        self.last_attn_stats = {}

    def forward(self, F_grid, D_grid, collect_attn_stats=False):
        """F_grid [B,384,h,w], D_grid [B,768,h,w] -> injected [B,384,h,w]."""
        b, cq, h, w = F_grid.shape
        n = h * w
        # [B,C,h,w] -> [B,N,C], row-major, matching dino_shared.tokens_to_grid
        q_tok = F_grid.flatten(2).transpose(1, 2)
        kv_tok = D_grid.flatten(2).transpose(1, 2)

        q = self.W_q(self.norm_q(q_tok))
        kv = self.norm_kv(kv_tok)
        k = self.W_k(kv)
        v = self.W_v(kv)

        def split(t):
            return t.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        qh, kh, vh = split(q), split(k), split(v)        # [B,heads,N,64]

        if collect_attn_stats:
            # the explicit path, only when the observations are wanted
            logits = (qh @ kh.transpose(-2, -1)) * self.scale
            attn = logits.softmax(dim=-1)               # [B,heads,Nq,Nk]
            ctx = attn @ vh
            with torch.no_grad():
                p = attn.clamp_min(1e-12)
                entropy = float(-(p * p.log()).sum(-1).mean())
                if attn.shape[-1] == attn.shape[-2]:
                    diag = attn.diagonal(dim1=-2, dim2=-1)
                    diag_mass = float(diag.mean())
                else:
                    diag_mass = float('nan')            # non-square: undefined
                self.last_attn_stats = {
                    'attn_entropy': entropy,
                    'attn_diag_mass': diag_mass,
                    'attn_uniform_entropy': float(math.log(attn.shape[-1])),
                }
        else:
            # same maths, fused kernel, no [B,heads,N,N] tensor materialised
            ctx = F_nn.scaled_dot_product_attention(qh, kh, vh)

        ctx = ctx.transpose(1, 2).reshape(b, n, cq)
        injected = self.W_o(ctx)                        # zero at step 0
        return injected.transpose(1, 2).view(b, cq, h, w)


class RestormerDinoCrossAttnRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        # surfaced in the YAML so the fusion operator is visible in the config,
        # not only in the class name.
        fusion = kwargs.pop('dino_fusion', 'crossattn')
        if fusion != 'crossattn':
            raise ValueError(
                f'RestormerDinoCrossAttnRender requires dino_fusion: crossattn, '
                f'got {fusion!r} (residual addition is '
                f'RestormerDinoSpatialRender, concat is '
                f'RestormerDinoConcatRender)')
        heads = int(kwargs.pop('dino_attn_heads', 6))
        # how often the [B,heads,N,N] attention matrix is materialised for the
        # entropy / diag-mass observations. Observations only; never a gate rule.
        self.attn_stats_freq = int(kwargs.pop('dino_attn_stats_freq', 5000))
        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion

        # the residual projection of the addition arm is REMOVED, not left
        # dangling: no P.* key reaches a checkpoint, no unused parameter joins
        # the optimizer. The parent built it inside its own RNG fence, so
        # deleting it here costs nothing.
        del self.P

        # ---- RNG FENCE around the attention module (see module docstring) ---
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.xattn = DinoCrossAttention(dim_q=self.latent_channels,
                                            dim_kv=self.dino_embed_dim,
                                            heads=heads)
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
                f'{list(inp_img.shape)}. crossattn-render needs '
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

        # ---- the ONLY difference from addition-render: the fusion operator ---
        # dino_prior is INHERITED unchanged: frozen B6 patch tokens from the
        # render, centered by the render mean, reshaped to the [B,768,g,g] grid,
        # with the no-interpolation gate asserted inside it. The grid enters
        # attention as keys and values -- never pooled, never broadcast.
        d_centered = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        d_centered = d_centered.to(inp_enc_level4.dtype)
        self._forward_count += 1
        want_stats = (self._capture_dino_io
                      or self._forward_count % self.attn_stats_freq == 1)
        injected = self.xattn(inp_enc_level4, d_centered,
                              collect_attn_stats=want_stats)
        guided = inp_enc_level4 + injected
        if want_stats:
            self.last_attn_stats = dict(self.xattn.last_attn_stats)
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
            # global residual against the RADAR, never the render
            out_dec_level1 = self.output(out_dec_level1) + radar

        return out_dec_level1
