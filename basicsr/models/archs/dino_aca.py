"""ACA — gated channel cross-attention for injecting a DINO prior at the latent.

DELIBERATELY NOT A `*_arch.py` FILE. `basicsr/models/archs/__init__.py` scans for
`*_arch.py` and imports every match into the network registry; this is a
building block, not a network, so it stays out of that scan. Any arm that wants
gated channel cross-attention imports `DinoAca` from here — the planned
single-layer `aca-L6` arm included — so there is exactly ONE implementation of
this operator in the repo and the arms differ only in what they feed it.

WHAT IT COMPUTES

    X  = LayerNorm(F)          F is the Restormer latent, [B, 384, h, w]
    X' = LayerNorm(D_proj)     the DINO prior, already projected to 384

    Q, K, V, Q'  <- X          (four projections of the feature)
    K', V'       <- X'         (two projections of the prior)

    F_sa = TransposedAttn(Q,  K,  V )      plain self-attention on the feature
    F_ca = TransposedAttn(Q', K', V')      the feature ATTENDING TO the prior

    alpha  = sigmoid(alpha_logit)          one learnable scalar
    guided = project_out(F_sa + alpha * F_ca) + F

Each projection is a 1x1 conv followed by a 3x3 DEPTHWISE conv -- Restormer's
own MDTA projection style (`restormer_arch.py:105-107`), with `bias` taken from
the same config key the trunk uses, so this module cannot drift from the trunk's
convention.

THE ATTENTION IS TRANSPOSED (CHANNEL), NOT SPATIAL. THIS IS THE POINT.

    rearrange to [B, heads, C/heads, H*W]
    L2-normalise along the TOKEN axis (dim=-1)
    attn = softmax( (Q @ K^T) * temperature )      -> [B, heads, C/heads, C/heads]

The attention matrix is **C/heads x C/heads and therefore independent of the
token count**. That is the whole reason this operator was chosen over the one in
`restormer_dino_crossattn_render_arch.py`, which used a spatial 256x256 softmax:
that arm scored +2.00 dB at crop128 and -7.56 dB at full256 FROM THE SAME
WEIGHTS, because the DINO grid goes 16x16 (256 tokens) at train to 32x32 (1024
tokens) at eval and a spatial softmax has to renormalise over four times as many
competitors. A channel softmax does not change shape at all between the two
regimes. `attn_shape()` is exposed so a smoke test can VERIFY that rather than
trust this paragraph.

TEMPERATURE follows Restormer exactly (`restormer_arch.py:103,124`): a per-head
`nn.Parameter(torch.ones(heads, 1, 1))` that MULTIPLIES the logits. It is
initialised to 1.0 and it is a scale, not a divisor -- do not "fix" it into a
division by temperature, that would invert its meaning. The self- and
cross-attention paths get SEPARATE temperatures, because they are two different
attention operations over two different key spaces.

ALPHA is the headline diagnostic of the whole arm. `alpha_logit` starts at -2.0,
so alpha starts at ~0.119: the DINO path opens quietly. Because it is a gate the
network can also CLOSE it again -- if alpha decays toward 0 the model is saying
the prior does not help and is falling back to plain self-attention, which is a
clean interpretable negative result rather than a collapse. The failed spatial
cross-attention arm had no such escape hatch.

ZERO-INIT ON `project_out` means the whole module returns exactly F at step 0,
so an arm using it starts bit-identical to E0. The cost is a one-step gradient
staircase: everything upstream of `project_out` reaches the loss only through it,
so Q/K/V and the AFFM see zero gradient on the first backward and learn from
step 2. That is expected and is asserted in the smoke test, not assumed away.
"""

import torch
import torch.nn as nn

from basicsr.models.archs.restormer_arch import LayerNorm


def mdta_projection(dim, bias):
    """Restormer's MDTA projection: 1x1 conv then a 3x3 depthwise conv.

    Kept as a function so every projection in this file is provably the same
    shape of thing, and so `bias` is threaded from the config in one place.
    """
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,
                  bias=bias))


class DinoAca(nn.Module):
    """Gated channel cross-attention. Returns (guided, ca_term, stats)."""

    def __init__(self, dim=384, heads=6, bias=False, alpha_init=-2.0,
                 LayerNorm_type='WithBias'):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'dim {dim} not divisible by heads {heads}')
        self.dim, self.heads = int(dim), int(heads)
        self.head_dim = self.dim // self.heads

        self.norm_f = LayerNorm(dim, LayerNorm_type)      # on the feature
        self.norm_d = LayerNorm(dim, LayerNorm_type)      # on the prior

        # four projections of the FEATURE, two of the PRIOR
        self.to_q = mdta_projection(dim, bias)
        self.to_k = mdta_projection(dim, bias)
        self.to_v = mdta_projection(dim, bias)
        self.to_q_cross = mdta_projection(dim, bias)
        self.to_k_cross = mdta_projection(dim, bias)
        self.to_v_cross = mdta_projection(dim, bias)

        # Restormer's convention: per-head, initialised to 1.0, MULTIPLICATIVE
        self.temperature_sa = nn.Parameter(torch.ones(self.heads, 1, 1))
        self.temperature_ca = nn.Parameter(torch.ones(self.heads, 1, 1))

        # the gate. -2.0 -> sigmoid ~ 0.119
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        nn.init.zeros_(self.project_out.weight)           # guided == F at init
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

    # ------------------------------------------------------------------
    def attn_shape(self, h, w):
        """The attention matrix shape this module will produce.

        Takes h, w and IGNORES them on purpose: that independence is the
        property the smoke test checks by calling this at both scales.
        """
        return (self.heads, self.head_dim, self.head_dim)

    def _split(self, t, b):
        return t.reshape(b, self.heads, self.head_dim, -1)

    def _attend(self, q, k, v, temperature, want_attn=False):
        b = q.shape[0]
        h, w = q.shape[-2:]
        q, k, v = self._split(q, b), self._split(k, b), self._split(v, b)
        # L2 along the TOKEN axis -> the matmul below contracts over tokens and
        # yields a CHANNEL x CHANNEL matrix, independent of how many tokens
        # there are.
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * temperature
        attn = attn.softmax(dim=-1)                       # [B,heads,C/h,C/h]
        out = (attn @ v).reshape(b, self.dim, h, w)
        return (out, attn) if want_attn else (out, None)

    # ------------------------------------------------------------------
    def forward(self, feat, prior, collect_stats=False):
        """feat [B,dim,h,w] (the latent), prior [B,dim,h,w] (projected DINO)."""
        if feat.shape != prior.shape:
            raise ValueError(f'ACA needs matching shapes, got feat '
                             f'{tuple(feat.shape)} and prior {tuple(prior.shape)}')
        x = self.norm_f(feat)
        xd = self.norm_d(prior)

        f_sa, attn_sa = self._attend(self.to_q(x), self.to_k(x), self.to_v(x),
                                     self.temperature_sa, collect_stats)
        f_ca, attn_ca = self._attend(self.to_q_cross(x), self.to_k_cross(xd),
                                     self.to_v_cross(xd),
                                     self.temperature_ca, collect_stats)

        alpha = torch.sigmoid(self.alpha_logit)
        ca_term = alpha * f_ca                            # the DINO contribution
        guided = self.project_out(f_sa + ca_term) + feat

        stats = {}
        if collect_stats:
            with torch.no_grad():
                sa_n = float(f_sa.norm())
                ca_n = float(ca_term.norm())
                stats = {
                    'aca_alpha': float(alpha),
                    'aca_ca_to_sa_ratio': ca_n / (sa_n + 1e-8),
                    'aca_injected_norm': float((guided - feat).norm()),
                }
                for tag, t in (('sa', self.temperature_sa),
                               ('ca', self.temperature_ca)):
                    for i, val in enumerate(t.detach().reshape(-1).tolist()):
                        stats[f'aca_temp_{tag}_h{i}'] = float(val)
                for tag, a in (('sa', attn_sa), ('ca', attn_ca)):
                    p = a.clamp_min(1e-12)
                    ent = (-(p * p.log()).sum(-1)).mean(dim=(0, 2))
                    for i, val in enumerate(ent.tolist()):
                        stats[f'aca_entropy_{tag}_h{i}'] = float(val)
        return guided, ca_term, stats
