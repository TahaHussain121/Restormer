"""DinoGate — a per-position, per-channel switch on the projected DINO prior.

DELIBERATELY NOT A `*_arch.py` FILE. `basicsr/models/archs/__init__.py` scans
for `*_arch.py` and registers every match as a network; this is a building
block, so it stays out of that scan. Both gated arms import `DinoGate` from
here, so there is exactly ONE implementation and the arms differ only in which
image DINO reads.

WHAT IT COMPUTES

    g      = sigmoid( G( concat[ F , P(D) ] ) )     [B, 384, h, w], in (0,1)
    guided = F + g * P(D)

`G` is a single 1x1 convolution from 768 channels (the feature and the
projected prior stacked) to 384. So the switch at one position and one channel
is decided by looking at BOTH the network's own feature there and the prior
there — it can compare them rather than judging the prior in isolation.

WHY THIS AND NOT THE OPERATORS ALREADY TESTED. Three arms have varied HOW the
prior is mixed and all three were nulls at matched depth (concat +/-0.02, ACA
+0.030 at 4.6x the parameters). What none of them varied is HOW MUCH prior
arrives, per position. The additive arms add all of it everywhere. AFFM chooses
the DEPTH MIX at each position but its weights sum to 1 by construction, so it
cannot add less in total. `DinoAca` has a gate, but ONE SCALAR for the whole
image, so it cannot vary across positions either. This module is the missing
degree of freedom, and it is the cheapest way to ask whether "let the model
decide" is worth anything on this data.

PARAMETERS. 768*384 + 384 = 295,296, i.e. it doubles addition-render's added
parameters and is about a fifth of `DinoAca`'s 1.05M. Reported honestly: this
arm is NOT parameter-matched to addition-render, so a gain here is confounded
with capacity exactly as the ACA arms are. The control that partly settles it
already exists — aca-L6 spent 4.6x the parameters for +0.030 dB, which says
capacity is roughly neutral at this data scale.

INITIALISATION, and the one-step staircase.

`G` is ZERO-initialised (weight and bias), so every logit is 0 and the gate
starts EXACTLY 0.5 at every position and channel — deterministic and
seed-independent, the same property AFFM's uniform 1/L start has. The arm
therefore begins as addition-render at half strength, and because `P` is itself
learned and zero-initialised, that factor of two is absorbed by `P` growing;
it is a starting point, not a handicap.

Step-0 equality with E0 is preserved by `P`, not by the gate: P(D) == 0 at
initialisation, so `guided == F` exactly.

The gradient staircase is ONE step deep, and it runs the opposite way round
from `DinoAca`'s. At step 1, d(g*P(D))/dP = g = 0.5, which is NON-ZERO, so `P`
learns immediately. But d(g*P(D))/dG carries a factor of P(D), which is 0, so
`G` receives EXACTLY zero gradient on the first backward and learns from step 2.
This is the same structure AFFM has and it is asserted in the smoke test rather
than assumed. Note it cannot deadlock the way `aca-L6-nosa` did: there, BOTH
paths to the loss passed through a zero-initialised weight, so nothing upstream
ever moved. Here `P` is live from step 1 and pulls `G` alive at step 2.

MONITORING. The arm logs `projected_norm` as ||g * P(D)||, the quantity that is
actually injected, so `injection_ratio` and the existing gate rules read the
same shape of thing they read in every other arm. The gate's own statistics —
mean, standard deviation across positions, and the fraction of positions below
0.1 and above 0.9 — are published as observations through the model wrapper's
generic hook. No stopping rule is defined on them, deliberately.
"""

import torch
import torch.nn as nn


class DinoGate(nn.Module):
    """Per-position, per-channel gate on a projected prior. Returns (guided,
    injected, stats)."""

    def __init__(self, dim=384, bias=True):
        super().__init__()
        self.dim = int(dim)
        # 2*dim in: the feature and the projected prior, stacked on channels.
        self.G = nn.Conv2d(2 * self.dim, self.dim, kernel_size=1, bias=bias)
        nn.init.zeros_(self.G.weight)          # -> logit 0 -> gate exactly 0.5
        if self.G.bias is not None:
            nn.init.zeros_(self.G.bias)

    def forward(self, feat, prior, collect_stats=False):
        """feat [B,dim,h,w] (the latent input), prior [B,dim,h,w] (P(D))."""
        if feat.shape != prior.shape:
            raise ValueError(f'gate needs matching shapes, got feat '
                             f'{tuple(feat.shape)} and prior {tuple(prior.shape)}')
        g = torch.sigmoid(self.G(torch.cat([feat, prior], dim=1)))
        injected = g * prior
        guided = feat + injected

        stats = {}
        if collect_stats:
            with torch.no_grad():
                stats = {
                    'gate_mean': float(g.mean()),
                    'gate_std': float(g.std()),
                    'gate_frac_closed': float((g < 0.1).float().mean()),
                    'gate_frac_open': float((g > 0.9).float().mean()),
                    # spread ACROSS POSITIONS of the channel-averaged map: the
                    # number that separates a real spatial gate from a scalar
                    'gate_spatial_std': float(g.mean(dim=1).std()),
                }
        return guided, injected, stats
