"""ACA WITHOUT the self-attention branch — the F_sa ablation.

WHY THIS EXISTS

`DinoAca` computes two attentions and sums them before one output conv:

    guided = project_out(F_sa + alpha * F_ca) + F

`F_ca` is the point of the block: the latent attending to the DINO prior.
`F_sa` is the latent attending to ITSELF — and that is Restormer's own MDTA,
step for step (1x1 + 3x3 depthwise projections, L2-normalise along the token
axis, per-head multiplicative temperature, channel x channel softmax). The
ACA is applied to `inp_enc_level4`, which is handed straight to `self.latent`
— EIGHT transformer blocks that each already run exactly that operation. So
`F_sa` is a ninth MDTA immediately in front of eight more, with its own
weights, trained from scratch.

WHAT IT COSTS, MEASURED, NOT ESTIMATED (net_g_272000.pth, dinolight-render):

    ACA self-attention half   (to_q/k/v + temperature_sa)      452,742
    ACA cross-attention half  (to_*_cross + temperature_ca)    452,742
    shared (norms, project_out, alpha)                         148,993
    ACA total                                                1,054,477
    the 8 latent blocks it feeds, MDTA attention only         4,801,600

WHAT THE TRAINED MODEL DID WITH IT. `dino_aca.py` already logs
`aca_ca_to_sa_ratio = ||alpha * F_ca|| / ||F_sa||`. Over dinolight-render's
300k that ratio rose from 0 (alpha and P are both zero-init, so F_ca starts at
exactly 0) to a mean of **8.157 over the last 100k iterations**. The DINO half
ends up carrying ~8x the magnitude of the self half.

THAT IS A MAGNITUDE, NOT A CAUSAL CLAIM, AND THE DISTINCTION MATTERS. A term
with a small norm can still matter: `F_sa` and `alpha * F_ca` are summed and
pushed through ONE shared `project_out`, so `F_sa` is also the baseline the
cross term is added onto. The ratio motivates this arm; it does not settle it.
Training this ablation is what settles it.

THIS IS NOT A CLAIM THAT DINOLight IS WRONG. dinolight-render reproduces a
published block faithfully and should keep doing so. This is a separate arm
that asks one question about it.

HOW THE ONE-FACTOR PROPERTY IS ACHIEVED

`DinoAcaNoSa` SUBCLASSES `DinoAca` and calls its `__init__` unchanged, then
deletes `to_q`, `to_k`, `to_v` and `temperature_sa`. That is deliberate and
buys two things a from-scratch rewrite could not:

  1. `_attend`, `_split`, `attn_shape`, both LayerNorms and `project_out` are
     INHERITED, so the cross-attention path is provably the same code aca-L6
     runs. There is still exactly one implementation of the operator.
  2. Because the parent constructs the SA projections FIRST and the cross
     projections SECOND, building the parent and then deleting the SA half
     leaves `to_q_cross` / `to_k_cross` / `to_v_cross` holding the SAME random
     draws they would hold in aca-L6 under the same seed. The two arms
     therefore start from an identical cross-attention branch, and the only
     difference at step 0 is the presence of F_sa.

The construct-then-delete costs a few hundred kB of transient allocation once
at build time. It is not a leak: the modules are unreferenced after `del` and
they never enter `state_dict()` or the optimizer, which is asserted by the
smoke test (zero `to_q.` / `to_k.` / `to_v.` / `temperature_sa` keys).

ZERO-INIT IS UNCHANGED. `project_out` is still zero, so `guided == feat` at
step 0 and an arm using this block is still bit-identical to E0 at step 0.
"""

import torch

from basicsr.models.archs.dino_aca import DinoAca


class DinoAcaNoSa(DinoAca):
    """`DinoAca` with the F_sa branch removed: guided = P(alpha*F_ca) + F."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the self-attention branch. Order matters only in that the
        # parent has already drawn the cross projections by this point, which
        # is exactly the property documented above.
        del self.to_q, self.to_k, self.to_v
        del self.temperature_sa

    def forward(self, feat, prior, collect_stats=False):
        """feat [B,dim,h,w] (the latent), prior [B,dim,h,w] (projected DINO).

        Returns the same 3-tuple as `DinoAca` — (guided, ca_term, stats) — so
        an arm can swap the block in without touching its forward().
        """
        if feat.shape != prior.shape:
            raise ValueError(f'ACA needs matching shapes, got feat '
                             f'{tuple(feat.shape)} and prior {tuple(prior.shape)}')
        x = self.norm_f(feat)
        xd = self.norm_d(prior)

        f_ca, attn_ca = self._attend(self.to_q_cross(x), self.to_k_cross(xd),
                                     self.to_v_cross(xd),
                                     self.temperature_ca, collect_stats)

        alpha = torch.sigmoid(self.alpha_logit)
        ca_term = alpha * f_ca                            # the DINO contribution
        guided = self.project_out(ca_term) + feat

        stats = {}
        if collect_stats:
            with torch.no_grad():
                stats = {
                    'aca_alpha': float(alpha),
                    # F_sa does not exist here. The tag is emitted as exactly
                    # 0.0 rather than omitted, so a plot of the ladder does not
                    # silently change length between arms -- and so that a
                    # reader who greps for it sees "absent", not "missing".
                    'aca_ca_to_sa_ratio': 0.0,
                    'aca_injected_norm': float((guided - feat).norm()),
                }
                for i, val in enumerate(
                        self.temperature_ca.detach().reshape(-1).tolist()):
                    stats[f'aca_temp_ca_h{i}'] = float(val)
                p = attn_ca.clamp_min(1e-12)
                ent = (-(p * p.log()).sum(-1)).mean(dim=(0, 2))
                for i, val in enumerate(ent.tolist()):
                    stats[f'aca_entropy_ca_h{i}'] = float(val)
        return guided, ca_term, stats
