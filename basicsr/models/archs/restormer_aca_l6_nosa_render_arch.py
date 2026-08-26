"""aca-L6-nosa: aca-L6 with the ACA self-attention branch REMOVED.

THE ONE FACTOR IS F_sa. NOTHING ELSE.

    addition-render   guided = F + P(D)
    aca-L6            guided = project_out(F_sa + alpha * F_ca) + F
    aca-L6-nosa       guided = project_out(       alpha * F_ca) + F     <- this

Same layer (B6), same render source, same centering mean, same injection point
(`inp_enc_level4`), same trunk, same schedule, same seed. Against **aca-L6**
this is a genuine one-factor ablation and it is NOT capacity confounded in the
usual direction: this arm is SMALLER by exactly the size of the removed branch.

WHY. `F_sa` is the latent attending to itself, which is Restormer's own MDTA
step for step, and the ACA output is handed straight to `self.latent` — eight
transformer blocks that each already run that operation. See
`dino_aca_nosa.py` for the parameter accounting and for the measured
`aca_ca_to_sa_ratio` of ~8.16 over dinolight-render's last 100k iterations
that motivated this arm.

  THAT RATIO IS A MAGNITUDE, NOT A CAUSAL RESULT. It is why this arm exists;
  it is not a substitute for running it. A term with a small norm can still
  matter, because `F_sa` is also the baseline the gated cross term is summed
  onto before a single shared `project_out`.

WHAT THIS ARM IS NOT. It is NOT a claim that dinolight-render is implemented
wrongly. dinolight-render reproduces a published block faithfully and should
continue to. This arm asks one question about that block on our data.

THREE COMPARISONS, THREE DIFFERENT CAVEATS -- DO NOT CONFLATE THEM

  vs aca-L6            ONE factor (F_sa), and the cross-attention branch is
                       initialised IDENTICALLY (see `DinoAcaNoSa`). This is
                       the comparison this arm was built for.
  vs addition-render   ONE factor in spirit (add vs gated cross-attention)
                       and a much smaller capacity gap than aca-L6's ~4.6x,
                       but still not parameter-matched. Report the delta.
  vs dinolight-render  THREE factors at once (layer count, AFFM, F_sa).
                       Meaningless as an ablation. Do not report it as one.

INHERITANCE. Everything comes from `RestormerAcaL6Render` -- including
`forward()` and the inherited `dino_prior`, so the prior this arm sees is
byte-for-byte the prior aca-L6 and addition-render see. Only `self.aca` is
replaced, and `DinoAcaNoSa` returns the same `(guided, ca_term, stats)` tuple,
so the forward path is provably identical apart from the block itself.

RNG. The parent builds its `DinoAca` inside an RNG fence and RESTORES the
state afterwards. Rebuilding here therefore starts from the same state, so
`to_q_cross` / `to_k_cross` / `to_v_cross` receive the SAME draws as aca-L6's
under the same seed, and the trunk is byte-identical to E0's as always. The
replacement is fenced again anyway, so nothing downstream can shift.

INITIALISATION is unchanged: `project_out` zero -> `guided == F` -> step-0
output bit-identical to E0. `alpha_logit = -2.0` -> alpha ~ 0.119, and the
gate can still close, which remains the clean interpretable negative result.

MONITORING. `projected_norm` still logs `||alpha * F_ca||` before the zero-init
output conv, so `injection_ratio` stays comparable to the OTHER ACA ARMS and
stays NOT comparable to the addition arms' ~1.03. `aca_ca_to_sa_ratio` is
emitted as exactly 0.0 rather than dropped, so the observation series keeps the
same shape across the ladder.
"""

import torch

from basicsr.models.archs.restormer_aca_l6_render_arch import (
    RestormerAcaL6Render)
from basicsr.models.archs.dino_aca_nosa import DinoAcaNoSa


class RestormerAcaL6NoSaRender(RestormerAcaL6Render):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---- RNG FENCE around the replacement -----------------------------
        # The parent already fenced and restored, so this draw sequence is the
        # one aca-L6's ACA consumed; fencing again keeps that true regardless
        # of what the parent does in future.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.aca = DinoAcaNoSa(dim=self.latent_channels,
                                   heads=self.aca_heads,
                                   bias=self.aca_trunk_bias,
                                   alpha_init=self.aca_alpha_init,
                                   LayerNorm_type=self.aca_trunk_ln)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    # forward() and dino_prior are INHERITED UNCHANGED. Do not override them:
    # the whole point is that only the block differs.
