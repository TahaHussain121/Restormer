# aca-L6-postlatent — does the ACA block do better at the LATER location?

`Holo_aca_postlatent_render_fixed128_B6`

Scope: **this arm only**. Written **before the run** (2026-09-12). The
prediction below is a pre-registration.

---

## THE DECISION

The SAME `DinoAca` block, imported not copied, moved after the latent stage.

```
aca-L6              guided = ACA(F, P(D_B6))       ; latent = self.latent(guided)
aca-L6-postlatent   latent_out = self.latent(F)    ; guided = ACA(latent_out, P(D_B6))
```

Unchanged: the block, B6, the render source, the centering means, `P` and its
zero-init, `alpha_init = -2.0`, 6 heads, the recipe, the seed. **Parameter count
identical to aca-L6** — the tensor either side of the latent stage has the same
shape. One factor moves.

## THE 2x2 THIS FILLS, AND THE COMPARISON THAT DECIDES IT

Fusion operator and injection location have only ever been varied one at a time,
leaving one cell empty (test full256):

| fusion \ location | before latent | after latent |
|---|---|---|
| addition, B6 | 24.081 | **24.387** |
| ACA, B6 | 24.111 | **this arm** |

**The decisive comparison is aca-L6-postlatent against postlatent-render — both
at the same location.** Comparing it to addition-render (the weaker
before-latent reference) would let a location gain masquerade as an operator
gain, which is the error the 2x2 exists to prevent.

## PRE-REGISTERED PREDICTION

**aca-L6-postlatent does not beat postlatent-render by more than 0.10 dB on test
full256, and it still loses significantly on crop128.**

Rationale: the operator has been a null at matched depth in every form tested
(concat −0.016; ACA +0.030 at 4.6x parameters; the per-position gate +0.042),
and every attention-family arm has lost on crop128 regardless of anything else.
Nothing measured so far suggests the location changes the operator's value; the
location effect appeared with plain addition.

Outcomes and what each licenses:

  * **Beats postlatent-render (> +0.10):** the ACA operator has value at this
    location that it lacked at the old one. This would be the first positive
    operator result in the project and would need the capacity confound stated
    in the same sentence (this arm is ~4.6x addition-render's added parameters).
  * **Beats aca-L6 but matches postlatent-render:** the LOCATION helped, exactly
    as it helped plain addition; an operator advantage remains unestablished.
    Expected.
  * **Still loses on crop128:** moving the block does not resolve the protocol
    penalty. Report it as unresolved, not as explained.

Both protocols reported. Selection on validation alone.

## WHAT THIS ARM CANNOT SETTLE, STATED UP FRONT

`DinoAca` is `project_out(F_sa + alpha*F_ca) + F`: self-attention AND gated
cross-attention behind one shared output projection. So this arm, like every ACA
arm before it, tests the WHOLE BLOCK against addition. **It does not isolate the
cross path.** Isolating it requires a matched self-attention control at the same
location — a separate pair of runs, deliberately not attempted here, and only
worth running if this arm leaves the question open.

The companion checkpoint interventions (`run_aca_interventions.sh`) attack the
same question from the other side and cost no training: they ask what the
ALREADY TRAINED aca-L6 depends on. Those establish dependence, not what a model
trained without the component would achieve.

## MONITORING CAVEAT

`latent_norm` measures the latent stage OUTPUT here, the tensor the block acts
on. `projected_norm` keeps aca-L6's convention of `||alpha * F_ca||`. Both are
comparable in KIND to aca-L6 but their 5,000-iteration references are not
numerically comparable to any before-latent arm. Compare this arm with itself
and with postlatent-render.
