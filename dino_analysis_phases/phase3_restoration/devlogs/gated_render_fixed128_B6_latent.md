# gated-render — does letting the model DECIDE beat adding everywhere?

`Holo_gated_render_fixed128_B6_latent`

Scope: **this arm only**. Written **before the run** (2026-09-07), so the
prediction below is a pre-registration, not a description of a result.

---

## THE DECISION

Addition-render with one change: a per-position, per-channel switch on the
projected prior.

```
addition-render   guided = F + P(D)
gated-render      guided = F + g * P(D),   g = sigmoid(G([F, P(D)])) in (0,1)
```

`G` is one `Conv2d(768 -> 384, 1x1)`: it reads the network's own feature and the
projected prior stacked, at each of the 256 positions, and returns an amount for
each of the 384 channels. Same DINO, same B6, same render, same means, same
zero-init `P`, same injection point, same recipe, same seed 100.

**+295,296 parameters over addition-render (a doubling of the added
parameters, about a fifth of `DinoAca`'s 1.05M). NOT parameter-matched. Report
it that way.** The partial control already exists: aca-L6 spent 4.6x the
parameters for +0.030 dB, so capacity is roughly neutral at this data scale.

## WHY THIS ARM, GIVEN FINDING 6

Three arms varied the fusion OPERATOR (concat, ACA at matched depth, the ACA
ladder) and all were nulls. The recurring objection to plain addition is that it
hands the network the whole prior everywhere. That objection is only half right:
`P` is a learned readout and can silence whole channels — but it applies ONE
rule at every position of every image. **No arm so far has varied HOW MUCH prior
arrives, per position.** AFFM cannot: its weights sum to 1 by construction, so
it chooses the depth mix but never the total. `DinoAca` has a gate, but a single
scalar for the whole image. This is the missing degree of freedom and the
cheapest way to test the objection.

## PRE-REGISTERED PREDICTION

**This arm does not beat addition-render by more than 0.10 dB on test full256.**

Rationale: every operator change measured on this data has been a null, and the
gate is another operator change. The prior here is a clean render whose value
is uniform across the frame, so there is little for a spatial switch to exploit.

  * If it lands within +/-0.10 (n.s.): the fusion-operator finding EXTENDS to
    selection, which is a stronger and more general statement than the current
    one, and the answer to "the model should choose" becomes an experimental no.
  * If it beats +0.10 significantly: selection matters after all, and the
    capacity confound must be stated in the same sentence.
  * If it LOSES: the extra parameters are actively harmful at this data scale,
    which would also qualify the ACA comparison.

**Both protocols are reported.** The additive family loses on neither; a gated
arm that loses on crop128 would put it with the attention family instead.

**SECONDARY OUTCOME, INDEPENDENT OF PSNR, and the more interesting one.** The
gate is observable. `gate_mean`, `gate_std`, `gate_spatial_std`,
`gate_frac_closed` (<0.1) and `gate_frac_open` (>0.9) are logged every 5,000
forwards. Three readings, all publishable:

  * the gate stays near 0.5 everywhere and `gate_spatial_std` stays near zero —
    the network found nothing to select, and the null has a mechanism;
  * the gate opens almost everywhere — it wants all of the prior, which is
    addition-render by another route and explains why addition works;
  * the gate develops real spatial structure — then WHERE it closes is a figure
    worth printing, and the crop-border prediction from Phase 5 becomes testable
    (does it close at the frame borders?).

## SMOKE

`smoke_tests_wave2.py --config <this> --device cpu`: **30/30 passed.** Step-0
output identical to E0 (max dev 0.0), trunk byte-identical to E0, gate map
[2, 384, 16, 16] and exactly 0.5 everywhere at init, all values inside (0,1),
parameter delta exactly +590,592, eval256 path 448 -> 32x32 with no
interpolation and the gate still per position.

**The staircase runs the RIGHT way round, and this was checked explicitly.** `P`
has non-zero gradient on the FIRST backward (dg/dP is the gate, 0.5, not zero);
`G` has exactly zero on the first (it multiplies P(D) = 0) and non-zero on the
second. This is NOT the `aca-L6-nosa` deadlock, where both paths to the loss
passed through a zero-initialised weight and nothing upstream ever moved.
