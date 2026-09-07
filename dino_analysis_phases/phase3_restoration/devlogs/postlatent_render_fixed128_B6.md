# postlatent-render — the INJECTION-POINT ablation

`Holo_postlatent_render_fixed128_B6`

Scope: **this arm only**. Written **before the run** (2026-09-07).

---

## THE DECISION

Addition-render with one line of the forward pass moved.

```
addition-render     guided = F + P(D)        ;  latent = self.latent(guided)
postlatent-render   latent_out = self.latent(F) ;  guided = latent_out + P(D)
```

The tensor on either side of the latent stage has the same shape
[B, 384, g, g], so `P` is unchanged and the parameter count is **EXACTLY
addition-render's, +295,296 over E0**. This is the cleanest one-factor arm in
the whole study: same DINO, same block, same render, same means, same
projection, same recipe, same seed, **zero parameter difference**.

## WHY IT MATTERS

Chapter §4.5 argues that injecting BEFORE the eight latent blocks strictly
contains what injecting after would provide: each transformer block carries an
identity path, so a signal added at the input rides the residual stream through
all eight and is still present at the output *in addition to* having been
processed. The argument is sound and **it was never tested** — the code
hard-refuses any other injection point, and §9 lists "one injection point" as a
limitation. This arm closes it.

The eight latent blocks hold **14,370,448 parameters, 55.0% of the network**.
Injecting after them means those blocks never see the prior; only the decoder
does.

## PRE-REGISTERED PREDICTION

**postlatent-render is WORSE than addition-render by more than 0.30 dB on test
full256, and remains well above E0.**

Rationale: the residual-path argument, plus the fact that the decoder receives
the prior only through the upsampling path and the skip connections it is
concatenated with come from the encoder, which never saw it.

  * A gap larger than 0.30: §4.5's reasoning is confirmed empirically and the
    limitation in §9 is discharged with a measurement.
  * A NULL: the more interesting outcome. It would mean the prior's benefit is
    realised in the decoder rather than in latent processing, which would
    reframe §4.5 from "strictly contains" to "the location does not matter", and
    would sit alongside the operator null as a second architectural
    non-finding.
  * BELOW E0: would indicate the prior actively disrupts the decoder, which no
    current reading predicts.

**MONITORING CAVEAT, WRITE IT DOWN.** `latent_norm` here measures the latent
stage OUTPUT, not its input, because that is the tensor being added to. The gate
rules are unchanged in form but their 5,000-iteration reference is **not
comparable with the other arms'**. Compare this arm only with itself.

## SMOKE

`smoke_tests_wave2.py`: **24/24 passed.** Step-0 identical to E0, trunk
byte-identical, parameter delta identical to addition-render's, and three checks
specific to this arm: the latent stage receives an UNGUIDED input, the prior is
added to `self.latent(F)` (verified by recomputing it), and `latent_norm` logs
the latent output.
