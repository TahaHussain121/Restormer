# postlatent-B3 — do the two wave-2 gains COMBINE?

`Holo_postlatent_render_fixed128_B3`

Scope: **this arm only**. Written **before the run** (2026-09-11). The
prediction below is a pre-registration.

---

## THE DECISION

Combine the two changes that each independently beat addition-render in wave 2:

```
addition-render      guided = F + P(D_B6)          ; latent = self.latent(guided)
postlatent-render    latent_out = self.latent(F)   ; guided = latent_out + P(D_B6)   test +0.306
addition-render B3   guided = F + P(D_B3)          ; latent = self.latent(guided)    test +0.228
postlatent-B3        latent_out = self.latent(F)   ; guided = latent_out + P(D_B3)   THIS ARM
```

**NOT a one-factor arm.** Against addition-render it moves two factors at once,
and it will be reported that way. Its references are the two constituents, not
the baseline.

Parameter count is **identical to addition-render's** (+295,296 over E0): the
tensor on either side of the latent stage has the same shape, and changing the
DINO block changes no shape. A difference here cannot be capacity.

**Config-only lowers implementation risk, not experimental risk.** The class and
the B3 means already exist and are smoke-tested (24/24 again for this config),
but this is still a full 300k run and a full commitment of a100 time.

## WHY THIS FIRST, AHEAD OF ANY NEW INJECTION POINT

Wave 2 left a tier of changes — B3 instead of B6, three depths, post-latent,
dinolight — that all sit roughly +0.23 to +0.31 dB above addition-render and are
statistically indistinguishable from one another. Whether they are independent
improvements or different routes to the same limit changes what the thesis
claims, and no further injection-point arm is interpretable until it is known.
This arm is the cheapest instrument that bears on it.

## PRE-REGISTERED PREDICTION

**postlatent-B3 does NOT beat postlatent-render alone by more than 0.10 dB on
test full256** — i.e. it lands below +0.41 against addition-render.

Rationale, and it is an inference rather than a strong prior: every member of
the tier improves ONLY on the out-of-regime protocol. Against addition-render on
test, post-latent is +0.306 full / +0.008 crop, B3 alone +0.228 / −0.006,
affm {3,6,9} +0.255 / +0.057, affm {3,6,9,12} +0.230 / +0.044. A shared
signature of that kind is more consistent with the arms addressing one common
weakness — plausibly robustness to the train-to-eval scale shift measured in
Phase 5 — than with four independent gains. If that reading is right the two
changes should overlap substantially rather than add.

**HOW EACH OUTCOME IS TO BE WRITTEN, fixed in advance:**

  * **Clearly above both constituents (> +0.10 over post-latent alone):** the
    improvements are at least partly complementary; depth choice and injection
    location act through different routes. This also yields the project's best
    model and the tier reading weakens.
  * **Level with post-latent alone:** write **"the improvements did not combine
    under this training recipe."** Do NOT write "we found a ceiling", "they hit
    the same limit", or any statement about a performance bound. One combined
    run cannot establish a limit; it can only fail to show addition.
  * **Below both:** the two changes interact negatively, which would be the most
    surprising outcome and would need its own follow-up before interpretation.

crop128 is predicted flat against post-latent alone (within ±0.10), as it is for
every tier member. Both protocols reported regardless.

Selection on validation alone, as every arm. Test read once, with any other arm
that finishes near it.

## SMOKE

`smoke_tests_wave2.py --config <this> --device cpu`: **24/24 passed** — reads
block B3 (0-indexed 2), loads the B3 render means in both regimes, parameter
delta identical to addition-render's, trunk byte-identical to E0, step-0 output
identical to E0, the latent stage receives an unguided input, the prior is added
to `self.latent(F)`, and the eval256 path is 448 -> 32x32 with no interpolation.
