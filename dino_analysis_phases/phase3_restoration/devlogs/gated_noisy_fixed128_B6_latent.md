# gated-noisy — can a gate rescue a prior that HURTS?

`Holo_gated_noisy_fixed128_B6_latent`

Scope: **this arm only**. Written **before the run** (2026-09-07). The
prediction below is a pre-registration.

---

## THE DECISION

E1-addition-noisy with one change: the same `DinoGate` as gated-render.

```
addition-noisy   guided = F + P(D)
gated-noisy      guided = F + g * P(D),   g = sigmoid(G([F, P(D)])) in (0,1)
```

DINO still reads the **1e5 radar tensor the network is restoring**, exactly as
addition-noisy does. Same B6, same 1e5 means, same zero-init `P`, same recipe,
same seed. Single-stream, so nothing can be misaligned by the training sub-crop.
**+590,592 over E0.**

## WHY THIS IS THE SHARPEST ARM IN THE WAVE

addition-noisy is the project's cleanest negative result: **-0.577 dB below the
no-prior baseline on test**. A merely useless prior should cost nothing — `P` is
zero-initialised and could stay near zero — yet this one costs real
performance, and it also overfits (train loss 0.0471 against E0's 0.0587,
validation peaking at 128k then declining, DEVLOG Step 38).

The standing explanation is that the network cannot cheaply ignore a prior added
at every position of every image. **This arm tests that explanation directly,
because a gate CAN switch the prior off and AFFM cannot.** It is the one place
where "let the model decide" has a concrete job to do.

## PRE-REGISTERED PREDICTION

**The gate recovers at least half the deficit: gated-noisy lands above 21.60 dB
on test full256** (addition-noisy 21.296, E0 21.873; half the 0.577 deficit is
21.58).

Rationale: the deficit is attributed to forced injection, and the gate removes
the forcing. Confidence is moderate — the competing explanation, that the
damage is done by what the prior CONTAINS rather than by the inability to
suppress it, predicts no recovery.

Both outcomes are publishable and neither is a failure:

  * **Recovery to ~baseline.** "The model can decide" is worth something exactly
    where the prior is bad. The mechanism behind the negative result is
    identified, and the gate earns a place even if it is inert on the render arm.
  * **Still below baseline.** Selection does NOT rescue a bad prior. The SOURCE
    finding gets STRONGER, not weaker: the damage is in the content, not in the
    architecture's inability to suppress it.

**The gate statistics decide between them regardless of PSNR.** If the arm
recovers, `gate_frac_closed` should be high and `gate_mean` well below 0.5. If
the arm does not recover but the gate DID close, that is the most informative
outcome of all: the network tried to switch the prior off and still lost, which
would point at the optimisation trajectory rather than at the injected signal.

## SMOKE

`smoke_tests_wave2.py`: **29/29 passed.** Step-0 output identical to E0, trunk
byte-identical, gate exactly 0.5 at init and per position and channel, DINO
source verified to be the radar tensor itself, parameter delta +590,592,
eval256 path clean, and the one-step staircase in the correct direction.
