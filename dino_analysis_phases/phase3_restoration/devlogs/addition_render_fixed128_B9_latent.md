# addition-render B9 — the single-depth ablation

`Holo_addition_render_fixed128_B9_latent`

Scope: **this arm only**. Written **before the run** (2026-09-07).

---

## THE DECISION

Addition-render with the block number changed from 6 to 9, and the two
centering means swapped for that block's files. **Config-only: no new code, the
same `RestormerDinoSpatialRender` class, and EXACTLY the same parameter count
(+295,296 over E0).** Not an AFFM arm and not an ACA arm — AFFM needs two or
more depths to have anything to choose between, and Finding 6 showed the
attention operator is inert, so putting a new depth behind it would move two
factors at once.

## WHY

B6 was locked by the Phase-1/2 criterion (centred same-scene 1e5<->1e7
correspondence at the training scale), never by restoration quality. The
**B3 arm was pre-registered in the Phase-3 README and never run**; it is the
project's oldest open item. B9 is its counterpart from the other side: the
four-depth AFFM arm gave B9 the LARGEST learned share (0.326 mean over the last
100 logged points), which is the only evidence on record that another single
depth might beat B6.

Measured fixed-128 correspondence, WO1 Task 1.3, n=339 (centred):

| block | same-scene | different-scene | advantage |
|---|---|---|---|
| B3 | 0.6091 | 0.4124 | **+0.1966** |
| B6 | **0.6694** | 0.5241 | +0.1453 |
| B9 | 0.5482 | 0.4235 | +0.1247 |

B3 wins the same-vs-different advantage — the criterion Phase 2's own
interpretation called "the one to trust" — while B6 wins raw correspondence.
**That conflict has stood unresolved since Phase 2 and these two arms settle it
empirically rather than by argument.**

## PRE-REGISTERED PREDICTION

**Neither single depth beats B6 by more than 0.10 dB on test full256, and both
stay well above E0.**

Rationale: three unrelated criteria already point at B6 (Phase-1/2 cross-source
consistency, the tightest learned AFFM weight, and Phase-5 crop robustness), and
the completed depth ladder shows the single-depth choice matters far less than
the depth COUNT — {3,6,9} beat B6 alone by +0.255 while {3,6} was a null.

  * All three single depths within ~0.1 dB: **the depth CHOICE is not the
    lever, the depth COUNT is.** That is a cleaner sentence than the current one
    and it retires the Phase-2 criterion conflict as immaterial.
  * B9 clearly beats B6: the layer lock was suboptimal, which must be stated
    plainly, and every arm built on B6 inherits a stated caveat.

Both protocols reported. Selection on validation alone, as every arm.

## SMOKE

`smoke_tests_wave2.py`: **20/20 passed.** Reads block B9 (0-indexed 8),
loads the B9 render means in both regimes, parameter delta identical to
addition-render's, trunk byte-identical to E0, step-0 output identical to E0,
eval256 path 448 -> 32x32 with no interpolation.
