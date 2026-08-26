# THE PHASE-3 STORY — what to actually write, in order

Written 2026-08-26. This is the NARRATIVE spine, not a data source.
Numbers of record live in `DEVLOG.md` (Steps 29-32) and `HANDOVER.md`;
if this file and those disagree, THOSE are right and this one is stale.

**One sentence.** A DINO prior helps radar holography restoration, but only
if you get three things right — WHERE it reads from, WHETHER it keeps spatial
structure, and HOW MANY depths you use. HOW you mix it in matters far less
than the literature suggests.

---

## The spine: each experiment answers ONE question

All PSNR below is full256. **Split is marked on every row** — three arms have
been read on the locked TEST split, the rest are VALIDATION only.

### Q1 — Does a foundation-model prior help at all?

| arm | split | PSNR | vs E0 |
|---|---|---|---|
| E0-Fixed (no DINO) | test | 21.873 | — |
| E1-addition-render | test | **24.081** | **+2.208** |

Improved on 298/338 images (88.2%). **Yes, and by a lot.**

### Q2 — Does it matter WHAT DINO looks at?

| arm | split | PSNR | vs E0 |
|---|---|---|---|
| E1-addition-**noisy** (DINO reads the 1e5 radar) | test | 21.296 | **-0.577** |
| E1-addition-**render** (DINO reads the render) | test | 24.081 | +2.208 |

**FINDING 1 — THE SOURCE DECIDES THE SIGN.** A prior computed on degraded
input is WORSE THAN NO PRIOR AT ALL. Same architecture, same parameter count,
same seed; only the image DINO reads changes. This is the strongest and
cleanest result in the project and it is on the locked test split.

### Q3 — Does the prior need SPATIAL structure?

| arm | split | PSNR | vs E0 |
|---|---|---|---|
| global-render (same render, POOLED to 1x768, broadcast) | val | 20.585 | **-1.49** |

**FINDING 2 — THE SPATIAL GRID IS THE VALUE.** Destroying the layout while
keeping the content drops it BELOW baseline. The claim is not "DINO knows
about chairs", it is "DINO knows WHERE".

### Q4 — Does the FUSION OPERATOR matter?

| arm | split | PSNR | vs addition | p |
|---|---|---|---|---|
| addition-render | val | 24.120 | — | — |
| concat-render | val | 24.143 | +0.023 | 0.49 |

**FINDING 3 — FANCIER MIXING BUYS NOTHING.** A statistical tie, paired over
n=339. Addition is enough. (This also serves as the method's control: a null
that reproduces as a null.)

### Q5 — Can ATTENTION do better?

| arm | crop128 | full256 |
|---|---|---|
| crossattn-render, iteration 178k, IDENTICAL WEIGHTS | **+2.00** | **-7.56** |

**FINDING 4 — SPATIAL ATTENTION FAILS SCALE TRANSFER.** The DINO grid is
16x16 (256 tokens) in training and 32x32 (1024 tokens) at evaluation, and a
spatial softmax must renormalise over four times as many competitors.

  DO NOT REPORT +2.00 dB AS A RESULT. Checkpoint selection uses the
  training-time full-256 validation -- the one regime this arm cannot do --
  so it selected iteration 4,000 out of 179,000, and every published
  crossattn number comes from that model. The honest statement is that the
  selection metric and in-distribution performance are ANTI-CORRELATED for
  this arm. priorquery-render (the papers' attention direction) was dropped
  at 90k and never evaluated.

**This failure is not a dead end — it is the motivation for the next chapter.**
It is why every later attention arm uses CHANNEL attention, whose matrix is
C/heads x C/heads and therefore independent of token count.

### Q6 — Does the LAYER COUNT matter?

| arm | params vs addition | split | PSNR | vs addition | 95% CI | p |
|---|---|---|---|---|---|---|
| affm-render, depths {3,6,9,12}, STILL PLAIN ADDITION | **+1.04%** | val | **24.404** | **+0.284** | [+0.176,+0.393] | 1.0e-09 |

**FINDING 5 — MORE DEPTHS HELP, ESSENTIALLY FOR FREE.** Four DINO depths,
each centred with its own train-only mean, combined by a per-position softmax
ACROSS LAYERS. The output stays 768 channels because it is a weighted SUM, so
the projection is unchanged and the arm is within 1% of addition-render's
parameter count. **A gain here CANNOT be blamed on capacity** — that is the
entire reason AFFM was used instead of a concat.

### Q7 — Does the PUBLISHED method beat it?

| arm | params vs addition | split | PSNR | vs addition | p |
|---|---|---|---|---|---|
| dinolight-render, {3,6,9,12} + gated channel cross-attention | **~4.6x** | val | 24.344 | +0.224 | 1.3e-04 |

**FINDING 6 — IT TRANSFERS, BUT UNDERPERFORMS THE SIMPLE VERSION AT 4.6x THE
COST.** Both gaps are statistically real. The ordering is the point:

> **Plain addition over four depths (+1% params) beats the published
> sophisticated fusion (+360% params).**

That contrast is the punchline of the chapter.

---

## Chapter 8 — what is running, and what it decides

dinolight-render changed TWO things at once versus addition-render (the depth
count AND the operator), so Q7 cannot say which caused what. The ladder
decomposes it. **Fusion is IDENTICAL across all four ACA arms** — the same
`DinoAca` block, imported never copied. Only the layer set changes.

| arm | isolates | status (2026-08-26) |
|---|---|---|
| aca-L6 {6} | the OPERATOR alone — one factor from addition-render | RUNNING ~8.5h, job 1793518 |
| aca-L36 {3,6} | the depth ladder | RUNNING, job 1793527 |
| aca-L6912 {6,9,12} | the depth ladder | RUNNING, job 1793528 |
| aca-L6-nosa | whether the ACA's SELF-attention branch is needed at all | smoke queued, job 1794461 |

**THE PREDICTION, WRITTEN DOWN BEFORE THE ANSWER ARRIVES.** affm (+0.284,
simple fusion) beat dinolight (+0.224, complex fusion), so the gain most
likely came from DEPTHS, NOT THE OPERATOR. If aca-L6 lands near
addition-render, that is confirmed, and the chapter concludes:

> *The operator is not where the value is. The layer count is.*

If aca-L6 instead beats addition-render clearly, the operator does carry
value and Finding 5 becomes "depths AND operator both help". Either way the
experiment is decisive, which is the point of running it.

### The F_sa sub-question (aca-L6-nosa)

`DinoAca` computes `guided = project_out(F_sa + alpha*F_ca) + F`. `F_ca` is
the point — the latent attending to the DINO prior. **`F_sa` is the latent
attending to ITSELF, which is Restormer's own MDTA step for step** — and the
block's output feeds `self.latent`, EIGHT transformer blocks that each already
run that operation. F_sa is 452,742 parameters, 43% of the fusion block.

dinolight's own logs show `||alpha*F_ca|| / ||F_sa||` rising to a mean of
**8.157** over its last 100k iterations: the DINO half carries ~8x the
magnitude of the self half.

  THAT IS A MAGNITUDE, NOT A CAUSAL RESULT. F_sa is also the baseline the
  gated cross term is summed onto before one shared `project_out`. The ratio
  MOTIVATES the ablation; it does not settle it. Do not write "F_sa does
  nothing" until aca-L6-nosa has run.

  IT IS ALSO NOT A CLAIM THAT DINOLight IS IMPLEMENTED WRONGLY. That arm
  reproduces a published block faithfully and must keep doing so.

---

## THE CAVEATS YOU MUST CARRY INTO THE WRITE-UP

1. **Split discipline.** Findings 1-2 are on the LOCKED TEST split. Findings
   3-6 are VALIDATION ONLY. The test split is UNREAD for affm-render,
   dinolight-render and the entire ACA ladder. Read it ONCE, for all arms
   together, and never let it drive checkpoint selection.

2. **Neither new arm clears the pre-registered bar.** The threshold for
   "meaningful" was **>+0.30 dB**. affm's +0.284 falls short and its CI
   [+0.176, +0.393] STRADDLES 0.30 — unresolved in both directions.
   dinolight's +0.224 is below it.

3. **affm-render's pre-registered prediction is FALSIFIED** ("not >0.10 dB";
   the CI lower bound is +0.176). dinolight's ("not >0.30 dB") HOLDS. A
   falsified prediction is a FINDING, not a failure — you wrote down that it
   would not help, and it did.

4. **The capacity confound is NOT uniform — never average over it.**
   affm-render is +1.04% parameters, so its result is NOT capacity-explained.
   dinolight and every ACA arm are ~4.6x, so THOSE comparisons ARE confounded.
   Across the ACA ladder alone the spread is 0.227%, so a difference THERE is
   not capacity either.

5. **E0-Fixed scores 0.53 dB BELOW the old progressive baseline** (21.873 vs
   22.405) because it never trains at the evaluation resolution. This was
   registered in advance and does NOT weaken the Phase-3 comparison, which is
   internal: every arm shares E0-Fixed's recipe exactly.

6. **Checkpoint top-5 spreads are tiny** (affm 0.0525 dB, dinolight 0.0369 dB).
   The selected checkpoint is barely distinguishable from four neighbours.

7. **`injection_ratio` is NOT comparable between ACA arms and addition arms.**
   The ACA arms log `||alpha*F_ca||` before the output conv (~0.05-0.3); the
   addition arms sit at ~1.03. Compare ACA to ACA.

---

## If someone asks "so what is the contribution?"

1. **The source of a foundation-model prior decides whether it helps or hurts.**
   Computing it on the degraded input is worse than using no prior. (test split)
2. **The prior's value is spatial.** Pooling it away drops below baseline. (val)
3. **Depth beats sophistication.** Four DINO depths with plain addition, at +1%
   parameters, outperform a published gated cross-attention fusion at +360%. (val)
4. **Spatial cross-attention over a prior grid does not survive the train-to-eval
   scale change** — a concrete negative result with a diagnosed mechanism, and
   the reason to prefer channel attention.
