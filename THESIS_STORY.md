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

### Q6a — Does the layer ablation INVALIDATE the Phase-1/2 layer study?

**NO. It extends it, and the learned weights support it.** Write this down,
because the panic reflex is to think the multi-layer result makes the earlier
depth study look like wasted work. It does not, for two independent reasons.

**REASON 1 — the headline result IS the Phase-1/2 layer.** The best model this
project has produced is `addition-render`: **24.081 on the LOCKED TEST SPLIT,
+2.208 vs E0, improved on 298/338 images (88.2%)**. That model uses **B6
alone** — the depth Phase 1/2 selected. The multi-layer result is a **+0.284 dB
refinement ON VALIDATION ONLY** that does not clear the pre-registered +0.30 dB
bar. The main claim rests on Phase 1/2; the layer ablation is a follow-up.

**REASON 2 — the learned AFFM weights put B6 in the stable core.** affm-render
logs a per-position softmax across the four depths (they sum to 1 at each of
the 256 positions). Mean and std over the LAST 100 logged points (iter > 200k),
where uniform would be 0.2500:

| depth | mean weight | std | min | max |
|---|---|---|---|---|
| B3 | 0.2161 | 0.0358 | 0.1453 | 0.2884 |
| **B6** | **0.2568** | **0.0155** | 0.2303 | 0.2812 |
| B9 | **0.3261** | 0.0182 | 0.2996 | 0.3678 |
| B12 | 0.2010 | 0.0342 | 0.1359 | 0.2749 |

Two things to read off it, and one not to.

  - **NO DEPTH IS EVER DISCARDED.** The minimum any weight reaches across the
    ENTIRE 300k run is **0.1028** (B3). The network keeps all four.
  - **THE MID-DEPTHS ARE THE STABLE CORE.** B6 and B9 carry the most weight AND
    are the most stable (std 0.0155 and 0.0182). B3 and B12 wander with roughly
    **twice** the std (0.0358, 0.0342). **B6 has the tightest weight of all
    four.**
  - **DO NOT rank B9 above B6 as a finding.** The weights are NON-STATIONARY:
    over the run B12 went from 0.3592 at 151k to 0.1729 at 299k. Only report a
    windowed mean, state the window, and say nothing about a strict ordering
    among the two wandering depths.

**THE SENTENCE TO WRITE:**

> Phase 1/2 identified B6 as the strongest single depth, and that choice powers
> the project's best test-split result. The layer ablation then shows the best
> single depth is not SUFFICIENT: a learned per-position combination over
> {3,6,9,12} adds a further +0.284 dB at +1% parameters, never discards any
> depth, and independently places B6 in the stable core of the learned
> weighting.

That is a progression — best-single, then best-combination — not a reversal.

### Q6b — Is plain addition "cheating"? Does it fail to SELECT?

Recorded because this doubt keeps recurring. Two separate worries, and they
have different answers.

**THE LEAKAGE WORRY IS ALREADY ANSWERED.** The render is an INPUT channel
available at test time, not derived from the 1e7 target. `CONTEXT.md:531`:
*"render is normally available, so this is a usable method, not only an
oracle."* The 1e7 is target only.

**THE "IT DOES NOT SELECT" WORRY IS MISTAKEN AS STATED.** Addition does not
hand the network the render. The prior passes through `P = Conv2d(768, 384, 1)`,
zero-initialised — a LEARNED readout free to suppress any channel. In
affm-render there is a SECOND selection stage on top: a per-position softmax
across depths. So affm-render selects twice — across channels and, per
position, across depths. What it does not do is softmax over spatial positions.

**AND THE UNCOMFORTABLE PART, WHICH IS THE ACTUAL RESULT.** "Selective
attention should beat naive addition" is a hypothesis these experiments are
currently FALSIFYING:

  - spatial attention broke on scale transfer (-7.56 dB at full256)
  - channel attention over the same four depths got **+0.224** where plain
    addition over those depths got **+0.284**, at a fifth of the parameters

That is a RESULT, and a more interesting one than confirming the assumption.
Note also that **no paper in the reference set does what these attention arms
do** — Perceive-IR's prior is a global 1x768 vector with no spatial tokens to
attend over at all, and DSGIR is paywalled and was never verified. These are
novel constructions, so this is not a failure to reproduce anyone.

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

**SCOPE — YOU DO NOT NEED TO RERUN THE LADDER FOR THIS.** One comparison
answers it: **aca-L6 vs aca-L6-nosa**, same depth, same everything, one branch
removed, and the cross branch verified byte-identical at initialisation
(0.000e+00). That licenses the claim *"the ACA's self-attention branch is
redundant given Restormer's trunk"*. `nosa` variants of L36 / L6912 /
dinolight would only test whether that redundancy also holds at other depth
counts — a robustness check, NOT a claim the thesis needs. Both required runs
are already submitted, so the answer costs no additional experiments.

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

7. **The AFFM layer weights are NON-STATIONARY.** Over the run B12 went from
   0.3592 at 151k to 0.1729 at 299k. Report a WINDOWED MEAN and state the
   window, or say nothing. The two claims that ARE stable: no depth is ever
   discarded (whole-run minimum 0.1028), and B6/B9 are the stable core with
   roughly half the std of B3/B12.

8. **`injection_ratio` is NOT comparable between ACA arms and addition arms.**
   The ACA arms log `||alpha*F_ca||` before the output conv (~0.05-0.3); the
   addition arms sit at ~1.03. Compare ACA to ACA.

---

## SCOPE DECISIONS — taken deliberately, recorded so they read as decisions

An unstated omission looks like an oversight; a stated scope decision looks
like a decision. Same fact, completely different reception. Write each of
these as a sentence in the thesis rather than leaving a reader to notice it.

**1. The DINO prior is a PREMISE of this work, not a hypothesis under test.**
The supervisor specified a DINO prior, so no no-DINO control (e.g. the render
supplied as a plain input channel) was run. The research question is therefore
**"given a DINO prior, how should it be injected?"** — and the four controlled
answers (source, spatial structure, depth count, fusion operator) ARE the
contribution. The consequence to state honestly: if asked whether raw render
pixels could achieve the same, the answer is *out of scope, not tested*. That
is fine, and only becomes a problem if a stronger claim is made anywhere.

  Sentence to use: *"The use of a DINO prior is a premise of this work; the
  contribution is the injection study."*

**2. E0-Fixed is the reference, deliberately, and the handicap is answered.**
Fixed 128 crops for every arm means the ONLY difference between arms is the
DINO branch, which is what makes the ladder internally valid; it also matches
common practice. E0-Fixed scores 0.53 dB BELOW the old progressive baseline
(21.873 vs 22.405) because it never trains at the evaluation resolution.
**Have the answer on a slide:** addition-render's 24.081 beats that older,
STRONGER baseline too, by **+1.676 dB**. Registered in advance; not a
weakness.

**3. Each arm was trained ONCE, at a single seed. Not being reported.**
Between-run variance was not measured. Checked against comparable theses in
the field, where single-run reporting is the norm. Recorded here so the choice
is visible rather than accidental.

  **THE MITIGATION THAT COSTS NOTHING — LEAD WITH THE TREND, NOT THE DELTA.**
  The depth ladder gives FOUR independently trained arms at increasing depth
  counts (aca-L6 {6}, aca-L36 {3,6}, aca-L6912 {6,9,12}, dinolight {3,6,9,12}),
  plus affm-render and dinolight-render both beating addition-render from
  DIFFERENT fusion operators (+0.284 and +0.224). A monotonic trend across
  several independent runs is a stronger argument than any single pairwise gap,
  and needs no seed repeats.

  Prefer: *"performance increases with the number of DINO depths across four
  independently trained arms"*
  Over:   *"+0.284 dB over addition-render"*

  **AND DO NOT CLAIM THE affm-vs-dinolight GAP.** +0.284 vs +0.224 is 0.06 dB.
  That is far below any plausible noise floor and cannot be claimed. What CAN
  be claimed is parameter efficiency: both beat addition-render, and affm does
  it at a FIFTH of the parameters. That claim does not depend on the small PSNR
  difference at all.

  Supporting evidence already in hand: concat-render reproduced as a clean null
  (+0.023, p=0.49), which shows the measurement does not manufacture
  differences — a method that finds nulls where nulls exist lends weight to its
  positives.

**4. crossattn-render and priorquery-render: not carried further.**
No additional runs, and no further analysis. If crossattn is EXCLUDED, exclude
it cleanly — do not leave it half-present in a results table. If it is KEPT,
two sentences are enough and it earns its place: it is a negative result with a
DIAGNOSED MECHANISM (a spatial softmax over 256 tokens does not survive
becoming 1024), and it is the REASON the later arms use channel attention.
Without it, ACA looks like an arbitrary choice rather than a fix for an
identified failure. priorquery was dropped at 90k and produced no metrics.

**5. Not attempted, and out of scope:** capacity-matched control for the ACA
arms (they are ~4.6x addition-render, so Finding 6 stays capacity-confounded;
Finding 5 does NOT, at +1%), DINO fine-tuning, alternative injection points,
other datasets or degradation levels, and inference-time cost.

## If someone asks "so what is the contribution?"

1. **The source of a foundation-model prior decides whether it helps or hurts.**
   Computing it on the degraded input is worse than using no prior. (test split)
2. **The prior's value is spatial.** Pooling it away drops below baseline. (val)
3. **Depth beats sophistication.** Four DINO depths with plain addition, at +1%
   parameters, outperform a published gated cross-attention fusion at +360%. (val)
4. **Spatial cross-attention over a prior grid does not survive the train-to-eval
   scale change** — a concrete negative result with a diagnosed mechanism, and
   the reason to prefer channel attention.
