# THE PHASE-3 STORY — what to actually write, in order

Written 2026-08-26, results completed 2026-08-31. This is the NARRATIVE spine, not a data source.
Numbers of record live in `DEVLOG.md` (Steps 29-32) and `HANDOVER.md`;
if this file and those disagree, THOSE are right and this one is stale.

**STATUS: every arm is trained and evaluated on BOTH splits. The study is
complete.** Only priorquery-render (dropped at 90k) has no numbers.

**One sentence.** A DINO prior helps radar holography restoration, but only
if you get three things right — WHERE it reads from, WHETHER it keeps spatial
structure, and HOW MANY depths you use. HOW you mix it in matters far less
than the literature suggests — and at matched depth the published attention
operator buys NOTHING at 4.6x the parameters.

---

## The spine: each experiment answers ONE question

All PSNR below is full256. **Split is marked on every row** — three arms have
been read on the locked TEST split, the rest are VALIDATION only.

### Q1 — Does a foundation-model prior help at all?

| arm | test full256 | vs E0 | test crop128 | vs E0 |
|---|---|---|---|---|
| E0-Fixed (no DINO) | 21.873 | — | 19.546 | — |
| addition-render | **24.081** | **+2.208** | **22.259** | **+2.713** |

Improved on 298/338 images. **Yes, and by a lot.**

### Q2 — Does it matter WHAT DINO looks at?

| arm | test full256 | vs E0 |
|---|---|---|
| E1-addition-**noisy** (DINO reads the 1e5 radar) | 21.296 | **-0.577** |
| E1-addition-**render** (DINO reads the render) | 24.081 | **+2.208** |

**FINDING 1 — THE SOURCE DECIDES THE SIGN.** A prior computed on degraded input
is WORSE THAN NO PRIOR AT ALL. Same architecture, same parameter count, same
seed; only the image DINO reads changes. Still the cleanest result in the
project.

Supported by the **mismatched-render control** (job 1776802, four conditions on
the trained arm): shuffling the renders so each image gets the WRONG one costs
**9.094 dB**, worse on **335/339**, and drops below E0's mean on 322/339. The
model uses each image's specific render, not a generic template.

### Q3 — Does the prior need SPATIAL structure?

| arm | test full256 | vs E0 |
|---|---|---|
| global-render (same render, POOLED to 1x768, broadcast) | 20.589 | **-1.284** |

**FINDING 2 — THE SPATIAL GRID IS THE VALUE.** Destroy the layout while keeping
the content and it falls BELOW baseline. The claim is not "DINO knows about
chairs", it is "DINO knows WHERE".

### Q4 — Does the FUSION OPERATOR matter? (part 1: concat)

| arm | test full256 | vs addition | p |
|---|---|---|---|
| concat-render | 24.065 | **-0.016** | 0.98 |

**FINDING 3 — FANCIER MIXING BUYS NOTHING.** A clean null, paired over n=338.
It also went +0.023 on validation, i.e. it lands on both sides of zero across
splits — exactly what a true null looks like, and evidence the measurement does
not manufacture differences. **Cite this when defending the positives.**

(On crop128 concat is +0.214, p=1.7e-4. Read it as noise around zero, not a
result: a null that is positive on one protocol and negative on the other.)

### Q5 — Can ATTENTION do better? The failure that redirects the thesis

| arm | crop128 | full256 |
|---|---|---|
| crossattn-render, iteration 178k, IDENTICAL WEIGHTS | **+2.00** | **-7.56** |

**FINDING 4 — SPATIAL ATTENTION FAILS SCALE TRANSFER.** The DINO grid is 16x16
(256 tokens) in training and 32x32 (1024 tokens) at evaluation, and a spatial
softmax must renormalise over four times as many competitors.

  DO NOT REPORT +2.00 dB AS A RESULT. Checkpoint selection uses the
  training-time full-256 validation -- the one regime this arm cannot do -- so
  it selected iteration 4,000 out of 179,000, and every published crossattn
  number comes from that model. The honest statement: the selection metric and
  in-distribution performance are ANTI-CORRELATED for this arm.

**This failure is the hinge of the whole thesis.** It is why every later arm
uses CHANNEL attention, whose matrix is C/heads x C/heads and therefore
independent of token count. Without it, the ACA arms look like an arbitrary
choice instead of a fix for a diagnosed failure.

### Q6 — Does the LAYER COUNT matter?

| arm | params vs addition | test full256 | vs addition | 95% CI | p |
|---|---|---|---|---|---|
| affm-render {3,6,9,12}, STILL PLAIN ADDITION | **+1.04%** | **24.311** | **+0.230** | [+0.122,+0.338] | 4.3e-07 |

**FINDING 5 — MORE DEPTHS HELP, ESSENTIALLY FOR FREE.** Four DINO depths, each
centred with its own train-only mean, combined by a per-position softmax ACROSS
LAYERS. The output stays 768 channels because it is a weighted SUM, so the
projection P is unchanged and the arm is within 1% of addition-render's
parameter count. **A gain here CANNOT be blamed on capacity** — that is the
entire reason AFFM was used instead of a concat.

Replicated across splits: **+0.284 on validation (p=1.0e-09), +0.230 on test
(p=4.3e-07)**. Same sign, same order of magnitude, both far from zero.

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

### Q7 — Does the PUBLISHED method beat it, and WHICH factor earned it?

dinolight-render changes TWO things at once versus addition-render — the depth
count AND the operator — so on its own it answers "does the published method
transfer", never "which factor matters". The ACA ladder separates them. **All
four arms share the IDENTICAL DinoAca block, imported never copied. Only the
layer set changes.**

| arm | depths | params vs addition | test full256 | vs addition | p |
|---|---|---|---|---|---|
| **aca-L6** | **1** | ~4.6x | 24.111 | **+0.030** | **0.22 — NULL** |
| aca-L36 | 2 | ~4.6x | 24.196 | +0.115 | 0.011 |
| aca-L6912 | 3 | ~4.6x | 24.071 | -0.010 | 0.88 |
| dinolight-render | 4 | ~4.6x | **24.390** | **+0.309** | 9.9e-08 |
| affm-render | 4 | **+1.04%** | 24.311 | +0.230 | 4.3e-07 |

**FINDING 6 — THE OPERATOR BUYS NOTHING.** aca-L6 is ONE FACTOR from
addition-render: same depth {6}, same layer, same mean, same injection point,
only the fusion operator differs. It returns **+0.030 dB, p=0.22, not
significant, for 4.6x the parameters.**

  CONFIRMED TWICE, INDEPENDENTLY: validation +0.075 (p=0.075, n.s.) and test
  +0.030 (p=0.22, n.s.). Two splits, same verdict.

**FINDING 7 — THE ATTENTION ARMS PAY A CROP128 PENALTY. NEVER REPORT full256
ALONE.**

| arm | test full256 | test crop128 |
|---|---|---|
| aca-L6 | +0.030 (n.s.) | **-0.304** (p=1.3e-06) |
| aca-L36 | +0.115 | **-0.130** (p=0.032) |
| aca-L6912 | -0.010 (n.s.) | **-0.389** (p=4.3e-08) |
| dinolight-render | **+0.309** | **-0.206** (p=0.0054) |
| **affm-render** | **+0.230** | **+0.044** (n.s.) |

**EVERY attention-based arm is SIGNIFICANTLY WORSE on the matched-crop
protocol, dinolight included** — the arm with the best headline number. They
gain on full256 and lose on crop128. **affm-render is the only multi-depth arm
that loses nowhere.**

This is why the headline claim is about parameter efficiency and protocol
robustness, NOT about a PSNR gap.

## The ending — the sentence the whole study earns

> Reading DINO at multiple depths improves restoration. The fusion operator does
> not: at matched depth, gated channel cross-attention is statistically
> indistinguishable from plain addition at 4.6x the parameters, and every
> attention-based arm degrades significantly on the matched-crop protocol. A
> per-position softmax over depths with plain addition — **+1% parameters** —
> captures the benefit without that penalty.

### The F_sa sub-question — attempted, and it FAILED TO RUN. Report it as such.

`DinoAca` computes `guided = project_out(F_sa + alpha*F_ca) + F`. `F_ca` is the
point of the block. **`F_sa` is the latent attending to ITSELF — Restormer's own
MDTA step for step** — and the block feeds `self.latent`, EIGHT transformer
blocks that each already run that operation. F_sa is **452,742 parameters, 43%
of the fusion block**, and dinolight's logs show `||alpha*F_ca|| / ||F_sa||`
rising to a mean of **8.157** over its last 100k iterations.

`aca-L6-nosa` was built to remove it. **It cannot train as designed.** With both
`P` and `project_out` zero-initialised, the cross branch outputs zero, so
`project_out` receives EXACTLY ZERO gradient and stays zero forever; nothing
upstream ever learns. Measured on a real backward pass:

    gradient reaching project_out:   aca-L6 (with F_sa) 1.754    nosa 0.000

The 6k integration smoke caught it — `projected_norm`, `injection_ratio` and
`aca_injected_norm` all exactly 0.0 at every print, attention entropy pinned at
ln(64). **The arm was never submitted; the cost was 49 minutes, not 38 hours.**

**THIS IS ITSELF A FINDING, and the honest way to report it:** `F_sa` is
LOAD-BEARING FOR OPTIMISATION, not only for representation — it is what breaks
the double-zero-init deadlock. A naive "remove F_sa" ablation is impossible
under this initialisation scheme. The fix (initialise `P` normally, keep
`project_out` at zero) preserves the step-0 identity with E0 but makes the arm
differ from aca-L6 in TWO ways, so it stops being a one-factor ablation.

  Do NOT write "F_sa does nothing". The magnitude ratio motivated the
  ablation; the ablation did not run; and the deadlock shows F_sa has a second,
  separate job. Do NOT claim dinolight is implemented wrongly either — it
  reproduces a published block faithfully.

## THE CAVEATS YOU MUST CARRY INTO THE WRITE-UP

1. **REPORT BOTH PROTOCOLS, ALWAYS.** The attention arms win on full256 and
   lose on crop128. Showing either alone misrepresents the result. This is the
   single most important reporting rule in the project.

2. **The test split was read in SEVERAL PASSES, not once.** The original three
   arms on 2026-08-14, then concat / crossattn / global as each finished, then
   affm / dinolight / the ACA ladder on 2026-08-31. The pre-registration asked
   for one pass. **What survived intact is the thing that matters: checkpoint
   selection NEVER touched test** — every arm was selected on validation alone,
   verified for all eleven evaluated arms. State it plainly: *"the test split
   was read in several passes as arms completed; selection was performed on
   validation alone throughout, so the test split never informed any modelling
   decision."* Do NOT claim a single clean read.

3. **DO NOT CLAIM THE affm-vs-dinolight GAP.** 24.311 vs 24.390 on test is
   0.079 dB with overlapping CIs. Unclaimable. The honest separators are
   **parameter cost** (+1% vs ~4.6x) and **crop128 behaviour** (flat vs
   -0.206), neither of which depends on that PSNR difference.

4. **The capacity confound is NOT uniform — never average over it.**
   affm-render is +1.04% parameters, so its result is NOT capacity-explained.
   Every ACA arm is ~4.6x, so THOSE comparisons ARE confounded. Across the ACA
   ladder alone the spread is 0.227%, so a difference THERE is not capacity
   either. **aca-L6 partly closes this**: a 4.6x arm landing statistically level
   with addition-render says the extra capacity is roughly NEUTRAL at this data
   scale — it neither rescues nor sinks the attention arms.

5. **The depth trend inside the ACA ladder is NOT clean.** +0.030, +0.115,
   -0.010, +0.309 for 1/2/3/4 depths. aca-L6912 (3 depths) falls BELOW aca-L36
   (2 depths) and the CIs overlap heavily. Report the endpoints (1 vs 4) and
   say the intermediate points do not resolve; do NOT draw a monotonic curve
   through four noisy points.

6. **Selection searched HALF the saved checkpoints.** Checkpoints save every
   2,000 iterations; validation runs every 4,000. So ~75 of 151 checkpoints per
   arm were ever scored, and every selected iteration is a multiple of 4,000.
   One line in the methods. The top-5 spreads (0.012-0.118 dB) say neighbouring
   checkpoints are near-interchangeable, so this is a limitation, not a flaw.

7. **BasicSR has no best-model tracking.** Validation runs automatically and
   logs a PSNR; nothing compares scores or marks a best. Selection is a
   post-hoc parse of the training log (`select_best_checkpoint.py`). Worth a
   sentence so nobody assumes the framework did it.

8. **The AFFM layer weights are NON-STATIONARY.** B12 ran 0.3592 at 151k to
   0.1729 at 299k. Report a WINDOWED MEAN and state the window. The two stable
   claims: no depth is ever discarded (whole-run minimum 0.1028), and B6/B9 are
   the stable core with roughly half the std of B3/B12. The weight MAPS confirm
   the choice is genuinely spatial — std across positions 0.10-0.16 against
   means of 0.20-0.30, so AFFM is nothing like the pooled global arm.

9. **`injection_ratio` is NOT comparable between ACA arms and addition arms.**
   The ACA arms log `||alpha*F_ca||` before the output conv (~0.05-0.3); the
   addition arms sit at ~1.03. Compare ACA to ACA.

10. **E0-Fixed scores 0.53 dB BELOW the old progressive baseline** (21.873 vs
    22.405) because it never trains at the evaluation resolution. Registered in
    advance. **Have the answer ready: addition-render's 24.081 beats that older,
    STRONGER baseline too, by +1.676 dB.**

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

  **THE MITIGATION THAT COSTS NOTHING — REPLICATION ACROSS SPLITS, NOT A TREND.**
  An earlier version of this file advised leading with a monotonic depth trend.
  **The completed results do not support that** — the ACA ladder runs +0.030,
  +0.115, -0.010, +0.309 for 1/2/3/4 depths, so the middle points do not order
  cleanly and the CIs overlap. Do not draw a curve through them.

  What the data DOES give you, free of seed repeats:

    * CROSS-SPLIT REPLICATION. affm-render is +0.284 on validation
      (p=1.0e-09) and +0.230 on test (p=4.3e-07). aca-L6's null replicates too:
      +0.075 (p=0.075) and +0.030 (p=0.22). Both findings survive an
      independent split — which is what a seed repeat would have bought.
    * A NULL THAT BEHAVES LIKE A NULL. concat-render lands +0.023 on validation
      and -0.016 on test: on both sides of zero. A method that finds nulls
      where nulls exist lends weight to its positives.
    * ENDPOINT COMPARISON, not a trend. 1 depth versus 4 depths is large and
      significant on both splits; the intermediate ladder points are reported
      as unresolved.

  Prefer: *"the effect replicates on an independently held-out split"*
  Over:   *"performance increases monotonically with depth"*

  **AND DO NOT CLAIM THE affm-vs-dinolight GAP.** 0.079 dB on test with
  overlapping CIs. The parameter-efficiency and crop128-robustness claims do
  not depend on it.

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
   Computing it on the degraded input is worse than using no prior at all:
   -0.577 dB versus +2.208 dB, from the same architecture with one tensor
   swapped. (test split)
2. **The prior's value is spatial.** Pooling the grid away drops performance
   below baseline (-1.284 dB), and shuffling renders between images costs
   9.094 dB. It is not "the model learned what chairs look like". (test split)
3. **Depth beats sophistication, and the operator contributes nothing.** At
   matched depth, gated channel cross-attention is statistically
   indistinguishable from plain addition (+0.030 dB, p=0.22) at 4.6x the
   parameters — while four depths with plain addition give +0.230 dB at +1%.
   (test split, confirmed on validation)
4. **Attention-based fusion carries a protocol penalty.** Every ACA arm is
   significantly worse on matched-crop evaluation, including the best-scoring
   one. (test split)
5. **Spatial cross-attention over a prior grid does not survive the
   train-to-eval scale change** — a negative result with a diagnosed mechanism
   (+2.00 dB at crop128 versus -7.56 dB at full256 from identical weights), and
   the reason to prefer channel attention.
6. **Both attention arms are novel constructions, not reimplementations.**
   Verified from the arXiv HTML: Perceive-IR's prior is a 1x768 GLOBAL VECTOR
   with no spatial tokens to attend over, so no paper in the reference set does
   spatial-token cross-attention between a prior grid and a feature grid. Say
   this explicitly — it means crossattn's failure is an original negative
   result, not a failed reproduction.
