# multilevel-aca-render — FINAL EXPERIMENT B

`Holo_multilevel_aca_render_fixed128_B6`

**Pre-registration, written 2026-09-14, before any training.** EXPLORATORY
follow-up informed by earlier results. Designed in a ChatGPT-assisted discussion
and authorised by the author as the second of the two final training runs; not a
supervisor instruction. **Its submission is NOT conditional on how
aca-L6-postlatent scores**: the purpose is a matched comparison of fusion
operators across the two injection layouts.

## What changes

Exactly experiment A's prior, extraction, sites and nearest-neighbour mapping,
with the additive injection at each site replaced by the existing `DinoAca`:

    projected_prior_s = expand_s( P_s(D) );   guided_s = ACA_s( feature_s, projected_prior_s )

Independent `DinoAca` instances at widths 384 / 192 / 96, each with the existing
six heads, LayerNorm and bias conventions, self-attention path, channel
cross-attention, scalar sigmoid gate from logit −2, and zero-initialised output
projection. No channel-wise gate, no AFFM, no new attention type, no new loss, no
direct additive bypass.

**Capacity.** Added trainable parameters over E0: **1,910,535**; total trainable
**28,034,587**. Against experiment A that is **3.70x the ADDED parameters and
+5.2% of TOTAL trainable parameters**. Runtime and memory are not inferred from
either ratio. Reference `aca-L6-postlatent`: +1,349,773.

Recipe identical to experiment A and to its reference arm.

## Pre-registered outcome and comparisons

Same primary outcome (paired test full256 PSNR), same **+0.10 dB practical
threshold**, same separate reporting of mean / interval / validation replication
/ threshold, same crop128 trade-off rule, same single-seed caveat.

Comparisons this arm enters:

  * **B. multi-level ACA − post-latent ACA** — does repeated delivery help ACA?
  * **C. multi-level ACA − multi-level addition** — operator at matched sites.
  * **D. the difference-in-differences of B and A** — does ACA gain more than
    addition from the extra sites?

**Expectation, a hypothesis:** limited incremental benefit, and no established
ACA advantage at matched sites, given that every attention-family arm so far has
been a full-frame null against addition at matched depth and has lost on
crop128. This lowers the expectation; it does not answer the question, which is
why the arm runs.

## Interpretation rules, fixed now

  * ACA outperforming addition at matched sites supports THIS ACA architecture in
    this setting. It does not isolate cross-attention as the cause: the block
    also contains a self-attention path and has more parameters.
  * Both improving similarly supports repeated guidance without an ACA advantage.
  * Protocol disagreement is a trade-off.
  * This tests the package, not each decoder stage.

**Intervention results elsewhere in the study must be quoted correctly:**
removing the cross branch measures the DEPENDENCE of a trained model; uniform
attention does NOT preserve output magnitude; and dividing the two PSNR drops
does NOT show that the learned mixing is "2% of the branch's value".

## Gradient flow — verified per stage before submission

Each stage has two zero-inits in series (`project_out`, `P_s`), hence the
three-step staircase: step 1 only `project_out`; step 2 `P_s` and the feature
path; step 3 every group including the cross path and alpha. Verified on a real
multi-step backward at EACH of the three stages. It cannot reproduce the
`aca-L6-nosa` deadlock because the self-attention path keeps `project_out`'s
gradient non-zero from step 1.

## Monitoring

The gate keeps aca-L6's convention at the post-latent site (`projected_norm` =
||alpha·F_ca||). Every site is recorded separately — feature norm, actual update
||guided − feature||, ratio, ||alpha·F_ca||, finiteness — and each stage's ACA
statistics are logged with pl_/d3_/d2_ prefixes.
