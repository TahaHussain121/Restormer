# affm-render {3,6} — a middle point on the ADDITIVE depth curve

`Holo_affm_render_fixed128_spatial_L36_latent`

Scope: **this arm only**. Written **before the run** (2026-09-03), so the
prediction below is a pre-registration, not a description of a result.

---

## WHY THIS ARM EXISTS

The operator that carries the Phase-3 depth finding — plain addition through a
zero-initialised 1x1 projection — has only TWO points on its depth curve:

| arm | depths | test full256 vs addition-render |
|---|---|---|
| addition-render | {6} | — |
| affm-render | {3,6,9,12} | +0.230 (p = 4.3e-07) |

Nothing says whether the gain is graded (each extra depth adds a little) or a
threshold (nothing until several depths are present). The attention ladder has
intermediate points, but Finding 6 shows that operator is inert, and its
ladder is noisy (+0.030, +0.115, −0.010, +0.309). This arm and its sibling
`affm-render {3,6,9}` put the missing points on the operator that matters.

**The chain is NESTED on purpose:** {6} → {3,6} → {3,6,9} → {3,6,9,12}. Each
step adds exactly one depth, so the four arms trace ONE curve with ONE factor
moving. B6 is never swapped out.

## THE DECISION

Config-only. Same class (`RestormerDinoAffmRender`), same per-position
softmax across layers, same zero-init `P = Conv2d(768, 384, 1)`, same
injection point (`inp_enc_level4`), same recipe, same seed 100. The config
differs from `affm_render_fixed128_spatial_L3691_latent.yml` in three places:
the name, `dino_layers: [3, 6]`, and the per-layer mean maps listing only B3
and B6 (the B6 files are byte-identical to addition-render's; the arch asserts
it).

Parameter delta over E0: **295,296 + 2 × 769 = 296,834** (addition-render
+295,296; +0.52%). Verified by `smoke_tests_affm_subset.py`.

At init the softmax is exactly 0.5 / 0.5 at every position, so the arm starts
as the unweighted mean of the centred B3 and B6 grids, and P(D) = 0 makes the
step-0 output identical to E0's.

## PRE-REGISTERED PREDICTION

Against addition-render on the test split, full256:

* **+0.05 to +0.20 dB** is the expected band. The four-depth arm reached
  +0.230, and the learned AFFM weights gave B3 the least stable and second
  smallest share (0.216 mean over the last 100 logged points), so one extra
  shallow depth should recover only part of the gain.
* If {3,6} ≥ +0.18: the gain saturates early; two depths already capture it.
* If {3,6} ≤ +0.05 (n.s.): the second depth alone buys nothing; the gain needs
  B9 or B12 (the sibling arm decides which).

Both protocols are reported. The pre-registered risk is the crop128 side:
affm-render {3,6,9,12} was the only multi-depth arm that lost nowhere; this
arm should also not lose on crop128 (|Δ| < 0.10, n.s.).

Selection: highest validation PSNR, as every other arm. The test split is read
once, together with the sibling arm, after both finish.

## THE CO-INFLATION STOP RULE — SPECIFIED, MEASURED, AND NOT ADOPTED

HANDOVER §5 and PHASE3_CHAPTER §9 carry an "unimplemented monitoring fix": abort
if `latent_norm` or `projected_norm` exceeds **5×** its own 5k-iteration
reference, because the failed cross-attention arm inflated both ~15× while the
ratio stayed flat. Adding it to this arm was the plan.

Before enabling it, the rule was replayed against every finished arm's
`dino_stability.csv` (2026-09-03):

| arm | latent growth vs 5k ref | projected growth |
|---|---|---|
| addition-render | **9.12×** | 8.19× |
| affm-render {3,6,9,12} | **9.72×** | 6.70× |
| concat-render | 9.07× | 7.40× |
| global-render | 9.99× | 6.11× |
| E1-addition-noisy | 12.53× | 6.21× |
| aca-L6 | 15.77× | 2.28× |
| dinolight-render | 15.86× | 2.03× |

**Every healthy arm — including the project's best model — grows its latent
norm 9–16× over the run.** The 5× rule would have aborted all of them, and the
crossattn arm's ~15× sits inside the healthy range. Norm growth under this
recipe is normal behaviour, not a failure signature, and the ratio rules that
exist are the right monitor. The rule is therefore **not added**, so this arm's
gate is byte-identical to every other arm's. The chapter's "known gap" should be
rewritten as this measurement.

## SMOKE

`smoke_tests_affm_subset.py --config <this> --device cpu` — see
`results/wo2_implementation/smoke_results_Holo_affm_render_fixed128_spatial_L36_latent.json`
and DEVLOG Step 37 for the outcome. No 6k GPU integration run: the class,
dataset, wrapper, gate and validation path are the ones the finished
{3,6,9,12} arm already ran for 300k iterations; only the length of one list
changed. The chain's early-crash guard (MIN_RUNTIME) covers the remaining risk.

## STATUS

Submitted 2026-09-03 as a self-chaining a100 job (see DEVLOG Step 37 for the
job id). a100 nodes were draining for a reboot at submission time.
