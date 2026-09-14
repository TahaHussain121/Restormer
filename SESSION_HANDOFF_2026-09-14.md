# SESSION HANDOFF — 2026-09-14

Covers the session of 2026-09-03 → 09-14 (DEVLOG Steps 37–50). Read this, then
`DEVLOG.md` Steps 42–50 for the numbers of record. **If this file and the DEVLOG
disagree, the DEVLOG is right.**

---

## 1. THE ONE THING THAT MUST NOT BE DROPPED

**The two FINAL training experiments are running. They close the architecture
training study. Launch no further training arm.**

| experiment | first job | successor | state at hand-over |
|---|---|---|---|
| A `Holo_multilevel_addition_render_fixed128_B6` | 1812561 | 1812563 | running on a100 since 2026-09-14 14:36, checkpoint 2,000 |
| B `Holo_multilevel_aca_render_fixed128_B6` | 1812562 | 1812564 | same |

About 39 h each, so expect `TRAINING_DONE` around **2026-09-16 morning**.

A persistent monitor in the old session runs selection and submits evaluation
automatically — **but monitors die with the session.** If you are reading this in
a new session, do it by hand:

```bash
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
PY=/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python
P3=dino_analysis_phases/phase3_restoration

# 1. finished and healthy?
for E in Holo_multilevel_addition_render_fixed128_B6 Holo_multilevel_aca_render_fixed128_B6; do
  ls experiments/Phase3_chain_state_$E/{TRAINING_DONE,CHAIN_ABORTED} 2>/dev/null
  ls experiments/$E/STABILITY_FAILURE* 2>/dev/null; done

# 2. validation-only selection (chain_core NEVER does this itself)
$PY $P3/scripts/select_best_checkpoint.py --name Holo_multilevel_addition_render_fixed128_B6
$PY $P3/scripts/select_best_checkpoint.py --name Holo_multilevel_aca_render_fixed128_B6

# 3. eight evaluation cells — check none already exist first (do not duplicate)
#    CFG = multilevel_addition_render_fixed128_B6 | multilevel_aca_render_fixed128_B6
for S in val test; do for PR in full256 crop128; do
  sbatch $P3/scripts/run_evaluate.sh $P3/configs/$CFG.yml experiments/$E/models/net_g_<ITER>.pth $S $PR
done; done

# 4. the pre-registered analysis (its defaults ARE the final 2x2)
$PY $P3/scripts/final_matched_pair.py
```

Then: record a DEVLOG step, **recommend the final thesis configuration using
VALIDATION evidence plus explicit complexity and trade-off criteria**, update the
results grid, update memory. v100 evaluation jobs have sat pending for hours this
session; that is queueing, not failure.

### The final pair — what was fixed before training (do not reinterpret)

Same B6 render prior, extracted once, at three sites: after the latent blocks,
and at the inputs of decoder levels 3 and 2 **after skip concatenation and channel
reduction**. No pre-latent, no decoder-level-1 injection. Decoder sites receive
the prior by **nearest-neighbour upsampling** (2x, 4x) — a new mapping, not the
old no-resizing rule. A fuses by addition, B by the existing ACA block at each
site.

| | added params | total trainable |
|---|---|---|
| A multi-level addition | 516,768 | 26,640,820 |
| B multi-level ACA | 1,910,535 | 28,034,587 |

B is 3.70x A's ADDED parameters and +5.2% of TOTAL. Frozen DINO (86,580,480) is
counted separately.

**Pre-registered** in `devlogs/multilevel_*_render_fixed128_B6.md`:

* primary outcome: paired test full256 PSNR;
* comparisons: **A** ml_add − pl_add · **B** ml_aca − pl_aca · **C** ml_aca −
  ml_add · **D** the difference-in-differences;
* **+0.10 dB** practical threshold, reported SEPARATELY from reliability (95%
  interval excludes zero AND validation has the same sign);
* a crop128 decline worse than −0.10 is a trade-off, never hidden;
* single seed: intervals measure images, not seeds; non-significance ≠
  equivalence;
* expectation, as a hypothesis: limited incremental benefit;
* an ACA win supports THIS ACA block, not cross-attention as the sole cause; it
  tests the multi-level PACKAGE, not each decoder stage.

Verification already passed: unit smokes 42/42 and 48/48 on real data (capacities
exact, trunk and step-0 identical to E0, one DINO extraction per forward, exact
nearest-neighbour mapping, the three-step gradient staircase at EVERY ACA stage);
2k-iteration integration smokes clean (Step 49).

---

## 2. RESULTS OF RECORD FROM THIS SESSION

Test split, n=338, full256, paired against addition-render (B6 before the
latent) unless stated. Every number is in the DEVLOG step named.

| arm | test full256 | vs ref | crop128 vs ref | step |
|---|---|---|---|---|
| affm {3,6} | 24.069 | −0.012 (val +0.140) — null | +0.066 | 40 |
| affm {3,6,9} | 24.336 | +0.255 | +0.057 | 40 |
| gated-render | 24.123 | +0.042 n.s. | **−0.252** | 42 |
| gated-noisy (vs addition-noisy) | 21.162 | **−0.134** | −0.281 | 42 |
| **postlatent-render** (after latent) | **24.387** | **+0.306**, p=1.8e-08 | +0.008 | 42 |
| **addition B3 alone** | 24.309 | **+0.228** | −0.006 | 42 |
| addition B9 alone | 24.140 | +0.059 n.s. | −0.083 | 43 |
| postlatent-B3 (vs postlatent-B6) | 24.480 | +0.094, CI incl. 0, val +0.002 | −0.055 | 50 |
| aca-L6-postlatent (vs postlatent-render) | 24.139 | **−0.247**, p=2.8e-05, val −0.138 | **−0.268** | 50 |

**Inference-only and feature-space results:**

* **ACA interventions** (aca-L6 at 236k, validation, self-check 339/339
  bit-identical): zeroing the cross term −6.52 dB; uniform cross attention
  −0.146 dB. Strong DEPENDENCE on the cross branch; uniform mixing costs a
  smaller, measurable amount. Never divide the two (Step 46/47).
* **global-render overfits**: train loss 24% below E0's, validation collapses
  0.86 dB from its 60k peak. The pooled B6 vector is a partial scene ID (top-1
  17.7% of 6,101, chance 0.016%); the papers' CLS token is sharper (39.5%)
  (Step 38).
* **The crop-vs-full DINO gap is mostly a learnable transform**: on held-out
  scenes a linear 768x768 map closes 71.5% of it, two 3x3 convs 86.8%;
  per-channel rescaling only 19% (Step 39).
* **The co-inflation stop rule was dropped on measurement**: every healthy arm
  grows its latent norm 9–16x (Step 37).

---

## 3. WHAT CHANGED IN THE STORY — the corrections

These overturn things THESIS_STORY and PHASE3_CHAPTER still say. Correction
banners mark them; **the prose rewrite has NOT been done and is the author's
call.**

1. **"Depth count is the lever / a threshold at three depths" — withdrawn.** B3
   alone is level with three depths (−0.027, n.s.). The {3,6} null happened
   because AFFM put 0.847 of its weight on B6, the weaker depth (Step 42).
2. **B6 is not the best single depth for restoration.** B3 beats it by +0.228;
   B9 and B6 are not separable. The AFFM weights ANTI-predict single-depth
   quality — never cite them as evidence of which depth is best. The Phase-2
   same-vs-different-scene ADVANTAGE was the criterion that picked the winner
   (Step 43).
3. **Chapter §4.5 is empirically wrong on this data.** Injecting AFTER the latent
   stage is +0.306 dB better at identical parameter count (Step 42).
4. **Finding 6 becomes location-dependent.** The operator was a null before the
   latent (+0.030); after it, the ACA block is significantly WORSE than addition
   (−0.247, replicated). The claim is about the tested ACA BLOCK, not
   cross-attention alone (Step 50).
5. **The +0.25 tier** (post-latent, B3, three depths, four depths, dinolight) is
   internally indistinguishable, and **every member gains only on full256 and is
   flat on crop128** — an observation, pointing at one shared mechanism
   (plausibly robustness to the train/eval scale shift), not proven (Step 44).
6. **Stacking is sub-additive.** Post-latent + B3 did not demonstrate an
   improvement exceeding the threshold over post-latent alone; the interaction
   is significantly negative on both splits (Step 50).
7. **"4.6x the parameters" is an ADDED-parameter ratio**: aca-L6 is +3.99% of
   TOTAL trainable parameters. 22 occurrences across THESIS_STORY, the chapter,
   HANDOVER and the results grid need the denominator (Step 47).
8. **Withdrawn: "the learned mixing is 2% of the branch's value"** (Step 47).
9. **Uniform attention does NOT preserve output magnitude** (the term measured
   1.153x). Corrected in DEVLOG; **still wrong in a code comment in
   `basicsr/models/archs/dino_aca.py`**, left because shared implementations may
   not be edited without authorisation (Step 48).
10. **Provenance.** The structured design reviews were **ChatGPT-assisted
    discussions** the author relayed and accepted — **not the supervisor's**.
    Corrected in Steps 44–46 (Step 47). The separate line that the supervisor
    specified the DINO prior is the author's own statement.
11. DINOLight DOES keep spatial features and inject at multiple stages, so
    hierarchical spatial injection is not broadly novel; our `dinolight-render`
    is a partial, one-site adaptation. "Global guidance is fatal" is too broad —
    only our pooled-B6 arm was tested (Step 44).

**What stands:** Finding 1 (the source decides the sign — now stronger, since a
gate did not rescue the noisy prior), Finding 2 (spatial structure), Finding 7
(the protocol split — now with a first additive-family member, the gate).

**Pre-registration scorecard, this session:** wave 1 one of two right; wave 2 two
of five; wave 3 both right. Every miss is recorded as a miss.

---

## 4. OPEN DECISIONS — all the author's

* **The narrative rewrite** of THESIS_STORY, the chapter (§4.5, §6.8, §7.2a,
  §7.5, §9, §10) and PROSE_ARGUMENTS, including the 22 "4.6x" phrasings. Best
  done after the final pair lands.
* **Disk cleanup.** `/home/woody` was 735.9 GB of a 1000 GB soft quota before the
  final pair (~+95 GB). Nothing may be deleted without an exact deletion manifest
  that the author has read.
* **The one-line comment fix in `dino_aca.py`** (item 9 above) needs explicit
  authorisation.
* **The results grid** (claude.ai artifact below) lacks the wave-3 arms and the
  final pair; republish once, after the pair.
* The crop-gap-correction training arm is justified by Step 39 but falls under
  the hard stop; it is closed unless the author reopens it.
* The Gmail and Google Calendar connectors are unauthorised; nothing depends on
  them.

---

## 5. TRAPS

* **Monitors and background watchers are session-local.** The chains are not.
* **`chain_core` never runs checkpoint selection.** A finished arm sits
  unselected until someone runs it.
* **Never duplicate an evaluation**: check `results/<exp>/metrics/` and the queue
  first.
* **Run exact step-0 smoke checks on CPU**: a100 TF32 has broken exact equality
  before.
* **The two stale `RUNNING_JOB` locks** (crossattn, priorquery) must stay.
* **The DEVLOG is append-only**, except provenance and precision corrections,
  which are made in place AND recorded in a later step. **A pre-registration is
  never edited after results exist.**
* **Wording rules:** name the denominator of every parameter ratio; never turn
  PSNR drops into percentages; near a threshold, separate observed / reliable /
  exceeds; mechanisms are hypotheses; "did not demonstrate an improvement
  exceeding the registered threshold", not "did not combine" and never
  "ceiling".
* **No `Co-Authored-By` trailer** on commits in this repo — the author's standing
  rule, which overrides the harness default.
* Selection on validation only. The test split has never driven a choice.

---

## 6. FILES AND LINKS

**New tracked code this session:** `dino_gate.py`, `dino_multilevel.py`, the
`restormer_{gated_render,gated_noisy,postlatent_render,aca_l6_postlatent_render,
multilevel_addition_render,multilevel_aca_render}_arch.py` archs; scripts
`paired_compare.py`, `final_matched_pair.py`, `global_vector_identifiability.py`,
`run_aca_interventions.sh`, `summarize_aca_interventions.py`,
`smoke_tests_{affm_subset,wave2,multilevel}.py`, `run_smoke2k_multilevel.sh`;
`phase5_crop_context/crop_gap_correction.py`; every arm's config, chain script
and pre-registration devlog.

**Edited shared files** (default-off, verified no-op): `dino_aca.py` (inference
interventions), `predict_phase3.py` (`--aca-intervention`).

**Artifacts (private, claude.ai):**
* results grid — https://claude.ai/code/artifact/6d4bfad6-890c-4d7e-8ab0-de601d081087 (stale: pre-wave-3)
* the arms as flow charts — https://claude.ai/code/artifact/ffac72e8-cd8f-46d3-b42f-3e076c1a9456
* AFFM versus the gate, worked example — https://claude.ai/code/artifact/d3da55d5-7ca3-4953-a5ce-89601c9368b6

Git: branch `dino_e2`, clean at hand-over; last commit before this file
`1f40226`.
