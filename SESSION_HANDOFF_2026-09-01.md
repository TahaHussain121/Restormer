# SESSION HANDOFF — 2026-09-01

Read `THESIS_STORY.md` first, then `DEVLOG.md` Steps 32–33. This file is the
delta for THIS session and what to pick up next.

---

## 1. PHASE 3 IS COMPLETE. THE QUEUE IS EMPTY.

**Every arm is trained AND evaluated on both splits and both protocols.** Only
priorquery-render has no numbers (dropped at 90k, by decision). Nothing is
running; nothing is pending.

The question the whole ACA ladder existed to answer is **answered**.

---

## 2. WHAT HAPPENED THIS SESSION

1. The three ACA arms were found finished (300k each) but **unselected** —
   `chain_core` manages training only and never runs checkpoint selection.
   Selection was run by hand.
2. Six validation cells + one missing gap cell were run (jobs 1799689–1799695).
3. **The test split was read for the five remaining arms** (jobs
   1799708–1799717): affm, dinolight, aca-L6, aca-L36, aca-L6912.
4. `aca-L6-nosa` was built, smoke-run, and **found unable to train**.
5. A new **Phase 5** was created: does DINO see the same thing in a crop as in
   the full frame? Three scripts, six figures, two of them replicating DSGIR.
6. The DSGIR paper was obtained and read. Several earlier assumptions corrected.

---

## 3. THE HEADLINE RESULT — THE OPERATOR BUYS NOTHING

`aca-L6` is ONE FACTOR from addition-render: same depth {6}, same layer, same
mean, same injection point, same seed — **only the fusion operator differs.**

    test full256:  +0.030 dB  p=0.22   NOT SIGNIFICANT
    validation:    +0.075 dB  p=0.075  NOT SIGNIFICANT
    cost:          ~4.6x the parameters

Two independent splits, same verdict. **The gain dinolight-render shows comes
from the DEPTH COUNT, not from the operator.**

### Full test table (n=338, uint16)

| arm | dep | full256 | vs E0 | crop128 | vs E0 |
|---|---|---|---|---|---|
| E0-fixed | — | 21.873 | — | 19.546 | — |
| E1-addition-noisy | 1 | 21.296 | −0.577 | 19.062 | −0.484 |
| global-render | 1 | 20.589 | −1.284 | 19.352 | −0.194 |
| addition-render | 1 | 24.081 | +2.208 | 22.259 | +2.713 |
| concat-render | 1 | 24.065 | +2.193 | 22.473 | +2.927 |
| aca-L6 | 1 | 24.111 | +2.238 | 21.955 | +2.409 |
| aca-L36 | 2 | 24.196 | +2.323 | 22.128 | +2.583 |
| aca-L6912 | 3 | 24.071 | +2.199 | 21.870 | +2.324 |
| dinolight-render | 4 | **24.390** | +2.517 | 22.053 | +2.507 |
| affm-render | 4 | 24.311 | +2.439 | **22.303** | +2.758 |

### Paired vs addition-render

| arm | test full256 | p | test crop128 | p |
|---|---|---|---|---|
| concat-render | −0.016 | 0.98 | +0.214 | 1.7e−04 |
| **aca-L6** | **+0.030** | **0.22** | **−0.304** | 1.3e−06 |
| aca-L36 | +0.115 | 0.011 | −0.130 | 0.032 |
| aca-L6912 | −0.010 | 0.88 | −0.389 | 4.3e−08 |
| dinolight-render | +0.309 | 9.9e−08 | −0.206 | 0.0054 |
| affm-render | +0.230 | 4.3e−07 | +0.044 | 0.14 |

### THE REPORTING RULE THAT MATTERS MOST

**EVERY attention arm is SIGNIFICANTLY WORSE on crop128**, dinolight included —
the arm with the best full256 number. **affm-render is the only multi-depth arm
that loses nowhere.** Never report full256 alone for these arms.

---

## 4. aca-L6-nosa CANNOT TRAIN — and that is itself a finding

Built to remove `F_sa` (the ACA's self-attention half, which duplicates
Restormer's own MDTA). The 6k integration smoke returned rc=0 but
`projected_norm`, `injection_ratio` and `aca_injected_norm` were **exactly 0.0
at every print**, entropy pinned at ln(64).

Cause, verified on a real backward pass:

    gradient reaching project_out:   with F_sa 1.754    without 0.000

With `P` and `project_out` both zero-init, the cross branch outputs zero, so
`project_out` gets zero gradient forever and starves everything upstream.
**`F_sa` is what breaks that deadlock — it is LOAD-BEARING FOR OPTIMISATION,
not only for representation.**

The architectural suite passed 29/29 because it checks that step 0 equals E0,
which is true; the arm never leaves step 0. **The 6k smoke caught it: cost 49
minutes, not 38 hours.**

  DO NOT write "F_sa does nothing" — the ablation never ran.
  DO NOT write that DINOLight is implemented wrongly — it is faithful.

The fix, if ever wanted: initialise `P` normally, keep `project_out` at zero.
Preserves the step-0 identity but makes the arm differ from aca-L6 in TWO ways,
so it stops being a one-factor ablation.

---

## 5. PHASE 5 — THE CROP-VERSUS-FULL DRIFT (new this session)

`dino_analysis_phases/phase5_crop_context/`

**Why:** Phase 3 trains on 128 crops (DINO at 224, 16x16 tokens) and evaluates
on full 256 frames (DINO at 448, 32x32). Nobody had checked whether DINO
produces the same features for the same patch under both. Finding 7's mechanism
was labelled interpretation, not result.

### 5a — the two-axis measurement (`analyze_crop_context_shift.py`)

CONTEXT axis: `cos(crop features, same region inside the full frame)`.
DEGRADATION axis: render/noisy vs clean, measured in EACH context.
INTERACTION `deg_crop − deg_full` is DSGIR's actual claim.

**INTERACTION, n=339, render vs clean:**

| layer | mean Δ | 95% CI | wilcoxon p | worse in crop |
|---|---|---|---|---|
| B3 | −0.0192 | [−0.0201,−0.0184] | 2.8e−57 | 337/339 |
| **B6** | **−0.0020** | [−0.0032,−0.0008] | 3.3e−04 | **193/339** |
| B9 | −0.0180 | [−0.0204,−0.0156] | 6.8e−33 | 269/339 |
| B12 | −0.0284 | [−0.0345,−0.0223] | 1.0e−15 | 225/339 |

**DSGIR's prediction is CONFIRMED on radar data** — degradation agreement is
significantly worse inside a crop, at every depth, worst at the deepest.

**CONTEXT axis:** crop-vs-full cosine 0.66–0.76 against a wrong-place floor of
0.35–0.53. Centred it falls to ~0.48–0.55 — raw cosine is inflated by the shared
layer mean, so the true shift is LARGER than raw numbers suggest, and the
regime-specific centring does NOT compensate for it.

### 5b/5c — six visualisations (`visualize_crop_drift.py`, `tsne_crop_context.py`)

  fig10_replica_dsgir_layers.png / _our_layers.png   DSGIR Fig. 10 replica
  fig12_replica_kde.png                              DSGIR Fig. 12 replica
  spatial_drift_heatmap.png                          WHERE the drift happens
  tsne_context.png                                   t-SNE + separability
  pca_maps_full_vs_crop.png                          joint-PCA, full vs crop

  `tsne_{patch,image}_{raw,centred}.png` + `tsne_separability.json` come from
  `tsne_crop_context.py`. That script was lost to a session interrupt on 09-01
  and was REGENERATED on 09-02 (job 1800304); the rewrite reproduces the
  original numbers exactly (patch_raw B3 0.990 / B6 0.849 / B9 0.929 / B12
  0.923; image_raw all 1.000), so the figures are reproducible again and all
  three scripts are tracked.

**Fig-10 replica, render~clean cosine by crop ratio:**

| layer | Full | 0.8 | 0.5 | 0.2 |
|---|---|---|---|---|
| B1 | 0.9556 | 0.9204 | 0.9031 | 0.9561 |
| B4 | 0.9004 | 0.8187 | 0.7856 | 0.8984 |
| B8 | 0.8254 | 0.7625 | 0.7269 | 0.8182 |
| B12 | 0.5512 | 0.4899 | **0.4085** | 0.5151 |

DSGIR's trend holds from Full → 0.8 → 0.5 at EVERY layer, and the gap widens
with depth (Full−0.5 is 0.053 at B1, **0.143 at B12**).

  **THE 0.2 RATIO REVERSES AND MUST BE EXPLAINED, NOT HIDDEN.** 0.2 of a 256
  frame is 51 px upsampled 4.4x to 224, which smooths the degradation away so
  both images look alike. It is a domain artifact of small source images, not a
  contradiction of DSGIR. Report the 1.0–0.5 range, which is the range training
  actually uses, and state the artifact.

**Aligned drift, this project's real pipeline (full 448 / crop 224):**

| layer | position cosine | border − centre | NN top-1 |
|---|---|---|---|
| B1 | 0.7987 | −0.0639 | 0.123 |
| B3 | 0.7591 | −0.0549 | 0.106 |
| B6 | 0.7081 | −0.0578 | 0.111 |
| B9 | 0.6568 | −0.1373 | 0.114 |
| B12 | 0.6721 | −0.1261 | 0.156 |

**BORDER − CENTRE IS NEGATIVE AT EVERY LAYER.** The drift concentrates at the
crop BORDERS — missing surrounding context, made spatial. **DSGIR asserts this
mechanism; this measures it. It goes beyond their paper.**

**NN position retrieval 0.045–0.156** against chance 1/1024 = 0.001. ~100x
chance, yet 85–95% of crop tokens still cannot identify their own position.

**Context separability** (logistic regression on 768-d features, 0.50 = chance):
image-level **0.97–1.00**, patch-level 0.78–0.99, surviving centring.

### B6 IS THE MOST CROP-ROBUST DEPTH — three independent measurements

  1. interaction −0.0020, vs −0.018 to −0.028 at every other depth
  2. context separability 0.859, the LOWEST of the seven layers measured
  3. Phase 1/2 selected it, for an unrelated reason (cross-source consistency)

**This independently vindicates the Phase-1/2 layer choice.** Write it up.

---

## 6. WHAT THE DSGIR PAPER ACTUALLY SAYS (now read, no longer second-hand)

Deng, Tian, Zhao, Liu. *DSGIR: Dual-semantic guided all-in-one image
restoration.* Neurocomputing 696 (2026) 134106.

  - **Fig. 8 is NOT the crop analysis.** It is t-SNE of DEGRADATION-TYPE
    representations from their DSE module. Do not cite it for crop drift.
  - **Fig. 10 IS the crop analysis**: layer-wise cosine, crop ratios
    Full/0.8/0.5/0.2, layers {1,4,8,12}.
  - **Fig. 12** is KDE of per-image similarity, local-crop vs full-image.
  - CSA adapts DINOv2 layers **{9,10,11,12}** with a **zero-initialised residual
    projection** — the same zero-init idea this project uses at `P`.
  - CSA is trained with **hybrid preprocessing**: each iteration takes, with
    equal probability, a full image resized to 224 OR a random 224 crop. That
    is how they make DINO robust to both regimes.
  - Content priors are injected **hierarchically**: `z^(12)` at the LATENT
    stage, `z^(8)`, `z^(4)`, `z^(1)` into successive DECODER stages.
  - **Their content prior is a GLOBAL VECTOR** — SGFM produces channel-wise
    affine γ, β broadcast spatially.

**Perceive-IR** (verified from arXiv HTML): prior is `F_l ∈ R^{1x768}` per level,
read at DINOv2 layers **i = 1,4,8,12**, injected at **four DECODER levels** via
ETB = PGCA + GDFN.

  **BOTH reference methods buy multi-level decoder injection by giving up
  SPATIAL structure. On this data a global prior is worse than no prior at all:
  global-render is 1.284 dB BELOW baseline.** That is the architectural
  trade-off to state, and it is why this project's design diverges from theirs.

---

## 7. WHAT TO DO NEXT

**Nothing is required. Phase 3 is complete and writable-up as it stands.**

Ranked, if there is appetite:

1. **Move `results/INTERPRETATION.md` somewhere tracked.** It explains every
   Phase-5 figure in plain language — what it means, why that figure, how it was
   made — and it currently sits in the GITIGNORED results folder. The numbers
   themselves are safe in DEVLOG Step 34; the explanations are not.
2. **A render-misalignment ablation.** Inference only, minutes. Shift the render
   by k pixels for k = 0,1,2,4,8,16 and measure degradation. Converts the binary
   shuffle control (−9.094 dB) into a dose-response curve, and answers the
   fragility question any reader will ask.
3. **Read DINO-IR** (Lin et al., arXiv 2312.01677) — directly adjacent, cited by
   DSGIR as [36], and **still absent from this repo**. Do not characterise it
   second-hand.
4. **A parameter-matched capacity control.** ~38 h. aca-L6 already gives partial
   evidence capacity is neutral (4.6x, level with addition-render); a dedicated
   control would settle it.
5. **A middle point on the ADDITIVE depth curve** (affm at {3,6} or {6,9}). The
   operator that carries the result has only two points, 1 depth and 4.

**DO NOT** run aca-L369 — it duplicates a depth point on the operator the data
shows is inert. **DO NOT** add more ACA arms.

---

## 8. TRAPS

- **`chain_core` NEVER runs checkpoint selection.** It writes `TRAINING_DONE`
  and exits. An arm without a deferred dispatcher sits finished and unselected.
- **BasicSR has NO best-model tracking.** Validation runs and logs a score;
  nothing compares scores. Selection is a post-hoc log parse.
- **Checkpoints save every 2000, validation runs every 4000.** Only ~75 of 151
  checkpoints per arm are ever scored; every selected iteration is a multiple
  of 4000.
- **Two stale `RUNNING_JOB` locks must stay**: crossattn (1784331) and
  priorquery (1785021). Deleting one lets a dead arm auto-resume.
- **`experiments/`, `**/results/`, `*.pth`, `*.png` are gitignored.** Every
  Phase-5 figure and JSON exists on disk only.
- **The Phase-3 matched-128 manifest is NOT token-aligned** — only 5 of 339
  crops have x,y divisible by 8. Phase 5 draws its own aligned manifest and does
  not touch the evaluation one.
- **The stability gate only constrains the RATIO** and is blind to co-inflation
  of latent and injected norms. The crossattn arm inflated both 15x/14x and the
  gate never fired. A co-inflation rule was specified and never implemented.
- **The test split was read in SEVERAL passes**, not once. Selection never
  touched test — verified for all eleven evaluated arms. Say it that way.
- **Do not claim the affm-vs-dinolight gap** (0.079 dB, overlapping CIs).
- **crossattn-render is being dropped from the narrative** (user decision). Cut
  it cleanly; do not leave it half-present in a table. Note that the stability
  gate paragraph currently uses it as its worked example.

---

## 9. FILES WRITTEN THIS SESSION

Tracked in git:

  THESIS_STORY.md         rewritten to test numbers; scope decisions; caveats
  PHASE3_CHAPTER.md       ~8,500-word long-form chapter with all equations
  PROSE_ARGUMENTS.md      copy-pasteable paragraphs, two parts
  DEVLOG.md               Step 33 appended
  SESSION_HANDOFF_2026-09-01.md   this file
  dino_analysis_phases/phase5_crop_context/analyze_crop_context_shift.py
  dino_analysis_phases/phase5_crop_context/tsne_crop_context.py
  dino_analysis_phases/phase5_crop_context/visualize_crop_drift.py
  basicsr/models/archs/dino_aca_nosa.py
  basicsr/models/archs/restormer_aca_l6_nosa_render_arch.py
  configs/aca_render_fixed128_L6_nosa_latent.yml
  scripts/chain_aca_render_fixed128_L6_nosa_latent.sh
  scripts/smoke_tests_aca_nosa.py
  scripts/run_smoke6k_aca_l6_nosa.sh
  scripts/make_arm_panel.py   (+ affm/dinolight arms, --arms, --exclude)

Two published artifacts (private, on claude.ai):

  The DINO Injection Pass   forward-pass walkthrough, 8 diagrams
  The Phase-3 Casebook      every experiment with equation, story, result
