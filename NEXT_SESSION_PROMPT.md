# Paste this at the start of a new session

---

I'm doing a master's thesis on using a frozen DINOv2 visual prior to improve
restoration of holographic radar images. Phase 3 is finished and fully
evaluated. Before answering anything, read these in order:

1. `SESSION_HANDOFF_2026-09-01.md` — current status, all results, traps
2. `THESIS_STORY.md` — the narrative spine, scope decisions, caveats
3. `DEVLOG.md` Steps 29–34 — the numbers of record
4. `PHASE3_CHAPTER.md` — the long-form write-up with every equation

If those disagree with anything below, **they are right and this prompt is
stale.** Don't trust my summary over the repo.

---

## What the project is

I restore holographic radar reconstructions. The input is computed from a low
ray budget (I call it **1e5**), the target from a 100× budget (**1e7**). The
degradation isn't sensor noise — it's missing interference fringes and lost
high-frequency structure, so the model has to hallucinate detail a bigger
compute budget would have produced.

The backbone is **Restormer**, unmodified — a 4-level U-Net of transformer
blocks, 44 blocks total, 26.1M parameters. The eight **latent** blocks hold 55%
of all parameters.

Alongside each radar image I have an aligned **render** — a clean rendered
picture of the same scene. It's available at inference, so this is *guided*
restoration, not blind. The backbone never sees the render as pixels: it reaches
the network **only** through DINOv2.

The DINO prior is injected at exactly one tensor — `inp_enc_level4`, the
384-channel input to the latent blocks. I call it **F**. Every experiment
modifies only that tensor; encoder, decoder, skips and refinement are
byte-identical across all of them. That's what makes the comparisons one-factor.

## Vocabulary I use

- **arm** = one trained experiment. They have names like `addition-render`.
- **E0** / **E0-fixed** = the no-DINO baseline.
- **1e5** = noisy input, **1e7** = clean target, **render** = the DINO source.
- **full256** = evaluate on the whole 256 frame. **crop128** = evaluate on a
  matched 128 crop. I always need BOTH reported.
- **B3, B6, B9, B12** = DINOv2 transformer blocks (1-indexed) I read features
  from.
- **the ladder** = the four ACA arms that vary depth count while holding the
  fusion operator fixed.
- **F_sa / F_ca** = the self- and cross-attention halves inside the ACA block.
- **injection point** / **the latent** = where the prior enters.
- **the gate** = the training-stability monitor on injected-vs-latent norm.

## The story, in the order the experiments answer it

1. **Does a foundation prior help?** Yes — `addition-render` is **+2.208 dB**
   over baseline on the locked test split, better on 298/338 images.
2. **Does the source matter?** Decisively. Point DINO at the *noisy radar*
   instead and it's **−0.577 dB** — worse than no prior at all. Same
   architecture, same parameter count, one tensor swapped.
3. **Does it need to be spatial?** Yes. `global-render` pools the feature grid to
   one vector: **−1.284 dB**, below baseline. Also: shuffling renders between
   images costs **9.094 dB**, so the model uses each image's specific render.
4. **Does the fusion operator matter?** No. Concat is a null (−0.016, p=0.98).
   And `aca-L6` — one factor from addition-render, only the operator differs —
   gives **+0.030 dB, p=0.22, not significant, at 4.6× the parameters**.
5. **Does depth matter?** Yes. `affm-render` reads four DINO depths instead of
   one, still with plain addition: **+0.230 dB on test (p=4.3e-07), +0.284 on
   validation (p=1.0e-09), for +1.04% parameters.**

**The conclusion: the gain comes from reading DINO at multiple depths, not from
the fusion operator.**

## Things I will get annoyed about if you get them wrong

- **Always report both protocols.** Every attention arm gains on full256 and is
  *significantly worse* on crop128 — dinolight included, the one with the best
  headline number. `affm-render` is the only multi-depth arm that loses nowhere.
  Reporting full256 alone misrepresents the result.
- **Don't claim the affm-vs-dinolight gap** (0.079 dB, overlapping CIs). The
  honest separators are parameter cost and crop128 behaviour.
- **Don't say "F_sa does nothing."** The ablation that would have shown that
  couldn't train — a double zero-init deadlock. F_sa turned out to be
  load-bearing for *optimisation*. Different claim.
- **Don't say DINOLight is implemented wrongly.** My arm reproduces it faithfully.
- **crossattn-render is dropped from the narrative** by my decision. Don't bring
  it back into results tables. (Its numbers came from iteration 4,000 of 179,000
  because checkpoint selection used a metric that arm structurally can't do.)
- **priorquery-render** was stopped at 90k and never evaluated. Excluded.
- **Test split honesty:** it was read in several passes as arms finished, not
  once. Checkpoint selection *never* touched test — validation only, verified for
  all eleven evaluated arms. Say it that way; don't claim a single clean read.
- **Single seed, deliberately.** I checked other theses and single-run reporting
  is the norm. Cross-split replication is my substitute. Don't push seeds again.
- **DINO is a premise**, set by my supervisor. There's no no-DINO control and
  that's a scope decision, not an oversight.

## Where things stand right now

Phase 3 complete — 11 arms trained and evaluated on both splits and both
protocols. Queue empty. Nothing running.

**Phase 5** (new) measured whether DINO describes a crop the same way it
describes the full frame, since I train on 128 crops and evaluate on full 256
frames. It doesn't. Highlights:

- DSGIR's crop-shift claim reproduces on my data — degradation agreement is
  significantly worse inside a crop at every depth.
- The drift concentrates at **crop borders** (border − centre negative at every
  layer). This goes beyond their paper.
- **B6 is the most crop-robust depth** on three independent measurements — and
  B6 is the layer my earlier Phase-1/2 study chose for an unrelated reason.

## Working style

- Run commands on the HPC yourself; don't hand me scripts to run. Slurm, sbatch,
  v100 and a100 partitions.
- I want honest nulls and real measurements. If something doesn't work, say so
  plainly. Don't soften a negative result.
- Verify against the code and the data before asserting. If a paper is paywalled
  or a file is missing, say so — don't fill the gap with a guess.
- **No `Co-Authored-By` trailer** on commits in this repo.
- Write findings into tracked files. `experiments/`, `**/results/`, `*.pth` and
  `*.png` are gitignored, so anything left only there is one cleanup from gone.
- My English isn't native and my messages are short and often have typos. Read
  for intent. If I ask for "easy words" or "simple", drop all jargon — don't
  just shorten.

## What I might pick up next

Nothing is required; the thesis is writable as it stands. Ranked options are in
the handoff. In short:

1. A **render-misalignment ablation** — inference only, minutes. Shift the render
   by k pixels and measure degradation. Turns the binary shuffle control into a
   dose-response curve.
2. **Read DINO-IR** (Lin et al., arXiv 2312.01677) — directly adjacent, cited by
   DSGIR, still absent from my repo.
3. A **parameter-matched capacity control** (~38 h) — the last methodological
   hole, though `aca-L6` already gives partial evidence capacity is neutral.
4. **Fold the Phase-5 findings into `PHASE3_CHAPTER.md`** — the chapter still
   describes the crop128 mechanism as interpretation, but I now have the
   measurement.

**Don't suggest:** more ACA arms, `aca-L369`, seed replication, or a DINO
auxiliary loss. All decided against.

---

Start by reading the four files above, then tell me what you understand the
current state to be and what you'd suggest. Don't start any long job without
asking.
