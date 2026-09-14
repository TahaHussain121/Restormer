# Paste this at the start of a new session

---

I'm doing a master's thesis on using a frozen DINOv2 visual prior to improve
restoration of holographic radar images. Before answering anything, read these
in order:

1. `SESSION_HANDOFF_2026-09-14.md` — live state, the running final experiments,
   and every correction made last session
2. `DEVLOG.md` Steps 42–50 — the numbers of record
3. `THESIS_STORY.md` — the narrative spine; **its correction banner overrides
   the text below it**
4. `PHASE3_CHAPTER.md` — the long-form write-up (§4.5 and §7.2a carry
   corrections)

If those disagree with anything below, **they are right and this prompt is
stale.**

---

## Where things stand

**Two FINAL training experiments are running and they close the training
study**: multi-level addition and multi-level ACA — the B6 prior after the latent
blocks and at the inputs of decoder levels 3 and 2 — jobs 1812561 and 1812562,
due around 16 September. **First job of the session: check whether they
finished; if so, run validation-only selection, the eight evaluation cells, and
`final_matched_pair.py`.** Exact commands are in the handoff. Do not launch any
other training arm.

## The project in one paragraph

Low-ray-budget radar reconstructions (**1e5**) are restored towards a 100x
budget target (**1e7**) with an unmodified Restormer. An aligned clean **render**
is available at inference and reaches the network only through frozen DINOv2
features. The study varies how that prior is injected: source, spatial
structure, depth, fusion operator, and location.

## What the evidence currently says

1. **The source decides the sign.** A render prior gives +2.2 dB; a prior
   computed on the noisy radar is worse than none — and a learned gate does not
   rescue it.
2. **The value is spatial.** Pooling the grid away drops below baseline and
   overfits.
3. **B6 injected before the latent blocks is the LOW point of the good arms.**
   Several single changes lift it by roughly +0.23 to +0.31 dB — B3 instead of
   B6, three depths, or injecting after the latent blocks — and they cannot be
   told apart. Every one of them gains on full frames only and is flat on crops.
4. **The fusion operator does not help, and at the post-latent location the ACA
   block is significantly worse than plain addition** (−0.247 dB).
5. The gains do not stack cleanly: post-latent plus B3 did not demonstrate an
   improvement exceeding the registered threshold.

Earlier claims now WITHDRAWN: "depth count is the lever", "B6 is the best
depth", "injecting before the latent is strictly better", "4.6x the parameters"
without a denominator (it's +4% of the network), and "the learned mixing is 2% of
the branch's value".

## Vocabulary

**arm** = one trained experiment. **E0** = no-DINO baseline. **full256** =
whole-frame evaluation, **crop128** = matched-crop evaluation — **always report
both**. **B3/B6/B9/B12** = DINOv2 blocks. **pre/post-latent** = the prior added
before/after the eight latent blocks. **ACA** = the self-attention plus gated
channel cross-attention block. **the tier** = the indistinguishable +0.25 dB
group.

## Things I will get annoyed about if you get them wrong

- **Report both protocols, always.**
- **Name the denominator of every parameter ratio** (added vs total).
- **Never turn PSNR drops into percentages.**
- **Near a threshold, keep observed / reliable / exceeds-threshold separate.**
- **Mechanisms are hypotheses** until measured.
- **Single seed, deliberately**; cross-split replication is the substitute. Don't
  push seeds.
- **Design reviews I paste are from ChatGPT, not my supervisor.** Attribute them
  that way. (That the DINO prior is a premise set by my supervisor is my own
  statement.)
- **Pre-registrations are never edited after results exist.**
- crossattn-render and priorquery-render stay out of the narrative.
- **No `Co-Authored-By` trailer** on commits.

## Working style

- Run commands on the HPC yourself; don't hand me scripts. Slurm, a100 for
  training, v100 for evaluation.
- Honest nulls and real measurements. Say plainly when something doesn't work or
  when you were wrong.
- Verify against code and data before asserting.
- Write findings into tracked files: `experiments/`, `**/results/`, `*.pth` and
  `*.png` are gitignored.
- My English isn't native and my messages are short. Read for intent. If I ask
  for "simple", drop the jargon.
- **Plan in plain language and ask before any new experiment. Nothing may be
  deleted without an exact manifest I've read.**

## Open, and mine to decide

The narrative rewrite (after the final pair), disk cleanup (manifest first), a
one-line comment fix in `dino_aca.py`, and republishing the results grid.

---

Start by checking the two final chains, then tell me the state and what you'd do
next. Don't start any long job without asking.
