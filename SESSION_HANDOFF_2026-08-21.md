# Session handoff — 2026-08-17 → 2026-08-21

Everything from this session, in the order someone picking it up cold would
want it. Numbers are read from the evaluation chain's CSVs; nothing here is
recalled from memory.

---

## 1. WHERE THE PROJECT STANDS

The fusion axis is now four arms deep and has produced **no operator
advantage**. The two axes that matter were already settled: the SOURCE (render
vs 1e5 radar) and the FORM (spatial vs pooled).

| arm | DINO reads | form | fusion | +params | test full256 | vs E0 | status |
|---|---|---|---|---|---|---|---|
| E0-Fixed | — | — | — | 0 | 21.873 | — | done (268k) |
| E1-addition-noisy | 1e5 | spatial | addition | 295,296 | 21.296 | −0.577 | done (128k) |
| **addition-render** | render | spatial | addition | 295,296 | **24.081** | **+2.208** | done (204k) |
| global-render | render | **pooled** | addition | 295,296 | 20.589 | −1.284 | done (60k) |
| concat-render | render | spatial | concat | 442,752 | 24.065 | +2.193 | done (292k) |
| crossattn-render | render | spatial | attn, radar-Q | 888,576 | 18.723 | **−3.150** | **FAILED, removed at 179k** |
| priorquery-render | render | spatial | attn, prior-Q | 741,120 | — | — | **queued, job 1785021** |

**The headline is unchanged:** a clean geometric view of the object, delivered
through frozen DINOv2 B6 patch tokens injected at the latent, gains +2.21 dB
over an identical no-prior baseline; the same mechanism fed the noisy radar
loses 0.58 dB.

---

## 2. WHAT THIS SESSION ADDED

### Built and run
- **concat-render** — trained to 300k, evaluated on all four cells. A
  statistical tie with addition (below).
- **crossattn-render** — built, verified, trained to 179k, **failed**, evaluated,
  removed. Full record in its devlog.
- **priorquery-render** — built and verified (52/52 smoke), queued, not yet run.

### Controls and diagnostics
- **Mismatched-render control on addition-render** (post-hoc, val): shuffled
  render −9.297 dB, 339/339 images worse, r=0.251; zero −8.901; train-mean
  −9.872. `‖P(D)‖` identical under shuffling by construction.
- **The same control on concat-render**: −9.153 dB, 339/339 worse. The fusion
  change did not alter the mechanism.
- **Weight surgery on concat**: forcing `W_F ← I` costs 1.70 dB; keeping only
  the diagonal recovers 0.42 of that; rank-32 (22% of spectral mass) recovers
  everything. So the learned radar mixing is load-bearing WITHIN that trained
  network — which is not the same as an advantage BETWEEN arms.
- **Input baseline**: the raw 1e5 scores 12.354 dB on test/full256. E0 is
  +9.518 over it; the best arm +11.727.

### Reporting
- All comparison figures now carry **both** full-image and foreground-masked
  PSNR, with deltas for each.
- New reproducible figure scripts: `make_arm_panel.py`,
  `make_results_summary.py`, `make_e0_report_figures.py`, `plot_val_curves.py`.

---

## 3. THE TWO NEW RESULTS, STATED CAREFULLY

### concat vs addition — a tie, not a win

| cell | metric | addition | concat | delta | 95% CI |
|---|---|---|---|---|---|
| full256/val | PSNR | 24.120 | 24.143 | +0.023 | [−0.071, +0.119] |
| full256/test | PSNR | 24.081 | 24.065 | −0.016 | [−0.116, +0.085] |
| full256/test | masked | 19.673 | 19.741 | +0.068 | [−0.024, +0.160] |
| crop128/test | PSNR | 22.259 | 22.473 | +0.214 | [+0.099, +0.334] ✓ |

Only crop128 separates from zero, and even there it misses the pre-registered
+0.30 dB threshold — bought with 50% more added parameters. **Prefer addition
on parsimony.** Caveat: n=1 seed per arm, and a 0.02–0.2 dB difference is
exactly the size seed variance could produce.

### crossattn — a failure with a visible mechanism

Best validation was **iteration 4,000 (18.567 dB)** and it never improved on it
in 179k iterations. Trained 87k iterations past the 92k LR restart — the phase
where every other arm found its last 0.26–0.76 dB — with no recovery.

The attention statistics explain it: `attn_entropy` **5.52 → 0.151** (97%
collapse, essentially one-hot) while `attn_diag_mass` reached only **0.27**. It
learned confident routing to the WRONG positions — the named risk of computing
queries from the degraded 1e5 latent.

**This does NOT license "cross-attention doesn't work here"**: one seed, one
direction, one layer, one injection point, 3x addition's parameters. The
prior-as-query direction is a separate, queued arm.

### The gate has a hole, demonstrated

Through that run `latent_norm` grew 1,218 → 20,058 and `projected_norm`
629 → 16,486 — both ~16x — while `injection_ratio` stayed 0.5–0.85, never near
the cap of 10. **No rule fired.** The arm lost 3.15 dB with the gate silent.
This is the "co-inflation has no gate rule" item listed as open in every
Phase-3 devlog; it is no longer hypothetical.

---

## 4. WHAT IS RUNNING / QUEUED

- **Job 1785021** `p3chain_pq` — priorquery-render, `PENDING (Priority)`,
  waiting for a free a100. Nothing else of ours is queued.
- crossattn's chain-state dir keeps a stale `RUNNING_JOB` lock. `touch
  experiments/Phase3_chain_state_Holo_crossattn_render_.../CHAIN_ABORTED` if an
  accidental resubmission must be made impossible.

### On hardware
v100 is free while a100 is saturated, and these arms fit v100 (25,147 MiB
measured, 32 GB card). **But** every other arm trained on a100 with TF32 on;
Volta has no TF32, so a v100 run would be the only arm in true fp32 — an
uncontrolled difference in a comparison designed to have exactly one. Also
`chain_*.sh` hardcodes `ARM_PART`, so an sbatch override moves only the first
job, not the successors. Recommendation: stay on a100 unless the wait becomes
days.

---

## 5. OPEN ITEMS, HIGHEST VALUE FIRST

1. **The B3 arm has never been run.** B6 was locked on noise-stability
   (same-scene 1e5↔1e7 correspondence); B3 wins on the SCENE-ADVANTAGE
   criterion (+0.1966 vs +0.1453 at the training scale). This is a documented
   criterion choice, not a measurement, and it is the highest-value outstanding
   experiment — more so than another fusion variant, given the fusion axis has
   produced nothing.
2. **Seed replication.** Every sub-0.3 dB comparison is currently
   unfalsifiable. 2–3 seeds on addition vs concat would settle the fusion axis.
3. **A gate rule for co-inflation** — now motivated by real evidence.
4. **Token-shuffle control**, still not run. Most informative for the attention
   arms, which are the only fusions that could in principle recover from it.
5. A **gated variant** is worth listing as future work: E1-noisy loses 0.58 dB
   overall but GAINS +0.47 on E0's hardest decile, which is the textbook case
   for a per-sample gate.

---

## 6. PROCESS LESSONS FROM THIS SESSION (both cost real time)

- **A decision point needs a mechanism that outlives the session.** The plan was
  to stop crossattn at 100k; the watcher was a background shell that died with
  the session, so it ran to 179k unattended — ~13 hours of a100 wasted. Use a
  SLURM dependency job or a cron.
- **`run_evaluate.sh` keys its output directory off the config NAME, not the
  checkpoint.** Evaluating two checkpoints of one arm concurrently makes them
  overwrite each other (jobs 1786216/1786220 did exactly this; that cell was
  discarded and re-run). For a second checkpoint use `predict_phase3.py` with
  its own `--out-root`.
- A latent bug was found and fixed: `predict_phase3.py` detected DINO arms by
  class-NAME prefix, so `RestormerDinoConcatRender` and
  `RestormerDinoCrossAttnRender` silently skipped `set_dino_mode()` and ran
  full256 in the train128 regime. The no-interpolation gate caught it loudly
  rather than producing a plausible wrong number. Detection is now duck-typed
  (`hasattr(net, 'set_dino_mode')`) with a cross-check against `dino_enabled`.

---

## 7. FIGURES TO LOOK AT

All under `dino_analysis_phases/phase3_restoration/results/comparisons/`:

| file | what |
|---|---|
| `all_arms_results_summary_test.png` | every arm, absolute + delta vs E0 with bootstrap CIs |
| `all_arms_with_crossattn_full256_test.png` | six arms side by side on 3 cases |
| `all_arms_3cases_{full256,crop128}_test.png` | large 3-case panels, both protocols |
| `E0_training_curves.png` | E0 val/loss/LR with the best checkpoint marked |
| `E0_full256_vs_crop128_test.png` | why the two protocols are not comparable |
| `val_psnr_curves_four_arms.png` | training curves, all arms |
