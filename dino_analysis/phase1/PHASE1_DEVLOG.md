# Phase 1 Devlog — Dataset-Level Spatial DINO Consistency

Scope: **Phase 1 only** — establishing, across the validation split, which DINO
depth most consistently preserves spatial structure across Very Noisy Radar
(1e5 rays), Clean Radar (1e7 rays) and Render.

Phase 2 (comparing DINO(1e5) and DINO(Render) as candidate structural priors
relative to DINO(1e7)) gets its own `dino_analysis/phase2/PHASE2_DEVLOG.md` and
must not be logged here.

Predecessor: the single-sample run in
`dino_analysis/DINO_ANALYSIS_DEVLOG.md` (sample 0196, 2026-08-09).

Entries below are **appended automatically by**
`dino_analysis/phase1/analyze_dino_spatial_consistency.py` at the end of a full
run, using the real local execution timestamp and the real numbers from that
run. Smoke tests (`--max-samples`) never write here. Nothing is written before
the analysis actually executes, so an empty log below means Phase 1 has not been
run yet.

---
