# Phase 2 Devlog — DINO Prior Source: 1e5 Radar vs Render

Scope: **Phase 2 only** — at the DINO depth Phase 1 selected, does DINO(1e5) or
DINO(Render) give a spatial representation better aligned with the clean
DINO(1e7) target?

Judged on two quantities, never one alone:
1. **absolute** same-scene corresponding-patch cosine to DINO(1e7)
2. **scene-specific advantage** = same-scene − different-scene cosine

Phase 1 showed different-scene pairs can score highly, so absolute similarity on
its own can reflect a generic shared component rather than scene content.

Out of scope: Restormer, training, and any PSNR/SSIM claim — that is Phase 3,
which gets its own `dino_analysis/phase3/PHASE3_DEVLOG.md`.

Predecessors: `dino_analysis/DINO_ANALYSIS_DEVLOG.md` (single sample) and
`dino_analysis/phase1/PHASE1_DEVLOG.md` (validation-set layer sweep).

Entries below are **appended automatically by**
`dino_analysis/phase2/analyze_dino_prior_source.py` at the end of a full run,
using the real local execution timestamp and the real numbers from that run.
Smoke tests (`--max-samples`) never write here. An empty log below means Phase 2
has not been run yet.

---
