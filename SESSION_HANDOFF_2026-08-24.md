# SESSION HANDOFF — 2026-08-24 11:00

Read `CONTEXT.md` and `HANDOVER.md` first; this file is the delta since
`SESSION_HANDOFF_2026-08-21.md` plus what a fresh session should do next.

---

## 1. WHAT IS RUNNING RIGHT NOW

| job | arm | node | progress | walltime left |
|---|---|---|---|---|
| **1788585** | affm-render | tg097 | **243,000 / 300,000** | ~16h49m |
| **1789321** | dinolight-render | tg091 | **241,000 / 300,000** | ~16h57m |
| 1791652 | affm successor | — | `PENDING (Dependency)` | — |
| 1791655 | dinolight successor | — | `PENDING (Dependency)` | — |

**The two PENDING jobs are correct and expected**, not a problem: `chain_core.sh`
queues a successor from *inside* each job at startup so a walltime kill cannot
break the chain. Each arm needs ~7 h and has ~17 h left, so both should finish
inside their current job, and `chain_core` **cancels the unused successor** when
`net_g_300000.pth` appears.

Both are on `CHAIN_COUNT=2` — job 1 of each hit its 24 h walltime at exactly
188k, wrote `188000.state`, and job 2 resumed from it. That is the design
working, not a fault.

**Projected finish: both ~2026-08-24 18:00.**

Nothing else is running. Every other experiment directory is idle.

---

## 2. THE IMMEDIATE NEXT TASK

When both hit 300k:

1. Confirm `TRAINING_DONE` exists in each
   `experiments/Phase3_chain_state_Holo_{affm,dinolight}_*/` and that jobs
   1791652 / 1791655 were auto-cancelled.
2. **Select checkpoints on VALIDATION only** — pre-registered, non-negotiable.
3. Run `scripts/run_evaluate.sh` for both arms, both protocols, on **val**.
4. **Do not touch the test split** until you are ready to read it once, for all
   arms together.
5. Only then compare against addition-render's 24.081.

---

## 3. WHAT THE TWO ARMS ARE

Both replace exactly one tensor — the one entering the latent at
`inp_enc_level4`, `[B,384,16,16]`. Trunk, decoder, skips and loss are untouched.

**affm-render** — the clean ONE-FACTOR arm. DINO read at **{3,6,9,12}** instead
of {6}, each layer centred with its own train-only mean, combined by a
per-position softmax **across layers** (weights sum to 1 at each of 256
positions). Output stays 768 channels because it is a weighted SUM, so the
injection is the SAME zero-init `Conv2d(768,384,1)` addition. **+298,372 params
= +1.04% over addition-render** — the point being that a loss here cannot be
blamed on capacity.

**dinolight-render** — DINOLight (arXiv 2603.12579) adapted to radar. Same AFFM
(it *imports* `DinoAffm`, so stage 1 is the same code object), then
`P` feeds a **gated channel cross-attention** instead of an addition:

```
guided = project_out(F_sa + sigmoid(alpha_logit) * F_ca) + F
```

**+1,352,849 params = 4.58x addition-render**, and it changes **TWO** factors at
once. It is a "does the published method transfer" arm, **NOT an ablation** —
say so whenever it is reported. The frequency-domain branch is deliberately
omitted (~0.31 dB by their own ablation, for double the surface).

---

## 4. THE NUMBERS, AND HOW NOT TO MISREAD THEM

On the **8-bit training-time validation** curve at 240k: affm **24.364**,
dinolight **24.239**, addition-render's best-ever **24.117**.

**That is not a win.** Four reasons:

1. Wrong metric — the published 24.081 test / 24.120 val come from the **uint16**
   evaluation path. Not comparable. Neither live arm has been evaluated yet.
2. Point noise here is **~0.4 dB** (E0 swings 21.738 -> 21.328 between 180k and
   188k). Sub-0.3 dB single-point differences mean nothing.
3. No checkpoint selection has run. addition peaked at 204k, concat at 292k.
4. concat-render looked +0.025 ahead on this same curve and the full evaluation
   showed a **statistical tie**.

Honest reading: both live arms sit in the render-arm cluster (~24 dB) with
addition and concat, far above E0 (~21.5), E1-noisy (~21), global-render (~19.7).
**Nothing separates them from plain addition.** Both pre-registered predictions
(affm "not >0.10 dB", dinolight "not >0.30 dB") are so far holding.

---

## 5. THE ACTUAL SURPRISE — AND IT IS NOT PSNR

Both arms log the AFFM per-layer weights every 5k iterations. At ~185k:

| | w_b3 | w_b6 | w_b9 | w_b12 |
|---|---|---|---|---|
| affm-render | **0.172** (lowest) | 0.229 | **0.333** (highest) | 0.266 |
| dinolight-render | 0.249 | **0.211** (lowest) | 0.231 | **0.309** (highest) |

**B6 — the locked layer — wins in NEITHER arm.** And **B12**, which WO1 placed
nearest the different-scene floor (0.2337 same-scene against a 0.1175 floor), is
fine and is the **top** layer in dinolight. The B6 lock rests on that
feature-space ranking; two arms optimising a completely different objective
disagree with it.

They also disagree **with each other** about which layer wins, which argues the
weights are **weakly determined** rather than that a new winner has been found.
Report it either way — it is pre-registered as a secondary outcome in both
devlogs.

**CORRECTION ON RECORD:** the 6k smokes showed B3 gaining, and that was read out
loud as an early signal. Over 185k it did **not** hold — it was noise. Do not
repeat that reading.

**dinolight's `alpha`** rose 0.1192 -> ~0.139 and plateaued: the DINO path is
being used, modestly, and the gate has not closed.

---

## 6. VERIFIED THIS SESSION (things you can rely on)

- **The channel-attention scale property is real, not asserted.** dinolight's
  attention matrix is `64x64` (C/heads) at BOTH the 16x16-token train regime and
  the 32x32-token eval regime, while the DINO grid demonstrably changes. On a
  trained 6k checkpoint: crop128 19.562 / **full256 21.645** — full256 HIGHER.
  Contrast crossattn-render: +2.00 dB crop128 vs **−7.56 dB** full256 from
  identical weights.
- dinolight smoke: **44/44** architectural checks, plus a 6000-iteration
  training smoke with the gate ENFORCED across the 5000 boundary, `rc=0`, no
  `STABILITY_FAILURE`.
- **ACA costs only +0.5% step time** vs affm on the same card (0.6837 vs 0.6800
  s/iter on v100). Peak VRAM 31,450 MiB on a 32 GB v100 — fits a100, would NOT
  fit a 10 GB card.
- Both arms' four mean files are **bit-identical** (verified), so they are
  comparable on the feature side.

---

## 7. TRAPS

- **Two stale `RUNNING_JOB` locks — LEAVE THEM ALONE**: crossattn-render
  (`1784331`) and priorquery-render. Both were cancelled by hand; deleting a lock
  lets a dead arm auto-resume.
- **priorquery-render is DROPPED** (cancelled 2026-08-23 at 90k). Leave its files.
- **Do not edit `chain_core.sh` or any shared arch while an arm is training.** A
  change is read by the NEXT RESUME, silently, hours later. That is why
  dinolight's Aug-28 deadline guard lives in its own wrapper.
- **affm-render has NO deadline guard**; dinolight does. Asymmetry, known.
- Observation tags (`dino/affm_w_*`, `dino/aca_*`) are a **step function** —
  measured every N forwards, reprinted every `print_freq`. De-duplicate before
  plotting.
- The `dino/` prefix on the AFFM/ACA tags is deliberate: the model wrapper
  prefixes everything with `dino/`, and an `affm/` prefix would have required
  editing a file five finished arms also run.

---

## 8. GIT

Branch `dino_e2`, tracking `origin/dino_e2`. Last commit `1f343a0`.
**9 items uncommitted** — the whole dinolight arm (8 new files) plus the
`chain_affm_render.sh` walltime change (23h -> 24h). Committing is safe while
the arms train; editing is not. No `Co-Authored-By` trailer on this repo.

---

## 9. THE DEADLINE

Cluster maintenance **2026-08-28**. ~85 h left as of this handoff. The two live
arms finish today. **Nothing else needing 300k can be started** — that is ~39 h
of compute plus queue wait, and the queue already cost 15 h once this week.
Plan the remaining time around **evaluation and writing**, not new training.

The cheapest open item remains the **crossattn eval256 attention probe** (~2 min)
— it converts the scale-transfer finding from a well-supported hypothesis into
something quotable.
