# SESSION HANDOFF — 2026-08-24 20:00

Read `CONTEXT.md` and `HANDOVER.md` too; this is the delta since
`SESSION_HANDOFF_2026-08-24.md` (midday) plus exactly what to do next.

---

## 1. NOTHING IS RUNNING. THE QUEUE IS EMPTY.

**affm-render and dinolight-render both COMPLETED 300,000 iterations.** Both
wrote `TRAINING_DONE`; `chain_core` auto-cancelled both successors (1791652,
1791655). No `STABILITY_FAILURE`, no gate trigger on either. Each took two jobs
— 24 h walltime kill at 188k, clean resume from `188000.state`.

Your account holds **no a100 GRES**, so the `AssocGrpGRES` block that stopped an
earlier aca-L6 submission is gone.

---

## 2. THE MOST URGENT THING: EVALUATE THE TWO FINISHED ARMS

**300k of compute currently exists only as `.pth` files.** No checkpoint
selection, no metrics, no figures. `experiments/` is gitignored, so this is the
work most at risk of being lost — exactly how the E1-FiLM runs became
unrecoverable (DEVLOG Step 28).

~30 min, runs on rtx3080/v100, no conflict with a100.

1. Confirm `TRAINING_DONE` in both `experiments/Phase3_chain_state_Holo_{affm,dinolight}_*/`.
2. **Select checkpoints on VALIDATION only** — pre-registered, non-negotiable.
3. `scripts/run_evaluate.sh` for both arms, both protocols, on **val**.
4. **Do not touch the test split** until ready to read it once, for all arms
   together.
5. Only then compare against addition-render's 24.081.

**Do not quote the training-time val numbers as results.** Last values on that
8-bit path: affm best **24.402 @ 224k**, dinolight best **24.304 @ 236k**, both
still climbing at 300k. That is a DIFFERENT METRIC from the uint16 evaluation
that produced 24.081 test / 24.120 val. Point noise on that curve is ~0.4 dB,
and concat-render once looked +0.025 ahead there and turned out to be a
statistical tie.

---

## 3. THREE ACA ARMS BUILT, SMOKE-PASSED, NOT SUBMITTED

Fusion is **identical** across all four ACA arms — the same `DinoAca` block,
imported never copied. **Only the layer set changes.**

| arm | layers | AFFM | delta over E0 | vs dinolight |
|---|---|---|---|---|
| **aca-L6** | {6} | **none** | 1,349,773 | −0.227% |
| aca-L36 | {3,6} | 2 convs | 1,351,311 | −0.114% |
| aca-L6912 | {6,9,12} | 3 convs | 1,352,080 | −0.057% |
| dinolight-render | {3,6,9,12} | 4 convs | 1,352,849 | — |

**aca-L6 is the one that matters.** It is **ONE factor from addition-render** —
same B6 layer, same mean, same injection point, only the fusion operator
differs. Nothing else in Phase 3 gives that comparison. **If only one ever runs,
it must be this one.**

**NO AFFM in aca-L6, deliberately** — a 1-layer softmax is identically 1.0, so a
degenerate module would be 769 dead parameters. The smoke asserts it is absent
and that **zero `affm_w_*` keys** appear in training. Not an inconsistency.

**TWO COMPARISONS, TWO CAVEATS — NEVER CONFLATE.** Across the ACA ladder the
spread is **0.227%**, so a difference *there* is NOT attributable to capacity.
Against addition-render every ACA arm is **~4.6x** the parameters, so THAT
comparison IS capacity confounded.

### Smoke results

- **Architectural: 31/31 on each arm.** Step-0 output bit-identical to E0
  (`0.000e+00`), 494/494 trunk tensors byte-identical, param counts exact,
  alpha/temperature read from dinolight's own module.
- **Scale check on aca-L6 PASSED**: attention matrix **64×64 at BOTH** the
  16×16-token and 32×32-token regimes while the grid demonstrably changes. This
  is the property crossattn-render lacked (+2.00 dB crop128 vs −7.56 dB full256
  from identical weights).
- **6k INTEGRATION smoke on aca-L6 PASSED** (job 1792098, v100, 1h13m): rc=0,
  gate **enforced** across iteration 5000, no `STABILITY_FAILURE`,
  `injection_ratio` 0.24–0.86, alpha 0.11920 → **0.12756** rising, **zero
  `affm_w_*` keys**, val 20.74 / 21.03 / 21.34 at 2k/4k/6k.
- **0.6831 s/iter on v100** (dinolight: 0.6837 — within 0.1%) → **~0.4495 s/iter
  on a100 → 37.5 h for 300k = two 24 h jobs.**
- **Peak VRAM 25,214 MiB vs addition-render's 25,117 on the same card and the
  same metric = +0.4%.**

**arms 2 and 3 have had the architectural smoke only, not the 6k integration
run.** They share the same `DinoAca` block aca-L6 just validated, but the affm
precedent is that integration bugs live where the architectural test cannot
reach.

### The two commands

```bash
# STAGE 1 — aca-L6, now
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer && \
jid=$(sbatch --parsable dino_analysis_phases/phase3_restoration/scripts/chain_aca_render_fixed128_L6_latent.sh) && \
python3 dino_analysis_phases/phase3_restoration/scripts/aca_manifest.py --arm aca_render_fixed128_L6_latent --job $jid --event submitted && \
echo "aca-L6 submitted as job $jid"

# STAGE 2 — arms 2 and 3, ONLY after aca-L6 looks healthy
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer && \
dino_analysis_phases/phase3_restoration/scripts/submit_aca_stage2.sh
```

---

## 4. A VRAM CORRECTION ON RECORD

On 2026-08-22 dinolight-render's peak VRAM was quoted as **31,450 MiB** and
compared against addition-render's **25,117 MiB**. **Those are different
metrics** — 31,450 was the job's `nvidia-smi` whole-process `max_memory_usage`
(CUDA context + cuDNN workspace + allocator); 25,117 is
`torch.cuda.max_memory_allocated`. The comparison **overstated the ACA's memory
cost**. Measured on the same metric, same card, same job: **+0.4%**, consistent
with the ACA's +0.5% step-time cost.

---

## 5. AUG-28 IS NO LONGER A HARD DEADLINE

Jobs resume after the maintenance window, so an unfinished arm is not wasted.
The three ACA arms carry **no deadline guard**, deliberately.

**One asymmetry, known and deliberately NOT fixed:**
`chain_dinolight_render.sh` still carries a `DEADLINE_REACHED` guard from when
the deadline was believed hard. It is unreachable (that arm is finished) and it
was left alone because editing a chain script lands on the next resume. **Now
that dinolight has `TRAINING_DONE` it is safe to remove** — one edit, zero risk.
No other arm has a guard.

---

## 6. TRAPS

- **Two stale `RUNNING_JOB` locks — LEAVE THEM**: crossattn-render (`1784331`)
  and priorquery-render (`1785021`), both `CANCELLED by 214675`, neither with
  `TRAINING_DONE`. Deleting a lock lets a dead arm auto-resume.
- **priorquery-render is DROPPED** (cancelled 2026-08-23 at 90k).
- **Do not edit `chain_core.sh` or a shared arch while an arm trains** — a change
  is read by the NEXT RESUME, silently, hours later.
- Observation tags (`dino/affm_w_*`, `dino/aca_*`) are a **step function** —
  measured every N forwards, reprinted every `print_freq`. De-duplicate before
  plotting.
- **The ACA arms' `injection_ratio` is NOT comparable to the addition arms'.**
  It logs `||alpha*F_ca||` before the output conv: ~0.05–0.3 where the addition
  arms sit at ~1.03. Compare ACA to ACA.
- **The AFFM layer weights are non-stationary.** Over 245k they wandered without
  settling; three different readings were given at three different iterations.
  Use final converged values, averaged over a window, or say nothing.

---

## 7. GIT

Branch `dino_e2`, tracking `origin/dino_e2`. **Everything through the ACA ladder
is committed and pushed.** No `Co-Authored-By` trailer on this repo.

`.gitignore` excludes `experiments/`, `tb_logger/`, `**/results/`, `*.pth`,
`*.pt`, `*.png`, `*.log`, `*.state` — checkpoints, curves, figures and the mean
tensors exist **on disk only**. The mean `*_meta.json` files ARE tracked, so a
lost `.pt` can be recomputed with a known recipe.

`ACA_MANIFEST.md` (+ `.jsonl`) under `results/wo2_implementation/` is
**append-only** and carries, per arm: config path, experiment name, job ids in
order, state, iterations reached, and **the exact hand-resume command**. Refresh
it with `aca_manifest.py --refresh`.
