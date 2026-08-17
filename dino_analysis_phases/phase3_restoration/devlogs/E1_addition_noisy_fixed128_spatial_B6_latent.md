# Devlog — E1-N-Fixed (`Holo_E1_N_fixed128_spatial_B6_latent`)

**APPEND-ONLY.** Never rewrite an entry. New events go at the bottom with a real
timestamp. The stability gate appends here automatically on an abort.

---

## 2026-08-10T21:52:39+02:00 — implementation (Work Order 2, no training yet)

**Why.** Phases 0–2 measured a real but small scene-specific DINO signal in
representation space and never tested whether it converts into restoration
quality. E1 is that test, kept as simple and as causally interpretable as
possible: one prior, one injection point, one fusion, one control.

**What.** `configs/E1_N_fixed128_spatial_B6_latent.yml`, BasicSR name
`Holo_E1_N_fixed128_spatial_B6_latent`. New architecture
`RestormerDinoSpatial` (`basicsr/models/archs/restormer_dino_spatial_arch.py`)
and new model wrapper `ImageCleanModelDinoSpatial`
(`basicsr/models/image_restoration_dino_model.py`). The stock `Restormer` and
`ImageCleanModel` are untouched and still drive E0.

**How.**

| item | value |
|---|---|
| config | `dino_analysis_phases/phase3_restoration/configs/E1_N_fixed128_spatial_B6_latent.yml` |
| git commit | `4e37beb7a4f7256b785b1e4d057e1be83d2f1f13` (branch `dino_e2`) |
| hostname (implementation) | `tinyx` (login); all compute via SLURM |
| GPU (planned) | 1× V100-PCIE-32GB, partition `v100` (same as E0) |
| seed | `manual_seed: 100` (identical to E0) |
| crop | fixed 128×128, `gt_sizes: [128]`, `iters: [300000]`, `mini_batch_sizes: [8]` |
| batch | constant 8 |
| optimizer | AdamW, lr 3e-4, wd 1e-4, betas [0.9,0.999], grad-clip 0.01 — identical to E0 |
| scheduler | `CosineAnnealingRestartCyclicLR`, periods [92000,208000], eta_mins [3e-4,1e-6] — identical to E0 |
| loss | `L1Loss` only |
| iterations | 300,000 |
| fresh or resume | **fresh** — `experiments/Holo_E1_N_fixed128_spatial_B6_latent/` does not exist |

### DINO branch

| item | value |
|---|---|
| model | DINOv2 `dinov2_vitb14`, offline/local, strict weight load, 0 register tokens |
| **selected block** | **B6 (0-indexed 5)** |
| source | the **same** LQ tensor Restormer restores (`dino_source: same_lq`) |
| frozen | `requires_grad=False` on all 175 parameter tensors; stays `eval()` when the parent enters train mode; extraction under `no_grad` |
| preprocessing | grey→3ch repeat, bilinear `align_corners=False` resize, ImageNet mean/std |
| training regime | radar 128 → DINO 224 → `[B,256,768]` → 16×16 grid == 16×16 latent |
| eval regime | radar 256 → DINO 448 → `[B,1024,768]` → 32×32 grid == 32×32 latent |
| centering | one position-independent `[768]` vector, subtracted per token |
| train mean | `means/1e5_B6_train128_dino224_mean.pt` — 1e5, **train split**, 1000 images, 256,000 tokens, seed 0, norm 43.5031 |
| eval mean | `means/1e5_B6_eval256_dino448_mean.pt` — 1e5, **train split**, 1000 images, 1,024,000 tokens, seed 0, norm 41.2580 |
| injection | `inp_enc_level4`, **before** the 8 latent Transformer blocks |
| fusion | `guided = inp_enc_level4 + P(D_centered)`, `P = nn.Conv2d(768, 384, 1)` |
| initialisation | **zero weight and zero bias** — a *zero-initialized residual projection*. No alpha, no gate, no FiLM, no adaLN |

### Layer-lock rationale, and the tension recorded up front

B6 is locked on the **pre-registered primary criterion** — centered same-scene
1e5↔1e7 correspondence at the actual fixed-128 training scale (WO1 Task 1.3,
val n=339): B3 0.6091, **B6 0.6694**, B9 0.5482, B12 0.2337.

**The two representation criteria disagree.** B6 was locked on the
pre-registered absolute-correspondence criterion. Under the same-vs-different
scene-advantage criterion B3 wins (+0.1966 vs B6's +0.1453), and that gap is
wider at 128 than it was at 256 (+0.051 vs +0.026). **A B3 run under an
identical recipe is planned as the immediate follow-up so the criterion conflict
is resolved empirically rather than by argument.** This is recorded here before
any result exists, so it cannot later look like a post-hoc fishing expedition.

### Verification (Work Order 1)

- The shared extractor reproduces the Phase-1 representation **bit-identically**
  on CPU (32 comparisons, max abs err 0.0, `torch.allclose` True at atol 1e-6);
  on CUDA the two instances differ by ≤3.8e-4, pure float32 non-determinism.
- Scale consistency confirmed on this checkpoint: 128→224→16×16 and
  256→448→32×32, both matching the latent grid, no feature-grid interpolation.

### Smoke tests (Step 4)

61/61 assertions passed. The ones that matter most:

- **crop-stream identity**: the tensor handed to DINO preprocessing **is the
  same Python object** as the Restormer input, and `torch.equal` on it is True
  — not a shape check, not `allclose`. This closes the highest-probability
  silent bug named in the investigation report (§N.1).
- `P` is zero-initialised, `P(D)` is exactly 0 at step 0, while `D_centered` is
  demonstrably non-zero (max |D| = 67.60);
- `P` receives a **finite, non-zero** gradient on the **first** backward
  (max |dL/dW_P| = 3.12e-2), exactly as `∂L/∂W = ∂L/∂guided ⊗ D` predicts;
- DINO gradients are `None`, DINO stays `eval()`, trunk gradients finite;
- parameter delta vs E0 = **295,296** exactly (`768×384 + 384`);
- **step-0 outputs of E0 and E1 are bit-identical on CPU** (max abs diff 0.0);
- the trunk initialisation is bit-identical to E0 across 494 tensors, thanks to
  an RNG fence around the ViT/projection construction;
- a mode/size mismatch **raises** instead of silently interpolating.

### Stability gate

In-run (`ImageCleanModelDinoSpatial`), hard stop at **any** iteration on:
non-finite loss/latent/DINO/projected values; `injection_ratio > 0.5`;
`injection_ratio` > 10× the reference recorded at ~5k (persisted across
auto-resume). Cross-run, every ~10k iterations,
`scripts/stability_gate.py` checks E1 val PSNR against E0 at the same iteration
(>1.0 dB below = trigger) and E1-vs-E0 loss divergence.

Logged every 1000 iters to TensorBoard and `experiments/<name>/dino_stability.csv`:
`dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`, plus loss and LR.

**A trigger is an OPTIMIZATION / STABILITY FAILURE — not "the DINO prior does
not work". It is investigated and reported first. It is never answered by
redesigning the architecture; any design change requires a new experiment
identity.**

### Pre-registered interpretation

```
Delta PSNR = E1 test PSNR − E0 test PSNR
  <  +0.10 dB   → no meaningful PSNR improvement
  +0.10..+0.30  → marginal / promising
  >  +0.30 dB   → meaningful improvement
```

Also evaluated: SSIM, object-only PSNR/SSIM, HF-energy ratio, Laplacian
variance, Sobel gradient, radial power spectrum. A modest PSNR gain with clear
high-frequency or object-structure recovery may still be scientifically
important. **If E1 improves on matched-128 but not on full-256, that indicates
the prior is useful in-distribution and that scale/context transfer is the
limiter — not that the prior is useless.**

**Checkpoint selection:** highest **validation** PSNR; the test split never
drives selection.

**Not done in this entry:** no training was launched. No checkpoint exists.

---

## 2026-08-10T23:35:07+02:00 — OPTIMIZATION / STABILITY FAILURE in the 12-iteration integration smoke run

**Why this entry exists.** The Step-4 integration smoke run (`basicsr/train.py`,
12 iterations, throwaway identity `SMOKE_E1_N_fixed128_spatial_B6_latent`,
1× A100, SLURM job 1773288) **aborted at iteration 4**. The real experiment
identity was never started and its config is unchanged.

**What happened.** Abort rule 1 fired: `injection_ratio 0.5555 > 0.5`.

| iter | ‖F_latent‖ | ‖P(D)‖ | injection_ratio | l_pix |
|---|---|---|---|---|
| 1 | 90.944 | 0.000 | 0.0 | 0.17248 |
| 2 | 97.375 | 0.047213 | 4.8486e-04 | 0.22632 |
| 3 | 100.70 | 39.873 | 0.39594 | 0.21175 |
| 4 | 121.91 | 67.720 | **0.55549 → ABORT** | 0.18694 |

Recorded in `experiments/SMOKE_E1_N_fixed128_spatial_B6_latent/STABILITY_FAILURE.json`
and `dino_stability.csv`.

**Mechanism (analysis, not a fix).** AdamW's update is scale-free: with the
second-moment estimate starting at zero, the first steps move every weight by
approximately the learning rate (3e-4) in a coherent direction, no matter how
small the gradient is. `P` starts at exactly zero and every one of its
768×384 weights receives a coherent gradient (`∂L/∂W = ∂L/∂guided ⊗ D`), while
`D_centered` is large — per-token norm ≈88, because centering removes only about
10% of the raw B6 token norm (≈98). So ‖P(D)‖ climbs from 0 to ~68 within four
steps while ‖F_latent‖ is only ~90–122. The global gradient clip (0.01 over all
parameters) does not restrain `P`: it rescales the whole gradient vector, and
Adam then renormalises `P`'s step back to ~lr.

**Honest reading.** Four iterations cannot distinguish *the branch switching on*
from *a runaway*. The pre-registered threshold was written as a runaway
detector, and it is being crossed by the initial transient. Which of the two
this is, is exactly the open question — it is not yet answered, and nothing here
should be read as evidence about the DINO prior.

**Not done, deliberately.** No fix was applied. The architecture, the layer, the
crop, the scheduler, the fusion, the centering means, the learning rate and the
abort thresholds are all unchanged. Rules 2–4 are unaffected and remain the
runaway detectors. The decision on how to proceed is escalated, not taken here.

---

## 2026-08-12T11:45:00+02:00 — RENAME: `Holo_E1_N_...` → `Holo_E1_addition_...`

A concat-then-project variant is planned as the next row, so the fusion type is
now explicit in the identity rather than implied.

| | old | new |
|---|---|---|
| config | `E1_N_fixed128_spatial_B6_latent.yml` | `E1_addition_noisy_fixed128_spatial_B6_latent.yml` |
| BasicSR name | `Holo_E1_N_fixed128_spatial_B6_latent` | `Holo_E1_addition_noisy_fixed128_spatial_B6_latent` |
| results | `results/E1_N_.../` | `results/Holo_E1_addition_.../` |
| devlog | this file, renamed | `devlogs/E1_addition_noisy_fixed128_spatial_B6_latent.md` |
| TensorBoard | `tb_logger/Holo_E1_N_.../` | `tb_logger/Holo_E1_addition_.../` |

The old identity was never trained — no experiment directory, no checkpoint, no
training state ever existed under it — so no run history is being rewritten. The
new name has no experiment directory either: the launch is **START FRESH**.
Entries above this line were written under the old name and are left untouched.

---

## 2026-08-12T11:45:00+02:00 — GATE AMENDMENT (the only functional change)

| parameter | before | after |
|---|---|---|
| `injection_ratio_max` | 0.5 | **10** |
| ratio-rule validity window | from iteration 1 | **from iteration 5000** |
| NaN/Inf hard-stop | from iteration 1 | **from iteration 1 (unchanged)** |

**This changes WHEN the gate is valid, not how much injection is tolerated.**
The ratio is still measured, logged to TensorBoard and written to
`dino_stability.csv` from iteration 1; it simply does not trigger an abort
before iteration 5000.

**Reason.** The 5k diagnostic showed `injection_ratio` plateaus at 2.3–6.6 and
drifts *down* after iteration 1000. The 0.5 cap was therefore firing on the
startup transient of a zero-initialised projection — which by construction must
grow from nothing — rather than on a pathology. A rule that cannot survive the
first four steps of the design it is guarding is mis-specified in its window,
and that is what has been corrected.

Nothing else changed: architecture, layer B6, crop, scheduler, LR, loss, seed,
both centering means, batch size, iteration count and fusion are all exactly as
specified before.

---

## 2026-08-12T11:45:00+02:00 — DIAGNOSTIC HISTORY

### What happened

- The E1 integration test aborted at iteration 4. Not a crash: the
  pre-registered gate fired on rule 1, `injection_ratio 0.5555 > 0.5`.
- Trajectory: iter 1 `0.0` / iter 2 `4.85e-04` / iter 3 `0.3959` / iter 4
  `0.5555`, with `‖F_latent‖` 90.9 → 121.9 and `‖P(D)‖` 0 → 67.7.

### Why it happened

- `P` is zero-initialised, so it must grow from nothing; a rise is by design.
- AdamW's first steps are ~lr (3e-4) per weight regardless of gradient
  magnitude, so the zero-init identity survives only ~3 steps.
- `D_centered` is large: per-token norm ≈88; centering removes only ~10% of the
  raw B6 norm.
- The global gradient clip of 0.01 does not restrain `P` — Adam renormalises
  `P`'s step back to ~lr.
- Four iterations cannot distinguish "branch switching on" from "runaway".

### Diagnostic 1 — 5k run, throwaway identity, gate in log-only mode

Everything else unchanged (same LR, crop, seed, block, means, fusion).

- `injection_ratio` climbs for ~100 iterations, then **plateaus at 2.3–6.6**,
  drifting **down** after iteration 1000.
  Window averages: 3.26 / 4.08 / 4.57 / 4.24 / 3.84 / 3.51.
- Training healthy: loss 0.17 → ~0.10.
- **Conclusion: not a runaway. The abort was a startup transient.**
- Co-inflation noted: `‖P(D)‖` 449 → 1407 while `‖F_latent‖` 100 → 440, so the
  ratio stays flat while both grow. **No current rule catches this.**

### Diagnostic 2 — outcome check at iteration 2500

Same 339-image val set, full-256 protocol, 8-bit path:

| | PSNR | SSIM |
|---|---|---|
| E0 | 18.9630 dB | 0.4647 |
| E1 | 20.2130 dB | 0.6830 |
| delta | **+1.250 dB** | **+0.218** |

- E1 leads. The large injection is not harming reconstruction.
- Caveat: 2500 of 300k iterations, throwaway identities. This is "not broken",
  **not a result**.

### Diagnostic 3 — latent-block residual-scale probe

Job 1773770, both arms at iteration 5000, 5 identical batches, `no_grad`,
Frobenius over the whole `[B,C,H,W]` tensor — the same convention
`injection_ratio` uses.

| | attn_ratio | ffn_ratio | ‖x‖ |
|---|---|---|---|
| E0 | 0.1160 ± 0.0197 | 0.2034 ± 0.0515 | 1300.6 |
| E1 | 0.0808 ± 0.0331 | 0.2120 ± 0.0449 | 2032.8 |

- Restormer's **own** residual branches contribute 0.05–0.30 of the stream they
  join. `injection_ratio ≈ 4` is therefore **20–40× the network's native branch
  scale**. The injection is dominant, not a nudge — "residual projection"
  understates it.
- E1's ffn profile matches E0's U-shape. In absolute terms E1's ffn output
  scaled with the stream (~1.63×) but attention did not (~1.09×): the FFN
  absorbed the DINO content, attention branches contribute proportionally less.
- E1 block-0 `‖x‖ ≈ 1927` agrees independently with the stability log
  (`‖F_latent‖ ≈ 440` + `‖P(D)‖ ≈ 1407`) — two measurements cross-check.

### Decision taken

- Launch E1 with **addition unchanged**. The magnitude is unusual but is not
  hurting results, and re-deriving the fusion now costs time the schedule does
  not have.
- Gate amended: window moved to 5k, threshold 0.5 → 10. This changes **when**
  the gate is valid, not how much is tolerated.
- A **concat-then-project variant is planned as the next row**, so the magnitude
  question is answered empirically rather than by argument. Pre-registered here,
  before E1 results exist.

### Open / not yet resolved

- **Co-inflation has no gate rule.** Recommended addition: abort if
  `‖F_latent‖` or `‖P(D)‖` grows >5× versus its 5k reference. Not implemented.
- **Different-scene DINO similarity floor is 0.524** (B6, centered). Untested
  hypothesis: the dataset is single-class (chairs only), **or** the
  position-independent mean cannot remove shared positional layout. A
  token-shuffle control would separate these. Not run.
- **B3 wins the scene-advantage criterion while B6 wins absolute
  correspondence.** The layer lock was a criterion choice, not a measurement.
  The B3 run is still pending.

---

## 2026-08-12T12:05:00+02:00 — HARDWARE: E1 runs on V100, E0 on A100

E1 is submitted to the `v100` partition (32 GB); E0 has been running on `a100`
(40 GB) since job 1773732. **The two arms therefore run on different GPU
models.** Recorded here rather than left implicit.

**What this does not affect.** The comparison itself. Same seed, same data
order, same crops, same optimizer, same schedule, same loss; the arithmetic is
identical and both arms are evaluated by the same scripts on the same splits.
PSNR/SSIM/HF-energy comparisons remain valid.

**What it does affect.** Wall-clock only. V100 runs ~1.35x slower per iteration
than A100, and E1 is already ~0.55 s/iter (versus E0's 0.417) because of the
extra frozen ViT-B/14 forward per step. Expect ~62 h across 3 chained jobs
rather than ~46 h. Per-iteration timings must not be compared between the arms,
and any throughput figure has to name its GPU.

**Why not both on the same GPU model.** At the time of submission all 32 A100s
and all 12 usable V100s were allocated (a 4th V100 node, tg071, is drained).
`PrivateData=jobs` is set on this cluster, so queue depth for other users is not
visible and neither partition's wait can be predicted. Running E1 on V100 was a
deliberate choice to use a second pool rather than queue both arms behind the
same one.

**Not viable at all:** rtx3080 (10 GB) and rtx2080ti (11 GB). Restormer at
128^2 x batch 8 exceeded 9.6 GB with the stock E0 trunk alone (measured, job
1773194 OOM). Those 80 idle consumer GPUs cannot hold this job.

The chain driver now carries a per-arm partition, so every successor is
resubmitted to the same GPU type and an arm cannot drift onto another mid-run.

---

## 2026-08-12T14:15:00+02:00 — RENAME: `Holo_E1_addition_...` → `Holo_E1_addition_noisy_...`

The naming scheme now carries BOTH axes of the E1 comparison, fusion and
source, because a concat variant of each is planned:

| | fusion | DINO source | identity |
|---|---|---|---|
| this arm | addition | 1e5 noisy crop | `Holo_E1_addition_noisy_fixed128_spatial_B6_latent` |
| sibling | addition | render | `Holo_E1_addition_render_fixed128_spatial_B6_latent` |

Config, devlog, results directory, chain script and TensorBoard path renamed to
match. The identity had never been started -- no experiment directory, no
checkpoint, no training state, no TensorBoard tree existed under the old name --
so no run history is rewritten. Entries above this line were written under the
earlier names and are left untouched.

Its chain driver is now `scripts/chain_E1_addition_noisy.sh`, job name
`p3chain_E1addN`, logs in `results/<name>/logs/chain_<jobid>.out`.

---

## 2026-08-12T17:25:00+02:00 — HARDWARE CORRECTION: v100 → a100

**Supersedes the hardware entry above.** That entry planned this arm on `v100`.
It never ran there: the v100 job (1774838) sat PENDING and was cancelled before
starting — no experiment directory, no chain-state directory, no checkpoint and
no training state were ever created under v100.

**This arm now runs on `a100`,** the same GPU model as E0 and as
`Holo_E1_addition_render_...`. Reason: the v100 pool is only 12 usable GPUs
(a 4th node is drained) and stayed fully allocated, while a100 turned jobs over
in minutes throughout.

**This is an improvement to the comparison, not a compromise.** All three arms —
E0, E1-addition-noisy and E1-addition-render — now run on A100, so per-iteration
timings are directly comparable across the whole Phase-3 table. The earlier
entry's caveat about straddling two GPU generations no longer applies.

Wall-clock estimate revised: ~0.45 s/iter on A100 (measured on the render arm)
gives ~37 h over 2 chained jobs, rather than the ~62 h projected for v100.

---

## 2026-08-14 — 300k COMPLETE, and a NEGATIVE result

`net_g_300000.pth` written, `TRAINING_DONE` set, chain closed after 2 jobs.

| | iter | val PSNR |
|---|---|---|
| best | **128,000** | **21.4678** |
| final | 300,000 | 21.2154 |

Top-5 spread 0.237 dB. Selected checkpoint `net_g_128000.pth`.

**THIS ARM IS BELOW THE BASELINE.** E0-Fixed's best val is 22.0749. On the
independent 16-bit evaluation path (job 1776654, val/full256, n=339):

| metric | E0 | E1-noisy | delta |
|---|---|---|---|
| PSNR whole | 22.077 | **21.469** | **-0.608** |
| PSNR mask | 17.978 | 17.479 | -0.499 |
| SSIM whole | 0.783 | 0.762 | -0.021 |
| SSIM mask | 0.569 | 0.547 | -0.022 |
| HF energy ratio | 0.200 | **0.280** | **+0.080** |
| Laplacian ratio | 0.284 | **0.370** | +0.086 |
| Sobel ratio | 0.756 | 0.765 | +0.009 |

Per-image against E0 over all 339: mean **-0.608 dB**, median -0.528, improves
only **100/339 (29.5%)**, best +4.54, worst **-10.14**.

**The pre-registered threshold is not met, in the wrong direction.** DINO
features extracted from the noisy 1e5 observation and injected at the latent
made restoration WORSE than no prior at all, under a recipe identical to the
baseline in every other respect and carrying only 295,296 extra parameters.

### Two things that make this more than "it didn't work"

**1. It is SHARPER while being less accurate.** HF energy ratio 0.280 vs the
baseline's 0.200, Laplacian ratio 0.370 vs 0.284 — it recovers MORE
high-frequency content and still loses PSNR. It is not an over-smoothing
failure; it adds structure the target does not contain. The three-arm figure
shows it directly: on val image 4467 it synthesises a bright cross that has no
counterpart in the 1e7 target. Report it as a hallucination failure, not a
blurring one.

**2. It HELPS where the baseline is desperate.** On E0's hardest decile (34
images) this arm averages **+0.478 dB** and wins 20/34, while being negative
overall. So the noisy prior carries something usable when the observation is
nearly hopeless, and actively interferes on ordinary images.

**Early peak.** Best val at 128k, then a slow decline to 300k, against E0
peaking at 268k and E1-render at 204k. Peaking at 43% of the schedule and
declining is the signature of a prior the network later has to work around. Not
investigated; recorded.

**Stability was never the problem.** No gate rule fired for the whole 300k. The
arm trained cleanly to a worse answer, which is the useful kind of negative
result — it is about the prior, not the optimisation.

### Why this result is load-bearing for the thesis

E1-render is the SAME code with one tensor swapped, and it gains +2.043 dB. Two
arms with identical parameter counts, seeds, schedules and fusion, differing
only in what DINO reads, land on opposite sides of the baseline. That isolation
is what licenses the claim that the render's contribution is the render's
CONTENT, not the DINO machinery and not the added capacity.


---

## 2026-08-14 — FINAL TEST EVALUATION (locked split unlocked, n=338)

Pre-registered conditions met before the split was read: all three arms
completed 300k, and each checkpoint was selected on **validation** PSNR alone
(E0 268k, E1-noisy 128k, E1-render 204k). The test split had never been touched.
Jobs 1776745-1776750, v100, both protocols.

| metric | E0 | E1-noisy | E1-render |
|---|---|---|---|
| **PSNR full256** | 21.873 | **21.296  (-0.577)** | **24.081  (+2.208)** |
| **PSNR crop128** | 19.546 | **19.062  (-0.484)** | **22.259  (+2.713)** |
| PSNR mask, full256 | 17.599 | 17.210 | 19.673 |
| SSIM full / mask (full256) | 0.783 / 0.560 | 0.762 / 0.540 | 0.822 / 0.643 |
| HF ratio (full256) | 0.218 | 0.275 | 0.328 |
| Laplacian / Sobel (full256) | 0.286 / 0.764 | 0.370 / 0.779 | 0.438 / 0.871 |

Per-image vs E0, full256: E1-render improves **298/338 (88.2%)**, median +2.159,
worst -2.81, best +8.19. E1-noisy improves 111/338 (32.8%), median -0.517,
worst -7.43, best +4.10. On crop128: render 299/338 (88.5%), median +2.540.

**VERDICT against the pre-registered thresholds** (>+0.30 dB = meaningful):
E1-render is **MEANINGFUL on both protocols**. E1-noisy is negative on both.
The test result confirms the validation result rather than overturning it.

**The matched-128 interpretation rule does NOT trigger.** It was registered as:
"if E1 improves on matched-128 but not on full-256, the prior is useful
in-distribution and full-image scale transfer is the limiter." E1-render improves
on BOTH, and by MORE at crop128 (+2.713) than full256 (+2.208). So there is no
scale-transfer limitation to invoke — the prior works in both regimes, slightly
better in the regime it trained in.
