# Devlog — E0-Fixed (`Holo_E0_fixed128_baseline`)

**APPEND-ONLY.** Never rewrite an entry. New events go at the bottom with a real
timestamp.

---

## 2026-08-10T21:52:39+02:00 — implementation (Work Order 2, no training yet)

**Why.** E1-N-Fixed needs a control that differs from it in exactly one thing:
the DINO branch. The existing `Holo_Baseline_Restormer_verynoisy` cannot be that
control — it was trained on the progressive `128→160→192→256` schedule with
mini-batches 8/5/4/2, and E1 trains on fixed 128 crops. A fair comparison needs
a baseline trained under E1's own conditions.

**What.** `configs/E0_fixed128_baseline.yml`, BasicSR name
`Holo_E0_fixed128_baseline`. Stock `Restormer` (`type: Restormer`, unchanged
file), `ImageCleanModel`, from scratch, 300k iterations.

**How.**

| item | value |
|---|---|
| config | `dino_analysis_phases/phase3_restoration/configs/E0_fixed128_baseline.yml` |
| git commit | `4e37beb7a4f7256b785b1e4d057e1be83d2f1f13` (branch `dino_e2`) |
| hostname (implementation) | `tinyx` (login); all compute via SLURM |
| GPU (planned) | 1× V100-PCIE-32GB, partition `v100` |
| seed | `manual_seed: 100` |
| crop | fixed 128×128, `gt_sizes: [128]`, `iters: [300000]`, `mini_batch_sizes: [8]`, `gt_size: 256` (dataset-level crop is a no-op on native 256² images) |
| batch | constant 8 for the whole run |
| optimizer | AdamW, lr 3e-4, weight_decay 1e-4, betas [0.9, 0.999], grad-clip 0.01 |
| scheduler | `CosineAnnealingRestartCyclicLR`, periods [92000, 208000], restart_weights [1,1], eta_mins [3e-4, 1e-6] |
| loss | `L1Loss`, weight 1, reduction mean |
| iterations | 300,000 |
| validation | every 4000 iters, full 256×256, 339 images, 8-bit PSNR/SSIM path |
| fresh or resume | **fresh** — `experiments/Holo_E0_fixed128_baseline/` does not exist |

**Scheduler note, verbatim:** *"The 92k restart is inherited for comparability
with the existing radar training recipe; it is not motivated by a crop
transition under fixed-128 training. Both arms share it, so it cannot confound
the comparison."* Work Order 1 established the independence mechanically
(`base_model.py:183-193` steps the scheduler on iteration count alone; the
progressive block builds a separate `groups` array from `datasets.train.iters`
and touches only the lq/gt tensors).

**Smoke test.** Assertions ran against real data through the real dataset and a
verbatim copy of `train.py`'s crop block: train split (6101 images), crop
**verified from the tensor** at `[8,1,128,128]` for both lq and gt, single
progressive stage, paired basenames identical, inputs in [0,1], forward,
backward, finite loss, finite gradients on 494 tensors, optimizer step changes
weights. Results in `results/wo2_implementation/smoke_results_*.json`.

**⚠ E0-Fixed is a NEW baseline.** It never trains at 256 and uses a constant
batch of 8, so the recorded 22.446 dB val / 22.405 dB test / 18.313 dB masked /
HF-energy 0.216 figures **do not describe this model**. Expect a lower absolute
PSNR than the old progressive baseline — that is expected and must be stated,
not hidden. The full over-smoothing characterisation (HF-energy ratio,
Laplacian variance, Sobel gradient, radial power spectrum) must be
**re-measured on E0-Fixed** via `scripts/run_evaluate.sh` once this run exists,
so that the project's motivation and its results describe the same model.

**Checkpoint selection (pre-registered).** Highest **validation** PSNR. The test
split never drives selection. Recorded by
`scripts/select_best_checkpoint.py --name Holo_E0_fixed128_baseline` into
`results/Holo_E0_fixed128_baseline/metadata/best_checkpoint.json`.

**Not done in this entry:** no training was launched. No checkpoint exists.

---

## 2026-08-13 — 300k COMPLETE, and the baseline re-characterisation

**Training finished.** `net_g_300000.pth` written (104.7 MB), chain wrote
`TRAINING_DONE` and cancelled its own successor. 151 checkpoints on disk, 44 GB.

| | iter | val PSNR | val SSIM |
|---|---|---|---|
| best | 268,000 | **22.0749** | 0.8006 |
| final | 300,000 | 22.0343 | 0.8001 |

Top-5 checkpoints span **0.041 dB** (268k / 292k / 272k / 296k / 300k). Nothing
should be read into "best" beyond the selection rule having been applied
consistently — the same caveat the Exp-2 post-mortem raised.

**Selected checkpoint: `net_g_268000.pth`**, on highest validation PSNR. The test
split was not read.

### Baseline re-characterisation (jobs 1775533 full256, 1775534 crop128, v100)

This is Work Order 2 Step 7, and the reason it is not optional: the project's
motivating over-smoothing figure (HF-energy ratio **0.216**) and the
22.446/22.405/18.313 dB numbers were all measured on the OLD progressive
baseline, which trained on a 128→256 schedule. E0-Fixed never trains at 256.
Quoting the old figure beside a Phase-3 result would describe two different
models as one.

Split: **val**, n=339. The test split stays locked until every arm has finished,
so all arms can be evaluated together under one protocol.

| metric | full256 | crop128 |
|---|---|---|
| PSNR whole image | 22.077 | 19.816 |
| PSNR object mask | 17.978 | 17.806 |
| SSIM whole image | 0.783 | 0.676 |
| SSIM object mask | 0.569 | 0.570 |
| **HF energy ratio** | **0.200** | 0.910 |
| Laplacian variance ratio | 0.284 | 0.285 |
| Sobel gradient ratio | 0.756 | 0.710 |

PSNR here is the 16-bit evaluation path and is NOT comparable to the 8-bit
training-time val PSNR, despite 22.077 landing near 22.075 by coincidence.

**THE PREMISE HOLDS.** E0-Fixed keeps **0.200** of the target's HF energy vs
**0.216** for the old baseline — marginally *worse*, not better. Over-smoothing
is the headline weakness of this baseline too, so motivation and results now
describe the same model. **From here, quote 0.200, not 0.216.**

**The Sobel/Laplacian gap is the finding, not noise.** Gradient ratio 0.756 vs
Laplacian ratio 0.284: first-order edges (silhouette, bright structural bars)
survive reconstruction; second-order detail and fine texture do not. The radial
power spectrum shows the same thing directly — the prediction tracks the target
only below ~0.1 of the maximum radius, then falls up to an order of magnitude
short all the way to Nyquist. The model is not adding spurious detail; it fails
to reproduce detail that is present in the target.

**crop128 HF ratio 0.910 is not a contradiction** and must not be quoted as one:
its std is 0.475, more than half the mean. On a 128 crop the HF band is narrow
and often nearly empty, so the ratio is unstable. full256 is the figure of record.

### Outputs

```
results/Holo_E0_fixed128_baseline/
├── metadata/best_checkpoint.json
├── metrics/{full256,crop128}_val_{summary.json,per_image.csv}
├── predictions/{full256,crop128}_val/raw/        339 PNGs each
└── visuals/ radial_power_{full256,crop128}_val.png
            cases_{full256,crop128}_val.png       (new, see below)
```

`scripts/make_single_arm_figures.py` was added: the four-panel
input/prediction/target/error figure for a SINGLE arm, cases chosen by score
from the per-image CSV (best, median, worst) plus fixed representatives in
filename order, so the sample cannot be curated after the fact.
`make_comparison_figures.py` remains the six-panel E0-vs-E1 tool of record once
a second arm finishes; this exists because E0 completed first.

Visible in the case figures: predictions recover the object's bright structure
and suppress speckle, but the fine ringing/fringe texture is smoothed away,
leaving fringe-shaped residue in every error map. The worst case (11.31 dB) is a
different failure — a ring-shaped object whose input is nearly pure speckle,
where the model returns a blurred blob carrying none of the target's structure.


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

### Correction to the re-characterisation framing (test numbers change it)

The val entry above said E0-Fixed keeps **0.200** HF energy against the old
progressive baseline's **0.216**, "marginally worse". The test evaluation makes
the apples-to-apples comparison available, and it is different:

| | HF energy ratio |
|---|---|
| old progressive baseline (test) | 0.216 |
| **E0-Fixed (test)** | **0.218** |
| E0-Fixed (val) | 0.200 |

The old 0.216 was measured on **test**. Compared test-to-test, E0-Fixed is
**0.218 vs 0.216 — indistinguishable**, not "marginally worse". The 0.200/0.216
gap was a val-vs-test artefact, not a difference between the models.

**The conclusion is unchanged and if anything cleaner:** the two baselines
over-smooth to the same degree, so the motivating weakness transfers exactly.
Quote **0.218 (test)** or **0.200 (val)**, matched to whichever split the
number beside it comes from — never mix them.

### The registered prediction about absolute PSNR held

The README registered in advance: "expect E0-Fixed to score lower in absolute
PSNR than the old baseline, because it never trains at the evaluation
resolution. That is expected and must be reported, not explained away."

  old progressive baseline, test, best-val ckpt   22.405 dB
  E0-Fixed, test, best-val ckpt                   21.873 dB   (-0.532)

Confirmed. E0-Fixed is 0.53 dB below the old baseline in absolute terms, for the
registered reason. This does not weaken the Phase-3 comparison, which is
internal: all arms share E0-Fixed's recipe exactly.

Worth recording alongside it: **E1-addition-render (24.081 dB test) beats the
OLD progressive baseline by +1.676 dB**, so the render prior does not merely win
its own internal comparison — it is the best model this project has produced.
