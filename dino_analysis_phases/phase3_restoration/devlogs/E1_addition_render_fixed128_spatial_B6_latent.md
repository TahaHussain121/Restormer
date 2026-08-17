# Devlog — E1-render (`Holo_E1_addition_render_fixed128_spatial_B6_latent`)

**APPEND-ONLY.** Never rewrite an entry. New events go at the bottom with a real
timestamp. The stability gate appends here automatically on an abort.

---

## 2026-08-12T13:05:00+02:00 — implementation (no training launched)

### Why this arm exists

**Source ablation.** E1-addition conditions DINO on the same noisy 1e5 crop the
network is already restoring, so the prior can only reorganise information
Restormer already has. E1-render conditions DINO on the *render* — clean
geometry, no radar noise — and asks the question that separates the two:

> Does a clean render give a better prior than the noisy observation the network
> already has?

This is the **render arm of the E1 source comparison**. E1-addition and
E1-render differ in exactly one scientific respect, the tensor DINO receives,
and share E0 as their common control.

**Not merely an oracle.** In this simulator setting the render is normally
available alongside the heatmap — the scene is what generated both. So a win
here is a usable method, not just an upper bound on what a perfect prior could
buy. That distinction matters for how the result is reported.

Phase 2's measurement is the prior expectation to beat: at B6 centered, the
render's *absolute* alignment to DINO(1e7) beat 1e5's by +0.0971 (dz +2.05,
render better on 98.2% of samples), but its *scene-specific* advantage lead
collapsed to +0.0089 (dz +0.163) — roughly 90% of the render's apparent
superiority also appears against wrong-scene targets. Phase 2 made no claim
about restoration; this arm is where that gets tested in PSNR.

### What changes, and what does not

| | E1-addition | E1-render |
|---|---|---|
| DINO input | 1e5 noisy crop | **render crop** |
| Restormer input | 1e5 crop | 1e5 crop (unchanged) |
| target | 1e7 | 1e7 (unchanged) |

The render is DINO input and nothing else: never a Restormer input, never a
target, never a centering statistic for another arm.

Identical to E1-addition: B6 (index 5), zero-init 1×1 conv 768→384, residual
addition at `inp_enc_level4` before the latent blocks, frozen DINO, fixed 128
crop, batch 8, 300k iterations, `CosineAnnealingRestartCyclicLR` periods
[92000, 208000], L1, seed 100, LR 3e-4, and the **amended gate inherited from
E1-addition** (`injection_ratio_max: 10`, ratio rules from iteration 5000,
NaN/Inf hard-stop from iteration 1).

Config diff, measured key by key over 98 flattened keys: **11 differ, 87
identical.** Scientific: `dino_source`, `dino_mean_train128`,
`dino_mean_eval256`. Plumbing, unavoidable for a separate identity: `name`,
`network_g.type`, both `datasets.*.type`, both `dataroot_render`,
`tb_logger_dir`, `dino_stability.devlog`.

### Crop alignment — the critical part

The render must receive the **same crop coordinates and the same geometric
augmentation** as the 1e5/1e7 pair. Rather than guard against desynchronisation,
this arm makes it impossible: `Dataset_PairedImage_uint16_RenderStacked` packs
the render into the LQ tensor as **channel 1**.

```
lq[:, 0] = 1e5 radar   -> Restormer
lq[:, 1] = render      -> DINO
```

`basicsr/train.py` sub-crops and subsamples `train_data['lq']` as one tensor, so
both channels receive byte-identically the same operation — not two coordinate
calculations that have to agree. The architecture splits the channels inside
`forward`. **No change to `train.py` was required**, which is also why E0 and
E1-addition, both in flight, are untouched.

**Proof, not assertion** (smoke test): the dataset returns the `(top, left,
flag)` it drew; the test re-loads the original render PNG from disk, applies
that crop and that dihedral flag, and compares with `torch.equal` against the
channel the dataset produced. Same for the radar channel. A shape check would
not catch a misaligned window.

### Centering means — new, and necessary

Render DINO statistics differ from 1e5 statistics, so the 1e5 means are **not**
reused. Both computed train split only, same method, sample count (1000 images)
and seed (0) as the 1e5 production means:

| file | images | tokens | ‖mu‖ | vs the 1e5 mean |
|---|---|---|---|---|
| `render_B6_train128_dino224_mean.pt` | 1000 | 256,000 | **56.8906** | +30.8% vs 43.5031 |
| `render_B6_eval256_dino448_mean.pt` | 1000 | 1,024,000 | **61.6951** | +49.5% vs 41.2580 |

Both `[768]`, all finite, train-only provenance asserted from the dataroot at
runtime.

**The separation is emphatic.** `cosine(mu_1e5, mu_render)` = **0.1799** at
train128 and **0.3288** at eval256 — the two domain means are nearly orthogonal,
and `‖mu_render − mu_1e5‖` is ~150% of `‖mu_1e5‖` in both regimes. Centering
render features with the 1e5 mean would have left a large, systematic offset in
every token. The separate means were required, not precautionary.

Render residual fractions after centering (0.75–0.86) are lower than 1e5's
(0.89–0.94): the render mean explains more of its domain's token energy, which
is consistent with the render being the more uniform, less noisy domain.

### Files

| | |
|---|---|
| config | `configs/E1_addition_render_fixed128_spatial_B6_latent.yml` |
| architecture | `basicsr/models/archs/restormer_dino_render_arch.py` → `RestormerDinoSpatialRender` |
| dataset | `basicsr/data/paired_radar_render_stacked_dataset.py` → `Dataset_PairedImage_uint16_RenderStacked` |
| means | `means/render_B6_{train128_dino224,eval256_dino448}_mean.pt` |
| model wrapper | `ImageCleanModelDinoSpatial`, reused unchanged |

One line of the shared parent arch changed: the `dino_source` validation now
accepts `'render'` as well as `'same_lq'`. Construction-time validation only —
it cannot alter the `same_lq` code path. Verified after the edit by re-running
the E1-addition assertion harness: **61/61 passed** (job 1774671, 12:51), before
E1-addition started.

### Hardware

`a100`. E1-addition runs on `v100` and E0 on `a100`. Per-iteration timings are
therefore not comparable across all three arms; PSNR/SSIM comparisons are
unaffected (same seed, same data, same math, same evaluation scripts).

### Not done in this entry

No training launched. No experiment directory, no checkpoint, no training state
exists for this identity.

---

## 2026-08-12T14:15:00+02:00 — RENAME: `Holo_E1_render_...` → `Holo_E1_addition_render_...`

The fusion is addition here too; only the SOURCE differs from the noisy arm.
The name now says both, because a concat variant of each is planned:

| | fusion | DINO source | identity |
|---|---|---|---|
| sibling | addition | 1e5 noisy crop | `Holo_E1_addition_noisy_fixed128_spatial_B6_latent` |
| this arm | addition | render | `Holo_E1_addition_render_fixed128_spatial_B6_latent` |

Config, devlog, results directory, chain script and TensorBoard path renamed to
match. Never started under the old name, so no run history is rewritten.

Its chain driver is now `scripts/chain_E1_addition_render.sh`, job name
`p3chain_E1addR`, logs in `results/<name>/logs/chain_<jobid>.out`.

---

## 2026-08-14 — 300k COMPLETE. The headline positive result of Phase 3.

`net_g_300000.pth` written, `TRAINING_DONE` set, chain closed after 2 jobs.

| | iter | val PSNR |
|---|---|---|
| best | **204,000** | **24.1171** |
| final | 300,000 | 24.0077 |

Top-5 spread 0.092 dB. Selected checkpoint `net_g_204000.pth`.

### Evaluation (job 1776655, val/full256, n=339, 16-bit path)

| metric | E0 | E1-render | delta |
|---|---|---|---|
| PSNR whole | 22.077 | **24.120** | **+2.043** |
| PSNR mask | 17.978 | **19.893** | +1.914 |
| SSIM whole | 0.783 | 0.818 | +0.035 |
| SSIM mask | 0.569 | **0.647** | +0.079 |
| HF energy ratio | 0.200 | **0.336** | +0.136 |
| Laplacian ratio | 0.284 | **0.449** | +0.165 |
| Sobel ratio | 0.756 | **0.873** | +0.117 |

**+2.043 dB, nearly 7x the pre-registered "+0.30 dB = meaningful" threshold.**

The gain is BROAD, not outlier-driven: improves **290/339 images (85.5%)**,
median +1.946 dB, best +9.19, worst regression only -3.72 (milder than the noisy
arm's -10.14).

**It also directly attacks the weakness the project was built around.** HF
energy ratio rises 0.200 -> 0.336, a 68% relative improvement in retained
high-frequency energy, with Laplacian ratio up 58%. The over-smoothing that
motivated the whole DINO line is measurably reduced, not merely traded for PSNR.

### Where it helps most: exactly where the baseline fails

On E0's hardest decile (34 images) this arm averages **+3.07 dB and wins 31/34**
— well above its own +2.043 average. On E0's six worst images individually:
+5.28, +7.23, +5.25, +6.98, +5.03 dB, and one loss of -0.04.

The three-arm figures (`results/comparisons/three_arm_harsh_full256_val.png`)
show what those numbers are: where E0 collapses to a featureless blob, this arm
reconstructs the object's actual geometry — the rounded chair back, the stool's
four legs, the slatted seat. Honest exception, kept in the figure: on val image
2886 all three arms fail and this one is marginally the worst.

### The claim this licenses, and the one it does not

**Does license it**, because E1-addition-noisy is the SAME code with one tensor
swapped and lands at **-0.608 dB**: identical parameter count (+295,296),
identical seed, schedule, crop, fusion and gate. The difference between +2.04
and -0.61 is attributable to the DINO input alone — not to added capacity, not
to "DINO features" generically.

**Does NOT license** the summary "DINO features help". The render is a clean
view of the same object, so it carries the target's geometry almost directly.
The defensible statement is narrower: *a clean geometric view of the object,
delivered through frozen DINO features, substantially improves restoration —
while the identical mechanism fed the noisy radar makes it worse.* Write it that
way. The render IS normally available in this pipeline, so this remains a usable
method rather than an oracle, but the source of the advantage must not be
overstated.

**Stability:** no gate rule fired across the full 300k. injection_ratio stayed
near ~1.0 throughout.


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

