# E1 — DINOv2 FiLM Guidance: Full Report

**Status as of 2026-08-09: not working. Three attempts, three failures. No usable model yet.**

This report covers the whole experiment from the start: what we wanted, what we
built, what we measured before training, what we ran, what broke each time, why,
and where things stand. Numbers are the measured ones, not rounded in our favour.

> Naming note: the repo labels this experiment **E1** (also `exp3_dino_film` on
> disk). That label is used throughout here.

---

## 1. What we were trying to do

### The starting point

The baseline (Exp 2) is a plain Restormer trained on the very noisy radar
heatmaps. It works:

| Exp 2 baseline (338 test images) | value |
|---|---|
| PSNR, full image | 22.405 dB |
| PSNR, masked (object only) | 18.313 dB |
| improvement over noisy input | +10.05 dB |
| high-frequency energy kept vs ground truth | **0.216** |

That last row is the problem. The model gets its +10 dB partly by **blurring**.
It keeps only about **22 %** of the fine detail that is really in the target. It
removes the noise and a lot of genuine structure along with it. This is the
classic L1 over-smoothing failure.

### The idea

Tell the network **what object it is looking at**, so it has a reason to put the
detail back.

We take a **frozen DINOv2 ViT-B/14** (a strong general-purpose vision model),
feed it an image, pool its features into one vector, and use a small MLP to
produce a per-channel scale and shift. Those modulate Restormer's features at
four points — the bottleneck and the three decoder stages:

```
F  ->  (1 + gamma) * F + beta
```

This is called **FiLM**. The MLP's last layer starts at zero, so at the very
first step `gamma = 0` and `beta = 0` and the whole network is *bit-identical*
to the baseline. Guidance can only grow from there.

Restormer is trained from scratch. DINOv2 is frozen and is the only pretrained
part.

### The one thing being studied

Everything is held identical to the baseline except **which image DINO looks at**:

| Arm | DINO sees | Question it asks |
|---|---|---|
| **A — renderDINO** | the clean black-background render, cropped and flipped exactly like the noisy input | does a clean object reference help? |
| **B — lqDINO** | the noisy input itself, the same tensor Restormer gets | does self-conditioning on the degraded image help? |

Same optimizer, same learning-rate schedule, same progressive crops
(128 → 160 → 192 → 256), same 300k iterations, same loss, same data split, same
seed. Only the DINO input differs.

### What we predicted, in writing, before training

- Any gain would be **small**: ≤ ~0.3 dB masked PSNR, plausibly a null result.
- renderDINO ≥ lqDINO if anything shows up at all.
- The FiLM head would probably need **centered** features to extract even that.

---

## 2. What we measured before spending any GPU time

Before training we checked whether the DINO features actually carry object
information for our data. They mostly do not — and this matters.

### The pooled feature is almost all constant

About **95 %** of each pooled feature vector is a single shared offset that is
the same for every image (‖mean‖ ≈ 106 vs ‖residual‖ ≈ 22). Every image looks
~0.9 cosine-similar to every other image, including empty crops. The actual
object information lives in the small leftover part.

### Does a render "point at" the same object's radar image?

Test: is DINO(render of object *i*) more similar to DINO(radar of object *i*)
than to DINO(radar of a different object *j*)? Higher `d` = more object
information.

| crop size | raw features | centered features |
|---|---|---|
| 128 | +0.027 (not significant) | **+0.249 (+8.6σ)** |
| 256 | −0.008 (null) | **+0.167 (+6.0σ)** |

**Raw features are object-blind. Centered features are not.** So the signal
exists, it is real, but it is modest and lives entirely in the residual.

For the lqDINO arm (noisy vs its own clean pair, a *different* test):

| crop size | centered |
|---|---|
| 128 | +0.183 (+3.7σ) |
| 256 | −0.123 (null) |

⚠️ **These two `d` numbers are not comparable scores.** lqDINO's measures noise
robustness inside the radar domain; renderDINO's measures correspondence between
two different image types. "+0.249 > +0.183" ranks nothing.

### One more thing we wrote down in advance

The signal **gets weaker as the crop gets bigger**, in both arms. And the
training schedule spends its last 96k iterations at crop 256 — the phase doing
the finest reconstruction, and the phase whose weights we keep. So the guidance
is strongest early and weakest exactly where it matters most. We recorded this
in advance as a possible explanation *if* E1 underperformed, and deliberately did
**not** change the schedule because of it.

---

## 3. Changes made before the first launch

| # | Change | Why |
|---|---|---|
| 1 | **Centering** — subtract a fixed per-arm mean feature vector | the measured signal exists only in the centered residual |
| 2 | No separate "raw features" arm | raw was already measured object-blind; ~3 GPU-days to confirm a known null |
| 3 | Run both arms | lqDINO matches published recipes and needs no render at inference |
| 4 | Re-verify identity-at-init after centering | confirm rather than assume |
| 5 | Write the crop-size decay into the pre-registration | so it can't become a post-hoc excuse |

The centering vector was built from **300 training crops**, with crop sizes drawn
in proportion to the training schedule (92/64/48/96 at 128/160/192/256), train
split only. It is stored as a buffer inside the model so it travels with the
checkpoint.

Measured on those crops:

| arm | ‖mean‖ | ‖residual‖ | share that is constant offset |
|---|---|---|---|
| renderDINO | 102.23 | 32.63 | 90.0 % |
| lqDINO | 85.63 | 35.40 | 84.9 % |

A **fixed** mean, deliberately not BatchNorm: the schedule drops the batch size
to 2 at 256px, and a two-sample mean is noise, not a mean.

**Checks that passed before launch:** identity at init exact (0.000e+00, both
arms, real DINOv2, real mean vectors); render/LQ crop alignment 200/200; DINO
weights genuinely loaded; a real forward+backward step with DINO receiving zero
gradients.

---

## 4. Attempt 1 — unbounded FiLM ❌

**Launched 2026-08-06.** renderDINO on A100, lqDINO on V100, both self-chaining
to 300k. **Cancelled 2026-08-07** after 44k and 73k iterations.

### It looked completely healthy

The training loss sat right on top of the baseline:

| iter | renderDINO | lqDINO | Exp 2 baseline |
|---|---|---|---|
| 41,000 | 5.40e-2 | 6.11e-2 | 5.43e-2 |

No crashes. Checkpoints landing. Nothing in the log looked wrong.

### Validation told a completely different story

| iter | renderDINO | lqDINO | baseline |
|---|---|---|---|
| 4,000 | 7.80 dB | 3.22 dB | **19.62 dB** |
| 44,000 | 4.73 dB | 4.96 dB | ~20.5 dB |
| 72,000 | — | 5.61 dB | ~20.7 dB |

The noisy input on its own is 12.35 dB. **Both arms were far worse than doing
nothing**, and flat instead of improving.

### The cause

We loaded the checkpoints and measured the modulation directly:

| lqDINO iter | \|γ\| max | \|β\| max | val PSNR |
|---|---|---|---|
| 2,000 | 30.7 | 7.2 | −74.3 |
| 10,000 | 125.4 | 33.9 | −138.7 |
| 40,000 | 255.7 | 64.0 | −167.1 |
| 72,000 | **325.5** | 88.4 | −172.4 |

`(1+γ)·F` with γ ≈ 325 multiplies features by ~326×. Model output reached
**±2.4 billion**. It was already broken by iteration **2,000**.

### Why the loss hid it

Stage 1 trains on **128px** crops; validation runs on full **256px** images. On
one validation image at 73k:

| what | PSNR |
|---|---|
| 128 crop, FiLM on | **20.55 dB** (looks fine) |
| 256 full, FiLM on | **−172 dB** (garbage) |

The network had found a knife-edge solution that only survives its training crop
size. The training loss never complained because it only ever saw 128px.

We also broke γ into "constant part" vs "varies with the image":

| arm | \|γ\| mean | variation across images | fraction that is constant |
|---|---|---|---|
| lqDINO | 86.6 | 25.2 | **71 %** |
| renderDINO | 0.38 | 0.45 | mostly varying, but tiny |

So for lqDINO, γ was not guidance at all — it was a giant fixed per-channel
rescale.

### Why nothing stopped it

Two guards existed and both are near-useless here:

- `use_grad_clip: true` (0.01) — **Adam normalizes per-parameter**, so uniformly
  shrinking the gradient barely changes the step size.
- `weight_decay: 1e-4` — with decoupled AdamW that's a pull of `lr·wd = 3e-8`
  per step. Nothing.

**Cost: ~20 GPU-hours.**

**Honest note:** our pre-launch checks tested identity at step 0 and a single
training step. Nothing tested stability *over* training. A few hundred iterations
watching |γ| would have caught this in minutes.

---

## 5. Attempt 2 — bounded γ and β ❌

### What changed

```python
gamma = 0.5 * tanh(raw)     # (1+gamma) confined to [0.5, 1.5]
beta  = 0.5 * tanh(raw)
```

Reasoning: FiLM has a **scale degeneracy** — you can multiply a feature map by
some number and let the next layers divide it back out, and the loss doesn't
change. That is a flat direction with nothing pushing back, so γ drifts along it
forever. `tanh` makes the runaway impossible by construction. `tanh(0) = 0`, so
identity-at-init is preserved exactly.

We also added:
- **Logging** of `|γ|max`, `|β|max`, and how much γ varies across the batch — so
  the modulation is visible instead of invisible.
- A **4,000-iteration gate**: a ~35-minute run that must reach ≥ 18 dB before
  any 3-GPU-day run is allowed to start.
- **v2 experiment names** — necessary, because the training code auto-resumes
  from the newest checkpoint in the experiment folder, and the old folders were
  full of the broken run's states.

### Result: the bound held, the model still failed

| arm | val @1k | @2k | @3k | @4k | baseline @4k |
|---|---|---|---|---|---|
| renderDINO | 15.12 | 14.00 | 12.88 | **14.82** | 19.62 |
| lqDINO | 12.68 | 7.93 | 7.15 | **6.69** | 19.62 |

max |γ| = 0.4999 — the bound worked perfectly, no explosion. But:

```
iter  200   |g|max=0.307
iter  400   |g|max=0.477
iter 1000   |g|max=0.496     <- pinned at the rail
iter 4000   |g|max=0.4998    g_std 0.073 -> 0.011
```

γ raced to the maximum allowed value within ~400 iterations and **stayed there**.
Once `tanh` saturates its gradient vanishes, so the head could no longer respond
to the image at all — γ froze into a near-constant ±0.5 mask. That is the worst
of both worlds: a big fixed distortion the backbone has to spend capacity
undoing, with no guidance benefit.

**Cost: ~1.5 GPU-hours.** The gate caught it instead of a 3-day run.

---

## 6. Attempt 3 — slow and late FiLM ❌

### The reasoning

Bounding capped *how bad* the distortion could get but never stopped the head
from racing to it. The real problem is a **race**: at iteration 200 the backbone
is still random, and the fastest way for a random FiLM head to reduce the loss is
that scale degeneracy. So it goes straight there.

Fix: don't let FiLM move until the backbone is worth conditioning.

| change | value | effect |
|---|---|---|
| **warmup** | 5,000 iters | FiLM completely off — the block is skipped, the head gets **zero gradient** and stays at its zero init |
| **ramp** | 5,000 iters | γ/β fade in linearly, full strength at iteration 10,000 |
| **raw scale** | 0.01 | shrinks the pre-tanh signal, so the modulation moves ~100× slower per step |

A nice side effect: during warmup the model *is* the plain baseline, so the early
validations act as a **built-in control** in every gate run.

We also extended the gate to **16,000 iterations** (a 4k gate would test nothing
when FiLM only turns on at 10k) and required two things to pass: final ≥ 18 dB
**and** final ≥ (warmup baseline − 1 dB).

### Result

| iter | renderDINO | lqDINO | phase |
|---|---|---|---|
| 2,000 | 17.85 | 17.39 | warmup — FiLM off = baseline |
| 4,000 | 19.29 | 19.32 | warmup — FiLM off = baseline |
| 8,000 | 19.33 | 19.38 | ramping |
| 12,000 | 18.69 | 17.62 | FiLM full |
| 16,000 | **20.00** | **17.60** | FiLM full |

**lqDINO failed clearly**: 17.60 vs its own 19.32 warmup baseline — turning FiLM
on cost **1.7 dB**, and γ sat at 99 % of the bound from 14k onward.

**renderDINO printed PASS — and that verdict was wrong.** The endpoint looked
good, but γ was mid-explosion:

| iter | \|γ\| max | % of bound |
|---|---|---|
| 13,000 | 1.19e-03 | 0.2 % |
| 14,000 | 1.18e-02 | 2.4 % |
| 15,000 | 8.10e-02 | 16.2 % |
| 16,000 | 2.08e-01 | **41.7 %** |

That is roughly **10× per 1,000 iterations**. The gate happened to stop right at
the knee of the curve. renderDINO is on exactly lqDINO's trajectory, just delayed
about 5,000 iterations by the warmup. A 300k run would certainly have degraded.

### Two bugs in our own gate, found and fixed

1. The verdict checked **only the final PSNR**, so a run one step from the cliff
   could pass. Added a third criterion: γ in the final quarter must be below half
   the bound **and** must not have grown more than 20× over the run.
2. The iteration labels were off by one, because the training code runs an extra
   end-of-training validation that duplicates the last one.

Re-judged under the corrected criteria: **both arms FAIL**.

**Cost: ~5 GPU-hours.**

---

## 7. Why it keeps failing

The important detail is that γ grows **geometrically** (~10× per 1,000
iterations) once FiLM switches on. That is far faster than the steady drift that
ordinary parameter growth under Adam would produce. Geometric growth means
**positive feedback**:

> As the modulation grows, the backbone adapts to *depend* on it. That
> dependence increases the gradient pushing the modulation further. Which makes
> the backbone depend on it more.

So the scale degeneracy is not merely unconstrained — it is **self-reinforcing**.
And it is self-reinforcing *precisely because the backbone is free to co-adapt*.

That explains why all three attempts behaved the same way:

| attempt | what it did | result |
|---|---|---|
| 1 — unbounded | nothing held γ back | γ → 325, total collapse |
| 2 — bounded | capped how bad it gets | γ pinned at the cap in 400 iters |
| 3 — bounded + late + slow | delayed the onset | γ started 5k later, then exploded anyway |

Each fix addressed a *symptom*. None removed the feedback loop.

**There is also a structural mismatch worth stating plainly.** FiLM and adapter
conditioning in the literature (ControlNet, T2I-Adapter) attach to a
**pretrained** backbone that is frozen or nearly frozen. Here the backbone is
trained from scratch *at the same time* as the conditioning head. That is what
creates the race, and it should have been flagged before the first launch rather
than after three failures.

---

## 8. What did work

Not everything failed. These all hold and are reusable:

- ✅ Zero-init identity — the guided model is bit-identical to the baseline at
  step 0, verified repeatedly at exactly `0.000e+00`, including after centering.
- ✅ Render/LQ alignment — 200/200 marker trials; the render really is cropped
  and flipped identically to the noisy input.
- ✅ DINOv2 loads correctly offline on compute nodes with no internet, and fails
  loudly if a path is wrong.
- ✅ Centering works as designed and is measured, not guessed.
- ✅ The **gate** works. It has caught two failures for ~6.5 GPU-hours total,
  instead of two 3-GPU-day runs.
- ✅ The **logging** works. Attempt 1 ran 73k iterations blind; attempts 2 and 3
  were diagnosed from the log within minutes.

---

## 9. Cost so far

| what | GPU time |
|---|---|
| Attempt 1 (unbounded, cancelled) | ~20 h |
| Attempt 2 gates (4k) | ~1.5 h |
| Attempt 3 gates (16k) | ~5 h |
| **Total** | **~27 GPU-hours** |

For comparison, one full 300k run per arm is roughly **3 GPU-days each**. The
gate has already saved far more than it cost.

---

## 10. Where we stand and what's next

**Nothing is running. No 300k run has ever been launched to completion. There is
no E1 result yet.**

### Recommended next step

Attach FiLM to the **already-trained Exp 2 backbone** (`net_g_292000.pth`),
frozen or fine-tuned at a much lower learning rate, instead of training the
backbone from scratch alongside it.

**Why this should work:** it removes the backbone's freedom to co-adapt, which is
the thing driving the feedback loop. It is also how this kind of conditioning is
normally done.

**Bonus:** it is far cheaper — tens of thousands of iterations, not 300k — which
matters with a September deadline.

**Cost:** it changes the pre-registration. The runs would no longer be "from
scratch", and the comparison against Exp 2 becomes "same backbone, with vs
without DINO guidance". That is arguably a *cleaner* isolation of what DINO
contributes, but it is a different experiment and needs an amendment written
before launching.

### Rules to keep

- Both arms must keep **identical** settings. Tuning one arm differently from the
  other would confound the only variable being studied.
- Nothing goes to 300k without passing the gate first.

### Still open, unrelated to the failures

- `test_holo.py` cannot yet load and pass the render, needed for renderDINO
  test-time evaluation.
- The citation for the "layers {1,4,8,12}" recipe is still marked
  **[SOURCE NEEDED]**.

---

## 11. One-paragraph summary

We wanted to fix the baseline's over-smoothing by telling Restormer what object
it was reconstructing, using a frozen DINOv2 and FiLM modulation, with the single
studied variable being whether DINO sees a clean render or the noisy input.
Before training we measured that the DINO signal is real but weak and only exists
after centering, and we predicted a small or null gain in writing. Three training
attempts all failed the same way: the FiLM scale factor runs away — to 325
unbounded, to the cap in 400 iterations when bounded, and to the cap again after
a delay when bounded and warmed up — because the modulation sits on a flat,
self-reinforcing direction in the loss that the from-scratch backbone actively
feeds. Bounding, slowing and delaying it treated symptoms, not the cause. The
proposed fix is to attach the guidance to the already-trained baseline backbone
so it can no longer co-adapt. Total cost so far is ~27 GPU-hours, most of it the
first attempt, before the gate existed to catch this cheaply.
