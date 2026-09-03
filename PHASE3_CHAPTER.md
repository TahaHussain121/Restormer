# Phase 3 — Injecting a Frozen Visual Foundation Prior into a Transformer Restoration Backbone

*A long-form draft of the Phase-3 chapter. Written 2026-08-31, after the locked
test split had been read for every completed arm.*

**Numbers of record live in `DEVLOG.md` (Steps 29–33). If this document and the
devlog disagree, the devlog is right and this is stale.** Test split n=338,
validation split n=339, PSNR computed on the uint16 evaluation path throughout
unless a figure is explicitly marked as the 8-bit training-time metric.

---

## 1. The problem, and the shape of the answer

The task is single-image restoration of holographic radar reconstructions. The
input is a reconstruction computed from a low ray budget — referred to
throughout as the 1e5 image — and the target is the reconstruction of the same
scene computed at a hundred times the budget, the 1e7 image. The degradation is
therefore not additive sensor noise in the usual sense. It is reconstruction
noise: speckle, missing interference fringes, and a characteristic loss of the
high-frequency structure that carries object boundaries. A network trained on
this pairing is not denoising in the Gaussian sense; it is being asked to
hallucinate the fringe structure that a larger ray budget would have produced,
constrained by whatever evidence survives in the low-budget image.

The motivating weakness of a plain restoration backbone on this data is
over-smoothing. A model trained with an L1 objective on a heavily degraded input
converges to something close to a conditional mean, which on this data means
soft, low-contrast object boundaries and a suppressed high-frequency band. The
project's earlier baselines exhibit exactly this, and it is measurable: the
ratio of high-frequency energy in the prediction to that in the target sits at
roughly 0.22 for a plain backbone, meaning the model reproduces less than a
quarter of the target's high-frequency content.

The hypothesis of Phase 3 is that a frozen visual foundation model can supply
the structural information the degraded input has lost. DINOv2 has been shown to
produce patch-level features that encode object extent, part boundaries and
figure-ground separation. If those features can be delivered to the restoration
backbone in a form it can use, the backbone should be able to place boundaries
it could not otherwise infer.

The question Phase 3 actually answers is narrower and more useful than "does a
foundation prior help". It is: **given that a DINO prior is used, what
determines whether it helps, and by how much?** Four factors are isolated and
tested one at a time — the *source image* the prior is computed from, the
*spatial structure* of the prior, the *number of network depths* it is read
from, and the *operator* used to inject it. The headline result is that the
first three matter, in one case decisively, and the fourth does not.

---

## 2. Background: what the reference methods actually do

Three works motivated the design, and it is worth being precise about them,
because the precision changed what could be claimed.

**Perceive-IR** applies a degradation-aware prior to a restoration backbone. Its
prior, verified by reading the arXiv HTML rather than working from memory, is a
vector `F_l ∈ ℝ^{1×768}` — **a single global descriptor per level, not a spatial
grid**. Its prior-guidance module applies a FiLM-style affine modulation to the
backbone feature *before* the attention step, so the query entering the
attention is the prior-modulated feature rather than the prior itself. Whether
the attention that follows uses a spatial softmax or a Restormer-style
transposed channel softmax is not stated in the paper; it writes the equation in
MDTA notation with hatted Q/K/V and a learnable temperature, and cites
Restormer, but never specifies the reshape. This ambiguity does not matter for
the present work, because a 1×768 prior has no spatial tokens to attend over
under either reading.

**DINOLight** supplies the two mechanisms this study reproduces most directly:
an adaptive feature fusion module that combines DINO features read at several
transformer depths using a per-position softmax across those depths, and a gated
channel cross-attention block that injects the fused prior. Both are implemented
here, cited rather than claimed as novel.

**DSGIR** is paywalled and was never verified. Everything the project records
about it is second-hand, and no claim in this chapter depends on it.

The consequence of the first point is worth stating explicitly because it
reframes one of this study's negative results. **No method in the reference set
performs spatial-token cross-attention between a prior grid and a feature
grid.** Perceive-IR cannot, because its prior has no spatial extent. The
attention arms built in this study are therefore original constructions rather
than reimplementations, and the failure of one of them is an original negative
result rather than a failure to reproduce prior work.

---

## 3. The backbone

The restoration backbone is Restormer, unmodified. It is a four-level encoder–
decoder of transformer blocks operating directly on feature maps rather than on
flattened token sequences.

A single-channel input of spatial size H×W enters a 3×3 overlapping patch
embedding producing 48 channels at full resolution. Three encoder levels follow,
each a stack of transformer blocks, separated by downsampling operations that
halve the spatial resolution and double the channel width. The configuration
used throughout is 4 blocks at 48 channels, 6 blocks at 96 channels, 6 blocks at
192 channels, and then a latent stage of 8 blocks at 384 channels operating at
H/8 × W/8. The decoder mirrors this: three upsampling steps, each concatenating
the corresponding encoder output as a skip connection, followed by transformer
blocks. Levels 3 and 2 follow their concatenation with a 1×1 convolution that
halves the width again; **level 1 does not**, which is why the final decoder
stage, the four refinement blocks and the output convolution all operate at 96
channels rather than 48. The output convolution maps 96 channels to 1, and its
result is added to the input radar image as a global residual, so the network
predicts a correction rather than an image.

There are 44 transformer blocks in total: 4 + 6 + 6 in the encoder, 8 in the
latent stage, 6 + 6 + 4 in the decoder, and 4 refinement blocks. The whole
backbone holds 26,124,052 parameters.

Each transformer block is two residual stages:

```
x = x + attn(norm1(x))
x = x + ffn(norm2(x))
```

The attention is **Multi-Dconv Head Transposed Attention (MDTA)**, and its
structure matters for a later argument. Query, key and value are produced by a
1×1 convolution followed by a 3×3 depthwise convolution. The tensors are
rearranged to `[B, heads, C/heads, H·W]`, and **Q and K are L2-normalised along
the token axis**. The matrix product therefore contracts over spatial positions
and yields an attention matrix of shape `[B, heads, C/heads, C/heads]` — a
channel-by-channel matrix whose size does not depend on how many spatial
positions exist. A per-head learnable temperature, initialised to 1.0,
multiplies the logits before the softmax. The feed-forward stage is a gated
convolutional network: a 1×1 expansion by a factor of 2.66 into two branches, a
3×3 depthwise convolution, a GELU gate of one branch against the other, and a
1×1 projection back.

The distribution of parameters across the network is extremely uneven, and this
turns out to be relevant to the injection design. A first-level block holds
31,279 parameters. A latent block holds **1,796,306**. The eight latent blocks
together hold 14,370,448 parameters — **55.0% of the entire network**.

---

## 4. The prior branch

### 4.1 Source and preprocessing

The dataset supplies, for each sample, a stacked two-channel tensor: channel 0
is the noisy radar reconstruction, channel 1 is an aligned rendered image of the
same scene on a black background. **The backbone only ever receives channel 0.**
The render reaches the network exclusively through the DINO branch and never
through the trunk or the global residual. This was a deliberate construction
decision: it guarantees that the difference between the baseline and any DINO
arm is confined to the prior pathway, and it also avoids a known hazard in the
training loop, where the random-crop routine sub-crops only the low-quality and
ground-truth tensors and would silently misalign a third aligned tensor.

The prior is computed by a frozen DINOv2 ViT-B/14. The source image is resized
bilinearly to 224×224 during training and 448×448 during evaluation, then
ImageNet-normalised. Because the patch size is 14, these give patch grids of
16×16 and 32×32 respectively — 256 and 1024 tokens, each of width 768.

### 4.2 The scale-matching constraint

The two resize targets are not arbitrary. The latent tensor at which the prior
is injected has spatial size H/8 × W/8, which is 16×16 when training on 128-pixel
crops and 32×32 when evaluating on full 256-pixel frames. Since 224/14 = 16 and
448/14 = 32, the DINO grid **exactly matches the latent grid in both regimes**,
and the number of radar pixels per DINO token stays at approximately 8 in both.

This means no interpolation of the feature grid is ever required. The property
is enforced rather than assumed: a runtime assertion compares the DINO grid
shape against the latent shape and raises if they disagree. This assertion has
already caught a real bug — an earlier arm detected DINO capability by class-name
prefix, which silently failed for two arms whose class names fell outside the
convention, leaving them in the 128-pixel regime while being fed 256-pixel
images and producing a 16×16 grid against a 32×32 latent. The detection is now
duck-typed on capability rather than name.

### 4.3 Centring

Raw DINO patch tokens have a large layer-dependent mean. Each layer's tokens are
centred by subtracting a mean vector computed **on the training split only**, at
the matching resolution. Separate means exist for the 224-pixel and 448-pixel
regimes and for each transformer depth read, and all of them are registered as
model buffers so they travel with the checkpoint. The mean tensors themselves
are not tracked in version control, but their metadata files are, so any lost
mean can be recomputed by a known recipe.

Centring matters because the injection is additive and zero-initialised: an
uncentred prior would inject a large constant offset into the latent as soon as
the projection left zero.

### 4.4 The projection

The centred grid `D` has 768 channels and must meet a 384-channel latent. A 1×1
convolution `P` performs the mapping, and is initialised to **exactly zero,
weight and bias**. Its parameter count is 768 × 384 + 384 = **295,296**.

The zero initialisation is load-bearing in three ways. It guarantees that at
step 0 every DINO arm produces output bit-identical to the baseline, verified to
0.000e+00 in the architectural smoke tests — so no arm inherits an advantage
from initialisation and every gain must be learned. It makes the injected
magnitude start at exactly zero, which gives the stability monitor a
well-defined reference. And it means an arm that finds the prior useless can
return to the baseline rather than being forced to use it.

The cost is a gradient staircase: everything upstream of a zero-initialised
layer reaches the loss only through it, so those parameters see zero gradient on
the first backward pass and begin learning from step 2. This is expected,
documented, and asserted rather than assumed away. It becomes decisive in
Section 6.9.

### 4.5 The injection point, and why it is before the latent stage

The prior is added to `inp_enc_level4`, the tensor produced by the final encoder
downsampling and consumed by the latent stage. Written F throughout, it has
shape `[B, 384, 16, 16]` in training and `[B, 384, 32, 32]` at evaluation.

An investigation of the backbone before any code was written established that
there are only **two** distinct candidate tensors at the bottleneck, not four.
The feature immediately after the final downsampling and the feature immediately
before the latent transformer blocks are the *same tensor* — nothing sits
between them. Likewise the feature immediately after the latent blocks and the
feature immediately before decoder upsampling are the same tensor. So the choice
is binary: inject before the eight latent blocks, or after them.

Before was chosen for two reasons. First, anything injected after the latent
stage is seen only by the decoder, never by the eight blocks that hold 55% of
the network's parameters. Second, because each transformer block carries an
identity path, a signal added at the input of the stage rides the residual
stream through all eight blocks and is still present at the output, *in addition
to* having been processed by them. Injecting before therefore strictly contains
what injecting after would provide: the model can learn to ignore a prior it was
given, but it cannot learn to process one it never saw.

Two honest qualifications. Spatial resolution is not the reason — the latent
stage preserves shape, so its output would match the DINO grid equally well.
And the choice is **untested**: the code hard-refuses any other injection point,
which keeps arms comparable but means there is no empirical comparison. It is a
reasoned design decision, and is reported as such.

---

## 5. Experimental protocol

### 5.1 Data

The dataset holds 6,778 scenes. Each exists as a 1e5 reconstruction, a 1e7
reconstruction and a render, all 256×256. The radar images are uint16 grayscale
PNGs; the renders are uint8 RGB with identical channels, on a black background.
The split is 6,101 training, 339 validation and 338 test, drawn with a fixed
seed. **The test split was locked before any Phase-3 arm was trained.**

### 5.2 Training

Every arm without exception uses: fixed 128×128 random crops, batch size 8,
300,000 iterations, AdamW at learning rate 3×10⁻⁴ with weight decay 10⁻⁴ and
betas (0.9, 0.999), a cosine-annealing schedule with restart periods of 92,000
and 208,000 iterations and minimum learning rates of 3×10⁻⁴ and 10⁻⁶,
an L1 pixel loss, gradient clipping, geometric augmentation, mixup disabled, and
manual seed 100. Validation runs every 4,000 iterations and checkpoints are
saved every 2,000.

The fixed-crop choice deserves comment. The project's older baseline used
progressive resizing, which trains at larger crops later in the schedule and
therefore performs better at the 256-pixel evaluation resolution. The Phase-3
baseline is deliberately fixed-crop so that **every arm shares one recipe**,
which is what licenses the one-factor claims. The cost, registered in advance,
is that the Phase-3 baseline scores 0.53 dB below the older baseline on the test
split — 21.873 against 22.405 — because it never trains at the resolution it is
evaluated at. This does not weaken any Phase-3 comparison, all of which are
internal, and the best Phase-3 arm still beats the older, stronger baseline by
1.676 dB.

### 5.3 Two evaluation protocols

Every arm is evaluated in two regimes.

**full256** feeds the complete 256×256 frame. This is the deployment-relevant
setting and the one that requires the model to generalise from its 128-pixel
training crops.

**crop128** feeds a fixed 128×128 crop drawn from a manifest, so every arm sees
identical crops at identical coordinates. This is the *matched* condition — the
model is evaluated at the scale it trained at.

Both are reported for every arm. As Section 7 shows, the conclusion differs
between them for an entire family of arms, and reporting only one would
misrepresent the study.

The two protocols also differ for the *prior*, not only for the backbone: DINO
sees 224 pixels under crop128 and 448 under full256. Section 4.2 guarantees the
resulting token grids match the latent in both cases, but says nothing about
whether the features themselves agree. Section 7.4 measures that directly, and
they do not.

### 5.4 Metrics

PSNR is computed on the uint16 path with a data range of 1.0, and this must
never be compared against the 8-bit training-time validation PSNR that appears
in the logs; the two are different quantities, and the training-time curve
carries roughly 0.4 dB of point-to-point noise.

Alongside full-image PSNR, a **foreground-masked PSNR** is reported, with the
mask defined as ground truth above 0.01, no morphological dilation, covering
approximately 32.5% of a frame. This matters because these images are largely
near-black background, which inflates full-image PSNR; the masked figure
reflects object reconstruction. SSIM is reported whole and masked. Sharpness is
characterised by the ratio of prediction to target for high-frequency spectral
energy, Laplacian variance and Sobel gradient magnitude, all of which are below
1.0 for an over-smoothing model.

### 5.5 Checkpoint selection

**Selection reads validation PSNR only. The test split never selects.** The
selection routine parses the training log for the highest validation PSNR and
records the corresponding iteration, the top-five spread, and whether the
checkpoint file exists.

Two mechanical facts about this are worth recording. The training framework has
**no best-model tracking of any kind** — validation runs automatically and logs
a score, but nothing compares scores or marks a checkpoint as best. Selection is
entirely a post-hoc parse. And because checkpoints save every 2,000 iterations
while validation runs every 4,000, **only about 75 of the 151 saved checkpoints
per arm are ever scored**, and every selected iteration is consequently a
multiple of 4,000. Top-five spreads range from 0.012 to 0.118 dB, so
neighbouring checkpoints are close to interchangeable and the practical cost of
this coarseness is small — but it is a limitation, not a non-issue.

### 5.6 Stability monitoring

Every DINO arm logs the Frobenius norm of the latent, the norm of the injected
prior, and their ratio, at fixed intervals. A gate aborts training if the ratio
exceeds 10, if it grows by more than a factor of 10 against a reference taken at
iteration 5,000, or on any NaN or infinity.

One honest finding about this gate is on record. When the spatial cross-attention
arm failed, its latent norm grew fifteen-fold and its projected norm fourteen-
fold against the 5,000-iteration reference, **yet the ratio never left the range
0.52–0.91** and the gate never fired. The gate constrains only the *ratio*, and
co-inflation of both quantities passes it undetected. A co-inflation rule was
identified as the fix and was never implemented.

---

### 5.7 Reproducibility and verification infrastructure

Because the study rests on one-factor claims, a substantial part of the work was
establishing that the arms really do differ in one factor. Four mechanisms
support this.

**Shared code rather than copied code.** Where two arms use the same component,
that component is imported, never duplicated. The gated cross-attention block
lives in one module and is imported by all four arms that use it; the layer
fusion module is imported from the arm that introduced it rather than
reimplemented, which also means the two arms sharing it are byte-identical on
the feature side. This has a documented cost — the fusion module carries a GELU
where a work order had specified a SiLU, and comparability between arms was
judged worth more than matching the reference activation — but it removes an
entire class of silent divergence.

**Architectural smoke tests.** Before any arm is submitted, a suite builds it
alongside the baseline under the same seed and verifies: that the step-0 output
is bit-identical to the baseline, to 0.000e+00, at both the 128 and 256 regimes;
that all 494 shared backbone tensors are byte-identical; that the parameter delta
matches an independently computed expected value exactly; that the attention
matrix shape is identical at both token counts while the grid demonstrably
changes; and that specific keys are present or absent as designed. The ACA arms
each pass 31 such checks. Construction of the fusion module is wrapped in a
random-number-generator fence that saves and restores state, so adding a module
cannot shift the draws that produce the backbone weights — which is what makes
"byte-identical to the baseline" achievable rather than approximate.

**Integration tests.** The architectural suite cannot reach the training loop.
A separate 6,000-iteration run exercises the real entry point over the real
configuration with only the run length and logging frequencies changed, under a
throwaway experiment name that cannot touch a real experiment directory or
append to a real log. The run length is chosen to cross the iteration at which
the stability gate's ratio rules begin, so the gate is genuinely enforced rather
than merely measured. This tier has caught real defects the architectural suite
missed: bare integer keys in a configuration file crashing the option parser
before iteration 1, an observation publishing condition that never fired at a
particular frequency, and — most consequentially — the dead-branch failure
described in Section 6.11.

**Self-chaining training with explicit state.** Each arm runs under a driver that
queues its own successor before training begins, so a walltime kill does not end
the run. State is kept in files — a running-job lock, a completion marker, an
abort marker — and the driver refuses to start a second trainer while one is
genuinely running, while treating a stale lock from a dead job as permission to
take over. An append-only manifest records, per arm, the configuration path,
the job identifiers in order, the state, the iterations reached and the exact
command to resume by hand.

One limitation of this infrastructure is worth recording because it caused real
work later. **The training driver manages training only.** It writes a completion
marker and exits; it never performs checkpoint selection or evaluation. One arm
had a deferred dispatcher attached that woke on training completion and ran
selection automatically; the arms submitted without such a dispatcher simply sat
finished and unselected until selection was run by hand. Combined with the fact
that the training framework has no best-model tracking at all, this means
"training finished" and "a best checkpoint exists" are entirely separate events,
and nothing in the system connects them unless a dispatcher is explicitly
attached.

---

## 6. The experiments

Every arm below differs from the baseline in exactly one respect unless stated
otherwise. Encoder, decoder, skips, refinement, the output convolution and the
global residual are byte-identical throughout, verified tensor by tensor at
494/494 in the smoke tests.

### 6.1 E0-fixed — the baseline

```
guided = F
```

No DINO branch exists. The latent stage receives the untouched encoder output.
**Test: 21.873 dB full256, 19.546 dB crop128. Masked PSNR 17.599, SSIM 0.7829,
high-frequency ratio 0.218.**

That last figure quantifies the motivating weakness: the model reproduces
roughly a fifth of the target's high-frequency energy.

### 6.2 E1-addition-noisy — the prior computed on the degraded input

```
D      = centred DINO block 6 of the 1e5 RADAR
guided = F + P(D)
```

Parameter cost +295,296. The natural first attempt: give the network a semantic
reading of its own input.

**Test: 21.296 dB full256 (−0.577 vs baseline), 19.062 crop128 (−0.484).**

It is worse than using no prior at all. DINO applied to a heavily degraded image
produces a degraded prior, and adding it costs half a decibel. Curiously the
high-frequency ratio *rises* to 0.275 against the baseline's 0.218 while PSNR
falls, indicating the arm injects structure that is high-frequency but wrong.

### 6.3 E1-addition-render — the prior computed on a clean modality

```
D      = centred DINO block 6 of the RENDER
guided = F + P(D)
```

Identical to the previous arm in architecture, parameter count, seed and
schedule. **One tensor is swapped.**

**Test: 24.081 dB full256 (+2.208), 22.259 crop128 (+2.713). Masked PSNR 19.673,
SSIM 0.8220, high-frequency ratio 0.328. Improved on 298 of 338 images (88.2%).**

**Finding 1: the source of the prior decides the sign of the effect.** The same
mechanism, at the same parameter count, is harmful when the prior is computed on
degraded input and strongly beneficial when it is computed on a clean modality.
The high-frequency ratio rises from 0.218 to 0.328 and the Laplacian ratio from
0.286 to 0.438, so the over-smoothing that motivated the work is measurably
reduced, not merely traded against PSNR.

A note on whether this constitutes an oracle: the render is an available input
at inference, not a derivative of the 1e7 target, so the method is usable rather
than merely diagnostic. The 1e7 image is used as a target and nowhere else.

### 6.4 The mismatched-render control

Not a trained arm but a post-hoc diagnostic on the trained addition-render
model, evaluated under four conditions: the correct render, a **shuffled**
render assigned by a Sattolo single-cycle derangement with the derangement
asserted programmatically, a zeroed prior, and the dataset-mean render.

**Shuffling costs 9.094 dB on average, is worse on 335 of 339 images, and falls
below the baseline's mean on 322 of them.**

This rules out the most obvious deflationary reading — that the model has merely
learned a dataset-level prior over object appearance. It is using the per-image
correspondence.

### 6.5 global-render — the spatial ablation

```
D_g    = broadcast( mean over the g×g positions of D )
guided = F + P(D_g)
```

The 16×16 grid is averaged to a single 768-vector and copied back to every
position, so every location receives the identical prior. Pooling is a mean and
a broadcast and adds **no parameters**, so this arm has exactly the same 295,296
parameter cost as addition-render. Centring and pooling commute, so no new mean
statistics are needed. The only difference in the entire model is whether the
prior varies across space.

**Test: 20.589 dB full256 (−1.284), 19.352 crop128 (−0.194).**

**Finding 2: the value of the prior is spatial.** Keeping the content and
destroying the layout drops performance *below* the no-DINO baseline. The claim
is not that the model learned what the objects look like; it is that the prior
tells it *where*.

### 6.6 concat-render — a more general fusion

```
guided = fuse( cat([F, D], dim=1) )          fuse: 1×1, 1152 → 384
       = conv(F, W_F) + conv(D, W_D) + b
```

The projection P is deleted entirely. This operator is **strictly more general
than addition**: it can transform the feature tensor as well as the prior, where
addition leaves F untouched. W_F is initialised to identity and W_D to zero, so
step 0 still equals the baseline. Parameter cost +442,752.

**Test: 24.065 dB full256, which is −0.016 against addition-render, p=0.98.
Validation: +0.023, p=0.49.**

**Finding 3: a strictly more expressive fusion buys nothing.** The arm lands on
both sides of zero across two independent splits, which is what a genuine null
looks like. This result is reported deliberately rather than buried: a method
that returns nulls where nulls exist lends credibility to the differences it
does detect. On crop128 it reads +0.214 (p=1.7×10⁻⁴), which should be read as
noise around zero rather than a result, precisely because the full256 figure is
negative.

### 6.7 crossattn-render — spatial cross-attention, and the failure that redirects the study

```
attn   = softmax( Q Kᵀ / √64 ) V        Q = F,  K = V = D,  6 heads of dim 64
guided = F + W_o( attn )
```

The intuition is that attention *selects* — it should pick the parts of the
prior relevant to each position rather than adding everything. Parameter cost
+888,576. The attention direction is deliberately reversed from the reference
papers: the radar feature queries the prior, so prior content reaches the output.

**From identical weights at iteration 178,000: +2.00 dB on crop128 and −7.56 dB
on full256.**

The mechanism is diagnosed rather than guessed. The attention matrix is spatial:
its size is the number of tokens squared. During training that is 256×256.
At evaluation the grid becomes 32×32, so the matrix becomes 1024×1024 and the
softmax must renormalise over four times as many competitors. The operation
changes shape between train and test, and the arm collapses in the regime it was
not trained in.

**The +2.00 dB must not be reported as a result.** Checkpoint selection uses the
full-256 validation metric — the one regime this arm cannot perform in — and
therefore selected **iteration 4,000 out of 179,000**. Every published number
for this arm comes from an essentially untrained model. The honest statement is
that the selection metric and in-distribution performance are *anti-correlated*
for this arm. A pre-registered escape clause existed, permitting matched-128
reporting if an arm improved there but not at full resolution; it was never
invoked, because invoking it after seeing the numbers would not have been
legitimate either.

**Finding 4: spatial cross-attention over a prior grid does not survive the
train-to-evaluation scale change.** This is the hinge of the study. Every
subsequent arm uses *channel* attention, whose matrix is independent of token
count, and that choice is a direct response to a diagnosed failure rather than
an arbitrary preference. It is also an original negative result: since
Perceive-IR's prior is a 1×768 global vector with no spatial tokens, no method
in the reference set performs this operation.

### 6.8 affm-render — the layer-count factor

```
s_l    = conv_l( gelu(D_l) )                 1×1, 768 → 1, one per layer
w      = softmax( [s₃, s₆, s₉, s₁₂] )        across LAYERS, per position
D      = Σ_l  w_l · D_l
guided = F + P(D)                            injection UNCHANGED
```

DINO is read at blocks 3, 6, 9 and 12, each centred with its own training-set
mean, and the four grids are combined by a softmax running across the *layer*
axis independently at each of the 256 grid positions. The four weights sum to
1 at every position.

The critical design property is that the combination is a **weighted sum, not a
concatenation**. The output therefore stays 768 channels wide, P is bit-identical
to addition-render's, and the entire additional cost is four scoring convolutions
at 769 parameters each — **3,076 parameters, an increase of 1.04%.** This is the
whole reason AFFM was chosen over a concatenation: *a gain here cannot be
attributed to capacity.*

**Test: 24.311 dB full256, +0.230 against addition-render, p=4.3×10⁻⁷.
Validation: +0.284, p=1.0×10⁻⁹. crop128: +0.044, not significant.**

**Finding 5: reading the prior at multiple depths improves restoration at
essentially no parameter cost.** The result replicates across two independent
splits with the same sign and comparable magnitude — which is precisely what a
seed repeat would have been run to establish.

**The learned weights are genuinely spatial.** Dumping the per-position weight
maps at 300,000 iterations gives standard deviations across positions of
0.159, 0.105, 0.105 and 0.113 for blocks 3, 6, 9 and 12, against means of 0.257,
0.246, 0.300 and 0.197. Variation of that size against those means means the
model really is selecting different depths at different locations rather than
applying a flat global mixture — which also distinguishes AFFM sharply from
global-render: **global-render pools over space; AFFM sums over layers,
per position.** They collapse orthogonal axes, which is why they land on
opposite sides of the baseline.

**No depth is ever discarded.** The minimum weight any layer reaches across the
entire run is 0.1028. Blocks 6 and 9 carry the most weight and are also the most
stable, with roughly half the variance of blocks 3 and 12; block 6 — the depth
identified by the project's earlier single-layer study — holds the tightest
weight of the four. The weights are non-stationary over training, with block 12
running from 0.359 at 151,000 iterations to 0.173 at 299,000, so only a windowed
mean with the window stated should be reported, and no strict ordering among the
less stable depths should be claimed.

### 6.9 The ACA block — gated channel cross-attention

This is DINOLight's fusion operator, shared by four arms.

```
X      = LayerNorm(F)              X' = LayerNorm(P(D))

F_sa   = Attn(Q,  K,  V )          all three projected from X
F_ca   = Attn(Q', K', V')          Q' from X;  K', V' from X'

alpha  = sigmoid(alpha_logit)
guided = project_out( F_sa + alpha · F_ca ) + F
```

Each projection is a 1×1 convolution followed by a 3×3 depthwise convolution,
matching the backbone's own MDTA convention with the bias flag threaded from the
same configuration key, so the module cannot drift from the trunk's style.

The attention is **transposed**. Tensors are reshaped to `[B, 6, 64, N]` where N
is the token count, Q and K are L2-normalised **along the token axis**, and the
product contracts over tokens to give an attention matrix of shape
`[B, 6, 64, 64]` — channel by channel, and **independent of N**. It is 64×64 at
both 256 and 1024 tokens. This is exactly the property the spatial arm lacked,
and it is verified rather than asserted: the shape function is called at both
scales in the smoke test while the grid demonstrably changes.

Separate per-head temperatures govern the two attentions, since they operate
over different key spaces. `project_out` is zero-initialised so step 0 still
equals the baseline. `alpha_logit` starts at −2.0, giving alpha ≈ 0.119: the
prior pathway opens quietly and, being a gate, **can close again**, which would
constitute a clean interpretable negative result rather than a collapse.

The block costs **1,054,477 parameters** — 452,742 for the self-attention half,
452,742 for the cross half, and 148,993 shared between them.

In the trained dinolight model, `alpha_logit` reaches −1.8470, giving
alpha = 0.1362 — the gate opened rather than closing.

### 6.10 The ladder that separates operator from depth

dinolight-render changes **two things at once** versus addition-render: the
number of depths and the fusion operator. It can therefore answer "does the
published method transfer" but never "which factor earned the gain". Four arms
resolve this. **All four import the identical `DinoAca` block — imported, never
copied — so only the layer set changes.**

| arm | depths | AFFM | Δparams vs baseline |
|---|---|---|---|
| aca-L6 | {6} | none | 1,349,773 |
| aca-L36 | {3,6} | 2 convs | 1,351,311 |
| aca-L6912 | {6,9,12} | 3 convs | 1,352,080 |
| dinolight-render | {3,6,9,12} | 4 convs | 1,352,849 |

aca-L6 carries **no AFFM at all**, deliberately: a softmax over a single layer is
identically 1.0, so a degenerate module would contribute 769 dead parameters and
a meaningless observation series. The smoke test asserts its absence and that no
layer-weight keys appear during training.

The spread across the ladder is **0.227%**, so a difference *within* it is not
attributable to capacity. Against addition-render, however, every ACA arm is
approximately **4.6×** the parameters, so *that* comparison is capacity-
confounded. These two facts must never be conflated.

**aca-L6 is the decisive arm.** It differs from addition-render in exactly one
respect: same block 6, same centring mean, same injection point, same seed, same
schedule — only addition becomes gated channel cross-attention.

```
guided = project_out( F_sa + alpha · F_ca ) + F      instead of   F + P(D)
```

**Test: +0.030 dB against addition-render, p=0.22 — not significant.
Validation: +0.075, p=0.075 — also not significant.**

**Finding 6: the fusion operator buys nothing.** Two independent splits, the
same verdict, at 4.6 times the parameter cost. The benefit that dinolight-render
shows comes from the depth count, not from the operator.

The full ladder gives +0.030, +0.115, −0.010 and +0.309 for one, two, three and
four depths. The endpoints separate clearly, but the intermediate points do not
order cleanly — the three-depth arm falls *below* the two-depth arm and the
confidence intervals overlap heavily. **No monotonic curve should be drawn
through these four points.** The defensible statement is the endpoint
comparison, with the intermediates reported as unresolved.

### 6.11 aca-L6-nosa — an ablation that could not run

Within the ACA equation, `F_ca` is the point of the block. `F_sa` is the feature
attending to *itself* — and that is the backbone's own MDTA, step for step: the
same projection style, the same L2 normalisation along tokens, the same per-head
multiplicative temperature, the same channel-by-channel softmax. The block's
output is handed directly to the latent stage, **eight transformer blocks that
each already perform that operation**. `F_sa` is therefore a ninth MDTA
immediately in front of eight more, with its own separately trained weights, at
452,742 parameters — 43% of the fusion block — against the 4,801,600 parameters
of MDTA attention already present in the eight blocks it feeds.

The trained model's own instrumentation quantifies its contribution. The logged
ratio ‖alpha·F_ca‖ / ‖F_sa‖ rises from 0 — both alpha and P are zero-initialised,
so F_ca begins at exactly zero — to a mean of **8.157 over the final 100,000
iterations**. The prior pathway ends up carrying roughly eight times the
magnitude of the self-attention pathway.

An arm was therefore built with the equation reduced to:

```
guided = project_out( alpha · F_ca ) + F
```

**It cannot train.** With P and `project_out` both zero-initialised, P(D) is
zero, so the value projection of the prior is zero, so F_ca is zero, so
`project_out` receives a zero input — and a convolution's weight gradient is
proportional to its input. `project_out` therefore receives **exactly zero
gradient** and never leaves zero, which in turn starves every parameter upstream
of it. Measured on a real backward pass:

```
gradient reaching project_out:    with F_sa   1.754
                                  without     0.000
```

In the working arms, `F_sa` is computed from the features rather than from the
zero prior, so it supplies `project_out` with a non-zero input on step 1 and the
cross pathway can begin learning on step 2 — the three-step gradient staircase
the architecture documents.

The 6,000-iteration integration test detected this immediately: the projected
norm, injection ratio and injected-delta were all exactly 0.0 at every print,
the attention entropy was pinned at ln(64) — perfectly uniform — and alpha moved
only by weight decay. **The arm was never submitted for a full run; the cost was
49 minutes rather than 38 hours.** The architectural test suite had passed 29/29,
because it verifies that step 0 equals the baseline — which is true. The arm
simply never leaves step 0.

**This is a finding, not merely a bug: `F_sa` is load-bearing for optimisation,
not only for representation.** It is what breaks the double zero-initialisation
deadlock. A naive removal ablation is impossible under this initialisation
scheme. The available fix — initialising P normally while keeping `project_out`
at zero — preserves the step-0 identity but makes the resulting arm differ from
aca-L6 in two respects rather than one, so it would no longer be a one-factor
ablation.

Two things must not be written. **Not** "F_sa does nothing" — the ablation never
ran, and the magnitude ratio is a magnitude, not a causal result; `F_sa` is also
the baseline onto which the gated cross term is summed before a single shared
output convolution. And **not** that DINOLight is implemented wrongly — the arm
reproduces a published block faithfully, and the redundancy is a property of
that design on this backbone.

---

## 7. Results

### 7.1 The test split

All figures are the locked test split, n=338, uint16 path, checkpoints selected
on validation alone.

| arm | depths | full256 | vs E0 | crop128 | vs E0 | mask PSNR | SSIM | HF |
|---|---|---|---|---|---|---|---|---|
| E0-fixed | — | 21.873 | — | 19.546 | — | 17.599 | 0.7829 | 0.218 |
| E1-addition-noisy | 1 | 21.296 | −0.577 | 19.062 | −0.484 | 17.210 | 0.7622 | 0.275 |
| global-render | 1 | 20.589 | −1.284 | 19.352 | −0.194 | 16.947 | 0.7327 | 0.265 |
| addition-render | 1 | 24.081 | +2.208 | 22.259 | +2.713 | 19.673 | 0.8220 | 0.328 |
| concat-render | 1 | 24.065 | +2.193 | 22.473 | +2.927 | 19.741 | 0.8227 | 0.321 |
| aca-L6 | 1 | 24.111 | +2.238 | 21.955 | +2.409 | 19.761 | 0.8211 | 0.266 |
| aca-L36 | 2 | 24.196 | +2.323 | 22.128 | +2.583 | 19.895 | 0.8238 | 0.258 |
| aca-L6912 | 3 | 24.071 | +2.199 | 21.870 | +2.324 | 19.648 | 0.8215 | 0.250 |
| dinolight-render | 4 | **24.390** | +2.517 | 22.053 | +2.507 | **19.972** | 0.8276 | 0.277 |
| affm-render | 4 | 24.311 | +2.439 | **22.303** | +2.758 | 19.923 | **0.8279** | 0.308 |

### 7.2 Paired comparisons against addition-render

Paired per-image, n=338, Wilcoxon signed-rank.

| arm | test full256 | p | test crop128 | p |
|---|---|---|---|---|
| concat-render | −0.016 | 0.98 | +0.214 | 1.7×10⁻⁴ |
| aca-L6 | +0.030 | **0.22** | −0.304 | 1.3×10⁻⁶ |
| aca-L36 | +0.115 | 0.011 | −0.130 | 0.032 |
| aca-L6912 | −0.010 | 0.88 | −0.389 | 4.3×10⁻⁸ |
| dinolight-render | +0.309 | 9.9×10⁻⁸ | −0.206 | 0.0054 |
| affm-render | +0.230 | 4.3×10⁻⁷ | +0.044 | 0.14 |

### 7.3 Finding 7 — the protocol split

**Every attention-based arm is significantly worse than addition-render on
crop128, including the arm with the best full256 figure.** The four ACA arms
read −0.304, −0.130, −0.389 and −0.206, all significant. They gain on full
frames and lose on matched crops.

**affm-render is the only multi-depth arm that loses nowhere**: +0.230 on
full256 and +0.044, not significant, on crop128.

This is the single most important reporting rule to come out of the study.
Presenting full256 alone would show the attention arms as the strongest;
presenting crop128 alone would show them as harmful. Both are reported for every
arm, always.

A plausible mechanism: the channel-attention arms are computing statistics
contracted over the token axis, and the token count differs by a factor of four
between the two protocols. Their matrix *shape* is invariant — that was verified
— but the *statistics* filling it are pooled over four times as many positions at
full resolution. Invariant shape is not the same as invariant content. The
additive arms have no such pooled statistic and show no such split.

That mechanism rests on a premise which had never been checked: that the prior
*itself* is not the same object in the two protocols. Section 7.4 measures it
directly.

---

### 7.4 The prior under cropping — a direct measurement

Every arm in this study is trained on 128-pixel crops, where DINO sees a
224-pixel image and returns a 16×16 token grid, and evaluated on full 256-pixel
frames, where DINO sees 448 pixels and returns 32×32. The scale-matching
constraint of §4.2 guarantees the grid *shape* lines up with the latent in both
regimes. It guarantees nothing about the *content* of those features. The
question left open is whether DINO describes a given region of the object the
same way when that region is a crop as when it is part of the whole frame.

**It does not.** The measurement is on the render stream, n=339 validation
images, with a token-aligned crop manifest drawn specifically for it — the
Phase-3 evaluation manifest is not token-aligned (only 5 of its 339 crops have
coordinates divisible by 8) and was deliberately not reused.

#### The interaction the reference work predicts

The quantity of interest is not how much the features move — features move for
many uninteresting reasons — but whether the *degraded-versus-clean agreement*
gets worse inside a crop than it is in the full frame. Writing `deg` for the
cosine between a degraded image's features and the clean image's features at the
same location, the interaction is `deg_crop − deg_full`:

| depth | mean Δ | 95% CI | Wilcoxon p | worse in crop |
|---|---|---|---|---|
| B3 | −0.0192 | [−0.0201, −0.0184] | 2.8×10⁻⁵⁷ | 337/339 |
| **B6** | **−0.0020** | [−0.0032, −0.0008] | 3.3×10⁻⁴ | **193/339** |
| B9 | −0.0180 | [−0.0204, −0.0156] | 6.8×10⁻³³ | 269/339 |
| B12 | −0.0284 | [−0.0345, −0.0223] | 1.0×10⁻¹⁵ | 225/339 |

Negative and significant at every depth, and largest at the deepest — which is
the shape DSGIR reports for natural images, reproduced here on radar renders.
A crop-ratio sweep reproduces their Figure 10 as well: the cosine declines
monotonically from full frame through ratio 0.8 to ratio 0.5 at every layer, and
the gap widens with depth, from 0.053 at B1 to 0.143 at B12.

> **The 0.2 crop ratio reverses, and is reported rather than dropped.** At ratio
> 0.2 the source is a 51-pixel region upsampled 4.4× to reach 224, which smooths
> the degradation away until both images look alike and the cosine rises again.
> That is an artifact of small source images, not a contradiction of the trend.
> The 1.0–0.5 range is the range this project's training actually occupies, and
> is what the claim rests on.

#### Where the drift is, spatially

Measured on this project's real pipeline — full frame at 448, crop at 224,
token-aligned because 256/32 = 8 pixels per token:

| depth | position cosine | border − centre | NN top-1 |
|---|---|---|---|
| B1 | 0.7987 | −0.0639 | 0.123 |
| B3 | 0.7591 | −0.0549 | 0.106 |
| B6 | 0.7081 | −0.0578 | 0.111 |
| B9 | 0.6568 | −0.1373 | 0.114 |
| B12 | 0.6721 | −0.1261 | 0.156 |

**Border minus centre is negative at every layer.** The drift is not spread
uniformly over the crop; it concentrates at the crop's edges, where the
surrounding context has been cut away. DSGIR asserts this mechanism — "absence
of global contextual support" — as an explanation. This measures it, spatially,
which goes beyond what that paper establishes.

Two further readings. Nearest-neighbour position retrieval — can a crop token
identify which full-frame token it corresponds to — runs 0.045 to 0.156 against
a chance rate of 1/1024 ≈ 0.001. That is roughly a hundred times chance, and yet
**85–95% of crop tokens still cannot locate themselves.** And a logistic
regression on the raw 768-dimensional features separates crop from full at
0.97–1.00 at image level and 0.78–0.99 at patch level, surviving centring: the
two regimes are not merely shifted, they are linearly distinguishable.

**Centring does not repair this.** Raw crop-versus-full cosine is 0.66–0.76,
which sounds tolerable, but falls to 0.48–0.55 once the layer mean is removed.
The raw figure is inflated by the large shared mean that §4.3 removes anyway, so
the true representational shift is *larger* than the raw numbers suggest, and the
regime-specific centring the arms already perform does not compensate for it.

#### B6 is the most crop-robust depth, on three independent measurements

1. its interaction is −0.0020, against −0.018 to −0.028 at every other depth
2. its context separability is 0.859, the lowest of the seven layers measured
3. Phase 1/2 selected it for an unrelated reason — cross-source consistency
   between the radar regimes, measured before any of this existed

The third is what makes the first two worth stating. A layer chosen on one
criterion turns out to be the most transfer-robust on a different criterion
measured later, and it is also the depth carrying the tightest learned weight in
`affm-render`'s per-position softmax. **The Phase-1/2 layer choice is
independently vindicated rather than merely reused.**

#### What this settles, and what it does not

It settles that the prior is genuinely a different object in the two protocols.
That was previously an assumption and is now a measurement, which upgrades §7.3's
mechanism from speculation to a grounded account.

**It does not, on its own, explain Finding 7, and should not be presented as
doing so.** Every arm receives the same shifted prior, including the additive
arms that show no protocol split at all. The measurement establishes that a
regime shift exists and is large; the step from there to "which fusion operators
are damaged by it" remains the interpretation offered in §7.3 — that pooling
statistics across the token axis is more exposed to a context-dependent prior
than a per-position addition is. That step is reasonable and is not tested here.

---

### 7.5 Where the gain actually comes from

Aggregate PSNR hides the distribution, and the per-image behaviour is not
uniform in a way that matters for interpretation.

Against the baseline, addition-render improves **298 of 338** images, or 88.2%.
That is close to a uniform benefit and is why the source finding is robust: it
is not driven by a subset.

The later comparisons are quite different. Against addition-render, affm-render
is better on **219 of 339** validation images (64.6%) and dinolight-render on
**198 of 339** (58.4%). So affm's mean advantage does not come from winning
nearly everywhere — it comes from winning by more when it wins. A method that
improves the mean while losing on a third of images is a different claim from
one that improves nearly all of them, and the two should not be described in the
same language.

Inspecting matched panels drawn from the baseline's own performance ranking —
so the selection cannot be biased by any arm's score — shows two consistent
patterns.

**On the hardest images, the multi-depth arms separate most.** On the image the
baseline does worst on, dinolight-render gains 6.80 dB over it and 1.5 dB over
addition-render, reconstructing concentric fringe structure that the simpler
arms smooth away. This is consistent with its high-frequency ratio and with the
intuition that a prior helps most where the evidence in the input is weakest.

**On the easiest images, every DINO arm slightly underperforms the baseline.**
On the image the baseline does best on, all four DINO arms are negative against
it, by 0.20 to 1.07 dB. Where the input is already clean, the injected prior is
a small perturbation to a solution that was nearly correct, and it costs a
little. This is a real and reportable characteristic: the prior buys the most
where the input is worst and costs slightly where the input is already good.

Together these explain the shape of the aggregate numbers. The source and
spatial findings are large and near-universal. The depth finding is smaller,
statistically solid, and concentrated in the harder part of the distribution.
The operator finding is a null in the mean and, on the matched-crop protocol,
a consistent small loss.

---

## 8. Synthesis

Holding the injection point, the prior and the backbone fixed and varying only
the fusion equation:

```
F + P(D)                              baseline for this comparison
fuse(cat([F, D]))                     −0.016 dB    p = 0.98
project_out(F_sa + α·F_ca) + F        +0.030 dB    p = 0.22,  4.6× parameters
```

No reliable improvement. A strictly more general linear fusion is a null, and a
gated channel cross-attention at matched depth is a null at 4.6 times the cost.

Holding the fusion equation fixed at `F + P(D)` and varying only the
construction of D:

```
one layer                             baseline for this comparison
per-position softmax over four        +0.230 dB    p = 4.3e-07,  +1.04% parameters
```

**The variable that matters is the construction of the prior, not the operator
that injects it.** Within that construction, two factors dominate — the source
image, which decides the sign of the entire effect, and spatial structure,
without which the prior is worse than useless — while depth count provides a
smaller but statistically solid additional gain at negligible cost.

The capacity question deserves a direct answer. The concern is that the
attention arms underperform because they are 4.6× larger and this dataset has
6,101 training images. aca-L6 addresses it: a 4.6× arm at matched depth lands
statistically level with the additive arm, slightly positive, with a confidence
interval comfortably spanning zero. Had the additional capacity been actively
harmful at this data scale, aca-L6 would sit significantly *below*
addition-render. It does not. The extra capacity appears roughly neutral —
neither rescuing nor sinking the attention arms. A dedicated parameter-matched
control with no new mechanism would be cleaner still, and was not run.

---

## 9. Limitations

**Single seed.** Each arm was trained once at seed 100. Between-run variance is
not measured, and differences below roughly 0.4 dB should be read with that in
mind — the baseline moves 0.41 dB between adjacent validation points *within* a
single run. The substitute available is cross-split replication: the depth
finding reads +0.284 on validation and +0.230 on test, and the operator null
reads +0.075 and +0.030, so both survive an independently held-out split.

**Multi-pass test reading.** The pre-registration called for the test split to be
read once for all arms together. In practice it was read in several passes as
arms completed. What survived intact is the property that matters: **checkpoint
selection never touched the test split** — every arm was selected on validation
alone, verified for all eleven evaluated arms. The test split never informed a
modelling decision. A single clean read should not be claimed.

**Capacity confound, unevenly.** affm-render at +1.04% is not
capacity-explained. Every ACA arm at ~4.6× is confounded against
addition-render, though not against each other, where the spread is 0.227%.
These must never be averaged over.

**Coarse checkpoint selection.** Validation every 4,000 iterations against
checkpoints every 2,000 means roughly half the saved checkpoints were never
scored.

**One injection point.** The latent input was chosen by architectural argument
and the code hard-refuses alternatives. There is no empirical comparison against
injecting after the latent stage.

**Frozen prior throughout.** No fine-tuning ablation was run.

**A train/evaluate regime mismatch in the prior, measured but not corrected.**
Every arm is trained on 128-pixel crops and evaluated on full 256-pixel frames,
so the prior is computed in one context and used in another. Section 7.4
measures the size of that mismatch and finds it substantial. No arm was trained
to be robust to it — DSGIR's hybrid crop/full preprocessing would be the obvious
remedy and is not applied here. Every arm in the study carries this equally, so
it does not confound the comparisons between them, but it does mean the absolute
full256 figures are obtained slightly out of the regime the priors were
conditioned in.

**No no-DINO render control.** The use of a DINO prior is a premise of this
work, so no arm supplies the render as a plain input channel without DINO. The
consequence, stated plainly: this study establishes that *the render, via DINO,*
helps — not that DINO specifically is necessary to exploit it.

**Narrow domain.** One dataset, one degradation level, one object class, one
backbone. No generalisation claim is available.

**One arm abandoned.** priorquery-render — the reference papers' attention
direction, with the prior as query and the feature as key and value, making DINO
a router over radar positions rather than a content source — was stopped at
90,000 iterations and never evaluated. It contributes nothing and is excluded.

**An unimplemented monitoring fix.** The stability gate constrains only the
injected-to-latent ratio and is blind to co-inflation of both quantities, as
the failed cross-attention arm demonstrated. A co-inflation rule was specified
and never implemented.

---

## 10. Conclusion

A frozen visual foundation prior substantially improves holographic radar
restoration — 2.208 dB over an identically trained baseline, improving 88.2% of
test images, with the high-frequency energy ratio rising from 0.218 to 0.328 and
the Laplacian ratio from 0.286 to 0.438, so the over-smoothing that motivated
the work is measurably reduced.

Whether it helps at all is decided by the source image: a prior computed on the
degraded input is worse than no prior, at −0.577 dB. Its value is spatial:
pooling the feature grid to a global descriptor drops performance 1.284 dB below
baseline, and shuffling renders between images costs 9.094 dB.

Given a well-constructed prior, **the operator that injects it does not matter.**
A strictly more general concatenation is a null. A gated channel cross-attention
at matched depth is a null at 4.6× the parameters, and every attention-based arm
degrades significantly under matched-crop evaluation. What does matter is
reading the prior at multiple network depths and combining them per position,
which yields +0.230 dB for a 1.04% parameter increase and is the only multi-depth
configuration that loses on neither protocol.

Along the way, two negative results with diagnosed mechanisms: spatial
cross-attention over a prior grid does not survive the fourfold token-count
change between training and evaluation, and the self-attention branch of the
published fusion block turns out to be load-bearing for optimisation — it breaks
a double zero-initialisation deadlock — independently of whatever it contributes
representationally.

Underneath the protocol split sits a property of the prior itself, measured
directly rather than assumed: DINO does not describe a crop the way it describes
the same region of the full frame. The degraded-versus-clean agreement is
significantly worse inside a crop at every depth tested, and the drift
concentrates at the crop's borders, where surrounding context has been removed.
The depth this project selected in an earlier phase, on an unrelated criterion,
turns out to be the most crop-robust of those measured — an independent
confirmation of a choice made for other reasons.

---

## 11. What would strengthen this, and what was declined

This section separates work that is genuinely open from work that was
*considered and declined*. The distinction matters: an unstated omission reads
as an oversight, while a stated decision reads as a decision. The items in the
second list are not gaps in the study.

### 11.1 Open

**A parameter-matched control with no new mechanism.** The single largest
remaining hole. An arm at the attention arms' parameter count that adds no
attention, no gate and no new operator — for instance a two-layer projection
with a hidden width sized to match — would isolate capacity from mechanism
directly. aca-L6 provides partial evidence that capacity is neutral here, but a
dedicated control would settle it rather than infer it.

**A middle point on the additive depth curve.** The operator that carries the
result has only two points, one depth and four. An arm at two or three depths
with plain addition would cost a few thousand parameters and would establish
whether the relationship is graded or a threshold. The equivalent ladder for the
attention operator exists but is noisy and, given Finding 6, describes an
operator that does not matter.

**Training at the evaluation resolution.** Section 7.4 establishes that DINO's
description of a crop differs systematically from its description of the same
region in the full frame. The clean fix is to train where the model is
evaluated, or to mix both resolutions during training, so that the radar stream
and the prior stream move to the new scale together.

> **DSGIR's hybrid preprocessing does not transfer directly, and the difference
> matters.** They draw each iteration from either a full image resized to 224 or
> a random 224 crop, and this is safe *for them* because their content prior is
> pooled to a global vector — there is no spatial correspondence to break. This
> project's prior is a token grid that must align with the latent grid position
> by position. Feeding DINO a full frame while the backbone receives a 128 crop
> would leave the prior describing one region and the latent another, and §7.6
> measures that even a half-token misregistration erases the prior's entire
> benefit. The transferable idea is mixed-resolution training of *both* streams,
> not hybrid preprocessing of the prior alone.

**Implementing the co-inflation gate rule.** The stability monitor is blind to
simultaneous growth of the latent and injected norms, which is exactly the
failure mode the cross-attention arm exhibited. The rule was specified and never
implemented; any future arm inherits the blind spot.

### 11.2 Considered and declined, with the reason

**Seed replication.** Each arm was trained once, at a single seed, and this is a
scope decision rather than an omission — single-run reporting is the norm in
comparable theses, and the compute was spent on additional one-factor arms
instead. The substitute is cross-split replication, which the study relies on
throughout: the depth finding reads +0.284 on validation and +0.230 on test, and
the operator null reads +0.075 and +0.030. This is not identical to a seed
repeat — an independent split tests generalisation of a *fixed* model, while a
second seed tests stability of the *training procedure* — and §9 states the
limitation plainly. It is recorded here so the choice is visible.

**The F_sa ablation.** Attempted, and it could not be run as a one-factor
ablation: under double zero-initialisation the arm receives exactly zero
gradient and never leaves step 0 (§6.11). The fix — initialising the projection
normally while keeping the output convolution at zero — preserves the step-0
identity but makes the arm differ from aca-L6 in two ways, at which point it no
longer answers the question it was built for. Given Finding 6, the operator it
belongs to has already been shown not to matter, so the value of resolving it is
low. The attempt is reported as a finding in its own right rather than left as
an open task.

**More arms on the attention operator.** A ladder point at depths {3,6,9} would
add a fourth noisy point to a curve describing an operator that Finding 6 shows
is inert. It would cost roughly 38 hours and could not change a conclusion.

**A no-DINO render control.** Out of scope by premise: the use of a DINO prior
was specified for this work, so no arm supplies the render as a plain input
channel. It is the first question a reader outside that premise will ask, and
the honest answer is that it was not tested — see §9.
