# PROSE ARGUMENTS — Phase 3, written to be copy-pasted

Plain paragraphs, each self-contained with its own evidence, so any single one
can be pasted without the rest. Written 2026-08-31, after the test split was
read for every arm.

**Numbers of record live in `DEVLOG.md` (Steps 29-33) and `THESIS_STORY.md`.**
If this file and those disagree, THOSE are right and this one is stale.
Test split n=338, validation n=339, PSNR on the uint16 path throughout.

---

## PART 1 — THE ARGUMENTS

### The setup

My baseline is a plain Restormer trained on fixed 128-pixel crops with no DINO
at all, scoring 21.873 dB on a locked test split of 338 images. Every
experimental arm inherits that exact recipe — same architecture, same seed, same
schedule, same crop, same optimiser — and differs only in the DINO branch
attached at one point: the 384-channel tensor entering the eight latent
transformer blocks. That shared recipe is what makes the comparisons
one-factor. My baseline does score 0.53 dB below an older progressive-training
baseline, because it never trains at the resolution it is evaluated at, but this
was registered in advance and does not weaken the comparison, which is internal.
For the record, my best arm also beats that older, stronger baseline by
1.676 dB.

### The source of the prior decides whether it helps or hurts

I ran two arms that are identical in every respect — same parameter count, same
seed, same schedule, same injection point — differing only in which image
DINOv2 reads. When DINO reads the noisy radar input, the model scores 21.296 dB,
which is 0.577 dB *worse* than using no prior at all. When DINO reads an aligned
rendered image instead, the same architecture scores 24.081 dB, an improvement
of 2.208 dB over baseline, better on 298 of 338 images. A foundation-model prior
computed on degraded input is actively harmful; the identical mechanism becomes
strongly beneficial when the prior is computed on a clean modality. This is a
controlled result on a locked test split.

### The prior's value is spatial, not semantic

To test whether the gain comes from DINO's spatial layout or merely from a
global "what is in this image" signal, I pooled the DINO feature grid to a
single 768-dimensional vector and broadcast it back to every position. Pooling
is a mean and a broadcast, so this arm has exactly the same parameter count as
the additive arm — the only difference is the presence or absence of spatial
variation. It scores 20.589 dB, which is 1.284 dB *below* the no-DINO baseline.
Destroying the layout while keeping the content makes the prior worse than
useless. The claim is not that the model learned what chairs look like; it is
that the prior tells it *where*.

### The model genuinely uses each image's specific prior

I ran a mismatched-render control on the trained model, using a Sattolo
derangement so that every image receives a different image's render, with the
derangement asserted programmatically. Performance drops by 9.094 dB on average,
is worse on 335 of 339 images, and falls below the baseline's mean on 322 of
them. The model is not exploiting a generic template or a dataset-level prior;
it is using the per-image correspondence.

### Concatenation buys nothing over addition

I replaced the additive injection with a concatenation followed by a learned 1x1
convolution mapping 1152 to 384 channels — strictly more general, since it can
transform the feature tensor as well as the prior, where addition leaves the
feature untouched. Paired per-image over 338 images it scores -0.016 dB against
addition with p=0.98, and on validation it scores +0.023 dB with p=0.49. It
lands on both sides of zero across two splits, which is exactly what a genuine
null looks like. I report this deliberately: a method that finds nulls where
nulls exist lends credibility to the differences it does detect.

### Spatial cross-attention does not survive the train-to-evaluation scale change

I built an arm where the radar feature queries the DINO grid through a standard
spatial softmax attention. From identical weights it scores +2.00 dB on matched
128-pixel crops and -7.56 dB on full 256-pixel frames. The mechanism is
diagnosed: the DINO patch grid is 16x16 (256 tokens) during training and 32x32
(1024 tokens) at evaluation, so a spatial softmax must renormalise over four
times as many competitors and the operation changes shape between train and
test. I do not report the +2.00 dB as a result, because checkpoint selection
uses the full-256 validation metric — the one regime this arm cannot do — and
therefore selected iteration 4,000 out of 179,000. The honest statement is that
the selection metric and in-distribution performance are anti-correlated for
this arm. This is an original negative result rather than a failed reproduction:
I verified from the arXiv HTML that Perceive-IR's prior is a single 1x768 global
vector with no spatial tokens to attend over, so no paper in my reference set
performs spatial-token cross-attention between a prior grid and a feature grid.

### Reading DINO at multiple depths improves restoration, at essentially no parameter cost

Instead of one transformer block, I read blocks 3, 6, 9 and 12, centre each with
its own training-set mean, and combine them with a softmax that runs across
*layers* independently at each of the 256 grid positions. Because the output is
a weighted sum rather than a concatenation, the tensor stays 768 channels and
the downstream projection is unchanged, so the arm costs 3,076 additional
parameters — an increase of 1.04% over the single-layer additive arm. It scores
+0.230 dB against that arm on test (p=4.3e-07) and +0.284 dB on validation
(p=1.0e-09). The result replicates across two independent splits with the same
sign and magnitude, and because the parameter increase is 1%, the gain cannot be
attributed to capacity.

### The learned depth weighting is genuinely spatial and discards nothing

Dumping the per-position weight maps at 300,000 iterations shows a standard
deviation across positions of 0.10 to 0.16 against means of 0.20 to 0.30, so the
model really is selecting different depths at different locations rather than
applying a flat global mix. No depth is ever switched off: the minimum weight
any layer reaches across the entire run is 0.1028. Block 6 — the depth my
earlier layer study identified as the strongest single choice — carries the most
stable weight of the four, with roughly half the variance of the shallowest and
deepest. The weights are non-stationary over training, so I report a windowed
mean and state the window rather than claiming a strict ordering.

### At matched depth, the published attention operator provides no measurable benefit

DINOLight's method changes two things at once versus my additive arm — the
number of DINO depths *and* the fusion operator — so it cannot say which factor
earned its gain. I therefore built the same gated channel cross-attention block
reading a single depth, block 6, holding everything else identical to the
additive arm: same layer, same centring mean, same injection point, same seed.
It scores +0.030 dB with p=0.22, statistically indistinguishable from plain
addition, while costing approximately 4.6 times the parameters. This replicates
on validation at +0.075 dB with p=0.075, also null. Two independent splits, the
same verdict. The benefit of the published method comes from the depth count,
not from the fusion operator.

### Every attention-based arm degrades on the matched-crop protocol

On full 256-pixel frames the attention arms range from -0.010 to +0.309 dB
against the additive baseline. On matched 128-pixel crops, all four are
significantly *worse*: -0.304 (p=1.3e-06), -0.130 (p=0.032), -0.389 (p=4.3e-08)
and -0.206 (p=0.0054), including the arm with the best full-frame number. The
multi-depth additive arm is the only one that loses nowhere, at +0.230 on full
frames and +0.044 (not significant) on crops. I report both protocols for every
arm, because the conclusion differs between them.

### The self-attention branch duplicates the backbone, and removing it is not straightforward

The block computes a self-attention on the feature and a gated cross-attention
to the prior, sums them, and passes the result through one output convolution.
The self-attention half is the same transposed channel attention the backbone's
own transformer blocks perform, and the block's output feeds eight such blocks
directly. It costs 452,742 parameters, 43% of the fusion block, while the
training logs show the cross-attention term carrying 8.157 times its magnitude
averaged over the final 100,000 iterations. I built an ablation removing it and
found it cannot train: with both the prior projection and the output convolution
zero-initialised, the cross branch outputs zero, and since a convolution's
weight gradient is proportional to its input, the output convolution receives
exactly zero gradient and never leaves zero. Measured on a real backward pass,
the gradient reaching that convolution is 1.754 with the self-attention branch
present and 0.000 without it. The self-attention branch is therefore
load-bearing for optimisation, not only for representation — it is what breaks
the double zero-initialisation deadlock. I report this as a finding about the
initialisation scheme, not as evidence that the branch is functionally useless,
because the ablation never ran.

### Limitations I state up front

Each arm was trained once at a single seed, so between-run variance is not
measured and differences below roughly 0.4 dB should be read with that in mind;
my two main findings each replicate across independent validation and test
splits, which is the substitute I have. The test split was read in several
passes as arms completed rather than in a single pass, but checkpoint selection
was performed on validation alone throughout for every arm, so the test split
never informed a modelling decision. The attention arms carry roughly 4.6 times
the parameters of the additive baseline, so comparisons involving them are
capacity-confounded; the multi-depth additive arm at +1% is not. Checkpoints
were saved every 2,000 iterations while validation ran every 4,000, so selection
searched roughly half the saved checkpoints. The study covers one dataset, one
degradation level and one object class, and DINOv2 was frozen throughout, so no
claim of generalisation beyond this setting is made.

---

## PART 2 — THE SAME STORY WITH THE EQUATIONS AND THE ATTACHMENT POINT

### Where everything attaches, and why there

The backbone is Restormer, a four-level U-Net of transformer blocks with base
width 48 and block counts 4, 6, 6, 8 in the encoder and latent stages. A
single-channel radar image passes through a 3x3 patch embedding and three
encoder levels, doubling in width and halving in resolution at each step, giving
tensors of 48 channels at full resolution, 96 at half, 192 at quarter, and
finally 384 channels at one-eighth resolution. That last tensor, produced by the
final downsampling and written `inp_enc_level4` in the code, is the injection
point. I call it F. It has shape [B, 384, 16, 16] when training on 128-pixel
crops and [B, 384, 32, 32] when evaluating on full 256-pixel frames. Every arm
in this study modifies exactly this one tensor and nothing else: encoder,
decoder, skip connections, refinement blocks, the output convolution and the
global residual against the radar are byte-identical across all arms including
the baseline. I inject before the eight latent transformer blocks rather than
after them for two reasons. First, those eight blocks hold 55% of the network's
parameters — 1.8 million each against 31 thousand for a first-level block — so
anything injected after them bypasses the majority of the model's capacity and
reaches only the decoder. Second, each block is a residual of the form
x = x + attn(norm(x)) followed by x = x + ffn(norm(x)), so a signal added at the
input rides the residual stream through all eight blocks and is still present at
the output, in addition to being processed by them. Injecting before therefore
strictly contains what injecting after would give. This is a reasoned design
decision, not a tested one; the code hard-refuses any other injection point, so
I have no empirical comparison.

### How the prior is produced, identically in every arm

The dataset supplies a stacked two-channel input: channel 0 is the noisy radar,
channel 1 is an aligned render. The backbone only ever sees channel 0, so the
render reaches the network exclusively through the DINO branch. The render is
resized to 224x224 when training and 448x448 when evaluating,
ImageNet-normalised, and passed through a frozen DINOv2 ViT-B/14. Because the
patch size is 14, 224/14 = 16 and 448/14 = 32, so the patch grid is 16x16 or
32x32 — exactly matching the latent grid in both regimes. This is enforced by an
assertion that throws if the two ever disagree, which means no interpolation of
the feature grid is ever performed. The patch tokens from a chosen block are
centred by subtracting a mean vector computed on the training set only, then
reshaped into a grid D of shape [B, 768, 16, 16]. A 1x1 convolution P maps 768
channels to 384 so the prior can meet the latent; P is initialised to exactly
zero, weight and bias, and costs 768 x 384 + 384 = 295,296 parameters.

### Baseline, E0-fixed

No DINO branch exists. The equation is simply `guided = F`, and the latent stage
receives the untouched encoder output. It scores 21.873 dB on the test split.
Because P is zero-initialised in every DINO arm, all of them produce output
bit-identical to this baseline at step 0, which I verify to 0.000e+00 in a smoke
test. Every gain therefore has to be learned rather than inherited from a
favourable initialisation.

### Additive injection, on the noisy radar and on the render

The equation is `guided = F + P(D)`, where D is the centred DINO grid from block
6, added residually at the injection point. The two arms are identical in
architecture, parameter count, seed and schedule; the only difference is which
image produces D. Reading the noisy radar gives 21.296 dB, which is 0.577 dB
below baseline. Reading the render gives 24.081 dB, an improvement of 2.208 dB,
better on 298 of 338 images. The parameter cost is 295,296 in both cases. This
is the cleanest one-factor comparison in the study: one tensor swapped, sign of
the effect reversed.

### Spatial ablation, global-render

The equation becomes `D_g = broadcast(mean over the g x g positions of D)`
followed by `guided = F + P(D_g)`. The 16x16 grid is averaged into a single
768-dimensional vector and copied back to every position, so every location
receives the identical prior. Pooling is a mean and a broadcast and introduces
no parameters, so this arm has exactly the same 295,296 parameter cost as the
additive render arm — the only difference in the entire model is whether the
prior varies across space. It scores 20.589 dB, which is 1.284 dB below the
no-DINO baseline. Centring and pooling commute, so no new mean statistics were
needed.

### Concatenation, concat-render

The projection P is deleted entirely and replaced by
`guided = fuse(cat([F, D], dim=1))`, where fuse is a 1x1 convolution mapping
1152 channels to 384. Expanded, this is `conv(F, W_F) + conv(D, W_D) + b`, so it
is strictly more general than addition: it can transform the feature tensor as
well as the prior, where addition leaves F untouched. W_F is initialised to
identity and W_D to zero, so step 0 still equals the baseline. It costs 442,752
parameters and scores 24.065 dB, which is -0.016 dB against additive injection
with p=0.98, and +0.023 dB on validation with p=0.49 — a null on both sides of
zero across two splits.

### Spatial cross-attention, crossattn-render

The equation is `attn = softmax(Q K^T / sqrt(64)) V` with Q taken from F and K
and V from D, across six heads of dimension 64, followed by
`guided = F + W_o(attn)`. This is a spatial softmax: the attention matrix is
256x256 during training, one entry per pair of grid positions. It costs 888,576
parameters. From identical weights it scores +2.00 dB on matched 128-pixel crops
and -7.56 dB on full 256-pixel frames, because the grid grows from 256 to 1024
tokens between train and evaluation and the softmax must renormalise over four
times as many competitors. I do not report the +2.00 dB as a result: checkpoint
selection uses the full-256 validation metric, the one regime this arm cannot
do, and therefore selected iteration 4,000 out of 179,000.

### Multi-depth fusion, affm-render

Instead of one block, DINO is read at blocks 3, 6, 9 and 12, each centred with
its own training-set mean. A 1x1 convolution mapping 768 channels to 1 scores
each layer at each position, the four scores are softmaxed across layers, and
the layers are combined as a weighted sum: `s_l = conv_l(gelu(D_l))`,
`w = softmax([s3, s6, s9, s12])` across the layer axis, `D = sum_l w_l * D_l`.
The injection itself is unchanged, `guided = F + P(D)`. Because the combination
is a sum rather than a concatenation, the tensor stays 768 channels wide and P
is identical to the additive arm's, so the entire additional cost is four
scoring convolutions at 769 parameters each, 3,076 in total — an increase of
1.04%. It scores 24.311 dB, +0.230 against additive injection with p=4.3e-07 on
test and +0.284 with p=1.0e-09 on validation.

### Gated channel cross-attention, the ACA block

This is DINOLight's fusion, used by four arms. Writing X = LayerNorm(F) and
X' = LayerNorm(P(D)), the block computes two attentions —
`F_sa = Attn(Q, K, V)` with all three projected from X, and
`F_ca = Attn(Q', K', V')` with Q' from X but K' and V' from X' — then combines
them through a single output convolution: `alpha = sigmoid(alpha_logit)` and
`guided = project_out(F_sa + alpha * F_ca) + F`. Each projection is a 1x1
convolution followed by a 3x3 depthwise convolution, matching the backbone's own
convention. Critically the attention is transposed: the tensors are reshaped to
[B, 6, 64, N] where N is the token count, Q and K are L2-normalised along the
*token* axis, and the matmul contracts over tokens to give an attention matrix
of shape [B, 6, 64, 64] — channel by channel, independent of N. It is 64x64 at
both 256 and 1024 tokens, which is precisely the property the spatial arm
lacked, and I verify it by calling the shape function at both scales in a smoke
test. The block costs 1,054,477 parameters: 452,742 for the self-attention half,
452,742 for the cross half, and 148,993 shared. `project_out` is zero-initialised
so step 0 still equals the baseline; alpha starts at sigmoid(-2.0) ~ 0.119 and
can close again if the prior does not help.

### The ladder that separates operator from depth

All four ACA arms import the same block, never a copy, and differ only in which
DINO layers feed it. aca-L6 reads block 6 alone with no layer-fusion stage,
since a softmax over a single layer is identically 1.0 and would add 769 dead
parameters; it costs 1,349,773 parameters over baseline. aca-L36 reads blocks 3
and 6, aca-L6912 reads 6, 9 and 12, and dinolight-render reads 3, 6, 9 and 12,
each adding one scoring convolution, giving 1,351,311, 1,352,080 and 1,352,849
respectively — a spread of 0.227% across the ladder, so differences within it
are not attributable to capacity. Against the additive arm, however, all four
are roughly 4.6 times the parameters, so that comparison is capacity-confounded.
aca-L6 is the decisive arm: same block 6, same centring mean, same injection
point, same seed as the additive render arm, with only the operator changed. It
scores +0.030 dB with p=0.22 on test and +0.075 with p=0.075 on validation —
null on both. The full ladder gives +0.030, +0.115, -0.010 and +0.309 for one,
two, three and four depths, so the endpoints separate but the intermediate
points do not order cleanly and I report them as unresolved.

### The self-attention branch, and why removing it fails

In the ACA equation `guided = project_out(F_sa + alpha * F_ca) + F`, the term
F_ca is the point of the block — the feature attending to the prior. The term
F_sa is the feature attending to itself, which is the same transposed channel
attention the backbone's own transformer blocks perform, and the block's output
feeds eight of them directly. It costs 452,742 parameters, 43% of the block,
while the logged ratio of ||alpha*F_ca|| to ||F_sa|| rises to a mean of 8.157
over the final 100,000 iterations. I built an arm with the equation reduced to
`guided = project_out(alpha * F_ca) + F` and it cannot train. With both P and
project_out zero-initialised, P(D) is zero, so the value projection of the prior
is zero, so F_ca is zero, so project_out receives a zero input — and a
convolution's weight gradient is proportional to its input, so project_out
receives exactly zero gradient and never leaves zero, which in turn starves
everything upstream of it. Measured on a real backward pass, the gradient
reaching project_out is 1.754 with F_sa present and 0.000 without it. F_sa is
therefore what breaks the double zero-initialisation deadlock, making it
load-bearing for optimisation and not only for representation. The integration
smoke test caught this before any full run was submitted.

### What the equations collectively show

Holding the injection point, the prior and the backbone fixed and varying only
the fusion equation from `F + P(D)` to `fuse(cat([F,D]))` to
`project_out(F_sa + alpha*F_ca) + F` produces no reliable improvement:
concatenation is a null at -0.016 dB, and gated channel cross-attention at
matched depth is a null at +0.030 dB while costing 4.6 times the parameters.
Holding the fusion equation fixed at `F + P(D)` and varying only what D is — one
layer versus a per-position softmax over four — produces +0.230 dB for 3,076
additional parameters. The variable that matters is the construction of D, not
the operator that injects it.
