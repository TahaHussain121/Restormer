# Phase 5 — guide to the crop-versus-full figures

*What each figure means, why that particular figure exists, and how it was
made. Written for a reader who does not want technical language.*

Written 2026-09-02.

> **This file is tracked in git.** The FIGURES it describes are not — they
> live in `results/`, which is gitignored, so a clean checkout has the
> explanations without the pictures. Re-run the three scripts listed at the
> bottom to rebuild every figure. The numbers are also in `DEVLOG.md` Step 34.

---

## Part 1 — The questions you should ask before looking at any of the figures

### Why are we looking at DINO features at all, instead of just the images?

Our model does not receive the render as pixels. It never sees the render. What
it receives is DINO's *description* of the render — a stack of numbers DINO
produces after looking at it.

So when we ask "does the model get good information?", the pixels are not the
thing to check. The thing to check is **what DINO says about those pixels**,
because that is the only part the model ever gets.

An analogy. Imagine you never see a photograph, but a friend describes it to you
over the phone, and you have to draw it. If the drawing comes out wrong, there
are two possible reasons: the photo was bad, or the *description* was bad.
Checking the photo tells you nothing about the description. Phase 5 checks the
description.

### Why do we keep talking about "layers"?

DINO is a stack of 12 processing stages, one after another. It is not one
description — it is twelve, and they say different things.

- **Early layers (1, 3, 4)** stay close to the picture. Edges, textures, light
  and dark. Almost like a filtered version of the image.
- **Middle layers (6, 8)** start describing parts. "This region belongs to one
  object, that region to another."
- **Late layers (9, 12)** are the most abstract. Closer to "this is a chair" than
  to "this is a bright edge".

This matters for us because we can choose which layers to feed the model, and
Phase 3 found that using **four layers instead of one** is worth +0.230 dB. So
"which layer" is a real design decision, not a detail — and any weakness we find
might affect some layers and not others. That is exactly what happened.

### Why cosine similarity, and what counts as "high"?

Every layer describes one small square of the picture with a list of 768
numbers. To ask "are these two descriptions the same?", we compare the two lists
and get a number between −1 and 1. **1 means identical, 0 means unrelated.**

**But you cannot judge a cosine number on its own.** In 768 dimensions, two
unrelated DINO descriptions do not score 0 — they often score 0.4 or 0.5, just
because DINO's outputs all share a common bias. So a score of 0.7 might sound
good and actually be poor.

That is why several figures include a **floor**: the same comparison done
against the *wrong* place in the image. The floor is what "no real agreement"
looks like. **Always read a cosine against its floor, never on its own.**

### Why does cropping matter enough to spend a whole phase on it?

Because our training and our testing do not match.

- **Training:** we cut a small 128-pixel square out of the picture and show DINO
  only that square.
- **Testing:** we show DINO the whole 256-pixel picture.

So the model learns from descriptions written one way, and is then tested on
descriptions written a different way. **Nobody had ever checked whether those two
kinds of description agree.** If they do not, the model is being handed something
subtly different at test time from what it was trained on.

This became urgent because of a strange Phase-3 result: every attention-based
model did *better* on full pictures and *worse* on small crops. We had a guess
about why, but it was labelled a guess in the write-up. Phase 5 replaces the
guess with a measurement.

### Why do these figures come in two different setups?

Two setups, answering two different questions. **Never mix their numbers.**

**Setup A — "sweep".** Every crop size is stretched to the same 224 pixels
before DINO sees it. So the only thing that changes is *how much of the scene is
visible*. This is what the DSGIR paper does, and it is the only way crop sizes
can be fairly compared to one another.

**Setup B — "aligned".** This is our actual pipeline, unchanged: full picture at
448, crop at 224. Here *both* the field of view and the level of detail differ.
Less clean as an experiment, but it describes what really happens in our system.

---

## Part 2 — Each figure, one at a time

---

### `fig10_replica_dsgir_layers.png` and `fig10_replica_our_layers.png`

**The question:** as we show DINO less and less of the picture, does its
description of a damaged image drift away from its description of the clean one?

**Why this figure exists.** This is a direct copy of Figure 10 from the DSGIR
paper, run on our radar data. They claim that cropping makes DINO's descriptions
unreliable. If that holds on our data too, their claim generalises and we can
build on it. If it does not, that is worth knowing before we lean on it.

**How it was made.** For each picture we cut out a square of a chosen size — the
whole thing, 80% of it, 50%, or 20%. We cut *the same square* from both the
damaged version and the clean version, stretch both to 224 pixels, and ask DINO
to describe each. Then we compare the two descriptions. High score means DINO
still sees the same thing despite the damage. Low means the damage has confused
it. Repeat for 339 pictures and average.

**Reading it.** The x-axis is DINO's layer, going deeper to the right. The
y-axis is agreement. Each coloured line is one crop size. Two versions of the
figure exist: one using the layers the DSGIR paper uses (1, 4, 8, 12) so the
comparison is like-for-like, and one using the layers our models actually use
(3, 6, 9, 12).

**What it shows.**

| layer | Full | 0.8 | 0.5 | 0.2 |
|---|---|---|---|---|
| B1 | 0.9556 | 0.9204 | 0.9031 | 0.9561 |
| B4 | 0.9004 | 0.8187 | 0.7856 | 0.8984 |
| B8 | 0.8254 | 0.7625 | 0.7269 | 0.8182 |
| B12 | 0.5512 | 0.4899 | **0.4085** | 0.5151 |

Three things:

1. **Every line slopes downward.** The deeper you go into DINO, the less its
   description of a damaged image resembles the clean one. At the last layer,
   even on the full picture, agreement is only 0.55 — barely better than the
   floor. **DINO's deepest layers are not reliable on this data at all.**
2. **Smaller crops sit lower.** Going from the full picture to 80% to 50%, the
   score drops at every single layer. Cropping does hurt. DSGIR's claim holds.
3. **The gap widens with depth.** Full-minus-50% is 0.053 at layer 1 but
   **0.143 at layer 12** — nearly three times bigger. Cropping hurts the abstract
   layers far more than the simple ones. Also as DSGIR predicted.

**The one thing that does not match, which you must explain rather than hide.**

The 20% crop line jumps back *up*, almost to the full-picture level. That breaks
the pattern. Here is why, and it is not a contradiction:

20% of a 256-pixel picture is a 51-pixel square. We then blow that up to 224
pixels — more than four times bigger. Enlarging a tiny square that much smears
everything into a blur. And once both the damaged and the clean version are
blurred into mush, they naturally look alike again. **The score goes up because
both images lost their detail, not because DINO is doing well.**

DSGIR does not hit this because their source pictures are much larger, so their
20% crop is still a decent-sized image.

**What to say:** report the range from full to 50%, which is the range our
training actually uses, and state the 20% artefact openly. Hiding it would be
worse than explaining it.

---

### `fig12_replica_kde.png`

**The question:** the previous figure showed averages. Are those averages hiding
anything?

**Why this figure exists.** An average can lie. Two very different situations
give the same average: everything shifting slightly, or most things staying put
while a few collapse badly. This is Figure 12 from the DSGIR paper, and it shows
the full spread instead of one number.

**How it was made.** Same measurements as the previous figure, but instead of
averaging, we plot the shape of the whole distribution — a smooth curve showing
how many pictures scored what. One curve for full pictures, one for 50% crops,
overlaid so you can see them move. One panel per layer.

**Reading it.** Left is low agreement, right is high. A tall narrow hill means
most pictures behave the same. A wide flat hill means they vary a lot. The
average is printed in the legend.

**What it shows.** The crop curve sits to the *left* of the full curve — the
whole population shifts, not a few outliers. And in the deeper layers the curves
also get wider, meaning pictures stop behaving alike. So the damage from
cropping is **broad and consistent**, not driven by a handful of bad cases. That
is important: it means you cannot fix it by excluding a few awkward images.

---

### `spatial_drift_heatmap.png`

**The question:** *where inside the crop* does DINO's description go wrong?

**Why this figure exists — this one is ours, not the paper's.** DSGIR says the
problem is "the absence of global contextual support" — meaning a crop is
missing the surrounding picture. That is a claim about *why*, and it makes a
prediction nobody tested: **if the problem really is missing surroundings, the
damage should be worst at the crop's edges**, where the missing context is
closest, and mildest in the middle.

Nobody has checked. This figure checks.

**How it was made.** Take a picture. Describe the whole thing with DINO. Then cut
out a 128 square and describe that separately. The crop covers a known 16×16
patch of squares in the full picture, so we can line them up exactly, square by
square. For every one of those 256 squares we ask: does the "seen in the crop"
description match the "seen in the full picture" description? That gives a 16×16
grid of scores. Average over 150 pictures and draw it as an image.

*(The exact line-up works because the full 256-pixel picture becomes a 32×32 grid
of squares — so one square is exactly 8 pixels. As long as the crop starts at a
multiple of 8, it lands perfectly on square boundaries. Our normal evaluation
crop list does not satisfy this — only 5 of 339 crops line up — so this analysis
draws its own crop list. It does not touch the evaluation one.)*

**Reading it.** Bright means the crop description matches the full-picture
description. Dark means it drifted. One panel per layer.

**What it shows.** The edges are darker than the middle, at **every layer**:

| layer | edge − middle |
|---|---|
| B1 | −0.064 |
| B3 | −0.055 |
| B6 | −0.058 |
| B9 | **−0.137** |
| B12 | **−0.126** |

Always negative. The damage really is concentrated at the borders, and it is
roughly **twice as bad** at the deep layers.

**Why this matters.** It confirms the *mechanism*, not just the effect. DINO
describes each little square partly by looking at its neighbours. Squares in the
middle of a crop still have all their neighbours. Squares at the edge have lost
half of them — the picture simply stops there. So they get described differently.

This is a real contribution beyond the paper: they asserted the mechanism, this
measures it, and it behaves exactly as the explanation predicts.

---

### `tsne_context.png`, and `tsne_{patch,image}_{raw,centred}.png`

**The question:** if you did not know which was which, could you tell a
"described inside a crop" feature from a "described inside the full picture" one?

**Why this figure exists.** The cosine numbers say the descriptions differ, but
they do not say whether the difference is *systematic*. Random noise would also
lower a cosine. This asks a sharper question: is there a consistent, learnable
signature of "this came from a crop"? If yes, the shift is a real, structured
change — not noise.

**How it was made.** Each description is a list of 768 numbers, which nobody can
picture. t-SNE squashes them onto a flat 2-D map so that things which were
similar end up near each other. Then we colour every dot by which image it came
from and shape it by which context — circle for full picture, triangle for crop.

**READ THIS BEFORE YOU TRUST THE PICTURE.** A t-SNE map is a *drawing*, not a
measurement. Distances between clusters mean nothing. Cluster sizes mean nothing.
Re-run it with different settings and the picture changes. You cannot conclude
anything from "the clusters look separate".

So each panel is also labelled with a real number: **context separability**. We
train a simple classifier to guess, from the raw 768 numbers, whether a feature
came from a crop or a full picture. Then we test it on data it has not seen.
**0.50 means it is guessing. 1.00 means it is always right.** This is computed on
the actual features, never on the 2-D drawing.

**What it shows.**

| view | B3 | B6 | B9 | B12 |
|---|---|---|---|---|
| whole images, raw | 1.000 | 1.000 | 1.000 | 1.000 |
| whole images, centred | 0.996 | 0.990 | 0.977 | 0.969 |
| single squares, raw | 0.990 | **0.849** | 0.929 | 0.923 |
| single squares, centred | 0.782 | **0.845** | 0.877 | 0.901 |

At whole-image level the classifier is **essentially perfect**. Crop features and
full-picture features are completely distinguishable. The difference is not
subtle and it is not noise — it is a systematic signature.

**Why there are four versions.** Two choices, each changing the meaning:

- **One square vs one whole image.** A single square is fine detail; a whole
  image averaged is the coarse overall description. The whole-image view is
  cleaner, which is why it separates perfectly.
- **Raw vs centred.** Our system subtracts an average from every description
  before use — and it subtracts a *different* average for crops than for full
  pictures. So a raw plot could separate the two just because those averages
  differ, which would be true but trivial. The centred version removes that.
  **Separation survives centring** (0.97–0.99 at image level), so the difference
  is genuine and not an averaging artefact.

**The one number worth remembering:** at single-square level, **B6 is the hardest
layer to classify** — 0.849 raw, the lowest of the four. Meaning B6's
descriptions change *least* when you crop. Hold that thought.

---

### `pca_maps_full_vs_crop.png`

**The question:** what does the drift actually look like?

**Why this figure exists.** Every other figure is a number. This one is for the
eye. Numbers convince a reviewer; a picture convinces a reader in two seconds.

**How it was made.** Each square is described by 768 numbers — impossible to
draw. PCA finds the three most important directions of variation and we use them
as red, green and blue. So each square becomes a colour, and the grid of squares
becomes a small colour image. Similar descriptions get similar colours.

**The important detail:** we fit **one** PCA on the full-picture features and the
crop features **together**. That is what makes the colours mean the same thing in
both rows. Fitting separately would produce two colour schemes that cannot be
compared, and any difference you saw would be meaningless.

**Reading it.** Top row: the region as described inside the full picture. Bottom
row: the *same* region as described inside the crop. One column per layer. If
DINO were context-independent, the two rows would look identical.

**What it shows.** They do not look identical. Early layers are broadly similar —
the same shapes in roughly the same colours. Deeper layers diverge visibly, with
the colour structure reorganising. This is the same conclusion as all the
numbers, in a form you can put on a slide.

---

## Part 3 — The measurement behind the figures

The figures come from `analyze_crop_context_shift.py`, which produced two
headline numbers.

### The interaction — the thing DSGIR actually claims

DSGIR's claim is not simply "crops are different". It is sharper: **damage hurts
more inside a crop than inside a full picture.** That is a claim about two
factors working together, so we measured both.

For each picture we computed how well the render's description matches the clean
one — twice. Once measured inside the full picture, once inside the crop. Then we
subtracted. **A negative number means cropping made the damage matter more.**

| layer | difference | worse in crop |
|---|---|---|
| B3 | −0.0192 | 337 of 339 pictures |
| **B6** | **−0.0020** | **193 of 339** |
| B9 | −0.0180 | 269 of 339 |
| B12 | −0.0284 | 225 of 339 |

Negative at every layer, and statistically solid. **DSGIR's claim holds on radar
data.** The effect is biggest at B12, the deepest layer, exactly as they say.

### The finding that matters most for our own work

**B6 barely moves.** −0.0020, roughly ten times smaller than every other layer,
and worse on only 193 of 339 pictures — near a coin flip.

**B6 is the layer our Phase-1/2 study chose**, for a completely different reason
(it was the most consistent across image sources). It now turns out to also be
the layer least disturbed by cropping.

Three independent measurements point the same way:

1. The interaction above: B6 is −0.002 against −0.018 to −0.028 elsewhere.
2. Context separability: B6 is the hardest layer to classify (0.849).
3. Phase 1/2 picked B6, on unrelated grounds.

**Our main result rests on B6, and it turns out to be the most crop-robust layer
in DINO on this data.** That is a genuinely lucky-looking outcome that is not
luck — it is two different notions of "stable feature" agreeing.

### One number that changes how you read all the others

Raw crop-vs-full agreement looks like 0.66–0.76. After removing the layer's
average it drops to **0.48–0.55**.

That is not the centring making things worse. Raw cosine is inflated because all
DINO descriptions share a large common component; once you remove it, you are
looking at the part that actually carries information. **The real disagreement is
bigger than the raw numbers suggest**, and our system's centring step does not
fix it.

### Nearest-neighbour position test

A sharper version of the same question, with a right answer. Take one square from
the crop. Search all 1024 squares of the full picture. Is the closest match the
square it actually came from?

Getting it right by luck would happen 1 time in 1024 — a rate of 0.001. We score
**0.045 to 0.156**. So roughly a hundred times better than chance, but still
**wrong 85–95% of the time**.

Descriptions keep *some* positional identity across contexts, and lose most of it.

---

## Part 4 — What all of this does and does not prove

**It proves:** the prior really does change between training and testing. That
was never measured before, and our Phase-3 write-up had it flagged as an
assumption.

**It does not prove:** that this explains why the attention models specifically
did worse on crops. **Every** model gets the same shifted descriptions, including
the addition models, which show no such problem. So the shift is necessary
background, not a complete explanation.

What it does do is make the existing explanation more solid. Attention works by
computing statistics across the whole grid of squares, so a grid whose
descriptions have shifted plausibly disturbs it more than a simple addition. That
is now a grounded argument rather than a guess — but it is still an argument,
and it should be written as one.

**Something practical, from the DSGIR paper.** They train DINO with a mixed diet:
half the time a full picture, half the time a crop. That is how they make it
robust to both. We do not do this. It costs no architecture change, and it is the
obvious next thing to try if we ever want to fix the shift rather than just
measure it.

---

## Where the source is

| file | what it does |
|---|---|
| `analyze_crop_context_shift.py` | the two-axis measurement, interaction, floors |
| `visualize_crop_drift.py` | figures 10, 12, heatmap, PCA maps, one t-SNE |
| `tsne_crop_context.py` | the four t-SNE views, patch/image × raw/centred |

All three sit in this folder and are tracked in git. **The figures and JSON in
`results/` are NOT tracked** — re-run the scripts to rebuild them. DEVLOG Step 34
holds the numbers independently.
