# crossattn-render — Holo_crossattn_render_fixed128_spatial_B6_latent

DINO reads the RENDER, spatial B6 patch tokens, injected at the latent — the
same as addition-render. The fusion operator is the one thing that differs.

---

## 2026-08-19 — BUILD, BEFORE ANY TRAINING

Written before launching. Nothing here has been trained; it records the
decision, the verified facts it rests on, the implementation and the checks.

### VERIFIED FIRST, FROM THE REPO — not from notes

Every number below was read from the config, the model code or a real forward
pass before a line of the new arm was written. All CONFIRMED, no mismatches:

| | value | source |
|---|---|---|
| latent at a 128 crop | `[B, 384, 16, 16]` | real forward pass through addition-render |
| DINO B6 at a 128 crop | `[B, 768, 16, 16]`, 256 tokens, 16×16 | same forward, `_dino_capture` |
| grids equal | 256 queries == 256 keys → square attention, `attn_diag_mass` well defined | same |
| batch / iters / crop / seed | 8 / 300,000 / 128 (`gt_sizes`) / 100 | config |
| scheduler | CosineAnnealingRestartCyclicLR, periods [92000, 208000], restart_weights [1,1], eta_mins [0.0003, 0.000001] | config |
| optimizer / loss | AdamW lr 3e-4, wd 1e-4, betas [0.9, 0.999]; L1Loss weight 1, mean | config |
| means | `render_B6_train128_dino224_mean.pt`, `render_B6_eval256_dino448_mean.pt` | config |
| E0 params | 26,124,052 trainable (addition-render +295,296) | constructed both, counted |
| gate + tags | `basicsr/models/image_restoration_dino_model.py`, tags `dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`, CSV `experiments/<name>/dino_stability.csv` | code |

`384 / 64 = 6` heads exactly, so the head count needed no adjustment.

### DECISION

**The fusion operator becomes multi-head cross-attention, radar as query, DINO
render tokens as key and value.**

```
addition-render   guided = F + P(D)                      P: 1x1, 768 -> 384
concat-render     guided = fuse(cat([F, D], dim=1))    fuse: 1x1, 1152 -> 384
crossattn-render  guided = F + W_o(MHCA(Q=F, K=D, V=D))
```

**Motivation.** Addition and concat both mix token *i* of DINO with token *i* of
the radar latent and nothing else — the fusion is strictly positionwise. Cross-
attention removes that constraint and lets every radar position read every DINO
position, which is the standard choice in the DINO-prior restoration literature.

**DIRECTION, REVERSED ON PURPOSE.** Radar queries; DINO supplies keys and
values. The published methods do the opposite — the prior is the query and the
restoration feature is key and value. We want DINO to inform the radar features,
not the radar to reorganise DINO's. This is not to be "fixed" to match the
papers.

**Related work** (from the papers, not from memory):

- **Perceive-IR** (TIP 2026, arXiv 2408.15994) — DINOv2 layers 1, 4, 8, 12, but
  each `F_l` is a **global 1×768 vector**, not spatial tokens. FiLM-style affine
  in PGM, then PGCA cross-attention with the **prior as query**. Injected at the
  latent *and* all three decoder stages.
- **DSGIR** (Neurocomputing 696, 2026) — DINOv2 layers 1, 4, 8, 12 with a
  trainable zero-initialised residual adapter on layers 9–12 to correct semantic
  drift; channel-wise affine modulation (SGFM) broadcast over space, then
  Perceive-IR's PGCA unchanged.
- **DINO-IR** (arXiv 2312.01677) — attention fusion, **DINO as query**.

Ours differs from all three: spatial patch tokens rather than a pooled vector, a
single DINO layer, a single injection point, a frozen encoder with no adapter,
no auxiliary loss, and the query/key direction reversed.

### THE MODULE

```
F : [B,384,16,16] -> 256 query tokens, dim 384
D : [B,768,16,16] -> 256 key/value tokens, dim 768

LayerNorm(384) on queries, LayerNorm(768) on keys/values  (pre-norm, affine)
W_q 384->384   W_k 768->384   W_v 768->384   W_o 384->384 (ZERO-INIT)
heads = 6, head_dim = 64, attn = softmax(Q K^T / sqrt(64)) V
guided = F + reshape(W_o(attn))
```

Written out rather than delegating to `nn.MultiheadAttention`, which requires
equal embed_dim for q/k/v (ours are 384 and 768), packs the projections into one
weight so `W_q`/`W_k`/`W_v` could not be asserted on separately, and does not
expose the per-head attention matrix that `attn_diag_mass` needs.

**The features stay SPATIAL.** The `[B,768,16,16]` B6 patch grid becomes 256
keys and values in row-major order. No pooling, no broadcast, no CLS token
(`dino_shared.extract_block` never returns CLS). The arm extends
`RestormerDinoSpatialRender`; nothing is inherited from the pooled global arm.
The residual projection `P` is deleted, so no `P.*` key reaches a checkpoint.

### THE GRADIENT STAIRCASE — different from every other arm

`W_o` is zero at the **output** of the fusion, so on the first backward
`dL/dW_q = dL/dW_k = dL/dW_v = 0`: their gradient has to travel through `W_o`,
which is still zero. Only `W_o` learns at step 1; q/k/v start at step 2.

**The step-1 assertion used for the concat arm would fail here, and that is
correct.** What is asserted instead:

```
step 0    output bit-identical to E0
step 1    W_o non-zero gradient; q/k/v zero -- EXPECTED
step 2    W_q, W_k, W_v all non-zero
step 10   q/k/v still alive; dead here would mean something IS broken -> FAIL
```

### RNG ORDER — approach used: EXPLICIT SAVE/RESTORE FENCE

`nn.Linear` and `nn.LayerNorm` draw from the generator, which would shift every
later draw (data order, crop offsets, augmentation flags) away from E0's for the
same seed. The whole attention module is built inside a save/restore fence —
`torch.get_rng_state()` / `cuda.get_rng_state_all()` before, restore in a
`finally` — the same device `RestormerDinoSpatial.__init__` already uses for the
ViT and `P`. The parent fences its own construction, so state on entry is
already E0's.

### PARAMETER COUNT

```
W_q   384*384 + 384 = 147,840
W_k   768*384 + 384 = 295,296
W_v   768*384 + 384 = 295,296
W_o   384*384 + 384 = 147,840
LN_q       384*2    =     768
LN_kv      768*2    =   1,536
------------------------------
              total   888,576
```

**KNOWN ASYMMETRY — recorded, not corrected.** The capacity ladder across the
fusion arms is now

```
addition 295,296  <  concat 442,752  <  crossattn 888,576
```

so **a cross-attention win is not by itself evidence that attention is the
better operator**. Cross-attention has 3.0× the extra parameters of addition and
2.0× those of concat. Any result from this arm must be read with that in view;
no attempt is made here to correct for it.

### MONITORING — the existing gate, reused unchanged

```
latent_norm     = ||F||                 exactly as the addition arms
projected_norm  = ||W_o(attn)||         the injected part only
injection_ratio = projected_norm / (latent_norm + eps)
```

into the same `last_dino_stats` keys, so the wrapper's TensorBoard tags
(`dino/latent_norm`, `dino/projected_norm`, `dino/injection_ratio`) and the same
`dino_stability.csv` are produced with no change to the gate.

Two additional **observations**, computed every `dino_attn_stats_freq` = 5000
forwards because they need the full `[B,heads,256,256]` attention matrix:

- **`attn_entropy`** — mean entropy of the 256-way softmax rows, in nats
  (uniform = log 256 = 5.545).
- **`attn_diag_mass`** — mean of the attention matrix diagonal. **This is the
  interesting one.** If it converges near 1.0, the attention has learned to read
  only its own spatial position — i.e. it has reduced itself to the addition
  arm. That is a RESULT worth reporting, not a failure.

Between measurements the forward uses the fused
`scaled_dot_product_attention` kernel, which computes the same maths without
materialising the attention matrix.

**No gate rule is defined on entropy or diagonal mass**, deliberately: they are
observations for this run, not stopping criteria. The gate stays exactly as
amended — `injection_ratio_max` 10.0, ratio rules from iteration 5000,
`growth_factor_max` 10.0, NaN/Inf hard-stop from iteration 1.

**One shared file was touched**, and it is worth stating plainly:
`basicsr/models/image_restoration_dino_model.py` gained 11 lines that copy any
`net.last_attn_stats` entries into `log_dict` under `dino/*`. It is additive and
a strict no-op for every existing arm — E0, addition-1e5, addition-render,
global-render and concat have no such attribute and take the empty default. No
arm was training when it was made. Nothing else in the repository was modified.

### CONFIG DIFF versus addition-render — machine-generated, key by key

**94 keys identical**, 3 added, 0 removed, 4 changed:

| key | addition-render | crossattn-render |
|---|---|---|
| `name` | Holo_E1_addition_render_… | Holo_crossattn_render_… |
| `network_g.type` | RestormerDinoSpatialRender | **RestormerDinoCrossAttnRender** |
| `network_g.dino_fusion` | *(absent)* | **crossattn** |
| `network_g.dino_attn_heads` | *(absent)* | **6** |
| `network_g.dino_attn_stats_freq` | *(absent)* | **5000** |
| `dino_stability.devlog` | …/E1_addition_render_….md | …/crossattn_render_….md |
| `logger.tb_logger_dir` | tb_logger/Holo_E1_addition_render_… | tb_logger/Holo_crossattn_render_… |

Only the fusion block and the two per-arm output paths differ. No mean, crop,
schedule, threshold, optimizer or seed moved.

### ISOLATION

No existing `experiments/Holo_crossattn_render_fixed128_spatial_B6_latent`, no
chain-state directory, no `tb_logger/` tree and no results directory for this
name — confirmed before writing this entry. E0-Fixed, addition-1e5,
addition-render, global-addition-render and concat-render were not touched,
renamed or interrupted. Concat-render had already reached 300k and written
`TRAINING_DONE`; only its post-training evaluation and diagnostic jobs were in
the queue, and none of them shares a file with this arm.

### OPEN / NOT YET RESOLVED

- **Co-inflation still has no gate rule.** The gate watches the injection ratio;
  it cannot see both streams growing together.
- **B3 vs B6 was a criterion choice, not a measurement.** Documented, still
  unresolved; this arm does not revisit it.
- **The token-shuffle control has not been run.** NOTE: this control becomes
  *much* more informative for this arm than for the others — cross-attention is
  the only fusion here that could in principle **recover** from a shuffled token
  order, by attending to wherever the content moved. Addition, concat and the
  pooled arm cannot. A shuffle control that leaves this arm's PSNR intact while
  collapsing the others would be a genuine mechanistic separation.
- Cross-attention is the largest arm by parameters, as recorded above.

---

## 2026-08-21 — RESULT: THE ARM FAILED. Stopped at 179k and removed.

Trained 2026-08-20 09:56 → 2026-08-21 08:50 (jobs 1784331, 1784336), reaching
**179,000 of 300,000 iterations** before being cancelled deliberately. This
entry is the durable output of the run; the checkpoints are not.

### The trajectory — it never improved on its own first validation

| iter | val PSNR | | iter | val PSNR |
|---|---|---|---|---|
| **4,000** | **18.567** ← best, ever | | 92,000 | 14.655 |
| 8,000 | 17.291 | | 100,000 | 14.417 |
| 20,000 | 15.177 | | 120,000 | 14.533 |
| 48,000 | 14.553 | | 140,000 | 14.735 |
| 80,000 | 14.642 | | 176,000 | 15.031 |

It fell for the first ~48k, then oscillated in a **14.3–15.8 dB band for 130k
iterations**. E0-Fixed sits at ~20.1–21.3 over the same range and
addition-render at ~22.6–23.4. The best validation point of the entire run is
**iteration 4,000**, i.e. the arm never beat its own near-initial state.

**THE LR-DECAY TEST WAS RUN AND FAILED.** The scheduler's first period anneals
3e-4 → 3e-4, so the learning rate is CONSTANT until the 92k restart; every other
arm found its last 0.26–0.76 dB during the decay that follows. This arm trained
**87,000 iterations past that restart**, with the LR down to 1.9e-4, and did not
recover. That removes the strongest objection to stopping it.

### Evaluation — uint16, the same unchanged metric chain

Best-validation checkpoint is `net_g_4000.pth` (18.567 dB val, selected on
validation alone as pre-registered — an almost untrained model, which is itself
the finding):

| cell | PSNR | SSIM | PSNR mask | HF ratio | vs E0 |
|---|---|---|---|---|---|
| full256 / val | 18.563 | 0.6404 | 15.878 | 0.592 | **−3.513** |
| full256 / test | 18.723 | 0.6372 | 15.966 | 0.610 | **−3.150** |
| crop128 / val | 18.284 | 0.5783 | 16.262 | 0.933 | −1.531 |
| crop128 / test | 18.115 | 0.5687 | 16.108 | 0.926 | −1.430 |

Per-image, full256/test: better than E0 on **30/338 (8.9%)**, median −2.892,
worst −9.95, best +2.96.

A LATE checkpoint (`net_g_176000.pth`, evaluated into a throwaway path) scores
**15.032 dB** on val/full256 — **3.53 dB below its own 4k checkpoint** and 7.04
below E0. The uint16 evaluation confirms the 8-bit val curve: training made it
monotonically worse.

Context worth keeping: even at 18.72 it is **+6.37 dB above the raw 1e5 input**.
It restores; it restores much worse than restoring with no prior at all.

**HF ratio 0.592** against E0's 0.218 and addition-render's 0.328 — nearly 3x
the baseline's high-frequency energy while being 3 dB less accurate. It
hallucinates structure at scale: the same failure mode as E1-addition-noisy,
far more extreme.

### THE MECHANISM: confident routing to the wrong positions

The two observations this arm was built to make are what explain it:

| | at init (4k) | at 178k |
|---|---|---|
| `attn_entropy` | 5.52 nats (uniform = 5.545) | **0.151** |
| `attn_diag_mass` | 0.0039 (= 1/256) | **0.273** |

Entropy collapsed **97%** — the attention is essentially one-hot, placing nearly
all mass on a single key per query. But diagonal mass reached only ~0.25 and has
been static since 48k. So it routes each radar position confidently to ONE DINO
token, and roughly **three quarters of the time that token is not the aligned
one**.

This is the precise risk of the radar-as-query direction, named before the run:
the query is computed from the DEGRADED 1e5 latent, so the routing signal is
unreliable, and the arm learned a confident wrong routing rather than a
positional one. The pre-registered alternative outcome -- diag_mass converging
to 1.0, i.e. attention reducing itself to the addition arm -- did NOT happen.

### CO-INFLATION, WITH THE GATE SILENT — the first real instance

| | 4k | 48k | 178k |
|---|---|---|---|
| latent_norm | 1,218 | 8,996 | **20,058** |
| projected_norm | 629 | 6,871 | **16,486** |
| injection_ratio | 0.517 | 0.764 | **0.822** |

Both streams grew ~16x together while the ratio stayed between 0.5 and 0.85 —
never within an order of magnitude of the cap of 10. **No gate rule fired at any
iteration**, no `STABILITY_FAILURE.json` was written, and no NaN/Inf occurred.

This is exactly the failure mode every Phase-3 devlog lists as unresolved:
"co-inflation still has no gate rule. The gate watches the injection ratio; it
cannot see both streams growing together." It has now happened for real, in an
arm that lost 3.15 dB while the gate reported nothing. That is a finding about
the GATE, not only about this arm, and it should be reported as one.

### What this licenses, and what it does not

**Does license:** the radar-as-query direction, at this layer and injection
point, with this schedule, fails on this data — badly and reproducibly across
179k iterations, with a mechanism visible in the attention statistics.

**Does NOT license** "cross-attention does not work for DINO-prior restoration".
This arm carries +888,576 parameters (3x addition's), one seed, one direction,
one layer and one injection point. The published methods that use attention also
use multiple layers, a gate over them, multiple injection points and an
auxiliary loss. The prior-as-query direction is a separate arm and is queued.

### PROCESS NOTE — the 100k stop was missed

The plan was to stop at 100k and decide. The watcher was a background shell in
an interactive session and did not survive the session ending, so nothing fired;
the run continued to 179k unattended, costing roughly 13 hours of a100 time.
**For any decision point, use a mechanism that outlives the session** — a SLURM
dependency job or a cron — not a background shell.

Separately, the first evaluation submission raced itself: `run_evaluate.sh`
derives its output directory from the config NAME, not the checkpoint, so
evaluating two checkpoints of one arm concurrently (jobs 1786216 and 1786220)
had both writing the same `predictions/full256_val/` and metrics files. That
cell was DISCARDED and re-run alone (job 1786222, 18.563 — it agreed, but was
unverifiable at the time). The late checkpoint now runs through
`predict_phase3.py` with its own `--out-root` under
`results/throwaway_crossattn_late_checkpoint/`, which cannot collide.

### STATUS

**Removed from the fusion axis.** Jobs cancelled at 179k; the chain-state
directory retains a stale `RUNNING_JOB` lock, so touch `CHAIN_ABORTED` there if
an accidental resubmission must be made impossible. The arm's checkpoints,
predictions and metrics are kept as the record of a documented negative result.
