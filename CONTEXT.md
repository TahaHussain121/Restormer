# Restormer Radar CFR — Project Context
> For Claude Code: Read this at the start of every session. Update the "Current State" and "Last Change" sections after every meaningful change. Do not rewrite the History section — only append to it.

## Who / What
Person: Taha — MSc Data Science, FAU Erlangen-Nürnberg, graduating Sept 2026.
Thesis: Radar CFR heatmap denoising using DINOv2-guided Restormer. Supervised by Prof. Belagiannis at FAU, company supervisor Christian at FiveD (Erlangen).
Repo being adapted: https://github.com/swz30/Restormer

## Problem Statement
Denoising holographic backprojection heatmaps from an SFCW MIMO radar.
- Target (clean): CFR heatmap at **1e7** ray count — cleaner reference
- Input (degraded): fewer rays => more Monte-Carlo noise + strong side lobes.
  `verynoisy` = **1e5** rays (the Exp 2 / E1 input). `noisy` = the intermediate
  level (Exp 1's input) — its exact count is [UNCONFIRMED], assumed 1e6.
  *** CORRECTED 2026-08-09 (user). These numbers were previously written the
  WRONG WAY ROUND (input 1e7 / target 1e6). More rays = LESS noise, so the
  clean target is the HIGH ray count. Verified empirically: PSNR against clean
  is 29.55 dB for `noisy` and 13.11 dB for `verynoisy` (n=30 val images), i.e.
  clean > noisy > verynoisy in quality. Only the LABELS were wrong — every
  experiment trained degraded -> clean and is unaffected. ***
- Data: ShapeNetCore.v2 chairs, rendered via shapenet_radar package
- Image format: single-channel grayscale PNG, 256x256
- Pairing: fully supervised — clean/noisy pairs exist

Using deraining config as template (not denoising) because side lobes are structured, object-dependent artifacts — closer to rain streaks than Gaussian noise.

DINOv2 injection planned for July. NOT now. Get baseline Restormer working first.

## Architecture Decisions
- Backbone: Restormer with deraining config as starting point
- Loss: L1Loss (mean reduction, weight 1) — verified from train.pixel_opt.type in the
  yml and self.cri_pix / l_pix in basicsr/models/image_restoration_model.py. This matches
  the upstream Restormer deraining recipe. (Earlier docs said "Charbonnier" — that was wrong.)
- inp_channels: 1, out_channels: 1 (CFR is single-channel)
- Everything else in yml identical to deraining baseline
- basicsr/ is NOT to be modified (except already-committed uint16 loader fixes)
- Progressive schedule CAPPED at native resolution (gt_size 256, patches <=256).
  Images are 256x256; gt_size 384 padded them to 384 via cv2.BORDER_REFLECT,
  injecting mirrored (physically fake) side-lobe structure into a large share of
  crops at ALL stages. So the 6-stage 128->384 schedule is truncated to 4 stages
  128->256: gt_sizes [128,160,192,256], iters [92000,64000,48000,96000],
  mini_batch_sizes [8,5,4,2]. *** Apply this same cap to the future
  denoising-schedule ablation config. ***

## Environment: HPC (current)
Cluster: tinygpu — FAU HPC (SLURM v25.11.2)
Login node: no GPU; submit jobs via sbatch

Partitions:
  a100      — A100, 8 nodes × 4 GPUs, 24 h walltime  ← default going forward
  v100      — V100-PCIE-32GB, 4 nodes × 4 GPUs, 24 h walltime
  rtx3080   — RTX 3080, 6 nodes × 8 GPUs, 24 h walltime
  work      — default/mixed (RTX 2080Ti + RTX 3080), 24 h walltime

Dataset path: /home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset/
  clean/        — 6778 uint16 PNGs (256×256, single-channel)
  noisy/        — 6778 uint16 PNGs
  splits/       — train.txt (6101), val.txt (339), test.txt (338)
  train_clean/, train_noisy/ — symlinks (6101 each), all resolve OK
  val_clean/,   val_noisy/   — symlinks (339 each),  all resolve OK
  test_clean/,  test_noisy/  — symlinks (338 each),  all resolve OK

  *** DO NOT touch test set until final model evaluation. ***
  test.txt must be committed to git for reproducibility.

Venv: /home/woody/iwnt/iwnt174h/thesis_dino/code/venv/
  PyTorch 2.5.1+cu121  (works on cluster CUDA 12.x drivers — confirmed)
  Activation (inside sbatch): module load python && conda activate /home/woody/iwnt/iwnt174h/thesis_dino/code/venv
  Direct call without activating: /home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python

Job script: train_holo.sh (repo root) — sbatch train_holo.sh to submit
Logs: experiments/Holo_Baseline_Restormer/slurm_<JOBID>.log

Training workflow (HPC):
  Submit:  sbatch train_holo.sh
  Monitor: squeue -u $USER   or   tail -f experiments/Holo_Baseline_Restormer/slurm_<JOBID>.log
  Resume:  edit resume_state in yml to experiments/Holo_Baseline_Restormer/training_states/ITER.state
           then resubmit: sbatch train_holo.sh
  State files: experiments/Holo_Baseline_Restormer/training_states/ITER.state (every 2k iters)
  Note: max walltime is 23 h. The 300k run will NOT fit one job — expect ~3 jobs.

  Auto-resume: basicsr/train.py (lines ~138-149) ALWAYS scans training_states/
  and resumes from the highest .state, overriding resume_state in the yml.
  So resume needs NO yml edit — just resubmit the same command.

  Self-chaining resume: Deraining_Holo/train_holo_chain.sh. Submit ONCE after a
  walltime kill; each job queues its successor (sbatch --dependency=afterany) and
  auto-resumes. Stops on TRAINING_DONE marker (300k done), after MAX_CHAIN=5 jobs,
  or via the crash guard (python exits < 1800s without the 300k ckpt => scancel
  successor + CHAIN_ABORTED marker). Markers + chain SLURM logs live in
  experiments/Holo_chain_state/ (NOT the experiment dir — keeps it separate from
  basicsr's fresh-start archiving): CHAIN_COUNT, TRAINING_DONE, CHAIN_ABORTED.
  To restart after an abort: rm experiments/Holo_chain_state/CHAIN_ABORTED.

  Effective-batch caveat (thesis): upstream Restormer deraining assumes 8 GPUs
  => effective batch 64. We train on 1 GPU => effective batch 8. Report this.

## Previous environment (fived08 — no longer used)
Machine: fived08 at FiveD, RTX 4070 Ti SUPER (16 GB), PyTorch 2.6.0+cu124
Training was tmux-based, no job scheduler.
A 300k-iter run was started and later cancelled before producing a usable checkpoint.
All config work happened on fived08; the actual first completed training run was on the HPC.

## Current State
- [x] Restormer repo cloned
- [x] Motion_Deblurring/ and Defocus_Deblurring/ deleted
- [x] Deraining_Holo/ directory created (copy of Deraining/)
- [x] yml modified: inp_channels=1, out_channels=1, name=Holo_Baseline_Restormer (controls experiments_root), dataset paths set
- [x] uint16 loading fixed — imfrombytes_uint16 + Dataset_PairedImage_uint16
- [x] val split created — ~6101 train / ~677 val, seed=42, symlink dirs
- [x] val split into val + test — 339 val / 338 test (seed=42); test locked
- [x] 1-batch smoke test passed — iter1 l_pix=7.23e-02, iter2 l_pix=8.14e-02, val PSNR=20.72 dB
- [x] Full training config finalized — 75k iters (~50 epochs), checkpoints every 2k iters, TensorBoard enabled
- [x] 75k baseline training run done — val PSNR 32.2650 dB — NOW OBSOLETE (progressive schedule never activated; all iters at 128px)
- [x] test_holo.py created — uint16 inference + heatmap visualization (inferno) + PSNR/SSIM output
- [x] reflect-padding contamination found (gt_size 384 padded 256 imgs -> mirrored fake structure in many crops at all stages); job 1694322 cancelled at iter ~28k
- [x] obsolete runs DELETED (75k run + 26-28k reflect-pad run + archives + tb_logger); numbers preserved in DEVLOG Step 17
- [x] yml schedule truncated to native 256: 4 stages, gt_size 256, gt_sizes [128,160,192,256], iters [92000,64000,48000,96000], mini_batch [8,5,4,2]
- [x] Fresh 4-stage 300k retrain COMPLETE (noisy baseline) — peak val 33.473 dB @ 224k, final 33.405 dB @ 300k
- [x] Evaluated on locked test set (224k ckpt) — full 33.50 dB / 0.9458, masked 29.31 dB / 0.8739 (n=338)
- [x] Verynoisy baseline COMPLETE (Step 19) — holographic_image_dataset, clean GT + verynoisy LQ, mixup OFF,
      experiment Holo_Baseline_Restormer_verynoisy — peak val 22.446 dB @ 292k, final 22.437 dB @ 300k, no overfitting
- [x] Test split carved for holographic_image_dataset (Step 19a) — 339 val / 338 test, seed 42, same recipe as Step 15
- [x] Verynoisy evaluated on the 338-image test set with net_g_292000.pth (best val, not final):
      full 22.405 dB / 0.7999, masked 18.313 dB / 0.5438, +10.05 dB over the noisy input
      CAVEAT: split was carved AFTER training, so the test half influenced checkpoint selection
      (not the weights). Exp 1's split predated training and is clean. See DEVLOG Step 19a.
- [x] Verynoisy checkpoints pruned (Step 19b) — kept net_g_292000.pth (best) + net_g_128000.pth only;
      all other .pth and ALL training states deleted (44 GB -> 201 MB). This run can no longer be resumed.
- [ ] Over-smoothing is the headline weakness: prediction retains only ~22% of GT high-frequency energy
- [x] E1 (DINOv2 FiLM injection) ATTEMPTED AND ABANDONED — failed 3x, code removed from this
      branch, preserved on `dino_prior`. See DEVLOG "Steps 20-26".
- [x] DINO prior analysis Phase 0/1/2 complete — dino_analysis_phases/
- [x] Phase 3 (restoration) BUILT AND VERIFIED — arch, model, dataset, configs, means,
      smoke tests, stability gate, per-arm self-chaining SLURM drivers. Layer locked to
      B6 (0-indexed 5) on the pre-registered criterion at the fixed-128 training scale.
      See HANDOVER.md and dino_analysis_phases/phase3_restoration/README.md.
- [x] *** PHASE 3 DECIDED ON THE LOCKED TEST SPLIT (2026-08-14, n=338) ***
        arm                 test full256      test crop128
        E0-Fixed            21.873 dB         19.546 dB     (baseline, no DINO)
        E1-addition-noisy   21.296 (-0.577)   19.062 (-0.484)   DINO reads 1e5 radar
        E1-addition-render  24.081 (+2.208)   22.259 (+2.713)   DINO reads render
      E1-render is MEANINGFUL on BOTH protocols against the pre-registered >+0.30 dB
      threshold and improves 298/338 images (88.2%); E1-noisy is negative on both.
      Test CONFIRMED validation. The pre-registered matched-128 escape clause does NOT
      apply: E1-render improves on both protocols, and by MORE at crop128 — so there is
      no scale-transfer limitation to invoke.
      E1-render (24.081) is the BEST MODEL THIS PROJECT HAS PRODUCED, beating the old
      progressive baseline (22.405 test) by +1.676 dB. E0-Fixed (21.873) is 0.53 dB BELOW
      that old baseline — exactly as registered in advance, since it never trains at 256.
      Checkpoints: E0 268k, E1-noisy 128k, E1-render 204k, all selected on VALIDATION.
- [x] *** FUSION AXIS: THREE MORE ARMS, NO OPERATOR ADVANTAGE FOUND (2026-08-21) ***
      test full256, best-val checkpoint, all vs E0-Fixed 21.873:
        addition-render     24.081  (+2.208)   the reference arm, +295,296 params
        concat-render       24.065  (+2.193)   +442,752 params  -> STATISTICAL TIE
        crossattn-render    18.723  (-3.150)   +888,576 params  -> FAILED, removed
        priorquery-render   queued              +741,120 params
      concat vs addition paired per-image: +0.023 dB val, -0.016 test, bootstrap
      95% CIs spanning zero on BOTH; only crop128/test separates (+0.214, CI
      [+0.099,+0.334]) and still misses the +0.30 dB "meaningful" threshold. For
      +147,456 parameters. THE FUSION OPERATOR DOES NOT MATTER ON THIS PROBLEM;
      what matters is the SOURCE (render vs 1e5) and the FORM (spatial vs pooled).
      crossattn-render (radar queries, DINO K/V) was stopped at 179k: its best
      validation was iteration 4,000 and it never improved on it. Mechanism is in
      its own attention statistics -- attn_entropy collapsed 97% (5.52 -> 0.151)
      while attn_diag_mass reached only 0.27, i.e. it learned CONFIDENT routing to
      the WRONG positions, which is the named risk of querying from the degraded
      1e5 latent. Full record in devlogs/crossattn_render_....md.
- [x] *** THE GATE HAS A HOLE, AND IT HAS NOW FIRED IN PRACTICE ***
      Through crossattn-render's 179k iterations, latent_norm grew 1,218 -> 20,058
      and projected_norm 629 -> 16,486 -- both ~16x -- while injection_ratio stayed
      between 0.5 and 0.85, never near the cap of 10. NO gate rule fired, no
      STABILITY_FAILURE.json, no NaN. The arm lost 3.15 dB with the gate silent.
      This is the "co-inflation has no gate rule" item that every Phase-3 devlog
      lists as open; it is no longer hypothetical. Report it as a finding about
      the GATE, not only about that arm.
- [x] MECHANISM OF THE RENDER GAIN, established by two post-hoc controls (val):
      MISMATCHED RENDER -- feeding another scene's render costs addition-render
      9.297 dB, 339/339 images worse, below E0's mean on 337/339, per-image
      correlation to the correct run only r=0.251, while ||P(D)|| is IDENTICAL by
      construction. Zero render -8.90, train-mean render -9.87. So a WRONG prior
      is worse than NO prior, and the gain is SCENE-SPECIFIC, not generic
      regularisation or added capacity. concat-render collapses identically
      (-9.153), so the fusion change did not alter the mechanism.
      POOLING -- global-render already showed the gain is SPATIAL: -1.284 dB, i.e.
      BELOW the no-DINO baseline.
- [x] The do-nothing floor is now measured: the raw 1e5 input scores 12.354 dB on
      test/full256. E0 is +9.518 over it (338/338 images improved); the best DINO
      arm is +11.727. The prior contributes ~19% of the total gain over doing
      nothing -- state it that way rather than quoting +2.2 dB in isolation.
- [x] Phase 3 THREE ARMS COMPLETE at 300k, validation numbers (2026-08-14, val n=339, full256):
        E0-Fixed            22.077 dB   HF 0.200   (baseline, best val 22.0749 @268k)
        E1-addition-noisy   21.469 dB   HF 0.280   -0.608 dB, improves only 100/339
        E1-addition-render  24.120 dB   HF 0.336   +2.043 dB, improves 290/339 (85.5%)
      *** THE RESULT: reading the RENDER gains +2.04 dB; reading the NOISY RADAR loses
      0.61 dB vs no prior at all. Same code, one tensor swapped, identical parameter
      count (+295,296), seed and schedule — so the gain belongs to the DINO INPUT, not
      to added capacity or to "DINO features" generically. ***
      NUANCE: E1-noisy is SHARPER but less accurate (HF 0.280 vs 0.200) — it hallucinates
      structure rather than over-smoothing; and it HELPS on E0's hardest decile (+0.48 dB,
      20/34) while hurting overall. E1-render gains most exactly where E0 fails worst
      (hardest decile +3.07 dB, 31/34 wins).
      CLAIM NARROWLY: the render is a clean view of the same object, so it carries the
      target's geometry. Not "DINO features help" but "a clean geometric view of the
      object, delivered through frozen DINO, helps — the same mechanism on noisy radar
      does not". The render IS normally available, so this is a usable method.
- [~] 4th arm global-render (pooled ablation of E1-render) at 90k/300k — tests whether
      the render's value is SPATIAL. Early signal: below baseline.
- [x] E0-Fixed over-smoothing re-characterisation DONE — HF ratio 0.218 (test) / 0.200 (val).
      CORRECTED FRAMING: the old baseline's 0.216 was measured on TEST, so test-to-test the
      two baselines are 0.218 vs 0.216 — INDISTINGUISHABLE, not "marginally worse". The
      earlier 0.200-vs-0.216 gap was a val-vs-test artefact. Quote the figure matched to the
      split it sits beside. Do NOT read crop128 HF ratios (~0.9 all arms, std ~0.47).
- [x] FINAL TEST EVALUATION DONE (jobs 1776745-1776750, both protocols, all three arms).
      Test split now read; global-render will need its own pass when it finishes.
- [x] Phase 3 IS committed as of 2026-08-21 (the earlier "NOT COMMITTED" line was stale
      from 2026-08-14; the archs/model/dataset landed on 2026-08-17, the rest today).
- [x] *** crossattn-render RE-DIAGNOSED 2026-08-21 — THE EARLIER ENTRY ABOVE IS WRONG
      ON MECHANISM AND MISLEADING ON MAGNITUDE. Read this one. ***
      (a) IT WAS NOT KILLED WHILE IMPROVING. Job 1784331 was scancel'd 16 min before
          walltime, but validation had been FLAT at 14.91 +- 0.41 dB for its last 34
          val points (132k iterations, trend -0.27 dB/100k). Stopping cost nothing.
      (b) THE PUBLISHED -3.150 dB IS A 4,000-ITERATION MODEL. Checkpoint selection uses
          the training-time FULL-256 validation, which is the one regime this arm cannot
          do, so it selected iteration 4,000 out of 179,000.
      (c) THE SAME WEIGHTS REVERSE SIGN WITH SCALE (val n=339, uint16):
            checkpoint      crop128        full256
            E0-Fixed 268k   19.816         22.077
            crossattn 4k    18.284         18.563
            crossattn 178k  21.812 (+2.00) 14.517 (-7.56)
          At its TRAINING scale the final checkpoint BEATS E0 by 2.00 dB. This is the
          pre-registered matched-128 escape clause, and nobody invoked it.
      (d) MECHANISM, from the real attention matrices at 178k (n=12 val crops): the heads
          are HETEROGENEOUS and the logged scalar averaged them into a misleading summary.
          head 4 diag_mass 0.933 (93% same-position routing — it re-derived the addition
          arm); head 2 0.471; heads 1/3 ~0.09; heads 0/5 ~0.01 with 65-74% of all mass on
          8 of 256 tokens (a pooled global read). NOT "confident routing to the wrong
          positions" as recorded on 2026-08-20.
      (e) FOUR INFERENCE-ONLY INTERVENTIONS on net_g_178000 (crop128/val, n=339; the
          pipeline reproduces the published E0 19.816 and crossattn-4k 18.284 to 0.0005 dB):
            wrong-scene render  -11.148 dB   73.7% of argmax keys move -> routing DOES
                                             track render content (refutes "arbitrary")
            temperature T=2/4    +-0.00      re-softening recovers NOTHING
            T=inf (uniform)      -8.009      the sharp routing is load-bearing
            attn = identity      -0.112      the learned routing is worth only 0.11 dB
            zero heads 0,5       -0.025      the "pooled" heads are neither help nor harm
          => entropy collapse, pooled heads and "no match signal" are all REFUTED as the
          cause. What remains: scale transfer of the softmax normalisation (256 -> 1024
          keys at full256) plus the checkpoint-selection interaction. NOT yet separated
          from "routing is tied to 16x16 geometry"; the separating test is the attention
          probe at eval256 on net_g_178000.
      Full record: dino_analysis_phases/phase4_crossattn_diagnosis/.
- [x] *** THE PAPERS DO NOT DO WHAT OUR ATTENTION ARMS DO. *** Verified from the arXiv
      HTML on 2026-08-21, not from memory. Perceive-IR's prior is `F_l in R^{1x768}` — a
      GLOBAL VECTOR per level — and its PGM applies a FiLM-style affine BEFORE PGCA, so
      the query into the attention is the prior-MODULATED FEATURE, not a prior token grid.
      Whether PGCA's softmax is spatial or Restormer-MDTA channel attention could NOT be
      settled: the paper never states the reshape, though it writes the equation in MDTA
      notation (hatted Q/K/V, learnable alpha) and cites Restormer. It does not matter for
      the conclusion: with a 1x768 prior there ARE no prior spatial tokens to attend over,
      so NO paper in our reference set does spatial-token cross-attention between a prior
      grid and a feature grid. Both our attention arms are novel constructions, not
      reimplementations — say so in the thesis. DSGIR (Neurocomputing 696) is paywalled
      and was NOT verified; what the repo records about it is second-hand.
- [x] THE GATE HOLE IS CONFIRMED QUANTITATIVELY: crossattn's latent_norm grew 15x and
      projected_norm 14x against the gate's own 5k reference, while injection_ratio never
      left 0.52-0.91 (cap 10). Only the RATIO is gated. Also note the ratio never even
      distinguished the failing arm: addition 0.89-1.47, concat 0.93-1.49, crossattn
      0.52-0.91, E1-noisy 1.90-4.46.
- [x] affm-render BUILT AND FULLY VERIFIED, NOT LAUNCHED (2026-08-21). The LAYER-COUNT
      ablation: DINO read at {3,6,9,12} instead of {6}, each centered with its own
      train-only mean, fused by DINOLight's AFFM (arXiv 2603.12579, cited not claimed) —
      a per-position softmax ACROSS LAYERS whose output stays 768 channels, so the
      injection stays the SAME zero-init Conv2d(768,384,1) addition. +298,372 params vs
      addition-render's +295,296 = +1.04%, so a decline across the layer ladder cannot be
      blamed on capacity (a naive 4-layer concat would have cost 1,180,032).
      Verified: 72/72 architectural checks (CPU+GPU); B6 mean recomputed through the new
      path matches addition-render's existing file to 1.2e-06 / 1.9e-06 (production file
      never overwritten); step-0 output bit-identical to E0; 494/494 trunk tensors
      byte-identical to E0 (RNG fence); peak VRAM 25,166 MiB vs addition's 25,117 on the
      same card (ratio 1.0020); 6000-iteration GPU smoke rc=0 with the gate ENFORCED over
      5000-6000, no STABILITY_FAILURE, injection_ratio settling at 0.94-1.19 against
      addition's ~1.03, val 22.06 dB at 6k.
      PRE-REGISTERED PREDICTION: multi-layer does NOT beat single-layer B6 by >0.10 dB.
      Contrary evidence recorded in advance: DINOLight's own ablation found multi-layer
      DID win (22.207 -> 22.414 -> 22.600 with AFFM).
      EARLY, NOISY, DO NOT OVERSTATE: in the 6k smoke the only layer GAINING weight is
      B3 (0.250 -> 0.334), which is the layer that wins the scene-advantage criterion —
      the documented tension behind the B6 lock. 6k of 300k, single-batch measurements.
- [x] *** affm-render AND dinolight-render BOTH LAUNCHED AND TRAINING (2026-08-24) ***
      affm-render      job 1788585, 243k/300k, layers {3,6,9,12} + AFFM, addition
                       injection UNCHANGED. +298,372 params = +1.04% over
                       addition-render, so a loss cannot be blamed on capacity.
      dinolight-render job 1789321, 241k/300k, the SAME AFFM plus DINOLight's
                       gated CHANNEL cross-attention (ACA) at the latent.
                       +1,352,849 params = 4.58x addition-render. Changes TWO
                       factors at once -- it is a "does the published method
                       transfer" arm, NOT an ablation. Say so when reporting it.
      Both gates clean (0 STABILITY_FAILURE), both projected to finish
      2026-08-24 ~18:00, inside the 2026-08-28 shutdown. Both hit their 24h
      walltime at 188k and resumed from 188000.state exactly as the chain is
      designed to.
- [x] NEITHER LIVE ARM IS SEPARATING FROM PLAIN ADDITION. On the 8-bit
      training-time val curve at 240k: affm 24.364, dinolight 24.239, against
      addition-render's best-ever 24.117. DO NOT read that as a win:
        (a) wrong metric -- the published 24.081 test is the uint16 path;
        (b) point noise on this curve is ~0.4 dB (E0 swings 21.738 -> 21.328
            between 180k and 188k);
        (c) no checkpoint selection has run -- addition peaked at 204k;
        (d) concat-render looked +0.025 ahead here and the full evaluation
            showed a statistical TIE.
      Both pre-registered predictions (affm "not >0.10 dB", dinolight "not
      >0.30 dB") are so far HOLDING.
- [x] *** THE AFFM LAYER WEIGHTS CONTRADICT OUR FEATURE-SPACE RANKING *** and
      this is a result in its own right, pre-registered in both devlogs. At
      ~185k, the per-position softmax over layers settles at:
        affm-render      B3 0.172 (lowest)  B6 0.229  B9 0.333 (highest)  B12 0.266
        dinolight-render B3 0.249  B6 0.211 (lowest)  B9 0.231  B12 0.309 (highest)
      B6 -- the locked layer -- wins in NEITHER arm, and B12, which WO1 put
      nearest the different-scene floor (0.2337 vs a 0.1175 floor), is fine and
      is the TOP layer in dinolight. The two arms also disagree with each other
      about which layer wins, which argues the weights are weakly determined
      rather than reading a strong signal.
      CORRECTION ON RECORD: the 6k smokes showed B3 gaining and that was read as
      an early signal. Over 185k it did NOT hold -- it was noise. Do not repeat
      that reading.
- [x] dinolight's ACA is CHANNEL-transposed attention, and the smoke test
      VERIFIED the property that motivated it: the attention matrix is
      64x64 (C/heads) at BOTH the 16x16-token train regime and the 32x32-token
      eval regime, while the DINO grid demonstrably changes. crossattn-render's
      spatial 256x256 softmax is exactly what could not survive that transition
      (+2.00 dB crop128 vs -7.56 dB full256 from identical weights). On a trained
      6k checkpoint dinolight scored crop128 19.562 / full256 21.645 -- full256
      HIGHER, not catastrophically lower.
      Its `alpha` gate (which the network can CLOSE, unlike crossattn) rose
      0.1192 -> ~0.139 and plateaued: the DINO path is used, modestly.
- [x] priorquery-render DROPPED 2026-08-23 at 90k/300k, cancelled by hand. Its
      files and stale RUNNING_JOB lock are left in place, untouched.
- [x] TWO BUGS CAUGHT BY THE PRE-LAUNCH SMOKE RUNS, both invisible to the architectural
      suite because they live in the basicsr integration path:
      (1) bare INTEGER yaml keys in the per-layer mean maps crashed
          basicsr/utils/options.py:109 (`k + ': '`) during init_loggers — the real 300k
          job would have died in its first seconds. Fixed by quoting the keys in the
          affm config; options.py NOT touched.
      (2) the observation publish condition `count % freq == 1` NEVER fires at freq == 1,
          which silently produced a run with no AFFM observations at all. Fixed to
          `(count - 1) % freq == 0` in the affm arch only. The real config (freq 5000) was
          never affected; crossattn carries the same idiom at 5000 and was left alone.

Config naming convention:
  Holo_Baseline_Restormer.yml       — pure Restormer, no DINOv2
  Holo_Baseline_Restormer_test.yml  — inference-only config (basicsr/test.py)
  DINO_analysis_data.yml            — NOT a training config. Data-only spec
                                      (datasets.val + five dino_* keys) read by
                                      dino_analysis_phases/ as its default --opt.

Manual commands log: COMMANDS.md at repo root — all commands to run by hand are recorded there.

Experiment results: Deraining_Holo/experiment_results/ — per-run figures, metrics CSVs and
results.md write-up. Tracked in git via a .gitignore exception, because experiments/, results/
and *.png are otherwise all ignored. Put anything worth keeping there, not in experiments/.

Dataset note: TWO datasets are now in play.
  holo_image_dataset/         — original; clean/noisy; HAS a locked test split (Step 15)
  holographic_image_dataset/  — newer; clean/noisy/verynoisy/renders; train.txt+val.txt ONLY,
                                no test split. train_/val_verynoisy built as symlinks from
                                splits/*.txt (see DEVLOG Step 19).

Last change: 2026-08-24 — TWO ARMS IN FLIGHT, FINISHING TODAY, AND A SURPRISE
THAT IS NOT ABOUT PSNR.

affm-render (243k/300k) and dinolight-render (241k/300k) are both training and
both finish ~18:00 today, inside the 2026-08-28 shutdown. Neither is separating
from plain addition on the training-time curve, and both pre-registered "does
not beat it" predictions are so far holding — but no checkpoint selection or
uint16 evaluation has run, so nothing is decided.

The interesting result so far is NOT the PSNR. It is that the AFFM layer weights
put B6 last or near-last in both arms, and rate B12 — the layer our own
feature-space analysis placed nearest the noise floor — highest in dinolight.
The B6 lock rests on that feature-space ranking. Two arms optimising a
completely different objective disagree with it, and disagree with each other,
which points at the weights being weakly determined rather than at a new winner.

IMMEDIATE NEXT TASK: when the two arms hit 300k, confirm TRAINING_DONE, select
checkpoints on VALIDATION only, run scripts/run_evaluate.sh on val for both
protocols, and do not touch the test split until ready to read it once.

Prior (2026-08-21) — THREE THINGS, IN ORDER OF HOW MUCH THEY CHANGE THE STORY.

1. crossattn-render is NOT the clean failure the 2026-08-20 entry describes. The same
   weights score +2.00 dB over E0 at the 128 training scale and -7.56 dB at full 256, and
   every published number for that arm came from a 4,000-iteration checkpoint that the
   selection rule picked BECAUSE the rule measures full-256. This is the pre-registered
   matched-128 escape clause, unclaimed. It also raises a methodological question the
   project has to answer deliberately rather than by drift: is best-val-on-full256 the
   right checkpoint rule for arms trained at 128? Changing it now would be changing a
   pre-registered rule after seeing results, so it needs a written decision either way.

2. No paper in our reference set does spatial-token cross-attention between a prior grid
   and a feature grid — Perceive-IR's prior is a 1x768 global vector and its FiLM step runs
   BEFORE the attention. Both our attention arms are therefore novel constructions, and
   "it works for them" was never evidence that it should work for us.

3. affm-render (layers {3,6,9,12}, AFFM fusion, addition injection, +1.04% params over
   addition-render) is built, verified end to end and NOT launched. Two bugs were caught
   before launch, one of which would have killed the 300k run at startup.

Prior (2026-08-14) — PHASE 3 HAS ITS ANSWER. All three original arms finished 300k and
were evaluated on val/full256 (n=339), best-validation checkpoint each. The DINO source is
the whole story:

  E0-Fixed            22.077 dB    baseline, no DINO
  E1-addition-noisy   21.469 dB    -0.608 dB   DINO reads the 1e5 radar
  E1-addition-render  24.120 dB    +2.043 dB   DINO reads the render

The negative arm is what makes the positive one usable: E1-noisy is the SAME code as
E1-render with one tensor swapped — same +295,296 parameters, same seed, schedule, crop,
fusion and gate — and they land on opposite sides of the baseline. E1-render also attacks
the project's motivating weakness directly: HF energy ratio 0.200 -> 0.336 (+68% relative),
Laplacian 0.284 -> 0.449, and it improves 290/339 images (85.5%), gaining most exactly where
the baseline fails worst (E0's hardest decile: +3.07 dB, 31/34 wins).

Do NOT summarise this as "DINO features help". The render is a clean view of the same object
and carries the target's geometry, so the defensible claim is narrower: a clean geometric
view of the object, delivered through frozen DINO features, substantially improves
restoration, while the identical mechanism fed the noisy radar makes it worse. The render is
normally available in this pipeline, so it remains a usable method rather than an oracle.

Two nuances that must survive into the write-up: E1-noisy is SHARPER than the baseline while
being less accurate (HF 0.280 vs 0.200) — it hallucinates structure instead of over-smoothing
— and it HELPS on E0's hardest decile (+0.48 dB, 20/34) while hurting overall. A fourth arm,
global-render (E1-render with the DINO grid pooled to one broadcast vector), is at 88k/300k
and testing whether the render's value is spatial; its early signal sits below the baseline.
Test split still UNTOUCHED. Figures: results/comparisons/three_arm_{harsh,median}_full256_val.png.

Prior (2026-08-13): Phase 3 running. Three arms train concurrently on a100, each with
its own config, devlog, chain script, job name, log directory, chain-state directory and
TensorBoard tree (logs are never overwritten). The single permitted difference between arms
is the DINO branch, and between the two E1 arms it is ONLY which tensor DINO reads:

  E0                  stock Restormer, fixed 128 crops, no DINO
  E1-addition-noisy   + frozen DINOv2 B6 from the SAME 1e5 crop
  E1-addition-render  + frozen DINOv2 B6 from the aligned RENDER (render is normally
                      available, so this is a usable method, not only an oracle)

Injection is a ZERO-INITIALIZED RESIDUAL PROJECTION at the latent — P = Conv2d(768,384,1),
zero weight and zero bias, added to inp_enc_level4 before the 8 latent blocks. Parameter
delta exactly 295,296; step-0 output identical to E0. It is NOT FiLM — none of the three
recorded FiLM failures apply. New files (all untracked so far):
  basicsr/models/archs/restormer_dino_spatial_arch.py   RestormerDinoSpatial
  basicsr/models/archs/restormer_dino_render_arch.py    RestormerDinoSpatialRender
  basicsr/models/image_restoration_dino_model.py        ImageCleanModelDinoSpatial + gate
  basicsr/data/paired_radar_render_stacked_dataset.py   render as channel 1 of the LQ tensor
  dino_analysis_phases/phase3_restoration/              configs, means, devlogs, scripts, results
The REPO_INVESTIGATION_REPORT.md N.1 hazard (train.py:241-270 sub-crops only lq/gt, so a third
aligned tensor is silently misaligned) is avoided BY CONSTRUCTION in both arms — the noisy arm
derives DINO's input from inp_img inside forward, and the render arm packs the render as
channel 1 of the LQ tensor so one slice hits both. Neither needed an edit to train.py, which
matters because all three arms re-read that file at every resume. Full detail: HANDOVER.md.

Prior (2026-08-10): E1 REMOVED from this branch (branch `dino_e2`). The DINOv2-FiLM
training experiment failed three times and is not being continued; its arch, model wrappers,
configs, launchers, gate scripts, design.md and E1_DINO_report.md were deleted here and are
preserved on the `dino_prior` branch. What was KEPT, because the DINO analysis line depends
on it and it is reusable for the next experiment — all three RENAMED after the E1 strip, so
the filename now states the function rather than the dead experiment:
  basicsr/models/archs/dinov2_feature_extractor.py  — was restormer_dino_arch.py. Loads the
                                                 frozen DINOv2 offline (strict=True) and owns
                                                 dino_preprocess / dino_denormalize. Holds no
                                                 Restormer subclass and registers nothing, so
                                                 it deliberately does NOT end in _arch.py.
  basicsr/data/radar_render_triplet_dataset.py — was paired_image_uint16_render_dataset.py.
                                                 Aligned 1e5 / 1e7 / render triplets; keeps the
                                                 _dataset.py suffix so the registry scans it.
  Deraining_Holo/Options/DINO_analysis_data.yml — was Holo_DINOv2_renderDINO_Restormer.yml.
                                                 Data-only spec, not a training config.
  dino_analysis_phases/                        — only the 3 import lines and 3 provenance
                                                 strings in visualize_dino_spatial_pca.py were
                                                 touched, for the renames. No logic changed.
Verified after the removal and after the renames: Phase 1 and Phase 2 both run end-to-end on
CPU against the real DINOv2 checkpoint, and Phase 1 still reproduces the Block 6 result.

NOTE: metadata JSONs already committed under dino_analysis_phases/*/outputs/ record the OLD
paths in their "config_read" / "pairing_source" fields. That is correct — they are provenance
records of what those runs actually read, and were deliberately not rewritten. Phase 2's
consistency check compares dino_model / dino_checkpoint / dino_hub_source only, so the stale
path strings do not affect it.

The two E1 pooled feature means SURVIVE the E1 removal, at
  Deraining_Holo/experiment_results/dino_pooled_means_reference/dino_feat_mean_{lq,render}DINO.pt
They are 3072-d (4 layers x 768) vectors mean-POOLED over patch tokens, computed over 300
training crops for E1's FiLM centering. The analysis never uses them for centering
(build_extractor passes feat_mean=None; spatial centering uses its own 768-d per-layer means
in dino_analysis_phases/dino_spatial_layer_means.pt). They are kept because
report_pooled_mean_incompatibility() reads them to RECORD why the pooled vectors cannot be
reused spatially — the result lands in metadata as centering.existing_pooled_means_examined.
Delete them and that field silently becomes [] on every future run, which would not match the
committed runs. Do not delete.

DELETED 2026-08-10, NOT RECOVERABLE (experiments/ and tb_logger/ are gitignored, so these
were never in any branch — dino_prior does NOT have them):
  experiments/Holo_DINOv2_{lq,render}DINO_verynoisy   24G + 15G  (E1 attempts 1-3)
  experiments/Holo_DINOv2_{lq,render}DINO_GATE        1.8G each  (gate rounds)
  experiments/Holo_chain_state_{lq,render}DINO{,_v2}, experiments/Holo_gate_state
  tb_logger/Holo_DINOv2_*
Worktree went 44G -> 1.6G. All E1 checkpoints, training states, training logs and TensorBoard
curves are gone; the only surviving record of E1 is the DEVLOG "Steps 20-26" entry and the
code on the `dino_prior` branch. KEPT: Holo_Baseline_Restormer (701M),
Holo_Baseline_Restormer_verynoisy (201M), their chain states and tb_logger dirs.

Removed 2026-08-10: the upstream Denoising/ task directory (Gaussian + real image denoising,
23 files). Nothing in this project referenced it, and Deraining/, Motion_Deblurring/ and
Defocus_Deblurring/ were already deleted back in DEVLOG Step 1. Dataset_GaussianDenoising
still lives in basicsr/data/paired_image_dataset.py and is untouched. Upstream README.md
still links to Denoising/README.md — those 4 links now dangle; README is upstream boilerplate
and was deliberately left alone.

Prior (2026-08-10): Phase 2 (DINO prior source: 1e5 radar vs render) complete after the
block/feature alignment fix. Phase 1: Block 6 leads, Block 9 does not generalise (339/339
val triplets). See dino_analysis_phases/DINO_ANALYSIS_DEVLOG.md and the per-phase devlogs.

Prior (2026-07-21): Verynoisy baseline complete (DEVLOG Step 19) — full 300k, peak val 22.446 dB @ 292k;
test 22.405 dB full / 18.313 dB masked; over-smoothing (HF ratio 0.216) is the weakness the
DINO line targets.

## Known Issues (must fix before smoke test)
1. ~~uint16 → uint8 truncation~~ FIXED via imfrombytes_uint16 + Dataset_PairedImage_uint16
2. ~~Channel mismatch~~ FIXED — same fix as above
3. ~~No val split~~ FIXED — 90/10 split, seed=42, symlink dirs created

## What NOT to do
- Do not resurrect E1 FiLM guidance without a reason the three recorded failures do not
  already cover (DEVLOG "Steps 20-26"; full detail on the `dino_prior` branch)
- Do not touch dino_analysis_phases/ — Phase 0/1/2 results are committed and Phase 2
  asserts against phase1_metadata.json
- Do not change the loss function
- Inside basicsr/, the only project-owned files are the uint16 loader
  (utils/img_util.py + data/paired_image_uint16_dataset.py),
  data/radar_render_triplet_dataset.py and models/archs/dinov2_feature_extractor.py.
  Everything else is stock Restormer — add new archs/models/datasets as NEW files;
  the registry auto-scans *_arch.py / *_model.py / *_dataset.py.
