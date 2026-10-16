# Branches and backups

Remote: `origin` = `github.com:TahaHussain121/Restormer` (fork of `swz30/Restormer`, kept as `upstream`).

The branches are one linear chain. Each branch contains everything in the one
before it, so **`dino_e2` holds the complete project**.

```
upstream/master ── baseline ── main ── dino_prior ── dino_e2   (current)
```

| branch | commits | last commit | what it adds on top of the previous branch |
|---|---|---|---|
| `baseline` | 49 | 2026-06-15 | Restormer adapted to holographic radar heatmap denoising: Holo dataset loader, `Deraining_Holo/Options` configs, training pipeline (`train_holo.sh`), test-set evaluation and training-curve scripts. L1 loss. |
| `main` | 50 | 2026-06-18 | One commit: the noisy-input baseline and the blur/sharpness analysis in test evaluation. |
| `dino_prior` | 81 | 2026-08-10 | The first DINOv2 work. Paper-ready verynoisy report; the E1 DINO-FiLM experiment (lqDINO / renderDINO) and its three failures (FiLM runaway); spatial patch-token PCA; **Phase 1** (Block 6 leads, Block 9 does not generalise) and **Phase 2** (DINO prior source, 1e5 radar vs render). |
| `dino_e2` | 166 | 2026-09-28 | The thesis study. E1-FiLM code removed (analysis kept, `dino_analysis` renamed to `dino_analysis_phases`). **Phase 3** restoration arms: addition, concat, crossattn, priorquery, AFFM, dinolight, ACA layer ladder, gated, post-latent, B3/B9 depth ladder, multi-level addition/ACA (DEVLOG Steps 28–53, study closed at Step 53). **Phase 4** crossattn diagnosis, **Phase 5** crop-vs-full-frame context. Residual and fg-balanced refiners on frozen E0 (Steps 51–52). Thesis results package `thesis_results_2026-09-17/`. Oracle masked-completion diagnostic `exp_masked_completion_20260923_230645_730d77/` (negative result). Project docs: DEVLOG, CONTEXT, HANDOVER, PHASE3_CHAPTER, THESIS_STORY, PROSE_ARGUMENTS. |

## What is not in git

Git tracks only source code, configs, docs, and curated small results. Checkpoints,
images, logs, `experiments/`, `tb_logger/`, data arrays and run caches are ignored
(see `.gitignore`). These live in the backups on
`/home/woody/iwnt/iwnt174h/thesis_dino/`:

| backup | size | contents |
|---|---|---|
| `BACKUP_20260903/` | 5.3 GB | **The dataset archive** (`dataset.tar`, split lists, `REBUILD_SPLIT_DIRS.sh`; the live copy is `thesis_dino/holographic_image_dataset/`), test images, all checkpoints as of 2026-09-03, results up to then. |
| `BACKUP_20260928/` | 7.3 GB | Supersedes the 09-03 backup for everything except the dataset. One validation-selected checkpoint per experiment (26), the frozen-E0 output caches, all results, figures, metadata and docs including the thesis package and the masked-completion audit snapshots, and a git bundle of all branches (`Restormer_repo.bundle`, restore with `git clone`). See its `MANIFEST.md`. |

To restore everything you need **both**: `BACKUP_20260928` plus `dataset.tar` (and the
split files) from `BACKUP_20260903`.
