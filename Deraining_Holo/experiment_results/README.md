# Experiment Results

Permanent, versioned record of every training run — figures, metrics and
write-ups. Tracked in git (via a `.gitignore` exception) so results survive
scratch cleanup. Model weights (`.pth`) are **not** here; they live under
`experiments/<name>/models/`.

## Layout

```
experiment_results/
├── results.md                    running log of ALL experiments (newest first)
│
├── exp1_noisy/                    Exp 1 — noisy input, holo_image_dataset
│   ├── report.md                 paper-ready: results + settings tables
│   ├── figures/                  training curves, overfit check, examples, ...
│   └── metrics/                  per-image CSVs
│
├── exp2_verynoisy/               Exp 2 — verynoisy input, holographic_image_dataset
│   ├── report.md                 paper-ready: results + settings tables
│   ├── figures/                  training curves, overfit, mask viz, spectrum
│   └── metrics/                  per-image CSVs
│
├── comparisons/                  cross-experiment figures
│   └── compare_noisy_vs_verynoisy.png
│
└── dataset_splits/               split files committed for reproducibility
    └── holographic/              train.txt / val.txt / test.txt
```

## Convention for a new experiment

Create `expN_<tag>/` with the same three parts:
- `report.md` — paper-ready results + settings tables (self-contained)
- `figures/` — plots (`.png`)
- `metrics/` — per-image data (`.csv`)

Then add a section to `results.md` (the running log) and, for anything spanning
runs, a figure in `comparisons/`. Keep the folder small — summary figures and
CSVs only, never checkpoints or raw prediction dumps.

## Quick index

| Experiment | Dataset | Input | Test PSNR / SSIM (full) | Report |
|---|---|---|---|---|
| Exp 1 | holo_image_dataset | noisy | 33.499 / 0.9458 | `exp1_noisy/report.md` |
| Exp 2 | holographic_image_dataset | verynoisy | 22.405 / 0.7999 | `exp2_verynoisy/report.md` |

> The two runs are **not directly comparable** (different dataset, noise level and
> mixup setting all differ). See the caveats in `results.md`.
