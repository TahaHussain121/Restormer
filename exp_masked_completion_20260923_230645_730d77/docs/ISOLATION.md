# Isolation and preservation record

Experiment root (created once, exclusively, and never reused):

    /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/exp_masked_completion_20260923_230645_730d77

Every artifact of this experiment — code, configs, documentation, checkpoints,
predictions, caches, temporary files, the scheduler scripts, stdout/stderr and
the reports — is inside that directory. Nothing existing is edited, moved,
renamed, deleted, retrained or resumed.

## How isolation is enforced, not just intended

* **Write guard.** `code/mc_common.py` routes every write through
  `assert_inside_root`, which resolves symlinks on the destination and on each
  existing parent and raises `OutsideRootError` for anything outside the root.
  The smoke run exercises it on three real paths (`/tmp/...`, a file inside
  `dino_analysis_phases/refiner_e0/`, and `<root>/../escape.txt`).
* **Path validation before training.** `train_completion.py` resolves every
  planned destination through the same guard before the first update and aborts
  on a violation.
* **Exclusive creation.** JSON records, checkpoints, CSVs, PNGs and the
  training markers are created with `open(..., 'x')` / an existence check.
  Checkpoints are saved as `checkpoints/ckpt_update_XXXXXX.pth`, one per
  validation check; there is no `best.pth` or `latest.pth` and no checkpoint is
  ever overwritten. The selection is recorded separately in
  `results/checkpoint_selection.json`.
* **Redirected caches.** `PYTHONPYCACHEPREFIX`, `PYTHONDONTWRITEBYTECODE`,
  `TMPDIR`/`TEMP`/`TMP`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`, `TORCH_HOME`,
  `HF_HOME` and `CUDA_CACHE_PATH` are set into `<root>/tmp` **before** torch,
  cv2 or matplotlib are imported (at the top of `mc_common`, and again in the
  SLURM scripts). `sys.pycache_prefix` and `sys.dont_write_bytecode` are set in
  process as well.
* **Scheduler outputs.** `--output` and `--error` are absolute paths inside
  `<root>/logs`. The jobs `cd` into the root, never the repository root, so no
  relative path can land outside.
* **No existing module is imported.** The utilities needed from
  `dino_analysis_phases/refiner_e0/refiner_common.py` were **copied** into
  `code/mc_common.py`, so no existing module executes (that module inserts paths
  at import time and holds output paths pointing into `refiner_e0/`). The frozen
  E0 network is never built: its outputs are read from the existing cache,
  memory-mapped read-only.
* **Git.** No `checkout`, `restore`, `reset`, `clean`, `stash`, `commit` or
  `add` was run; the index is untouched. The experiment root is an untracked
  directory.
* **Jobs.** Only this experiment's own jobs were submitted. No other job was
  submitted, cancelled or altered (the queue was empty for this user at the
  start).

## Read-only reuse

Existing splits (6,101 / 339 / 338) and preprocessing (uint16 / 65535); the
frozen E0 checkpoint `net_g_268000.pth` resolved from
`phase3_restoration/results/Holo_E0_fixed128_baseline/metadata/best_checkpoint.json`
and md5-verified against `KEPT_CHECKPOINTS.json`; E0's cached raw float32
full-frame outputs (train, val) from
`experiments/Holo_E0_frozen_noisy_output_residual_refiner/e0_cache/`, opened
with `mmap_mode='r'`; E0's recorded validation predictions; the fixed window
populations and cases in `refiner_e0/qualitative_cases.json`; and the saved
validation outputs of `refiner_e0` and `refiner_fgbal`.

## Audit coverage, stated honestly

`code/snapshot.py` records `audit/snapshot_before.json` before execution and
`audit/snapshot_after.json` at the end, then `audit/audit_diff.json`.

Covered: **102,975 files** — every file under `dino_analysis_phases/`,
`Deraining_Holo/`, `basicsr/`, `experiments/Holo_E0_fixed128_baseline/`,
`experiments/Holo_E0_frozen_noisy_output_residual_refiner/`,
`experiments/Holo_E0_frozen_noisy_output_fgbalanced_refiner/`, and the
validation/test dataset directories, by **full sha256** when the file is
≤ 64 MiB; files above that (the 1.6 GB E0 train cache) carry size, mtime and a
sha256 of their first and last 8 MiB — a partial fingerprint, not a full hash.
The training dataset directories (12,202 files) are covered by **size and mtime
only**, not by content hash. Git HEAD, branch, `status --porcelain` and the
hash of `.git/index` are recorded on both sides.

**Not covered:** the rest of the repository root (the top-level `*.md` files,
`tb_logger/`, the other ~60 `experiments/` subdirectories, `thesis_results_*`),
anything outside these trees, and file ownership/ACLs. Statements about
preservation extend only as far as this coverage.

## Audit result (2026-09-23, after the run)

`audit/audit_diff.json`, comparing `snapshot_before.json` (23:16, before the
smoke and training jobs) with `snapshot_after.json` (23:48, after every job):

* files removed: **0**
* files added: **0**
* files whose content or size changed: **0**
* files whose mtime changed with identical content: **0**
* git HEAD, branch, `status --porcelain` and `.git/index` hash: **identical**
  (the only difference in the status output is the new untracked experiment
  root itself)
* 102,975 files covered on both sides

Within the coverage stated above, no existing file was modified, created or
removed. Files outside that coverage were not checked, and this audit cannot
speak for them.

## Jobs

Submitted: 1820607 (v100, cancelled by me while pending, never started),
1820609 (a100, cancelled by me while held by an account GRES limit, never
started), 1820610 (ran the smoke checks, which failed on a bug in the smoke
script itself; training was correctly not started), 1820611 (smoke 28/28 then
the single training run), 1820615 (cancelled while pending, wrong partition),
1820616 (evaluation, failed on this experiment's own overwrite guard; its
partial outputs were moved to `results/_crashed_attempt_1820616/` and are
superseded), 1820618 (the evaluation of record plus the sensitivity run).
All of them are this experiment's own jobs. No other job was submitted,
cancelled or altered; the user's queue was empty when this session started.
