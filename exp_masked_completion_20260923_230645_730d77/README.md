# Oracle-assisted masked completion — one isolated diagnostic

`Holo_E0_oracle_masked_completion_diagnostic`, run 2026-09-23. Everything in
this directory; nothing outside it was created, edited or deleted.

    docs/PROTOCOL.md    written BEFORE training: hypothesis, architecture,
                        corruption recipe, loss, selection rule, oracle mask,
                        controls, metrics, budget, splits, limitations,
                        continue/stop criteria
    docs/ISOLATION.md   how isolation is enforced and what the audit covers
    docs/REPORT.md      the result
    code/               all code (utilities copied, nothing existing imported)
    configs/            the one fixed configuration
    cache/              the fixed synthetic validation task (masks + damaged images)
    checkpoints/        ckpt_update_XXXXXX.pth, one per validation check
    results/            logs, metrics, predictions, figures, smoke checks, selection
    logs/               SLURM stdout/stderr
    audit/              before/after file snapshots and their diff
