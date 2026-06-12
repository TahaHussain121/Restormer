# Commands Reference

Recurring procedures only (resume, TensorBoard, evaluation). One-off setup
commands are not kept here — see DEVLOG.md for what was done when.
All commands run from the repo root unless stated otherwise.

---

## Session variables (export at the start of each session)

    export DATASET=/home/woody/iwnt/iwnt174h/thesis_dino/holo_image_dataset
    export VENV=/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python
    cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer

---

## Submit / monitor training

    sbatch train_holo.sh                 # train_holo.sh is at the REPO ROOT
    squeue -u $USER -o "%.10i %.9P %.10j %.2t %.10M %.10L %.10l %R"
    tail -f experiments/Holo_Baseline_Restormer/train_*.log    # basicsr log (real progress)

Note: the SLURM stdout (slurm_<jobid>.out) ends up wherever basicsr leaves the
experiment dir; the file to watch for training progress is train_*.log above.

---

## Resume after a 23h walltime kill — SELF-CHAINING (preferred)

After the current job (1694322) dies at walltime, submit the chain ONCE:

    sbatch Deraining_Holo/train_holo_chain.sh

The chain handles the rest: each job auto-submits its successor (afterany
dependency) and basicsr auto-resumes from the latest .state. It stops when the
300k checkpoint exists (TRAINING_DONE marker) or after MAX_CHAIN=5 jobs.
NO yml edits — basicsr/train.py auto-detects the highest training_states/*.state.

Inspect chain state any time (markers live in experiments/Holo_chain_state/):

    cat experiments/Holo_chain_state/CHAIN_COUNT      # how many chain jobs so far
    ls  experiments/Holo_chain_state/TRAINING_DONE    # exists => chain has stopped
    ls  experiments/Holo_chain_state/CHAIN_ABORTED    # exists => crashed; rm it to retry
    squeue -u $USER                                   # see queued/dependent jobs

### Manual resume (fallback, if not using the chain)
basicsr auto-resumes from the latest state regardless of the yml, so just:

    sbatch train_holo.sh

On resume, basicsr continues the SAME experiment dir in place (no archiving).

---

## TensorBoard

    tensorboard --logdir experiments/Holo_Baseline_Restormer/ --port 6006
    # SSH tunnel from your laptop:
    #   ssh -L 6006:localhost:6006 <user>@<hpc-login-host>

---

## Evaluate a trained checkpoint on the TEST set (only at final evaluation)

    cd Deraining_Holo
    $VENV test_holo.py \
      --input_dir $DATASET/test_noisy \
      --gt_dir    $DATASET/test_clean
    # uses default weights ../experiments/Holo_Baseline_Restormer/models/net_g_latest.pth
    # outputs: raw uint16 preds + heatmap grids + mean PSNR/SSIM
