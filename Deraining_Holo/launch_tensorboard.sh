#!/bin/bash
tensorboard --logdir experiments/Holo_Baseline_Restormer/tb_logger \
            --port 6006 \
            --bind_all
