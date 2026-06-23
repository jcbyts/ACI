#!/usr/bin/env bash
set -euo pipefail

RUN_DATE=$(date +%F)
RUN_NAME="tracking_2eye_recurrent_actuated_seed0_baseline_$(date +%H%M%S)"
RUN_DIR="logs/${RUN_DATE}/${RUN_NAME}"

N_ENVS=4
EVAL_FREQ=6250   # 6250 vector steps × 4 envs = every 25,000 timesteps

# bash scripts/run_tracking_baseline.sh

bash scripts/run_tracking_baseline.sh actuated-2eye-rppo \
  seed=0 \
  expname="$RUN_NAME" \
  trainer.n_envs="$N_ENVS" \
  trainer.total_timesteps=500000 \
  trainer.callbacks.eval_callback.eval_freq="$EVAL_FREQ" \
  +eval_env.renderer.save_mode=MP4 