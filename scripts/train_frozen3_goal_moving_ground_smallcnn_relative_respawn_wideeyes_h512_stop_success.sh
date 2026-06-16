#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen3_goal_moving_ground_smallcnn_relative_respawn_wideeyes_h512_60k}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
GOAL_SPEED="${GOAL_SPEED:--0.15}"
APPROACH_REWARD="${APPROACH_REWARD:-8.0}"
LEARNING_RATE="${LEARNING_RATE:-0.00025}"
ENT_COEF="${ENT_COEF:-0.01}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-20.0}"

EXPNAME="${EXPNAME}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
EVAL_FREQ="${EVAL_FREQ}" \
GOAL_SPEED="${GOAL_SPEED}" \
APPROACH_REWARD="${APPROACH_REWARD}" \
LEARNING_RATE="${LEARNING_RATE}" \
ENT_COEF="${ENT_COEF}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_moving_ground_smallcnn_relative_respawn.sh" \
  trainer.max_episode_steps=512 \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_reward_threshold_callback.reward_threshold="${SUCCESS_THRESHOLD}" \
  'env.agents.agent.eyes.eye.num_eyes=[1,3]' \
  'env.agents.agent.eyes.eye.lon_range=[-60,60]' \
  'env.agents.agent.eyes.eye.fov=[70,70]' \
  "$@"
