#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_120k}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
GOAL_SPEED="${GOAL_SPEED:--0.45}"
APPROACH_REWARD="${APPROACH_REWARD:-8.0}"
LEARNING_RATE="${LEARNING_RATE:-0.00025}"
ENT_COEF="${ENT_COEF:-0.01}"
CONTACT_REWARD="${CONTACT_REWARD:--1.0}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-25.0}"
POLICY_KWARGS="${POLICY_KWARGS:-spatial_cnn_shared}"

EXPNAME="${EXPNAME}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
EVAL_FREQ="${EVAL_FREQ}" \
GOAL_SPEED="${GOAL_SPEED}" \
APPROACH_REWARD="${APPROACH_REWARD}" \
LEARNING_RATE="${LEARNING_RATE}" \
ENT_COEF="${ENT_COEF}" \
CONTACT_REWARD="${CONTACT_REWARD}" \
POLICY_KWARGS="${POLICY_KWARGS}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_moving_ground_smallcnn_relative_respawn.sh" \
  trainer.max_episode_steps=512 \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_reward_threshold_callback.reward_threshold="${SUCCESS_THRESHOLD}" \
  'env.agents.agent.eyes.eye.num_eyes=[1,2]' \
  'env.agents.agent.eyes.eye.lat_range=[0,0]' \
  'env.agents.agent.eyes.eye.lon_range=[-18.75,18.75]' \
  'env.agents.agent.eyes.eye.fov=[45,67.5]' \
  'env.agents.agent.eyes.eye.sensorsize=[0.008284271247461903,0.008909048505590654]' \
  'env.agents.agent.eyes.eye.resolution=[20,30]' \
  eval_env.reward_fn.penalize_if_has_contacts.reward=0.0 \
  "$@"
