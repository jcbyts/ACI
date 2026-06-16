#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_only_easy_ground_spatialcnn_relative_speed_80k}"
N_ENVS="${N_ENVS:-2}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_FREQ="${EVAL_FREQ:-2500}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
ENT_COEF="${ENT_COEF:-0.01}"
N_EPOCHS="${N_EPOCHS:-5}"
SPEED_REWARD="${SPEED_REWARD:-0.01}"

EXPNAME="${EXPNAME}" \
N_ENVS="${N_ENVS}" \
N_STEPS="${N_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
EVAL_FREQ="${EVAL_FREQ}" \
LEARNING_RATE="${LEARNING_RATE}" \
ENT_COEF="${ENT_COEF}" \
N_EPOCHS="${N_EPOCHS}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_only_easy_ground_smallcnn_relative_fixedspeed.sh" \
  trainer/model/policy_kwargs=spatial_cnn_light_shared \
  'trainer.wrappers.constant_action_wrapper.constant_actions={}' \
  '+env.reward_fn.forward_action._target_=cambrian.envs.reward_fns.reward_fn_action' \
  '+env.reward_fn.forward_action._partial_=true' \
  '+env.reward_fn.forward_action.for_agents=[agent]' \
  +env.reward_fn.forward_action.reward="${SPEED_REWARD}" \
  +env.reward_fn.forward_action.index=0 \
  +env.reward_fn.forward_action.normalize=true \
  +eval_env.reward_fn.forward_action.disable=true
