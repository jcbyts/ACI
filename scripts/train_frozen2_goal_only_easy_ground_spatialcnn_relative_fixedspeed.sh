#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_only_easy_ground_spatialcnn_relative_fixedspeed_80k}"
N_ENVS="${N_ENVS:-4}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
ENT_COEF="${ENT_COEF:-0.005}"

EXPNAME="${EXPNAME}" \
N_ENVS="${N_ENVS}" \
N_STEPS="${N_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
LEARNING_RATE="${LEARNING_RATE}" \
ENT_COEF="${ENT_COEF}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_only_easy_ground_smallcnn_relative_fixedspeed.sh" \
  trainer/model/policy_kwargs=spatial_cnn_shared
