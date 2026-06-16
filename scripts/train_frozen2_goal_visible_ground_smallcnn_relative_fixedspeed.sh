#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_visible_ground_smallcnn_relative_fixedspeed_40k}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-40000}"
EVAL_FREQ="${EVAL_FREQ:-2500}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
ENT_COEF="${ENT_COEF:-0.005}"
MAX_RELATIVE_HEADING="${MAX_RELATIVE_HEADING:-0.4}"

EXPNAME="${EXPNAME}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
EVAL_FREQ="${EVAL_FREQ}" \
N_STEPS="${N_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
LEARNING_RATE="${LEARNING_RATE}" \
ENT_COEF="${ENT_COEF}" \
MAX_RELATIVE_HEADING="${MAX_RELATIVE_HEADING}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_only_easy_ground_smallcnn_relative_fixedspeed.sh" \
  env/mazes@env.mazes.maze=GOAL_ONLY_VISIBLE_FORWARD \
  env.agents.agent.perturb_init_pos=false \
  'env.agents.agent.init_quat=[1,0,0,0]' \
  '+eval_env.agents.agent.perturb_init_pos=false' \
  '+eval_env.agents.agent.init_quat=[1,0,0,0]' \
  "$@"
