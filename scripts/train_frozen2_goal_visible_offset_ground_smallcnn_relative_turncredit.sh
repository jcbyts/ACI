#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_visible_offset_ground_smallcnn_relative_turncredit_40k}"
TURN_REWARD="${TURN_REWARD:-0.5}"
BEARING_REWARD="${BEARING_REWARD:-0.15}"
ENT_COEF="${ENT_COEF:-0.03}"
MAX_RELATIVE_HEADING="${MAX_RELATIVE_HEADING:-0.6}"

EXPNAME="${EXPNAME}" \
BEARING_REWARD="${BEARING_REWARD}" \
ENT_COEF="${ENT_COEF}" \
MAX_RELATIVE_HEADING="${MAX_RELATIVE_HEADING}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_visible_offset_ground_smallcnn_relative_bearing.sh" \
  '+env.reward_fn.turn_to_goal._target_=cambrian.envs.reward_fns.reward_fn_action_heading_to_agent' \
  '+env.reward_fn.turn_to_goal._partial_=true' \
  '+env.reward_fn.turn_to_goal.for_agents=[agent]' \
  '+env.reward_fn.turn_to_goal.to_agents=[goal0]' \
  +env.reward_fn.turn_to_goal.reward="${TURN_REWARD}" \
  +env.reward_fn.turn_to_goal.action_index=1 \
  +eval_env.reward_fn.turn_to_goal.disable=true \
  "$@"
