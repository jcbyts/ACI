#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_visible_offset_ground_smallcnn_relative_bearing_40k}"
BEARING_REWARD="${BEARING_REWARD:-0.08}"

EXPNAME="${EXPNAME}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_visible_offset_ground_smallcnn_relative_slow_recovery.sh" \
  '+env.reward_fn.face_goal._target_=cambrian.envs.reward_fns.reward_fn_heading_to_agent' \
  '+env.reward_fn.face_goal._partial_=true' \
  '+env.reward_fn.face_goal.for_agents=[agent]' \
  '+env.reward_fn.face_goal.to_agents=[goal0]' \
  +env.reward_fn.face_goal.reward="${BEARING_REWARD}" \
  env.reward_fn.penalize_if_has_contacts.reward=0.0 \
  +eval_env.reward_fn.face_goal.disable=true \
  "$@"
