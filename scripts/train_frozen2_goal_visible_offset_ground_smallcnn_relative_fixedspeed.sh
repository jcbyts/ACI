#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_visible_offset_ground_smallcnn_relative_fixedspeed_40k}"

EXPNAME="${EXPNAME}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_visible_ground_smallcnn_relative_fixedspeed.sh" \
  env/mazes@env.mazes.maze=GOAL_ONLY_VISIBLE_OFFSET \
  "$@"
