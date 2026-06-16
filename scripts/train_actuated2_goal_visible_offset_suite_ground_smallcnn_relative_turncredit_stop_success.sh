#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_actuated2_goal_visible_offset_suite_ground_smallcnn_relative_turncredit_stop_success_80k}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-19.5}"

EXPNAME="${EXPNAME}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD}" \
bash "${SCRIPT_DIR}/train_actuated2_goal_visible_offset_ground_smallcnn_relative_turncredit_stop_success.sh" \
  env.mazes.maze.enabled=false \
  +env/mazes@env.mazes.offset0=GOAL_ONLY_VISIBLE_OFFSET_EVAL_0 \
  +env/mazes@env.mazes.offset1=GOAL_ONLY_VISIBLE_OFFSET_EVAL_1 \
  +env/mazes@env.mazes.offset2=GOAL_ONLY_VISIBLE_OFFSET_EVAL_2 \
  +env/mazes@env.mazes.offset3=GOAL_ONLY_VISIBLE_OFFSET_EVAL_3 \
  +env/mazes@env.mazes.offset4=GOAL_ONLY_VISIBLE_OFFSET_EVAL_4 \
  +env/mazes@env.mazes.offset5=GOAL_ONLY_VISIBLE_OFFSET_EVAL_5 \
  eval_env.n_eval_episodes=6 \
  "$@"
