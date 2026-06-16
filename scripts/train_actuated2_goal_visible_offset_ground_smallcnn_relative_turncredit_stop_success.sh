#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_actuated2_goal_visible_offset_ground_smallcnn_relative_turncredit_stop_success_40k}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-19.5}"

EXPNAME="${EXPNAME}" \
bash "${SCRIPT_DIR}/train_actuated2_goal_visible_offset_ground_smallcnn_relative_turncredit.sh" \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_reward_threshold_callback.reward_threshold="${SUCCESS_THRESHOLD}" \
  "$@"
