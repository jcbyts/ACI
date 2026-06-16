#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_contactterm_80k}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
CONTACT_REWARD="${CONTACT_REWARD:--2.0}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-25.0}"

EXPNAME="${EXPNAME}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
EVAL_FREQ="${EVAL_FREQ}" \
CONTACT_REWARD="${CONTACT_REWARD}" \
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_stop_success.sh" \
  '+env.truncation_fn.truncate_if_has_contacts._target_=cambrian.envs.done_fns.done_if_has_contacts' \
  '+env.truncation_fn.truncate_if_has_contacts._partial_=true' \
  '+env.truncation_fn.truncate_if_has_contacts.for_agents=[agent]' \
  '+eval_env.truncation_fn.truncate_if_has_contacts._target_=cambrian.envs.done_fns.done_if_has_contacts' \
  '+eval_env.truncation_fn.truncate_if_has_contacts._partial_=true' \
  '+eval_env.truncation_fn.truncate_if_has_contacts.for_agents=[agent]' \
  "$@"
