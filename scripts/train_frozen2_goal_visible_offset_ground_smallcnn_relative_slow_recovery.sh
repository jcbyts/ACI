#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_frozen2_goal_visible_offset_ground_smallcnn_relative_slow_recovery_40k}"
FORWARD_ACTION="${FORWARD_ACTION:--0.5}"
MAX_RELATIVE_HEADING="${MAX_RELATIVE_HEADING:-0.2}"
CONTACT_REWARD="${CONTACT_REWARD:--1.0}"

EXPNAME="${EXPNAME}" \
FORWARD_ACTION="${FORWARD_ACTION}" \
MAX_RELATIVE_HEADING="${MAX_RELATIVE_HEADING}" \
CONTACT_REWARD="${CONTACT_REWARD}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_visible_offset_ground_smallcnn_relative_fixedspeed.sh" \
  'env.truncation_fn.truncate_if_has_contacts.for_agents=[]' \
  "$@"
