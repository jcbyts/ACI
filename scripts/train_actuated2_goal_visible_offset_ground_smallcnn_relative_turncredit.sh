#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPNAME="${EXPNAME:-exp_actuated2_goal_visible_offset_ground_smallcnn_relative_turncredit_40k}"
PAN_RANGE="${PAN_RANGE:-[-10,10]}"
TILT_RANGE="${TILT_RANGE:-[-5,5]}"

EXPNAME="${EXPNAME}" \
bash "${SCRIPT_DIR}/train_frozen2_goal_visible_offset_ground_smallcnn_relative_turncredit.sh" \
  env/agents@env.agents.agent=point_relative_eye \
  env/agents/eyes@env.agents.agent.eyes.eye=binocular_actuated \
  env.agents.agent.eye_action_mode=binocular \
  "env.agents.agent.eyes.eye.pan_range=${PAN_RANGE}" \
  "env.agents.agent.eyes.eye.tilt_range=${TILT_RANGE}" \
  "$@"
