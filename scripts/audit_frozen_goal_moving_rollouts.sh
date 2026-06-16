#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/jake/opt/anaconda3/envs/aci/bin/python}"
EXPDIR="${EXPDIR:?Set EXPDIR to the existing experiment directory containing best_model.zip}"
MODEL_PATH="${MODEL_PATH:-${EXPDIR}/best_model.zip}"
AUDIT_EPISODES="${AUDIT_EPISODES:-6}"
AUDIT_OUT="${AUDIT_OUT:-${EXPDIR}/rollout_audit.json}"
POLICY_KWARGS="${POLICY_KWARGS:-spatial_cnn_shared}"
GOAL_SPEED="${GOAL_SPEED:--0.15}"
CONTACT_REWARD="${CONTACT_REWARD:-0.0}"

mkdir -p /private/tmp/aci-mpl /private/tmp/aci-fc

MPLCONFIGDIR=/private/tmp/aci-mpl \
XDG_CACHE_HOME=/private/tmp/aci-fc \
MUJOCO_GL="${MUJOCO_GL:-glfw}" \
HYDRA_FULL_ERROR=1 \
MODEL_PATH="${MODEL_PATH}" \
AUDIT_EPISODES="${AUDIT_EPISODES}" \
AUDIT_OUT="${AUDIT_OUT}" \
"${PYTHON_BIN}" scripts/audit_policy_rollouts.py \
  example=tracking \
  env/agents@env.agents.agent=point_relative \
  env/agents/eyes@env.agents.agent.eyes.eye=multi_eye \
  trainer/model/policy_kwargs="${POLICY_KWARGS}" \
  trainer.n_envs=1 \
  trainer.max_episode_steps=512 \
  eval_env.n_eval_episodes="${AUDIT_EPISODES}" \
  expname="$(basename "${EXPDIR}")" \
  logdir="$(dirname "${EXPDIR}")" \
  expdir="${EXPDIR}" \
  '~env.agents.adversary0' \
  '~eval_env.reward_fn.penalize_if_adversary_respawned' \
  +env.truncation_fn.truncate_if_close_to_adversary.disable=true \
  eval_env.truncation_fn.truncate_if_close_to_adversary.disable=true \
  +env.termination_fn.terminate_if_close_to_goal.disable=true \
  eval_env.termination_fn.terminate_if_close_to_goal.disable=true \
  env.agents.agent.instance.max_relative_heading=0.6 \
  env.agents.agent.use_action_obs=true \
  env.agents.agent.use_contact_obs=false \
  env.agents.agent.check_contacts=true \
  env.agents.goal0.instance.speed="${GOAL_SPEED}" \
  'env.agents.agent.eyes.eye.num_eyes=[1,2]' \
  'env.agents.agent.eyes.eye.lat_range=[0,0]' \
  'env.agents.agent.eyes.eye.lon_range=[-45,45]' \
  'env.agents.agent.eyes.eye.fov=[70,100]' \
  'env.agents.agent.eyes.eye.resolution=[20,30]' \
  env.agents.agent.eyes.eye.noise_std=0 \
  env.agents.agent.eyes.eye.integration_factor=0 \
  '+env.step_fn.respawn_objects_if_agent_close._target_=cambrian.envs.step_fns.step_respawn_agents_if_close_to_agents' \
  '+env.step_fn.respawn_objects_if_agent_close._partial_=true' \
  '+env.step_fn.respawn_objects_if_agent_close.for_agents=[goal0]' \
  '+env.step_fn.respawn_objects_if_agent_close.to_agents=[agent]' \
  '+env.step_fn.respawn_objects_if_agent_close.distance_threshold=1.0' \
  '+env.reward_fn.time_penalty.disable=true' \
  '+env.reward_fn.approach_goal.disable=true' \
  '+env.reward_fn.reward_if_goal_respawned._target_=cambrian.envs.reward_fns.reward_fn_agent_respawned' \
  '+env.reward_fn.reward_if_goal_respawned._partial_=true' \
  '+env.reward_fn.reward_if_goal_respawned.for_agents=[goal0]' \
  '+env.reward_fn.reward_if_goal_respawned.reward=10.0' \
  '+env.reward_fn.reward_if_goal_respawned.scale_by_quickness=false' \
  env.reward_fn.penalize_if_has_contacts.reward="${CONTACT_REWARD}" \
  '+eval_env.reward_fn.approach_goal.disable=true' \
  '+eval_env.reward_fn.time_penalty.disable=true' \
  eval_env.reward_fn.reward_if_goal_respawned.scale_by_quickness=false \
  eval_env.reward_fn.reward_if_goal_respawned.reward=10.0 \
  eval_env.reward_fn.penalize_if_has_contacts.reward="${CONTACT_REWARD}" \
  "$@"
