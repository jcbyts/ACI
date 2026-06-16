#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/jake/opt/anaconda3/envs/aci/bin/python}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000}"
N_ENVS="${N_ENVS:-1}"
N_STEPS="${N_STEPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
EXPNAME="${EXPNAME:-exp_frozen2_goal_ground_smallcnn_single_goal_80k}"
RENDER="${RENDER:-true}"
SAVE_MODE="${SAVE_MODE:-MP4}"
GOAL_REWARD="${GOAL_REWARD:-20.0}"
CONTACT_REWARD="${CONTACT_REWARD:--4.0}"

mkdir -p /private/tmp/aci-mpl /private/tmp/aci-fc

MPLCONFIGDIR=/private/tmp/aci-mpl \
XDG_CACHE_HOME=/private/tmp/aci-fc \
MUJOCO_GL="${MUJOCO_GL:-glfw}" \
HYDRA_FULL_ERROR=1 \
"${PYTHON_BIN}" -m cambrian.main --train \
  example=tracking \
  trainer/model/policy_kwargs=small_cnn_shared \
  trainer.total_timesteps="${TOTAL_TIMESTEPS}" \
  trainer.n_envs="${N_ENVS}" \
  trainer.model.n_steps="${N_STEPS}" \
  trainer.model.batch_size="${BATCH_SIZE}" \
  trainer.model.learning_rate=0.001 \
  trainer.callbacks.eval_callback.eval_freq="${EVAL_FREQ}" \
  trainer.callbacks.eval_callback.render="${RENDER}" \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_no_improvement_callback.max_no_improvement_evals=100000 \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_reward_threshold_callback.reward_threshold=10000 \
  expname="${EXPNAME}" \
  env.mazes.maze.floor_texture=checker \
  'env.mazes.maze.floor_texrepeat=[10,10]' \
  +eval_env.renderer.save_mode="${SAVE_MODE}" \
  'env.agents.agent.eyes.eye.num_eyes=[1,2]' \
  'env.agents.agent.eyes.eye.lat_range=[0,0]' \
  'env.agents.agent.eyes.eye.lon_range=[-15,15]' \
  'env.agents.agent.eyes.eye.fov=[45,45]' \
  'env.agents.agent.eyes.eye.resolution=[20,20]' \
  'env.agents.agent.eyes.eye.noise_std=0' \
  'env.agents.agent.eyes.eye.integration_factor=0' \
  +env.reward_fn.reward_if_done.disable=false \
  env.reward_fn.reward_if_done.scale_by_quickness=false \
  env.reward_fn.reward_if_done.termination_reward="${GOAL_REWARD}" \
  env.reward_fn.reward_if_done.truncation_reward="${CONTACT_REWARD}" \
  '+env.reward_fn.approach_goal._target_=cambrian.envs.reward_fns.reward_fn_euclidean_delta_to_agent' \
  '+env.reward_fn.approach_goal._partial_=true' \
  '+env.reward_fn.approach_goal.reward=3.0' \
  '+env.reward_fn.approach_goal.to_agents=[goal0]' \
  '+env.reward_fn.approach_goal.for_agents=[agent]' \
  env.reward_fn.penalize_if_has_contacts.reward="${CONTACT_REWARD}" \
  '+env.truncation_fn.truncate_if_has_contacts._target_=cambrian.envs.done_fns.done_if_has_contacts' \
  '+env.truncation_fn.truncate_if_has_contacts._partial_=true' \
  '+env.truncation_fn.truncate_if_has_contacts.for_agents=[agent]' \
  '+env.truncation_fn.truncate_if_close_to_adversary.disable=true' \
  '+env.termination_fn.terminate_if_close_to_goal.disable=false' \
  'eval_env.step_fn.respawn_objects_if_agent_close.for_agents=[]' \
  +eval_env.reward_fn.reward_if_done.disable=false \
  +eval_env.reward_fn.reward_if_done.scale_by_quickness=false \
  +eval_env.reward_fn.reward_if_done.termination_reward="${GOAL_REWARD}" \
  +eval_env.reward_fn.reward_if_done.truncation_reward="${CONTACT_REWARD}" \
  eval_env.reward_fn.penalize_if_has_contacts.reward="${CONTACT_REWARD}" \
  '+eval_env.truncation_fn.truncate_if_has_contacts._target_=cambrian.envs.done_fns.done_if_has_contacts' \
  '+eval_env.truncation_fn.truncate_if_has_contacts._partial_=true' \
  '+eval_env.truncation_fn.truncate_if_has_contacts.for_agents=[agent]' \
  'eval_env.truncation_fn.truncate_if_close_to_adversary.disable=true' \
  'eval_env.termination_fn.terminate_if_close_to_goal.disable=false' \
  'eval_env.reward_fn.penalize_if_adversary_respawned.reward=0.0'
