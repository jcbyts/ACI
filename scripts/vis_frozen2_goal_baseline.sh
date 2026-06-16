#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/jake/opt/anaconda3/envs/aci/bin/python}"
RUN_DIR="${RUN_DIR:-logs/2026-06-11/exp_frozen2_goal_baseline_fast_80k}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/best_model}"
EXPNAME="${EXPNAME:-exp_frozen2_goal_baseline_best_vis}"

mkdir -p /private/tmp/aci-mpl /private/tmp/aci-fc

MPLCONFIGDIR=/private/tmp/aci-mpl \
XDG_CACHE_HOME=/private/tmp/aci-fc \
MUJOCO_GL="${MUJOCO_GL:-glfw}" \
HYDRA_FULL_ERROR=1 \
"${PYTHON_BIN}" -m cambrian.main --train \
  example=tracking \
  trainer/model=loaded_model \
  trainer.model.path="${MODEL_PATH}" \
  trainer.total_timesteps=64 \
  trainer.n_envs=1 \
  +trainer.model.n_steps=64 \
  +trainer.model.batch_size=32 \
  +trainer.model.learning_rate=0.0 \
  trainer.callbacks.eval_callback.eval_freq=64 \
  trainer.callbacks.eval_callback.render=true \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_no_improvement_callback.max_no_improvement_evals=100000 \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_reward_threshold_callback.reward_threshold=10000 \
  expname="${EXPNAME}" \
  'env.agents.agent.eyes.eye.num_eyes=[1,2]' \
  'env.agents.agent.eyes.eye.lat_range=[0,0]' \
  'env.agents.agent.eyes.eye.lon_range=[-15,15]' \
  'env.agents.agent.eyes.eye.fov=[45,45]' \
  'env.agents.agent.eyes.eye.resolution=[20,20]' \
  'env.agents.agent.eyes.eye.noise_std=0' \
  'env.agents.agent.eyes.eye.integration_factor=0' \
  '+env.step_fn.respawn_objects_if_agent_close._target_=cambrian.envs.step_fns.step_respawn_agents_if_close_to_agents' \
  '+env.step_fn.respawn_objects_if_agent_close._partial_=true' \
  '+env.step_fn.respawn_objects_if_agent_close.for_agents=[goal0,adversary0]' \
  '+env.step_fn.respawn_objects_if_agent_close.to_agents=[agent]' \
  '+env.step_fn.respawn_objects_if_agent_close.distance_threshold=1.0' \
  '+env.reward_fn.approach_goal._target_=cambrian.envs.reward_fns.reward_fn_euclidean_delta_to_agent' \
  '+env.reward_fn.approach_goal._partial_=true' \
  '+env.reward_fn.approach_goal.reward=3.0' \
  '+env.reward_fn.approach_goal.to_agents=[goal0]' \
  '+env.reward_fn.approach_goal.for_agents=[agent]' \
  '+env.reward_fn.reward_if_goal_respawned._target_=cambrian.envs.reward_fns.reward_fn_agent_respawned' \
  '+env.reward_fn.reward_if_goal_respawned._partial_=true' \
  '+env.reward_fn.reward_if_goal_respawned.reward=20.0' \
  '+env.reward_fn.reward_if_goal_respawned.for_agents=[goal0]' \
  '+env.reward_fn.reward_if_goal_respawned.scale_by_quickness=false' \
  '+env.reward_fn.penalize_if_adversary_respawned._target_=cambrian.envs.reward_fns.reward_fn_agent_respawned' \
  '+env.reward_fn.penalize_if_adversary_respawned._partial_=true' \
  '+env.reward_fn.penalize_if_adversary_respawned.reward=0.0' \
  '+env.reward_fn.penalize_if_adversary_respawned.for_agents=[adversary0]' \
  '+env.reward_fn.reward_if_done.disable=true' \
  'env.reward_fn.penalize_if_has_contacts.reward=0.0' \
  '+env.truncation_fn.truncate_if_close_to_adversary.disable=true' \
  '+env.termination_fn.terminate_if_close_to_goal.disable=true' \
  '+eval_env.reward_fn.approach_goal.disable=true' \
  '+eval_env.reward_fn.reward_if_done.disable=true' \
  'eval_env.reward_fn.penalize_if_has_contacts.reward=0.0' \
  'eval_env.reward_fn.penalize_if_adversary_respawned.reward=0.0' \
  'eval_env.reward_fn.reward_if_goal_respawned.scale_by_quickness=false' \
  '+eval_env.renderer.save_mode=MP4'
