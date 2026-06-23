#!/usr/bin/env bash
set -euo pipefail

mkdir -p launch_logs

GPUS=(0 1 2 3)
MODES=(
  fixed-2eye-mlp
  fixed-2eye-mlp
  actuated-2eye-mlp
  actuated-2eye-mlp
)
SEEDS=(0 100 0 100)

TOTAL_TIMESTEPS=500000
N_ENVS=8
EVAL_EVERY_TIMESTEPS=25000
EVAL_FREQ=$((EVAL_EVERY_TIMESTEPS / N_ENVS))

MAX_REL_HEADING=0.25
STAMP=$(date +%Y%m%d_%H%M%S)

OVERLAY_ARG='overlay=[tracking_documented_reward,tracking_hunger_pressure,textured_ground]'

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    MODE="${MODES[$i]}"
    SEED="${SEEDS[$i]}"

    if [[ "${MODE}" == "actuated-2eye-mlp" ]]; then
        AGENT_OVERRIDE='env/agents@env.agents.agent=point_relative_eye'
    else
        AGENT_OVERRIDE='env/agents@env.agents.agent=point_relative'
    fi

    RUN_NAME="${MODE}_hunger_relative_heading${MAX_REL_HEADING}_seed${SEED}_nenv${N_ENVS}_${STAMP}"

    echo "Launching ${MODE} on GPU ${GPU}"
    echo "  seed: ${SEED}"
    echo "  controller: ${AGENT_OVERRIDE}"
    echo "  max_relative_heading: ${MAX_REL_HEADING}"
    echo "  overlay: ${OVERLAY_ARG}"
    echo "  run: ${RUN_NAME}"
    echo "  log: launch_logs/${RUN_NAME}.log"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID="${GPU}" \
    bash scripts/run_tracking_baseline.sh "${MODE}" \
        "${OVERLAY_ARG}" \
        "${AGENT_OVERRIDE}" \
        env.agents.agent.instance.max_relative_heading="${MAX_REL_HEADING}" \
        seed="${SEED}" \
        expname="${RUN_NAME}" \
        trainer.n_envs="${N_ENVS}" \
        trainer.total_timesteps="${TOTAL_TIMESTEPS}" \
        trainer.callbacks.eval_callback.eval_freq="${EVAL_FREQ}" \
        +trainer.model.device=cuda:0 \
        +eval_env.renderer.save_mode=MP4 \
        > "launch_logs/${RUN_NAME}.log" 2>&1 &

    sleep 5
done

wait