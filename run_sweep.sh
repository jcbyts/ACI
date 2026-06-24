#!/usr/bin/env bash
set -euo pipefail

mkdir -p launch_logs

# GPU 0: fixed eyes, seed 0
# GPU 1: fixed eyes, seed 100
# GPU 2: actuated eyes, seed 0
# GPU 3: actuated eyes, seed 100
GPUS=(0 1 2 3)

MODES=(
  fixed-2eye-mlp
  fixed-2eye-mlp
  actuated-2eye-mlp
  actuated-2eye-mlp
)

AGENT_CONFIGS=(
  point_yaw_rate
  point_yaw_rate
  point_yaw_rate_eye
  point_yaw_rate_eye
)

SEEDS=(0 100 0 100)

N_ENVS=8

# Use 250k for the initial controller test.
# Increase to 500000 after confirming sensible translation and steering.
TOTAL_TIMESTEPS=1500000

EVAL_EVERY_TIMESTEPS=25000
EVAL_FREQ=$((EVAL_EVERY_TIMESTEPS / N_ENVS))

STAMP=$(date +%Y%m%d_%H%M%S)

# Restore the controlled reward condition.
OVERLAY_ARG='overlay=[tracking_documented_reward,textured_ground]'

PIDS=()

cleanup() {
    echo "Stopping child processes..."
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    MODE="${MODES[$i]}"
    AGENT_CONFIG="${AGENT_CONFIGS[$i]}"
    SEED="${SEEDS[$i]}"

    AGENT_OVERRIDE="env/agents@env.agents.agent=${AGENT_CONFIG}"
    RUN_NAME="${MODE}_yawrate_seed${SEED}_nenv${N_ENVS}_${STAMP}"
    LOG_FILE="launch_logs/${RUN_NAME}.log"

    echo "Launching:"
    echo "  GPU:        ${GPU}"
    echo "  mode:       ${MODE}"
    echo "  controller: ${AGENT_CONFIG}"
    echo "  seed:       ${SEED}"
    echo "  run:        ${RUN_NAME}"
    echo "  log:        ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID="${GPU}" \
    bash scripts/run_tracking_baseline.sh "${MODE}" \
        "${OVERLAY_ARG}" \
        "${AGENT_OVERRIDE}" \
        seed="${SEED}" \
        expname="${RUN_NAME}" \
        trainer.n_envs="${N_ENVS}" \
        trainer.total_timesteps="${TOTAL_TIMESTEPS}" \
        trainer.callbacks.eval_callback.eval_freq="${EVAL_FREQ}" \
        trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_no_improvement_callback.max_no_improvement_evals=1000 \
        +trainer.model.device=cuda:0 \
        +eval_env.renderer.save_mode=MP4 \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=("$!")

    sleep 5
done

echo "Started ${#PIDS[@]} jobs: ${PIDS[*]}"

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "Job PID ${pid} failed." >&2
        FAILED=1
    fi
done

exit "${FAILED}"