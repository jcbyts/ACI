#!/usr/bin/env bash
set -euo pipefail

mkdir -p launch_logs

GPUS=(2 3 2 3)
MODES=(
  fixed-2eye-rppo
  fixed-2eye-rppo
  actuated-2eye-rppo
  actuated-2eye-rppo
)
SEEDS=(0 100 0 100)

N_ENVS=8
TOTAL_TIMESTEPS=750000
EVAL_EVERY_TIMESTEPS=25000
EVAL_FREQ=$((EVAL_EVERY_TIMESTEPS / N_ENVS))
STAMP=$(date +%Y%m%d_%H%M%S)
OVERLAY_ARG='overlay=[tracking_documented_reward,textured_ground]'

PIDS=()
cleanup() {
    echo "Stopping launched jobs..."
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    MODE="${MODES[$i]}"
    SEED="${SEEDS[$i]}"
    RUN_NAME="${MODE}_cyclopean_lstm_seed${SEED}_nenv${N_ENVS}_${STAMP}"
    LOG_FILE="launch_logs/${RUN_NAME}.log"

    echo "Launching ${MODE}, seed ${SEED}, on GPU ${GPU}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID="${GPU}" \
    bash scripts/run_tracking_baseline.sh "${MODE}" \
        "${OVERLAY_ARG}" \
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

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "Job PID ${pid} failed." >&2
        FAILED=1
    fi
done

exit "${FAILED}"
