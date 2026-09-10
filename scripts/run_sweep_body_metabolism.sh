#!/usr/bin/env bash
set -euo pipefail

mkdir -p launch_logs

# Four-GPU sweep: same working recurrent actuated agent, increasing body/eye
# metabolic pressure.  Use this after the unpenalized cyclopean recurrent baseline.
GPUS=(0 1 2 3)
SEEDS=(0 0 100 100)
OVERLAYS=(
  "overlay=[tracking_documented_reward,textured_ground]"
  "overlay=[tracking_documented_reward,textured_ground,metabolism_idle_light]"
  "overlay=[tracking_documented_reward,textured_ground,metabolism_body_heavy_weak]"
  "overlay=[tracking_documented_reward,textured_ground,metabolism_body_heavy_strong]"
)
LABELS=(
  no_metabolism
  idle_light
  body_heavy_weak
  body_heavy_strong
)

N_ENVS="${N_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-750000}"
EVAL_EVERY_TIMESTEPS="${EVAL_EVERY_TIMESTEPS:-25000}"
EVAL_FREQ=$((EVAL_EVERY_TIMESTEPS / N_ENVS))
STAMP=$(date +%Y%m%d_%H%M%S)
MODE="${MODE:-actuated-2eye-rppo}"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    SEED="${SEEDS[$i]}"
    OVERLAY_ARG="${OVERLAYS[$i]}"
    LABEL="${LABELS[$i]}"
    RUN_NAME="${MODE}_${LABEL}_seed${SEED}_nenv${N_ENVS}_${STAMP}"
    LOG_FILE="launch_logs/${RUN_NAME}.log"

    echo "Launching ${RUN_NAME} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID="${GPU}" \
    bash scripts/run_tracking_baseline.sh "${MODE}" \
        "${OVERLAY_ARG}" \
        seed="${SEED}" \
        expname="${RUN_NAME}" \
        trainer.n_envs="${N_ENVS}" \
        trainer.total_timesteps="${TOTAL_TIMESTEPS}" \
        trainer.callbacks.eval_callback.eval_freq="${EVAL_FREQ}" \
        trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_no_improvement_callback.min_evals=10 \
        trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_no_improvement_callback.max_no_improvement_evals=20 \
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
