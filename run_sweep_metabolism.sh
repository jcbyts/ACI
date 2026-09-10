#!/usr/bin/env bash
set -euo pipefail

mkdir -p launch_logs

# Four-condition overnight test:
#
# GPU 0: baseline recurrent actuated agent
# GPU 1: idle metabolism only
# GPU 2: weak motor metabolism + heavier body physics
# GPU 3: strong motor metabolism + heavier body physics

GPUS=(2 3 2 3)

LABELS=(
  baseline
  idle_metabolism
  weak_body_heavy
  strong_body_heavy
)

OVERLAYS=(
  'overlay=[tracking_documented_reward,textured_ground]'
  'overlay=[tracking_documented_reward,textured_ground,metabolism_idle_light]'
  'overlay=[tracking_documented_reward,textured_ground,metabolism_body_heavy_weak]'
  'overlay=[tracking_documented_reward,textured_ground,metabolism_body_heavy_strong]'
)

AGENTS=(
  point_yaw_rate_eye
  point_yaw_rate_eye
  point_yaw_rate_heavy_eye
  point_yaw_rate_heavy_eye
)

SEEDS=(0 0 0 0)

MODE="actuated-2eye-rppo"
N_ENVS=8
TOTAL_TIMESTEPS=750000
EVAL_EVERY_TIMESTEPS=25000
EVAL_FREQ=$((EVAL_EVERY_TIMESTEPS / N_ENVS))

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DATE=$(date +%F)

PIDS=()
RUN_DIRS=()

cleanup() {
    echo "Stopping launched jobs..."
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    LABEL="${LABELS[$i]}"
    OVERLAY_ARG="${OVERLAYS[$i]}"
    AGENT="${AGENTS[$i]}"
    SEED="${SEEDS[$i]}"

    RUN_NAME="${MODE}_${LABEL}_seed${SEED}_nenv${N_ENVS}_${STAMP}"
    RUN_DIR="logs/${RUN_DATE}/${RUN_NAME}"
    LOG_FILE="launch_logs/${RUN_NAME}.log"

    RUN_DIRS+=("${RUN_DIR}")

    echo "Launching:"
    echo "  GPU:       ${GPU}"
    echo "  label:     ${LABEL}"
    echo "  mode:      ${MODE}"
    echo "  agent:     ${AGENT}"
    echo "  seed:      ${SEED}"
    echo "  overlay:   ${OVERLAY_ARG}"
    echo "  run:       ${RUN_NAME}"
    echo "  log:       ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID="${GPU}" \
    bash scripts/run_tracking_baseline.sh "${MODE}" \
        "${OVERLAY_ARG}" \
        "env/agents@env.agents.agent=${AGENT}" \
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

echo "Started jobs: ${PIDS[*]}"

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "Job PID ${pid} failed." >&2
        FAILED=1
    fi
done

echo "Training finished. Run dirs:"
printf '%s\n' "${RUN_DIRS[@]}"

# Run posthoc analysis on completed runs.
# This uses the best checkpoint by default if the analysis script follows the patched eval path.
for i in "${!RUN_DIRS[@]}"; do
    run_dir="${RUN_DIRS[$i]}"
    if [[ -d "${run_dir}" ]]; then
        echo "Analyzing ${run_dir}"
        MODE="${MODE}" EPISODES=100 \
            bash scripts/analyze_tracking_behavior.sh "${run_dir}" \
                "${OVERLAYS[$i]}" \
                "env/agents@env.agents.agent=${AGENTS[$i]}" \
                "seed=${SEEDS[$i]}" \
            > "${run_dir}/behavior_analysis.log" 2>&1 || {
                echo "Analysis failed for ${run_dir}" >&2
                FAILED=1
            }
    else
        echo "Missing run dir: ${run_dir}" >&2
        FAILED=1
    fi
done

exit "${FAILED}"