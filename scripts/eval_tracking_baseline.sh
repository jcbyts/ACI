#!/usr/bin/env bash
set -euo pipefail

# Evaluate a checkpoint under the same explicit task/architecture mode used for
# training and record an MP4 rollout. MODE defaults to the requested actuated baseline.
RUN_DIR="${1:?usage: [MODE=actuated-2eye-rppo] eval_tracking_baseline.sh RUN_DIR [Hydra overrides ...]}"
shift
MODE="${MODE:-actuated-2eye-rppo}"

case "${MODE}" in
  original-3eye-mlp)
    CONFIG="tracking"
    LOADED_MODEL_CONFIG="loaded_model"
    ;;
  fixed-2eye-mlp)
    CONFIG="tracking_2eye_mlp_fixed"
    LOADED_MODEL_CONFIG="loaded_model"
    ;;
  fixed-2eye-rppo)
    CONFIG="tracking_2eye_recurrent_fixed"
    LOADED_MODEL_CONFIG="loaded_recurrent_model"
    ;;
  actuated-2eye-rppo)
    CONFIG="tracking_2eye_recurrent_actuated"
    LOADED_MODEL_CONFIG="loaded_recurrent_model"
    ;;
  actuated-2eye-mlp)
    CONFIG="tracking_2eye_mlp_actuated"
    LOADED_MODEL_CONFIG="loaded_model"
    ;;
  *)
    echo "Unknown MODE: ${MODE}" >&2
    echo "Expected one of: original-3eye-mlp, fixed-2eye-mlp, fixed-2eye-rppo, actuated-2eye-rppo" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/best_model}"
if [[ ! -f "${MODEL_PATH}.zip" ]]; then
  echo "Missing checkpoint: ${MODEL_PATH}.zip" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ -z "${MUJOCO_GL:-}" ]]; then
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    export MUJOCO_GL=cgl
  else
    export MUJOCO_GL=egl
  fi
fi

exec "${PYTHON_BIN}" -m cambrian.main --eval \
  "example=${CONFIG}" \
  "trainer/model=${LOADED_MODEL_CONFIG}" \
  "trainer.model.path=${MODEL_PATH}" \
  "expname=$(basename "${RUN_DIR}")" \
  "logdir=${RUN_DIR}" \
  "expdir=${RUN_DIR}" \
  eval_env.save_filename=eval_best \
  +eval_env.renderer.save_mode=MP4 \
  "$@"
