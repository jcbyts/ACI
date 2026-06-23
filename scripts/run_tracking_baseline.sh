#!/usr/bin/env bash
set -euo pipefail

# One launcher; experiment identity lives in versioned Hydra config files rather than
# inherited shell-script overrides.
MODE="${1:-actuated-2eye-rppo}"
shift || true

case "${MODE}" in
  original-3eye-mlp)
    CONFIG="tracking"
    ;;
  fixed-2eye-mlp)
    CONFIG="tracking_2eye_mlp_fixed"
    ;;
  fixed-2eye-rppo)
    CONFIG="tracking_2eye_recurrent_fixed"
    ;;
  actuated-2eye-rppo)
    CONFIG="tracking_2eye_recurrent_actuated"
    ;;
  actuated-2eye-mlp)
    CONFIG="tracking_2eye_mlp_actuated"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: original-3eye-mlp, fixed-2eye-mlp, fixed-2eye-rppo, actuated-2eye-rppo" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
if [[ -z "${MUJOCO_GL:-}" ]]; then
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    export MUJOCO_GL=cgl
  else
    export MUJOCO_GL=egl
  fi
fi

exec "${PYTHON_BIN}" -m cambrian.main --train "example=${CONFIG}" "$@"
