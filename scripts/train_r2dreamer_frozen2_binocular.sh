#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/jake/opt/anaconda3/envs/aci/bin/python}"
R2DREAMER_ROOT="${R2DREAMER_ROOT:-/private/tmp/r2dreamer}"
EXPNAME="${EXPNAME:-exp_r2dreamer_frozen2_binocular_seed0_100k}"
LOGDIR="${LOGDIR:-logs/$(date +%Y-%m-%d)/${EXPNAME}}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-100000}"
ENV_NUM="${ENV_NUM:-1}"
EVAL_EPISODE_NUM="${EVAL_EPISODE_NUM:-6}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_LENGTH="${BATCH_LENGTH:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-200000}"
TRAIN_RATIO="${TRAIN_RATIO:-32}"
REP_LOSS="${REP_LOSS:-r2dreamer}"
DETER="${DETER:-512}"
HIDDEN="${HIDDEN:-128}"
DISCRETE="${DISCRETE:-8}"
DEPTH="${DEPTH:-8}"
UNITS="${UNITS:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00004}"
ACT_ENTROPY="${ACT_ENTROPY:-0.0003}"

mkdir -p /private/tmp/aci-mpl /private/tmp/aci-fc "$(dirname "${LOGDIR}")"

if [[ ! -f "${R2DREAMER_ROOT}/dreamer.py" ]]; then
  git clone https://github.com/NM512/r2dreamer.git "${R2DREAMER_ROOT}"
fi

MPLCONFIGDIR=/private/tmp/aci-mpl \
XDG_CACHE_HOME=/private/tmp/aci-fc \
MUJOCO_GL="${MUJOCO_GL:-glfw}" \
HYDRA_FULL_ERROR=1 \
"${PYTHON_BIN}" scripts/train_r2dreamer_frozen2_binocular.py \
  --r2dreamer-root "${R2DREAMER_ROOT}" \
  --logdir "${LOGDIR}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --env-num "${ENV_NUM}" \
  --eval-episode-num "${EVAL_EPISODE_NUM}" \
  --eval-every "${EVAL_EVERY}" \
  --final-eval-episodes "${FINAL_EVAL_EPISODES}" \
  --batch-size "${BATCH_SIZE}" \
  --batch-length "${BATCH_LENGTH}" \
  --buffer-size "${BUFFER_SIZE}" \
  --train-ratio "${TRAIN_RATIO}" \
  --rep-loss "${REP_LOSS}" \
  --deter "${DETER}" \
  --hidden "${HIDDEN}" \
  --discrete "${DISCRETE}" \
  --depth "${DEPTH}" \
  --units "${UNITS}" \
  --learning-rate "${LEARNING_RATE}" \
  --act-entropy "${ACT_ENTROPY}" \
  "$@"
