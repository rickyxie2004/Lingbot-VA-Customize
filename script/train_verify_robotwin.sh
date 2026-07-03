#!/usr/bin/env bash
set -euo pipefail

# Start RoboTwin verify training from cached WAN-VA latents.
# Usage:
#   bash script/train_verify_robotwin.sh train
#   bash script/train_verify_robotwin.sh debug
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=2 STEPS=20000 bash script/train_verify_robotwin.sh train
#   BACKGROUND=1 bash script/train_verify_robotwin.sh train

MODE="${1:-train}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SUCCESS_ROOT="${SUCCESS_ROOT:-/mnt/public/xieruiqi/datasets/lingbot-va/robotwin/robotwin-clean-and-aug-lerobot}"
FAILURE_ROOT="${FAILURE_ROOT:-/mnt/public/xieruiqi/datasets/lingbot-va/robotwin/robotwin-verify-failure-lerobot}"
RUN_NAME="${RUN_NAME:-verify_robotwin_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/Lingbot-VA-Customize/outputs/verify_train/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${SAVE_ROOT}/logs}"
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

if [[ "${MODE}" == "debug" ]]; then
  STEPS="${STEPS:-10}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  LATENT_TOKEN_MODE="${LATENT_TOKEN_MODE:-pooled}"
  MAX_TOKENS="${MAX_TOKENS:-256}"
  MAX_FAILURE_RECORDS="${MAX_FAILURE_RECORDS:-64}"
  EXTRA_ARGS+=(--max-failure-records "${MAX_FAILURE_RECORDS}")
elif [[ "${MODE}" == "train" ]]; then
  STEPS="${STEPS:-10000}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  LATENT_TOKEN_MODE="${LATENT_TOKEN_MODE:-spatial}"
  MAX_TOKENS="${MAX_TOKENS:-8192}"
  EXTRA_ARGS=()
else
  echo "Unknown mode: ${MODE}. Use train or debug." >&2
  exit 2
fi

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
OBS_WINDOW="${OBS_WINDOW:-6}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-16}"
POSITIVE_PROB="${POSITIVE_PROB:-0.5}"
CACHE_LATENT_ITEMS="${CACHE_LATENT_ITEMS:-128}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260630}"

CMD=(
  python wan_va/train_verify.py
  --success-root "${SUCCESS_ROOT}"
  --failure-root "${FAILURE_ROOT}"
  --save-root "${SAVE_ROOT}"
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --save-interval "${SAVE_INTERVAL}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --obs-window "${OBS_WINDOW}"
  --action-chunk-size "${ACTION_CHUNK_SIZE}"
  --positive-prob "${POSITIVE_PROB}"
  --cache-latent-items "${CACHE_LATENT_ITEMS}"
  --latent-token-mode "${LATENT_TOKEN_MODE}"
  --d-model "${D_MODEL}"
  --nhead "${NHEAD}"
  --dim-feedforward "${DIM_FEEDFORWARD}"
  --dropout "${DROPOUT}"
  --max-tokens "${MAX_TOKENS}"
  "${EXTRA_ARGS[@]}"
)

LOG_FILE="${LOG_DIR}/${MODE}.log"
echo "Run name: ${RUN_NAME}"
echo "Save root: ${SAVE_ROOT}"
echo "Log file: ${LOG_FILE}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${BACKGROUND:-0}" == "1" ]]; then
  nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
  echo "Started in background with PID $!"
else
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
