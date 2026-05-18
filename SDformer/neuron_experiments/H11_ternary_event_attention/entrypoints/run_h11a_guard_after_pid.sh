#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 WAIT_PID" >&2
  exit 2
fi

WAIT_PID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="neuron_experiments/H11_ternary_event_attention/results/h11a_event_score_h9a_core_guard120_bs8_${STAMP}"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/train.log"
CONFIG="neuron_experiments/H11_ternary_event_attention/configs/h11a_event_score_h9a_core_guard120.yml"
BASE_CKPT="experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
SAVE_PATH="$RUN_DIR/checkpoint_epoch{}.pth"

{
  echo "[H11] waiting for PID ${WAIT_PID} before starting guard"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[H11] wait PID finished; starting guard at $(date -Is)"
  echo "[H11] config: ${CONFIG}"
  set +e
  /opt/conda/envs/sdformerflow/bin/python -u \
    neuron_experiments/H11_ternary_event_attention/entrypoints/train.py \
    --config "$CONFIG" \
    --prev_runid "$BASE_CKPT" \
    --save_path "$SAVE_PATH"
  code="$?"
  set -e
  echo "$code" > "$RUN_DIR/exit_code.txt"
  echo "[H11] guard finished at $(date -Is)"
  exit "$code"
} > "$LOG_FILE" 2>&1
