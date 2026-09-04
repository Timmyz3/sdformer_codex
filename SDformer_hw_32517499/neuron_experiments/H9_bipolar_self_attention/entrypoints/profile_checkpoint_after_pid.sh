#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 WAIT_PID CONFIG CHECKPOINT OUTPUT_DIR LOG_FILE" >&2
  exit 2
fi

WAIT_PID="$1"
CONFIG="$2"
CHECKPOINT="$3"
OUTPUT_DIR="$4"
LOG_FILE="$5"

cd /root/private_data/work/sdformer_codex/SDformer

{
  echo "[profile-after-pid] waiting for PID ${WAIT_PID}"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[profile-after-pid] starting at $(date -Is)"
  echo "[profile-after-pid] checkpoint=${CHECKPOINT}"
  set +e
  /opt/conda/envs/sdformerflow/bin/python -u \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples 40 \
    --batch-size 1 \
    --num-workers 4 \
    --metric AEE \
    --metric AAE
  code="$?"
  set -e
  echo "[profile-after-pid] exit=${code} at $(date -Is)"
  exit "$code"
} > "$LOG_FILE" 2>&1
