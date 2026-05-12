#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/private_data/work/sdformer_codex/SDformer}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
WAIT_PID=${WAIT_PID:-}
POLL_SEC=${POLL_SEC:-120}
NUM_SAMPLES=${NUM_SAMPLES:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SNN_BACKEND=${SNN_BACKEND:-auto}

H6_EXP="$BASE/neuron_experiments/H6_attention_ternary_binary_highsops"
H6_CFG="$H6_EXP/configs/h6a_qk_ternary_mlp_down_binary_allparams_full.yml"
H6_CKPT="$H6_EXP/results/h6a_qk_ternary_mlp_down_binary_allparams_full_20260511_171228_setsid/checkpoint_epoch29.pth"

H8_EXP="$BASE/neuron_experiments/H8_ffn_block_search"
H8_CFG="$H8_EXP/configs/generated_full/h8m_stage3_block0_all_120_full_from_20260511_180615.yml"
H8_DIR="$H8_EXP/results/h8m_stage3_block0_all_120_full_from_20260511_180615_setsid"

echo "[deferred-profile $STAMP] started at $(date -u)"
echo "[deferred-profile $STAMP] samples=$NUM_SAMPLES batch=$BATCH_SIZE workers=$NUM_WORKERS backend=$SNN_BACKEND"

if [[ -n "$WAIT_PID" ]]; then
  echo "[deferred-profile $STAMP] waiting for pid $WAIT_PID"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep "$POLL_SEC"
  done
fi

echo "[deferred-profile $STAMP] waiting for h8m training processes to exit"
while pgrep -f "entrypoints/train.py.*h8m_stage3_block0_all_120_full_from_20260511_180615" >/dev/null 2>&1; do
  sleep "$POLL_SEC"
done

sleep 30
echo "[deferred-profile $STAMP] GPU before profile:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

if [[ ! -f "$H6_CKPT" ]]; then
  echo "[deferred-profile $STAMP] missing H6 checkpoint: $H6_CKPT" >&2
  exit 1
fi

H6_OUT="$H6_EXP/results/profile_h6a_all_epoch29_valid${NUM_SAMPLES}_${STAMP}"
echo "[deferred-profile $STAMP] profiling H6a-all epoch29 -> $H6_OUT"
/opt/conda/envs/sdformerflow/bin/python -u "$H6_EXP/entrypoints/profile_sops.py" \
  --config "$H6_CFG" \
  --checkpoint "$H6_CKPT" \
  --output-dir "$H6_OUT" \
  --split valid \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --snn-backend "$SNN_BACKEND" \
  --metric AEE \
  --metric AAE

H8_CKPT=$(ls -1v "$H8_DIR"/checkpoint_epoch*.pth | tail -1)
H8_EPOCH=$(basename "$H8_CKPT" .pth | sed "s/checkpoint_//")
H8_OUT="$H8_EXP/results/profile_h8m_full_${H8_EPOCH}_valid${NUM_SAMPLES}_${STAMP}"
echo "[deferred-profile $STAMP] profiling H8m full $H8_EPOCH -> $H8_OUT"
/opt/conda/envs/sdformerflow/bin/python -u "$H8_EXP/entrypoints/profile_sops.py" \
  --config "$H8_CFG" \
  --checkpoint "$H8_CKPT" \
  --output-dir "$H8_OUT" \
  --split valid \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --snn-backend "$SNN_BACKEND" \
  --metric AEE \
  --metric AAE

echo "[deferred-profile $STAMP] done at $(date -u)"
