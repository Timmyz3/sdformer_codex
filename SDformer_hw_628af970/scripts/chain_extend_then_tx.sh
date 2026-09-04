#!/usr/bin/env bash
# Chain: wait for extend training → eval best → launch TX fine-tuning
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
EXTEND_DIR="$ROOT/experiments/baseline_stride_upstream/extend"
TX_CONFIG="$ROOT/neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h41_tx_s02c_beta040.yml"
TX_OUT="$ROOT/experiments/baseline_stride_upstream/h41_tx_stride"
EVAL_CONFIG="$ROOT/configs/generated/upstream_baseline_eval.yml"

export SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy

echo "[chain] waiting for extend training to finish..."
while ps aux | grep -q "[t]rain_flow_parallel_supervised_SNN.*extend"; do
  sleep 30
done
echo "[chain] extend training finished at $(date)"

# Find best checkpoint by validation loss
echo "[chain] finding best extend checkpoint..."
BEST_CKPT=""
BEST_LOSS=999
for ckpt in "$EXTEND_DIR"/checkpoint_epoch*.pth; do
  [ -f "$ckpt" ] || continue
  ep=$(basename "$ckpt" .pth | sed 's/checkpoint_epoch//')
  # Just use the last saved checkpoint - upstream saves only on improvement
  BEST_CKPT="$ckpt"
done

if [ -z "$BEST_CKPT" ]; then
  echo "[chain] ERROR: no extend checkpoint found"
  exit 1
fi
echo "[chain] best extend ckpt: $BEST_CKPT"

# Run eval on extend best
echo "[chain] running eval on extend best..."
python "$ROOT/third_party/SDformerFlow/eval_DSEC_flow_SNN.py" \
  --config "$EVAL_CONFIG" \
  --checkpoint "$BEST_CKPT" 2>&1 | grep -E "SPARSITY|^[0-9]+\.[0-9]"

# Launch TX fine-tuning
echo "[chain] launching TX stride fine-tuning..."
mkdir -p "$TX_OUT"
python -u "$ROOT/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py" \
  --config "$TX_CONFIG" \
  --prev_runid "$BEST_CKPT" \
  --save_path "$TX_OUT/checkpoint_epoch{}.pth" \
  > "$TX_OUT/train.log" 2>&1 &

TX_PID=$!
echo "[chain] TX training PID=$TX_PID"
echo "[chain] log: tail -f $TX_OUT/train.log"
echo "[chain] done at $(date)"
