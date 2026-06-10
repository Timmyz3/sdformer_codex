#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/private_data/work/sdformer_codex/SDformer"
EP59="$ROOT/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"

echo "[chain] waiting for combo rapid_screen to finish..."
while ps aux | grep -q "[r]apid_screen.*nsc10_m0[0-9]_k0"; do sleep 60; done
echo "[chain] done at $(date)"

LATEST=$(ls -td "$ROOT/neuron_experiments/H9_bipolar_self_attention/results/rapid_screen_"*/ 2>/dev/null | head -1)
SUMMARY="$LATEST/summary.md"
echo "[chain] reading: $SUMMARY"

# Pick best AEE from valid10 profiles
BEST_AEE=999; BEST_NAME=""
for p in "$LATEST"/profiles/*/sops_summary.json; do
  [ -f "$p" ] || continue
  name=$(echo "$p" | grep -oP 'nsc10_m\d+_k\d+' | head -1)
  aee=$(python3 -c "import json; d=json.load(open('$p')); m=d.get('metrics',{}); print(m.get('AEE',999))" 2>/dev/null)
  if [ -n "$aee" ] && python3 -c "exit(0 if float('$aee') < float('$BEST_AEE') else 1)" 2>/dev/null; then
    BEST_AEE="$aee"; BEST_NAME="$name"
  fi
done
echo "[chain] best: $BEST_NAME AEE=$BEST_AEE"

if [ -z "$BEST_NAME" ]; then echo "[chain] no valid profiles"; exit 1; fi

CFG="$ROOT/neuron_experiments/H9_bipolar_self_attention/configs/generated/${BEST_NAME}.yml"
FULL_DIR="$ROOT/neuron_experiments/H9_bipolar_self_attention/results/${BEST_NAME}_full30_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FULL_DIR"

echo "[chain] launching full 30ep: $BEST_NAME"
nohup env \
  SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy \
  PYTHONPATH="$ROOT/third_party/SDformerFlow:$PYTHONPATH" \
  python -u "$ROOT/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py" \
    --config "$CFG" --prev_runid "$EP59" \
    --save_path "$FULL_DIR/checkpoint_epoch{}.pth" \
  > "$FULL_DIR/train.log" 2>&1 &
echo "[chain] PID=$! launched: $FULL_DIR"
