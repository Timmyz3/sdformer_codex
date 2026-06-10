#!/usr/bin/env bash
# Wait for batch 3 rapid_screen, pick best, launch full 30ep training
set -euo pipefail
ROOT="/root/private_data/work/sdformer_codex/SDformer"
EP59="$ROOT/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"

echo "[chain] waiting for batch 3 rapid_screen to finish..."
while ps aux | grep -q "[r]apid_screen.*nsc10_motion\|rapid_screen.*nsc10_dir\|rapid_screen.*nsc10_kmag"; do
  sleep 60
done
echo "[chain] batch 3 finished at $(date)"

# Find latest rapid_screen summary
LATEST=$(ls -td "$ROOT/neuron_experiments/H9_bipolar_self_attention/results/rapid_screen_"*/ 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  echo "[chain] no rapid_screen results found, aborting"
  exit 1
fi

SUMMARY="$LATEST/summary.md"
echo "[chain] reading: $SUMMARY"

# Find best by score/AAE - check if any passed gate (valid40)
BEST_CFG=$(grep -E "^\| [0-9]" "$SUMMARY" 2>/dev/null | head -1 | grep -oP '(?<=/configs/)[^/]+(?=_steps)' | head -1)

if [ -z "$BEST_CFG" ]; then
  echo "[chain] no config promoted to valid40, checking valid10 results..."
  # Check profiles for valid10 results and pick best AEE
  BEST_AEE=999
  BEST_PROFILE=""
  for p in "$LATEST"/profiles/*/sops_summary.json; do
    [ -f "$p" ] || continue
    aee=$(python3 -c "import json; d=json.load(open('$p')); m=d.get('metrics',{}); print(m.get('AEE',999))" 2>/dev/null)
    if [ -n "$aee" ] && python3 -c "exit(0 if float('$aee') < float('$BEST_AEE') else 1)" 2>/dev/null; then
      BEST_AEE="$aee"
      BEST_PROFILE="$p"
    fi
  done
  if [ -z "$BEST_PROFILE" ]; then
    echo "[chain] no valid profiles found, nothing to launch"
    exit 1
  fi
  # Extract config name from profile path
  CFG_NAME=$(echo "$BEST_PROFILE" | grep -oP 'nsc10_\w+' | head -1)
  echo "[chain] best valid10: $CFG_NAME AEE=$BEST_AEE"
else
  CFG_NAME="$BEST_CFG"
  echo "[chain] best valid40: $CFG_NAME"
fi

# Find matching config
CFG="$ROOT/neuron_experiments/H9_bipolar_self_attention/configs/generated/${CFG_NAME}.yml"
if [ ! -f "$CFG" ]; then
  echo "[chain] config not found: $CFG"
  exit 1
fi

# Launch full training
FULL_DIR="$ROOT/neuron_experiments/H9_bipolar_self_attention/results/${CFG_NAME}_full30_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FULL_DIR"
echo "[chain] launching full 30ep: $CFG_NAME -> $FULL_DIR"

nohup env \
  SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy \
  PYTHONPATH="$ROOT/third_party/SDformerFlow:$PYTHONPATH" \
  python -u "$ROOT/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py" \
    --config "$CFG" \
    --prev_runid "$EP59" \
    --save_path "$FULL_DIR/checkpoint_epoch{}.pth" \
  > "$FULL_DIR/train.log" 2>&1 &

echo "[chain] FULL TRAINING PID=$!"
echo "[chain] log: tail -f ${FULL_DIR#$HOME/}/train.log"
echo "[chain] done at $(date)"
