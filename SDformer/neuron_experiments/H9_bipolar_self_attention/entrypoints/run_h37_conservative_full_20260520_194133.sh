#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
CONFIG="$ROOT/neuron_experiments/H9_bipolar_self_attention/configs/h37_strict_bsa_qkv_sqrt_signv_conservative_reviewed_full_20260520_194133.yml"
PREV="$ROOT/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
OUT="$ROOT/neuron_experiments/H9_bipolar_self_attention/results/h37_strict_bsa_qkv_sqrt_signv_conservative_reviewed_full_20260520_194133_bs8_20260520_194133_setsid"

mkdir -p "$OUT"
cd "$ROOT"

echo "[runner] started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[runner] config=$CONFIG"
echo "[runner] prev=$PREV"
echo "[runner] out=$OUT"

"$PY" -u "$ROOT/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py" \
  --config "$CONFIG" \
  --prev_runid "$PREV" \
  --save_path "$OUT/checkpoint_epoch{}.pth"

latest_ckpt="$(ls -1 "$OUT"/checkpoint_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "$latest_ckpt" ]]; then
  echo "[runner] profiling latest checkpoint: $latest_ckpt"
  "$PY" -u "$ROOT/neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py" \
    --config "$CONFIG" \
    --checkpoint "$latest_ckpt" \
    --output-dir "$OUT/profile_latest_valid40" \
    --split valid \
    --num-samples 40 \
    --batch-size 1 \
    --num-workers 4 \
    --metric AEE \
    --metric AAE
else
  echo "[runner] no checkpoint found, skip profile"
fi

echo "[runner] finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
