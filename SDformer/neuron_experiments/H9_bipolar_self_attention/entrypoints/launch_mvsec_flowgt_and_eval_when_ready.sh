#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

SEQ="${1:-indoor_flying3}"
NPZ="third_party/SDformerFlow/data/Datasets/MVSEC/${SEQ}/${SEQ}_gt_flow_dist.npz"
GT_H5="third_party/SDformerFlow/data/Datasets/MVSEC/${SEQ}/${SEQ}_gt.hdf5"
DATA_H5="third_party/SDformerFlow/data/Datasets/MVSEC/${SEQ}/${SEQ}_data.hdf5"
FLOWGT="third_party/SDformerFlow/data/Datasets/MVSEC/MVSEC_test/${SEQ}/flowgt_dt1"
MIN_NPZ_BYTES=800000000
EVAL_START=314
EVAL_END=2199

flowgt_eval_ready() {
  python3 - <<PY
from pathlib import Path
flowgt = Path("${FLOWGT}")
needed = list(range(${EVAL_START}, ${EVAL_END}))
missing = [i for i in needed if not (flowgt / f"{i}.npy").exists()]
raise SystemExit(0 if not missing else 1)
PY
}

ensure_gt_hdf5() {
  if [[ -s "$GT_H5" ]]; then
    echo "[skip] gt.hdf5 ready: $GT_H5"
    return 0
  fi
  if [[ -f "$NPZ" ]]; then
    size=$(stat -c%s "$NPZ")
    if [[ "$size" -ge "$MIN_NPZ_BYTES" ]]; then
      echo "[convert] npz -> gt.hdf5"
      python3 scripts/mvsec_npz_to_gt_hdf5.py --npz "$NPZ" --output "$GT_H5"
      return 0
    fi
    echo "[wait] npz too small: ${size}/${MIN_NPZ_BYTES}"
    return 1
  fi
  if [[ -f "third_party/SDformerFlow/data/Datasets/MVSEC/${SEQ}/${SEQ}_gt.bag" ]]; then
    echo "[convert] gt.bag -> gt.hdf5 (local flow_dist)"
    python3 scripts/mvsec_gt_flow_from_bag.py --sequence "$SEQ" \
      --mvsec-root third_party/SDformerFlow/data/Datasets/MVSEC
    return 0
  fi
  return 1
}

if flowgt_eval_ready; then
  echo "[ready] flowgt eval frames already present under ${FLOWGT}"
else
  echo "[encode] flowgt missing for ${SEQ}; preparing gt + encoder"
  while ! ensure_gt_hdf5; do
    echo "[wait] gt source not ready (need gt.hdf5, valid npz, or gt.bag)"
    sleep 120
  done
  if [[ ! -s "$DATA_H5" ]]; then
    python3 scripts/mvsec_bag_to_hdf5.py --sequence "$SEQ" \
      --mvsec-root third_party/SDformerFlow/data/Datasets/MVSEC --data-only
  fi
  cd third_party/SDformerFlow
  python MDR_dataloader/MVSEC_encoder.py \
    --save-dir data/Datasets/MVSEC \
    --out-dir data/Datasets/MVSEC/MVSEC_test \
    --save-env "$SEQ" \
    --dt 1 \
    --sparse_print
  cd "$REPO"
fi

bash neuron_experiments/H9_bipolar_self_attention/entrypoints/launch_mvsec_eval_direct.sh