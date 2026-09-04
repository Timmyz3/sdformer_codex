#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/local5_qsilent_12block_frame_20260813}"
mkdir -p "$OUT"
cd "$ROOT"

python3 scripts/model_local5_qsilent_12block_frame.py \
  --qsilent-log results/local5_qsilent_score_rtl_20260813/tcfm5_l1_verilator.log \
  --baseline-log results/local5_score_projection_rtl_20260813/tcfm5_l1_verilator.log \
  --vector-manifest tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813/manifest.json \
  --output-dir "$OUT"

echo "PASS Local5 Q-silent 12-block model"
