#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_multihead_decoder_projection"
RESULT="$ROOT/results/gatestack_scale162_activity_20260715"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
mkdir -p "$BUILD" "$RESULT"
cd "$ROOT"

sim_hitflow/run_gatestack_multihead_decoder_projection_checks.sh \
  >"$BUILD/activity_prerequisite.log" 2>&1
rm -f "$BUILD/scale162.vcd"
vvp "$BUILD/tb_scale162.vvp" +dump_vcd | tee "$BUILD/iverilog_scale162_activity.log"
test -s "$BUILD/scale162.vcd"
"$PYTHON" scripts/summarize_vcd_activity.py "$BUILD/scale162.vcd" \
  --json "$RESULT/activity.json" \
  --markdown "$RESULT/activity.md"
echo "PASS: GateStack scale162 VCD与中文活动审计已生成"
