#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ANALYZER="system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
CONTRACT="contracts/m504_h67_single_port_parent_scratch_execution_contract_r3_20260827.json"
OUT="results/m504_h67_single_port_parent_scratch_r3_20260827"
PYTHON="/opt/anaconda3/envs/pytorch310/bin/python"

EXPECTED_ANALYZER="9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"
EXPECTED_CONTRACT="64f1ac425520816af5250647d251c14a34e28a715723c98a50b4234b01bd9a5d"

test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = "$EXPECTED_ANALYZER"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$EXPECTED_CONTRACT"
test ! -e "$OUT"
test -x "$PYTHON"

exec "$PYTHON" "$ANALYZER" \
  --contract "$CONTRACT" \
  --workers 3 \
  --chunksize 2 \
  --out "$OUT"
