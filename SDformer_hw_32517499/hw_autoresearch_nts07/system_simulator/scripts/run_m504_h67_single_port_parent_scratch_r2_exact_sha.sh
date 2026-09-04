#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ANALYZER="system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
CONTRACT="contracts/m504_h67_single_port_parent_scratch_execution_contract_r2_20260827.json"
OUT="results/m504_h67_single_port_parent_scratch_r2_20260827"
PYTHON="/opt/anaconda3/envs/pytorch310/bin/python"

EXPECTED_ANALYZER="3017dbc290db06924d4f05be7346ef2c4955169afa94fb9d24287bafd353f8df"
EXPECTED_CONTRACT="a6bddb1c94c5e2e5379e8886abfc65349bbb6a0cceb45376efb16672df9e64a1"

test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = "$EXPECTED_ANALYZER"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$EXPECTED_CONTRACT"
test ! -e "$OUT"
test -x "$PYTHON"

exec "$PYTHON" "$ANALYZER" \
  --contract "$CONTRACT" \
  --workers 3 \
  --chunksize 2 \
  --out "$OUT"
