#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ANALYZER="system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
CONTRACT="contracts/m504_h67_single_port_parent_scratch_execution_contract_r1_20260827.json"
OUT="results/m504_h67_single_port_parent_scratch_r1_20260827"
PYTHON="/opt/anaconda3/envs/pytorch310/bin/python"

EXPECTED_ANALYZER="3120cb600210548a19fc9756add0e45a5ab900fe776b5ec131dd53c9b9854e1e"
EXPECTED_CONTRACT="162e3bfdc1ae45f03d9d8da0aad64d819bbb1d6842fe925836547bf7eb7c35d6"

test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = "$EXPECTED_ANALYZER"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$EXPECTED_CONTRACT"
test ! -e "$OUT"
test -x "$PYTHON"

exec "$PYTHON" "$ANALYZER" \
  --contract "$CONTRACT" \
  --workers 3 \
  --chunksize 2 \
  --out "$OUT"
