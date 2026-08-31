#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ANALYZER="system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
CONTRACT="contracts/m505_h67_liveness_aware_single_port_parent_scratch_contract_r1_20260827.json"
OUT="results/m505_h67_liveness_aware_single_port_parent_scratch_r1_20260827"
PYTHON="/opt/anaconda3/envs/pytorch310/bin/python"

EXPECTED_ANALYZER="9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"
EXPECTED_CONTRACT="3c1e769fbb9f99e3b3bf50ee7d4658d62ae70aedcc736d5b5d59708f9b0bd5a5"

test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = "$EXPECTED_ANALYZER"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$EXPECTED_CONTRACT"
test ! -e "$OUT"
test -x "$PYTHON"

exec "$PYTHON" "$ANALYZER" \
  --contract "$CONTRACT" \
  --workers 3 \
  --chunksize 2 \
  --out "$OUT"
