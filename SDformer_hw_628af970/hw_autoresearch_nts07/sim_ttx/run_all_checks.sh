#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

./sim_ttx/run_iverilog.sh
./sim_ttx/run_verilator_lint.sh
python3 scripts/ttx_zaf_reference.py \
  --output sim_ttx/build/ttx_zaf_reference_summary.json
./sim_ttx/run_yosys.sh

echo "PASS: all TTX RTL checks completed"
