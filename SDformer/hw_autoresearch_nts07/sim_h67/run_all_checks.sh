#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

./sim_h67/run_iverilog.sh
./sim_h67/run_verilator_lint.sh
./sim_h67/run_verilator_assertions.sh
./sim_h67/run_gatelevel_sim.sh
python3 scripts/h67_score_reference.py \
  --output results/h67_score_reference.json
./sim_h67/run_yosys.sh

echo "PASS: H67 incremental RTL checks completed"
