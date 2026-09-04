#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

./sim_h68/run_structural_audit.sh
./sim_h68/run_iverilog.sh
./sim_h68/run_verilator_lint.sh
./sim_h68/run_verilator_assertions.sh
./sim_h68/run_gatelevel_sim.sh
python3 scripts/h68_score_reference.py \
  --output results/h68_score_reference.json
python3 scripts/audit_h68_deploy_contract.py \
  --repo "$ROOT/.." \
  --output results/h68_deploy_contract.json
./sim_h68/run_yosys.sh

echo "PASS: H68部署RTL全部增量检查完成"
