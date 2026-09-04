#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
./run_verilator_lint.sh
./run_iverilog.sh
./run_verilator_assertions.sh
./run_erie_static_lint.sh
./run_yosys.sh
