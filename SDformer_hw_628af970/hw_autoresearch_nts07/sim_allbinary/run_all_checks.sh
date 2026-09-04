#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/sim_allbinary/run_iverilog.sh"
"$ROOT/sim_allbinary/run_verilator_lint.sh"
"$ROOT/sim_allbinary/run_yosys.sh"

echo "PASS: all UniBin-H60 RTL checks completed"
