#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare.py > prepare.log
verilator --version > tool_version.txt
verilator --cc --exe --top-module interval_tile -Wall -Wno-fatal --Mdir build interval_tile.sv tb.cpp > build.log 2>&1
make -C build -f Vinterval_tile.mk -j2 > compile.log 2>&1
./build/Vinterval_tile > simulation.log 2>&1
/opt/anaconda3/bin/python3.12 analyze_results.py > analysis.log
