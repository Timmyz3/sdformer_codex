#!/usr/bin/env bash
set -euo pipefail
# Run only in a fresh copy: this script writes its local results/ directory.
r33_hw="$(cd "$(dirname "$0")" && pwd)"
cd "$r33_hw"
python3 prepare.py
./run_rtl.sh > results/RTL_RUN_LOG.txt
./run_dc.sh > results/DC_RUN_LOG.txt
python3 build_supply.py
if ! g++ -O3 -std=c++17 -Wall -Wextra bitmap_supply.cpp -o bitmap_supply > results/supply_compile.log 2>&1; then cp results/supply_compile.log results/failures/supply_compile_failure.log; exit 1; fi
if ! ./bitmap_supply > results/SUPPLY_REPLAY_V1.json 2> results/supply.stderr; then cp results/supply.stderr results/failures/supply_run_failure.log; exit 1; fi
if ! python3 check_results.py > results/CHECK_LOG.txt 2> results/check.stderr; then cp results/check.stderr results/failures/count_check_failure.log; exit 1; fi
