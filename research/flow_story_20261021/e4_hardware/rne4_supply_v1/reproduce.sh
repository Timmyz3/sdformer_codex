#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare.py
g++ -O3 -std=c++17 -Wall -Wextra rne4_supply.cpp -o rne4_supply
./rne4_supply > results/SUPPLY_REPLAY_V1.json
python3 check_and_report.py
