#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash build.sh
./obj_dir/Vjoined_core 4 4 2 1 2 ../joined_zero.bin small.csv zero_response 1 >small.log 2>&1
./obj_dir/Vjoined_core 34 4 1 1 2 inputs_expanded.bin expanded.csv zero_response 1 >expanded.log 2>&1
/opt/anaconda3/bin/python3.12 profile.py >profile.log
/opt/anaconda3/bin/python3.12 finalize.py
