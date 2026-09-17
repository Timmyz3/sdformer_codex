#!/bin/bash
set -eu
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare.py
verilator --cc --exe --unroll-count 512 -Wall -Wno-fatal --top-module subset_psn subset_psn.sv tb.cpp -CFLAGS '-O2 -std=c++17' > build.log 2>&1
make -C obj_dir -f Vsubset_psn.mk -j4 >> build.log 2>&1
./obj_dir/Vsubset_psn 1 smoke > smoke.log 2>&1
./obj_dir/Vsubset_psn 37 all > run.log 2>&1
/opt/anaconda3/bin/python3.12 summarize.py
