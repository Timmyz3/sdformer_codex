#!/bin/bash
set -eu
cd "$(dirname "$0")"
test -f ../../../support_lut_execution_20260915/cases.bin
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 implement.py
/opt/anaconda3/bin/python3.12 make_tb.py
verilator --cc --exe --unroll-count 512 -Wall -Wno-fatal --top-module joined_fc1 joined_fc1.sv tb.cpp -CFLAGS '-O2 -std=c++17' > build.log 2>&1
make -C obj_dir -f Vjoined_fc1.mk -j4 >> build.log 2>&1
./obj_dir/Vjoined_fc1 smoke > smoke.log 2>&1
./obj_dir/Vjoined_fc1 all > run.log 2>&1
./obj_dir/Vjoined_fc1 swap > swap.log 2>&1
/opt/anaconda3/bin/python3.12 summarize.py > summary.log 2>&1
