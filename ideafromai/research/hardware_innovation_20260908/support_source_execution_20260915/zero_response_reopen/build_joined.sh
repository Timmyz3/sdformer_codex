#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare_joined.py > prepare_joined_zero.log
/opt/anaconda3/bin/python3.12 prepare_joined.py --variant l2 > prepare_joined_l2.log
verilator --cc --exe --top-module joined_core --Mdir joined_obj --unroll-count 512 --unroll-stmts 100000 -Wno-fatal -CFLAGS '-std=c++17 -O2' joined_core.sv ../source_classifier.sv support_fc1_zero.sv joined_tb.cpp > joined_build.log 2>&1
make -C joined_obj -f Vjoined_core.mk -j2 >> joined_build.log 2>&1
