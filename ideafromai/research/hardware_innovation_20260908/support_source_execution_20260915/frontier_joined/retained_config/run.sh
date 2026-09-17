#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 implement.py
verilator --cc --exe --top-module joined_core --unroll-count 512 --unroll-stmts 100000 -Wno-fatal -CFLAGS '-std=c++17 -O2' joined_core.sv frontier_source.sv ../../../support_lut_execution_20260915/support_fc1.sv tb.cpp > build.log 2>&1
make -C obj_dir -f Vjoined_core.mk -j2 >> build.log 2>&1
./obj_dir/Vjoined_core 34 4 1 1 1 ../inputs_expanded.bin expanded_old.csv response_class 1 > expanded_old.log 2>&1
./obj_dir/Vjoined_core 34 4 1 1 1 ../inputs_adapt_expanded.bin expanded_new.csv adapt_response_class 1 > expanded_new.log 2>&1
