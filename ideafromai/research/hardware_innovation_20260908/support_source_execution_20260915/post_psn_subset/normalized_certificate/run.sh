#!/bin/bash
set -eu
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 implement.py
/opt/anaconda3/bin/python3.12 make_tb.py
/opt/anaconda3/bin/python3.12 prepare_diagnostic.py
verilator --cc --exe --unroll-count 512 -Wall -Wno-fatal --top-module normalized_fc1 normalized_fc1.sv normalized_bound.sv tb.cpp -CFLAGS '-O2 -std=c++17' > build.log 2>&1
make -C obj_dir -f Vnormalized_fc1.mk -j4 >> build.log 2>&1
./obj_dir/Vnormalized_fc1 smoke > smoke.log 2>&1
./obj_dir/Vnormalized_fc1 all > run.log 2>&1
./obj_dir/Vnormalized_fc1 swap > swap.log 2>&1
verilator --cc --exe --unroll-count 512 -Wall -Wno-fatal --top-module normalized_leaf --Mdir obj_leaf normalized_leaf.sv normalized_bound.sv leaf_tb.cpp -CFLAGS '-O2 -std=c++17' > build_leaf.log 2>&1
make -C obj_leaf -f Vnormalized_leaf.mk -j4 >> build_leaf.log 2>&1
./obj_leaf/Vnormalized_leaf 2 diagnostics diagnostics.bin > diagnostics.log 2>&1
/opt/anaconda3/bin/python3.12 summarize.py > summary.log 2>&1
