#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
/opt/anaconda3/bin/python3.12 implement.py
verilator --cc --exe -CFLAGS "-std=c++17 -O2" --assert -O3 -Wno-fatal --top-module frontier_source --Mdir obj_dir distance_source.sv tb.cpp > build.log 2>&1
make -C obj_dir -f Vfrontier_source.mk -j2 >> build.log 2>&1
./obj_dir/Vfrontier_source ../joined_chain/inputs.bin old_small.csv old_class 4 2 1 3 > old_small.log
./obj_dir/Vfrontier_source ../joined_chain/inputs_adapt.bin adapt_small.csv adapt_class 4 1 2 3 > adapt_small.log
./obj_dir/Vfrontier_source ../zero_response_reopen/joined_zero.bin zero_small.csv zero_class 4 2 2 3 > zero_small.log
./obj_dir/Vfrontier_source ../frontier_joined/inputs_expanded.bin old_expanded.csv old_class 32 1 1 3 > old_expanded.log
./obj_dir/Vfrontier_source ../frontier_joined/inputs_adapt_expanded.bin adapt_expanded.csv adapt_class 32 1 2 3 > adapt_expanded.log
./obj_dir/Vfrontier_source ../zero_response_reopen/frontier_retained/inputs_expanded.bin zero_expanded.csv zero_class 32 1 2 3 > zero_expanded.log
/opt/anaconda3/bin/python3.12 summarize.py > summary.log
