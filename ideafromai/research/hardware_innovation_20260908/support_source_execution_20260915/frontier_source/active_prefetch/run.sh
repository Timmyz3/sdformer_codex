#!/bin/bash
set -eu
cd "$(dirname "$0")"
verilator --cc --exe --unroll-count 512 -Wall -Wno-fatal --top-module frontier_source frontier_source.sv tb.cpp -CFLAGS '-O2 -std=c++17' > build.log 2>&1
make -C obj_dir -f Vfrontier_source.mk -j4 >> build.log 2>&1
./obj_dir/Vfrontier_source 67 small_cycles.csv > small.log 2>&1
./obj_dir/Vfrontier_source 1027 expanded_old_cycles.csv ../../source_class_adapt/expanded_sources/old_class/source.bin --expanded > expanded_old.log 2>&1
./obj_dir/Vfrontier_source 1027 expanded_new_cycles.csv ../../source_class_adapt/expanded_sources/source.bin --expanded > expanded_new.log 2>&1
/opt/anaconda3/bin/python3.12 verify.py > verify.log 2>&1
