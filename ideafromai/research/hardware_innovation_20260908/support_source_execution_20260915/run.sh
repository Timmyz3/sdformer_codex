#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 export_source_tb.py
verilator --cc --exe --top-module source_classifier -Wno-fatal -CFLAGS '-std=c++17 -O2' source_classifier.sv tb_source.cpp > build.log 2>&1
make -C obj_dir -f Vsource_classifier.mk -j2 >> build.log 2>&1
./obj_dir/Vsource_classifier "$@"
