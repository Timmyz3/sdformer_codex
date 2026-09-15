#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 export_tb.py
verilator --cc --exe --top-module support_fc1 --unroll-count 512 --unroll-stmts 100000 -Wno-fatal -CFLAGS '-std=c++17 -O2' support_fc1.sv tb.cpp > build.log 2>&1
make -C obj_dir -f Vsupport_fc1.mk -j2 >> build.log 2>&1
./obj_dir/Vsupport_fc1 "$@"
if [ "$#" -eq 0 ]; then /opt/anaconda3/bin/python3.12 summarize.py; fi
