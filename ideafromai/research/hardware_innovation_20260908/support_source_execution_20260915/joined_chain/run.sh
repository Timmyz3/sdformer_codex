#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare.py > prepare.log
verilator --cc --exe --top-module joined_core --unroll-count 512 --unroll-stmts 100000 -Wno-fatal -CFLAGS '-std=c++17 -O2' joined_core.sv ../source_classifier.sv ../../support_lut_execution_20260915/support_fc1.sv tb.cpp > build.log 2>&1
make -C obj_dir -f Vjoined_core.mk -j2 >> build.log 2>&1
./obj_dir/Vjoined_core "$@"
