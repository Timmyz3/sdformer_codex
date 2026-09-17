#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
verilator --cc --exe --top-module joined_core --Mdir neutral_obj --unroll-count 512 --unroll-stmts 100000 -Wno-fatal -CFLAGS '-std=c++17 -O2' joined_core.sv source_classifier_neutral.sv support_fc1_zero.sv joined_tb_neutral.cpp > neutral_build.log 2>&1
make -C neutral_obj -f Vjoined_core.mk -j2 >> neutral_build.log 2>&1
