#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
verilator --cc --exe -Wno-fatal --top-module test_top --Mdir obj_dir operator_slice.sv test_top.sv tb.cpp >build.log 2>&1
make -C obj_dir -f Vtest_top.mk -j4 >>build.log 2>&1
./obj_dir/Vtest_top >functional.log 2>&1
cat functional.log
