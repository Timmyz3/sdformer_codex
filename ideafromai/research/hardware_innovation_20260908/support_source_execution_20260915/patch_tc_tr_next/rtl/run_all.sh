#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
/opt/anaconda3/bin/python3.12 prepare.py > prepare.log
verilator --cc --exe -CFLAGS '-std=c++17 -O2' --assert -O3 -Wno-fatal --top-module patch_core --Mdir obj_dir patch_core.sv tb.cpp > build.log 2>&1
make -C obj_dir -f Vpatch_core.mk -j2 >> build.log 2>&1
./obj_dir/Vpatch_core ordinary_r32.bin ordinary_small.csv ordinary_r32 small 2 > ordinary_small.log
./obj_dir/Vpatch_core regional_r96_a48.bin regional_small.csv regional_r96_a48 small 2 > regional_small.log
./obj_dir/Vpatch_core ordinary_r32.bin ordinary_full.csv ordinary_r32 full 1 > ordinary_full.log
./obj_dir/Vpatch_core regional_r96_a48.bin regional_full.csv regional_r96_a48 full 1 > regional_full.log
/opt/anaconda3/bin/python3.12 profile.py > profile.log
