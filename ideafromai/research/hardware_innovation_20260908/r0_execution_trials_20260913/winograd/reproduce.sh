#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 prepare_real.py
verilator --cc --exe -Wall --top-module winograd_tile --Mdir obj_dir winograd_tile.sv tb.cpp > build_verilator.log 2>&1
make -C obj_dir -f Vwinograd_tile.mk -j2 CXXFLAGS=-std=c++11 > build_make.log 2>&1
/opt/anaconda3/bin/python3.12 run.py > run.log 2>&1
/opt/anaconda3/bin/python3.12 summarize.py
