#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare_source.py > prepare_source.log
verilator --cc --exe --top-module source_classifier --Mdir source_obj -Wno-fatal -CFLAGS '-std=c++17 -O2' ../source_classifier.sv source_tb.cpp > source_build.log 2>&1
make -C source_obj -f Vsource_classifier.mk -j2 >> source_build.log 2>&1
