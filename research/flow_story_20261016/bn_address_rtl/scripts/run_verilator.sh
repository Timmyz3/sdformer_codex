#!/usr/bin/env bash
set -euo pipefail
flow_addr_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$flow_addr_root"
mkdir -p results/verilator/obj
verilator --cc --exe --top-module bn_reduce_addr -Wall --Mdir results/verilator/obj rtl/bn_reduce_addr.sv "$flow_addr_root/tb/tb_addr.cpp" >results/verilator/build.log 2>&1
make -C results/verilator/obj -f Vbn_reduce_addr.mk -j2 >>results/verilator/build.log 2>&1
results/verilator/obj/Vbn_reduce_addr cases.txt results/verilator/cases.csv | tee results/verilator/run.log
