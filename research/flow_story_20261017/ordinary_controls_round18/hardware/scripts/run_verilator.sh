#!/usr/bin/env bash
set -euo pipefail
flow_k1_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$flow_k1_root"
mkdir -p results/verilator/obj
verilator --cc --exe --top-module k1_f64 -Wall --Mdir results/verilator/obj rtl/k1_f64.sv "$flow_k1_root/tb/tb_k1.cpp" >results/verilator/build.log 2>&1
make -C results/verilator/obj -f Vk1_f64.mk -j2 >>results/verilator/build.log 2>&1
results/verilator/obj/Vk1_f64 "$@" >results/verilator/receipt.json
