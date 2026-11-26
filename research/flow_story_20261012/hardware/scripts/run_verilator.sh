#!/usr/bin/env bash
set -euo pipefail
hw_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for lanes in 4 16; do
 mkdir -p "$hw_root/results/verilator_lanes$lanes"
 (
 cd "$hw_root/results/verilator_lanes$lanes"
 verilator --cc --exe --top-module dot_residual -GLANES=$lanes -CFLAGS "-DPE_LANES=$lanes" -Wall -Wno-fatal --Mdir obj "$hw_root/rtl/dot_residual.sv" "$hw_root/tb/dot_tb.cpp" >build.log 2>&1
 make -C obj -f Vdot_residual.mk -j2 >>build.log 2>&1
 ./obj/Vdot_residual >run.log 2>&1
 )
done
