#!/usr/bin/env bash
set -euo pipefail
# Run from any directory. A new output location is mandatory.
e4_root="$(cd "$(dirname "$0")" && pwd)"
e4_out="${1:?usage: reproduce.sh /absolute/new/output}"
if [ -e "$e4_out" ]; then echo 'Output directory already exists' >&2; exit 1; fi
mkdir -p "$e4_out/obj"
g++ -O3 -std=c++17 -Wall -Wextra "$e4_root/supply_replay.cpp" -o "$e4_out/supply_replay"
"$e4_out/supply_replay" > "$e4_out/SUPPLY_REPLAY_V1.json"
verilator --cc --exe -Wno-fatal --top-module pe_compare_top --Mdir "$e4_out/obj" "$e4_root/rtl/plane_pe.sv" "$e4_root/rtl/tb_plane_pe.cpp" > "$e4_out/elaborate.log" 2>&1
make -C "$e4_out/obj" -f Vpe_compare_top.mk -j2 > "$e4_out/compile.log" 2>&1
"$e4_out/obj/Vpe_compare_top" "$e4_root/../../flow_story_20261020/round30/actual_v1/final_vector_export_v1/final_vectors_v1.txt" > "$e4_out/RTL_INTEGER_V1.json"
# To rerun DC: create another new empty directory, cd there, then run:
# FLOW_TOP=binary_plane_pe FLOW_SOURCE="$e4_root/rtl/plane_pe.sv" \
# /opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f "$e4_root/rtl/synth_pe.tcl" > run.log 2>&1
# The shipped check_results.py validates the frozen results/ files. It does not
# overwrite the existing freeze or silently substitute a rerun for that record.
