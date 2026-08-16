#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_single_head_ibf"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/hitflow_implicit_bias_finalizer_accumulator.sv
  rtl_hitflow/gatestack_single_head_projection_top.sv
)
TB=tb_hitflow/tb_gatestack_single_head_projection_top.sv

iverilog -g2012 -Wall \
  -Ptb_gatestack_single_head_projection_top.IMPLICIT_BIAS_FINALIZE_ENABLE=1 \
  -s tb_gatestack_single_head_projection_top \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"
grep -Eq 'PASS: single-head req/rsp projection bsf=0 ibf=1 .*bias=8 req=2 rsp=2 .*final_stalls=[1-9][0-9]*' \
  "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  "-GIMPLICIT_BIAS_FINALIZE_ENABLE=1'b1" \
  --top-module tb_gatestack_single_head_projection_top \
  -Mdir "$BUILD/verilator_obj" "${RTL[@]}" \
  verif_hitflow/gatestack_single_head_projection_assertions.sv \
  verif_hitflow/bind_gatestack_single_head_projection_assertions.sv \
  verif_hitflow/hitflow_implicit_bias_finalizer_assertions.sv \
  verif_hitflow/bind_hitflow_implicit_bias_finalizer_assertions.sv \
  "$TB" >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_single_head_projection_top" \
  | tee "$BUILD/verilator.log"
grep -Eq 'PASS: single-head req/rsp projection bsf=0 ibf=1 .*bias=8 req=2 rsp=2 .*final_stalls=[1-9][0-9]*' \
  "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys_check.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_single_head_projection_top -chparam IMPLICIT_BIAS_FINALIZE_ENABLE 1; proc; flatten; opt; memory -nomap; opt; check -assert; stat"

for mode in current ibf; do
  enable=0
  if [[ "$mode" == ibf ]]; then enable=1; fi
  yosys -q -l "$BUILD/${mode}_nangate45.log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_single_head_projection_top -chparam IMPLICIT_BIAS_FINALIZE_ENABLE $enable; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert; stat -liberty $LIB"
done

echo 'RESULT suite=single_head_ibf status=PASS exact=PASS ibf_requests=2 iverilog=PASS verilator_sva=PASS yosys=PASS mapping=PASS'
