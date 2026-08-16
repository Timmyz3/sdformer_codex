#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ppdi_ibf_dctf96_banklocal_projection_top"
LIB="third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
RTL=(
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter.sv
  rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv
  rtl_hitflow/gatestack_ppdi_token_bank.sv
  rtl_hitflow/gatestack_ppdi_term_event_adapter_2c.sv
  rtl_hitflow/gatestack_dctf_term_fabric.sv
  rtl_hitflow/gatestack_ppdi_dctf_term_fabric.sv
  rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/hitflow_implicit_bias_finalizer_accumulator.sv
  rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv
)

mkdir -p "$BUILD"
cd "$ROOT"

map_mode() {
  local name="$1"
  local ppdi="$2"
  local ibf="$3"
  local log="$BUILD/map_${name}.log"

  if [[ !( "${FORCE_IBF:-0}" == 1 && "$ibf" == 1 ) ]] &&
     grep -Fq "Chip area for module '\\gatestack_dctf96_banklocal_projection_top'" \
      "$log" 2>/dev/null; then
    echo "SKIP: $name mapping already complete"
    return
  fi

  yosys -q -l "$log" -p \
    "read_liberty -lib $LIB; read_verilog -sv ${RTL[*]}; chparam -set ADAPTER_CONTEXTS 2 -set PPDI_ENABLE $ppdi -set IMPLICIT_BIAS_FINALIZE_ENABLE $ibf gatestack_dctf96_banklocal_projection_top; hierarchy -check -top gatestack_dctf96_banklocal_projection_top; proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; check -assert; stat -liberty $LIB"
  grep -Fq "Chip area for module '\\gatestack_dctf96_banklocal_projection_top'" "$log"
  echo "PASS: $name open-library logic mapping"
}

map_mode scalar_rmw 0 0
map_mode ppdi_rmw 1 0
map_mode scalar_ibf 0 1
map_mode ppdi_ibf 1 1
