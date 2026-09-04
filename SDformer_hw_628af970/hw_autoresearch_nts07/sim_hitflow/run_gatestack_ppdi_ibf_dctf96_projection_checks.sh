#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_ppdi_ibf_dctf96_banklocal_projection_top"
TB="tb_hitflow/tb_gatestack_dctf96_banklocal_projection_top.sv"
TOP="tb_gatestack_dctf96_banklocal_projection_top"
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
SVA=(
  verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv
  verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv
  verif_hitflow/gatestack_dctf96_term_datapath_top_assertions.sv
  verif_hitflow/bind_gatestack_dctf96_term_datapath_top_assertions.sv
  verif_hitflow/gatestack_ppdi_term_event_adapter_2c_assertions.sv
  verif_hitflow/bind_gatestack_ppdi_term_event_adapter_2c_assertions.sv
  verif_hitflow/gatestack_ppdi_dctf_term_fabric_assertions.sv
  verif_hitflow/bind_gatestack_ppdi_dctf_term_fabric_assertions.sv
  verif_hitflow/gatestack_ppdi_dctf32_bank_executor_assertions.sv
  verif_hitflow/bind_gatestack_ppdi_dctf32_bank_executor_assertions.sv
  verif_hitflow/hitflow_implicit_bias_finalizer_assertions.sv
  verif_hitflow/bind_hitflow_implicit_bias_finalizer_assertions.sv
)

mkdir -p "$BUILD"
cd "$ROOT"

run_icarus() {
  local name="$1"
  local ppdi="$2"
  local ibf="$3"

  iverilog -g2012 -Wall -s "$TOP" \
    -P"$TOP.ADAPTER_CONTEXTS=2" \
    -P"$TOP.PPDI_ENABLE=$ppdi" \
    -P"$TOP.IMPLICIT_BIAS_FINALIZE_ENABLE=$ibf" \
    -P"$TOP.STATIONARY_BIAS_TEST=1" \
    -o "$BUILD/$name.vvp" "${RTL[@]}" "$TB" \
    >"$BUILD/${name}_iverilog_build.log" 2>&1
  if grep -Eiq '(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
      "$BUILD/${name}_iverilog_build.log"; then
    cat "$BUILD/${name}_iverilog_build.log" >&2
    exit 1
  fi
  vvp "$BUILD/$name.vvp" | tee "$BUILD/${name}_iverilog.log"
  grep -q '^PASS DCTF96 BANKLOCAL PROJECTION ' \
    "$BUILD/${name}_iverilog.log"
}

run_icarus scalar_rmw 0 0
run_icarus ppdi_rmw 1 0
run_icarus scalar_ibf 0 1
run_icarus ppdi_ibf 1 1

rm -rf "$BUILD/verilator_obj"
verilator --binary --timing --assert -Wall \
  --top-module "$TOP" \
  -GADAPTER_CONTEXTS=2 \
  "-GPPDI_ENABLE=1'b1" \
  "-GIMPLICIT_BIAS_FINALIZE_ENABLE=1'b1" \
  "-GSTATIONARY_BIAS_TEST=1'b1" \
  -Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" "$TB" "${SVA[@]}" \
  >"$BUILD/ppdi_ibf_verilator_build.log" 2>&1
if grep -Eiq '(%Warning|%Error|warning:|error:|fatal:)' \
    "$BUILD/ppdi_ibf_verilator_build.log"; then
  cat "$BUILD/ppdi_ibf_verilator_build.log" >&2
  exit 1
fi
"$BUILD/verilator_obj/V$TOP" | tee "$BUILD/ppdi_ibf_verilator.log"
grep -q '^PASS DCTF96 BANKLOCAL PROJECTION ' \
  "$BUILD/ppdi_ibf_verilator.log"

yosys -q -l "$BUILD/ppdi_ibf_yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; chparam -set ADAPTER_CONTEXTS 2 -set PPDI_ENABLE 1 -set IMPLICIT_BIAS_FINALIZE_ENABLE 1 gatestack_dctf96_banklocal_projection_top; hierarchy -check -top gatestack_dctf96_banklocal_projection_top; proc; flatten; opt; memory -nomap; opt; check -assert; stat"
grep -Ei '(^Warning:|^ERROR:|fatal)' "$BUILD/ppdi_ibf_yosys.log" \
  | grep -Ev '^Warning: Replacing memory \\[^ ]+ with list of registers\.' \
  >"$BUILD/ppdi_ibf_yosys_unexpected.log" || true
if [[ -s "$BUILD/ppdi_ibf_yosys_unexpected.log" ]]; then
  cat "$BUILD/ppdi_ibf_yosys_unexpected.log" >&2
  exit 1
fi

echo "PASS: four-way PPDI/IBF Icarus ablation, combined Verilator assertions, and Yosys hierarchy/check/stat"
