#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/ppdi_ibf_real_trace_20260801"
LOGS="$OUT/logs"
BUILD="$OUT/build"
VECTORS="$ROOT/results/gatestack_dctf96_real_trace_20260720/vectors"
TB="tb_hitflow/tb_gatestack_dctf96_banklocal_projection_real_trace.sv"
TOP="tb_gatestack_dctf96_banklocal_projection_real_trace"
HEADS=(3 6 12 24)
MODES=(scalar_rmw ppdi_rmw scalar_ibf ppdi_ibf)
PPDI=(0 1 0 1)
IBF=(0 0 1 1)
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

mkdir -p "$LOGS" "$BUILD"
cd "$ROOT"

for mode_index in 0 1 2 3; do
  mode="${MODES[$mode_index]}"
  mkdir -p "$LOGS/$mode" "$BUILD/$mode"
  for stage in 0 1 2 3; do
    iverilog -g2012 -Wall -s "$TOP" \
      -P"$TOP.STAGE=$stage" \
      -P"$TOP.HEADS=${HEADS[$stage]}" \
      -P"$TOP.ADAPTER_CONTEXTS=2" \
      -P"$TOP.PPDI_ENABLE=${PPDI[$mode_index]}" \
      -P"$TOP.IMPLICIT_BIAS_FINALIZE_ENABLE=${IBF[$mode_index]}" \
      -o "$BUILD/$mode/s$stage.vvp" "${RTL[@]}" "$TB" \
      >"$LOGS/$mode/iverilog_build_s$stage.log" 2>&1
    if grep -Eiq '(^|[^[:alpha:]])(error:|syntax error)([^[:alpha:]]|$)' \
        "$LOGS/$mode/iverilog_build_s$stage.log"; then
      cat "$LOGS/$mode/iverilog_build_s$stage.log" >&2
      exit 1
    fi
    vvp "$BUILD/$mode/s$stage.vvp" "+VECTOR_DIR=$VECTORS/s$stage" \
      | tee "$LOGS/$mode/icarus_s$stage.log"
    grep -q "^PASS DCTF96 REAL TRACE stage=S$stage " \
      "$LOGS/$mode/icarus_s$stage.log"
  done
done

rm -rf "$BUILD/ppdi_ibf/verilator_obj"
verilator --binary --timing --assert -Wall \
  --top-module "$TOP" \
  -GSTAGE=0 -GHEADS=3 -GADAPTER_CONTEXTS=2 \
  "-GPPDI_ENABLE=1'b1" \
  "-GIMPLICIT_BIAS_FINALIZE_ENABLE=1'b1" \
  -Mdir "$BUILD/ppdi_ibf/verilator_obj" \
  "${RTL[@]}" "$TB" "${SVA[@]}" \
  >"$LOGS/ppdi_ibf/verilator_build_s0.log" 2>&1
if grep -Eiq '(%Warning|%Error|warning:|error:|fatal:)' \
    "$LOGS/ppdi_ibf/verilator_build_s0.log"; then
  cat "$LOGS/ppdi_ibf/verilator_build_s0.log" >&2
  exit 1
fi
"$BUILD/ppdi_ibf/verilator_obj/V$TOP" "+VECTOR_DIR=$VECTORS/s0" \
  | tee "$LOGS/ppdi_ibf/verilator_s0.log"
grep -q '^PASS DCTF96 REAL TRACE stage=S0 ' \
  "$LOGS/ppdi_ibf/verilator_s0.log"

python scripts/summarize_ppdi_ibf_real_trace.py \
  --log-root "$LOGS" --output-dir "$OUT" \
  --mapping-report results/ppdi_ibf_projection_20260801/report.json

echo "PASS: PPDI/IBF四路Motion真实S0-S3逐元素回放与组合S0动态SVA"
