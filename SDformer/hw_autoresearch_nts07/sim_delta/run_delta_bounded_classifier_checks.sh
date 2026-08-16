#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_delta/bounded_classifier"
ERIE_LINT="/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py"
mkdir -p "$BUILD"
rm -rf \
  "$BUILD/verilator_obj" \
  "$BUILD/delta4_verilator_obj" \
  "$BUILD/stream_verilator_obj" \
  "$BUILD/shared_verilator_obj" \
  "$BUILD/real_trace_verilator_obj" \
  "$BUILD/composite_verilator_obj" \
  "$BUILD/local5_composite_verilator_obj" \
  "$BUILD/dual_mode_verilator_obj"
cd "$ROOT"

iverilog -g2012 -Wall \
  -s tb_delta_bounded_classifier \
  -o "$BUILD/tb_delta_bounded_classifier.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  tb_delta/tb_delta_bounded_classifier.sv
vvp "$BUILD/tb_delta_bounded_classifier.vvp" \
  | tee "$BUILD/iverilog.log"

iverilog -g2012 -Wall \
  -s tb_alpha_xnor_delta4 \
  -o "$BUILD/tb_alpha_xnor_delta4.vvp" \
  rtl_delta/alpha_xnor_delta4.sv \
  tb_delta/tb_alpha_xnor_delta4.sv
vvp "$BUILD/tb_alpha_xnor_delta4.vvp" \
  | tee "$BUILD/delta4_iverilog.log"

iverilog -g2012 -Wall \
  -s tb_delta_bounded_classifier_stream \
  -o "$BUILD/tb_delta_bounded_classifier_stream.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  tb_delta/tb_delta_bounded_classifier_stream.sv
vvp "$BUILD/tb_delta_bounded_classifier_stream.vvp" \
  | tee "$BUILD/stream_iverilog.log"

iverilog -g2012 -Wall \
  -s tb_delta_shared_pipeline \
  -o "$BUILD/tb_delta_shared_pipeline.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  tb_delta/tb_delta_shared_pipeline.sv
vvp "$BUILD/tb_delta_shared_pipeline.vvp" \
  | tee "$BUILD/shared_iverilog.log"

iverilog -g2012 -Wall \
  -s tb_delta_shared_real_trace \
  -o "$BUILD/tb_delta_shared_real_trace.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  tb_delta/tb_delta_shared_real_trace.sv
vvp "$BUILD/tb_delta_shared_real_trace.vvp" \
  | tee "$BUILD/real_trace_iverilog.log"

iverilog -g2012 -Wall \
  -s tb_h67_tare4_composite_real_trace \
  -o "$BUILD/tb_h67_tare4_composite_real_trace.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/h67_tare4_composite_top.sv \
  tb_delta/tb_h67_tare4_composite_real_trace.sv
vvp "$BUILD/tb_h67_tare4_composite_real_trace.vvp" +STALL=0 \
  | tee "$BUILD/composite_nostall_iverilog.log"
vvp "$BUILD/tb_h67_tare4_composite_real_trace.vvp" +STALL=1 \
  | tee "$BUILD/composite_stall_iverilog.log"

iverilog -g2012 -Wall \
  -s tb_local5_tare4_composite_synthetic \
  -o "$BUILD/tb_local5_tare4_composite_synthetic.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/local5_tare4_composite_top.sv \
  tb_delta/tb_local5_tare4_composite_synthetic.sv
vvp "$BUILD/tb_local5_tare4_composite_synthetic.vvp" +STALL=0 \
  | tee "$BUILD/local5_composite_nostall_iverilog.log"
vvp "$BUILD/tb_local5_tare4_composite_synthetic.vvp" +STALL=1 \
  | tee "$BUILD/local5_composite_stall_iverilog.log"

iverilog -g2012 -Wall \
  -s tb_dual_mode_tare4_composite \
  -o "$BUILD/tb_dual_mode_tare4_composite.vvp" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/dual_mode_tare4_composite_top.sv \
  tb_delta/tb_dual_mode_tare4_composite.sv
vvp "$BUILD/tb_dual_mode_tare4_composite.vvp" +STALL=0 \
  | tee "$BUILD/dual_mode_nostall_iverilog.log"
vvp "$BUILD/tb_dual_mode_tare4_composite.vvp" +STALL=1 \
  | tee "$BUILD/dual_mode_stall_iverilog.log"

verilator --lint-only --timing -Wall \
  --top-module delta_bounded_classifier \
  rtl_delta/delta_bounded_classifier.sv \
  >"$BUILD/verilator_lint.log" 2>&1
verilator --lint-only --timing -Wall \
  --top-module alpha_xnor_delta4 \
  rtl_delta/alpha_xnor_delta4.sv \
  >"$BUILD/delta4_verilator_lint.log" 2>&1

verilator --binary --timing --assert -Wall \
  --top-module tb_delta_bounded_classifier \
  -Mdir "$BUILD/verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  tb_delta/tb_delta_bounded_classifier.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_delta_bounded_classifier" \
  | tee "$BUILD/verilator_assert.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_alpha_xnor_delta4 \
  -Mdir "$BUILD/delta4_verilator_obj" \
  rtl_delta/alpha_xnor_delta4.sv \
  verif_delta/alpha_xnor_delta4_assertions.sv \
  verif_delta/bind_alpha_xnor_delta4_assertions.sv \
  tb_delta/tb_alpha_xnor_delta4.sv \
  >"$BUILD/delta4_verilator_build.log" 2>&1
"$BUILD/delta4_verilator_obj/Vtb_alpha_xnor_delta4" \
  | tee "$BUILD/delta4_verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_delta_bounded_classifier_stream \
  -Mdir "$BUILD/stream_verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  tb_delta/tb_delta_bounded_classifier_stream.sv \
  >"$BUILD/stream_verilator_build.log" 2>&1
"$BUILD/stream_verilator_obj/Vtb_delta_bounded_classifier_stream" \
  | tee "$BUILD/stream_verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_delta_shared_pipeline \
  -Mdir "$BUILD/shared_verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  verif_delta/alpha_xnor_delta4_assertions.sv \
  verif_delta/bind_alpha_xnor_delta4_assertions.sv \
  tb_delta/tb_delta_shared_pipeline.sv \
  >"$BUILD/shared_verilator_build.log" 2>&1
"$BUILD/shared_verilator_obj/Vtb_delta_shared_pipeline" \
  | tee "$BUILD/shared_verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_delta_shared_real_trace \
  -Mdir "$BUILD/real_trace_verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  verif_delta/alpha_xnor_delta4_assertions.sv \
  verif_delta/bind_alpha_xnor_delta4_assertions.sv \
  tb_delta/tb_delta_shared_real_trace.sv \
  >"$BUILD/real_trace_verilator_build.log" 2>&1
"$BUILD/real_trace_verilator_obj/Vtb_delta_shared_real_trace" \
  | tee "$BUILD/real_trace_verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_h67_tare4_composite_real_trace \
  -Mdir "$BUILD/composite_verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/h67_tare4_composite_top.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  verif_delta/alpha_xnor_delta4_assertions.sv \
  verif_delta/bind_alpha_xnor_delta4_assertions.sv \
  verif_delta/tare4_residual_composite_assertions.sv \
  verif_delta/bind_tare4_residual_composite_assertions.sv \
  tb_delta/tb_h67_tare4_composite_real_trace.sv \
  >"$BUILD/composite_verilator_build.log" 2>&1
"$BUILD/composite_verilator_obj/Vtb_h67_tare4_composite_real_trace" \
  +STALL=0 | tee "$BUILD/composite_nostall_verilator.log"
"$BUILD/composite_verilator_obj/Vtb_h67_tare4_composite_real_trace" \
  +STALL=1 | tee "$BUILD/composite_stall_verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_local5_tare4_composite_synthetic \
  -Mdir "$BUILD/local5_composite_verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/local5_tare4_composite_top.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  verif_delta/alpha_xnor_delta4_assertions.sv \
  verif_delta/bind_alpha_xnor_delta4_assertions.sv \
  verif_delta/tare4_residual_composite_assertions.sv \
  verif_delta/bind_tare4_residual_composite_assertions.sv \
  tb_delta/tb_local5_tare4_composite_synthetic.sv \
  >"$BUILD/local5_composite_verilator_build.log" 2>&1
"$BUILD/local5_composite_verilator_obj/Vtb_local5_tare4_composite_synthetic" \
  +STALL=0 | tee "$BUILD/local5_composite_nostall_verilator.log"
"$BUILD/local5_composite_verilator_obj/Vtb_local5_tare4_composite_synthetic" \
  +STALL=1 | tee "$BUILD/local5_composite_stall_verilator.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_dual_mode_tare4_composite \
  -Mdir "$BUILD/dual_mode_verilator_obj" \
  rtl_delta/delta_bounded_classifier.sv \
  rtl_delta/alpha_xnor_delta4.sv \
  rtl_delta/alpha_xnor_raw32.sv \
  rtl_delta/tare4_residual_composite_core.sv \
  rtl_delta/dual_mode_tare4_composite_top.sv \
  verif_delta/delta_bounded_classifier_assertions.sv \
  verif_delta/bind_delta_bounded_classifier_assertions.sv \
  verif_delta/alpha_xnor_delta4_assertions.sv \
  verif_delta/bind_alpha_xnor_delta4_assertions.sv \
  verif_delta/tare4_residual_composite_assertions.sv \
  verif_delta/bind_tare4_residual_composite_assertions.sv \
  verif_delta/dual_mode_tare4_composite_assertions.sv \
  verif_delta/bind_dual_mode_tare4_composite_assertions.sv \
  tb_delta/tb_dual_mode_tare4_composite.sv \
  >"$BUILD/dual_mode_verilator_build.log" 2>&1
"$BUILD/dual_mode_verilator_obj/Vtb_dual_mode_tare4_composite" \
  +STALL=0 | tee "$BUILD/dual_mode_nostall_verilator.log"
"$BUILD/dual_mode_verilator_obj/Vtb_dual_mode_tare4_composite" \
  +STALL=1 | tee "$BUILD/dual_mode_stall_verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv rtl_delta/delta_bounded_classifier.sv; hierarchy -check -top delta_bounded_classifier; proc; opt; memory -nomap; check; stat"
yosys -q -l "$BUILD/delta4_yosys.log" -p \
  "read_verilog -sv rtl_delta/alpha_xnor_delta4.sv; hierarchy -check -top alpha_xnor_delta4; proc; opt; memory -nomap; check; stat"
yosys -q -l "$BUILD/composite_yosys.log" -p \
  "read_verilog -sv rtl_delta/delta_bounded_classifier.sv rtl_delta/alpha_xnor_delta4.sv rtl_delta/alpha_xnor_raw32.sv rtl_delta/tare4_residual_composite_core.sv rtl_delta/h67_tare4_composite_top.sv; hierarchy -check -top h67_tare4_composite_top; proc; opt; memory -nomap; check; stat"
yosys -q -l "$BUILD/local5_composite_yosys.log" -p \
  "read_verilog -sv rtl_delta/delta_bounded_classifier.sv rtl_delta/alpha_xnor_delta4.sv rtl_delta/alpha_xnor_raw32.sv rtl_delta/tare4_residual_composite_core.sv rtl_delta/local5_tare4_composite_top.sv; hierarchy -check -top local5_tare4_composite_top; proc; opt; memory -nomap; check; stat"
yosys -q -l "$BUILD/dual_mode_yosys.log" -p \
  "read_verilog -sv rtl_delta/delta_bounded_classifier.sv rtl_delta/alpha_xnor_delta4.sv rtl_delta/alpha_xnor_raw32.sv rtl_delta/tare4_residual_composite_core.sv rtl_delta/dual_mode_tare4_composite_top.sv; hierarchy -check -top dual_mode_tare4_composite_top; proc; opt; memory -nomap; check; stat"

python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/delta_bounded_classifier.sv \
  | tee "$BUILD/erie_rtl.log"
python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/alpha_xnor_delta4.sv \
  | tee "$BUILD/delta4_erie_rtl.log"
python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/alpha_xnor_raw32.sv \
  | tee "$BUILD/raw32_erie_rtl.log"
python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/tare4_residual_composite_core.sv \
  | tee "$BUILD/composite_core_erie_rtl.log"
python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/h67_tare4_composite_top.sv \
  | tee "$BUILD/h67_composite_erie_rtl.log"
python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/local5_tare4_composite_top.sv \
  | tee "$BUILD/local5_composite_erie_rtl.log"
python3 "$ERIE_LINT" --mode rtl --external none \
  rtl_delta/dual_mode_tare4_composite_top.sv \
  | tee "$BUILD/dual_mode_erie_rtl.log"

if grep -Eq '%Warning|%Error' \
  "$BUILD/verilator_lint.log" "$BUILD/delta4_verilator_lint.log" \
  "$BUILD/verilator_build.log" "$BUILD/delta4_verilator_build.log" \
  "$BUILD/stream_verilator_build.log" \
  "$BUILD/shared_verilator_build.log" \
  "$BUILD/real_trace_verilator_build.log" \
  "$BUILD/composite_verilator_build.log" \
  "$BUILD/local5_composite_verilator_build.log" \
  "$BUILD/dual_mode_verilator_build.log"; then
  echo "FAIL: Verilator存在warning/error" >&2
  exit 1
fi

echo "PASS: delta bounded classifier功能、断言、lint和综合可读性检查"
