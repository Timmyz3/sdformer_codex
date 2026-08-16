#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULT_DIR="$ROOT/results/local5_relation_memo_rtl_20260806"
BUILD_DIR="$ROOT/build_qfit/relation_memo_20260806"
mkdir -p "$RESULT_DIR" "$BUILD_DIR"

cd "$ROOT"

COMMON_RTL=(
  rtl_qfit/qfit_retirement_scheduler.sv
  rtl_qfit/qfit_sync_1r1w_bank.sv
  rtl_qfit/qfit_relation_transpose_leaf.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_exposure_relation_vault.sv
  rtl_qfit/qfit_fcsr_relation_memo_top.sv
)

PROJECTION_RTL=(
  "${COMMON_RTL[@]}"
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_tcfm5_acc_bank.sv
  rtl_qfit/qfit_tcfm5_projection_top.sv
  rtl_qfit/qfit_fcsr_relation_memo_projection_top.sv
)

ENGINE_RTL=(
  "${PROJECTION_RTL[@]}"
  rtl_qfit/qfit_relation_memo_tile_controller.sv
  rtl_qfit/qfit_local5_relation_memo_tile_engine.sv
)

python -m unittest tests/test_model_local5_relation_vault.py \
  2>&1 | tee "$RESULT_DIR/model_unittest.log"

iverilog -g2012 -Wall \
  -s tb_qfit_exposure_relation_vault \
  -o "$BUILD_DIR/tb_vault.vvp" \
  rtl_qfit/qfit_sync_relation_bank.sv \
  rtl_qfit/qfit_exposure_relation_vault.sv \
  tb_qfit/tb_qfit_exposure_relation_vault.sv \
  2>&1 | tee "$RESULT_DIR/iverilog_vault_compile.log"
vvp "$BUILD_DIR/tb_vault.vvp" \
  2>&1 | tee "$RESULT_DIR/iverilog_vault_run.log"

iverilog -g2012 -Wall \
  -s tb_qfit_fcsr_relation_memo_top \
  -o "$BUILD_DIR/tb_fcsr_memo.vvp" \
  "${COMMON_RTL[@]}" \
  tb_qfit/tb_qfit_fcsr_relation_memo_top.sv \
  2>&1 | tee "$RESULT_DIR/iverilog_fcsr_compile.log"
vvp "$BUILD_DIR/tb_fcsr_memo.vvp" \
  2>&1 | tee "$RESULT_DIR/iverilog_fcsr_run.log"

iverilog -g2012 -Wall \
  -s tb_qfit_fcsr_relation_memo_projection_top \
  -o "$BUILD_DIR/tb_projection_miter.vvp" \
  "${PROJECTION_RTL[@]}" \
  tb_qfit/tb_qfit_fcsr_relation_memo_projection_top.sv \
  2>&1 | tee "$RESULT_DIR/iverilog_projection_compile.log"
vvp "$BUILD_DIR/tb_projection_miter.vvp" \
  2>&1 | tee "$RESULT_DIR/iverilog_projection_run.log"

iverilog -g2012 -Wall \
  -s tb_qfit_local5_relation_memo_tile_engine \
  -o "$BUILD_DIR/tb_tile_engine.vvp" \
  "${ENGINE_RTL[@]}" \
  tb_qfit/tb_qfit_local5_relation_memo_tile_engine.sv \
  2>&1 | tee "$RESULT_DIR/iverilog_tile_engine_compile.log"
vvp "$BUILD_DIR/tb_tile_engine.vvp" \
  2>&1 | tee "$RESULT_DIR/iverilog_tile_engine_run.log"

rm -rf "$BUILD_DIR/verilator_vault"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "$BUILD_DIR/verilator_vault" \
  --top-module tb_qfit_exposure_relation_vault \
  rtl_qfit/qfit_sync_relation_bank.sv \
  rtl_qfit/qfit_exposure_relation_vault.sv \
  verif_qfit/qfit_exposure_relation_vault_assertions.sv \
  tb_qfit/tb_qfit_exposure_relation_vault.sv \
  2>&1 | tee "$RESULT_DIR/verilator_assert_compile.log"
"$BUILD_DIR/verilator_vault/Vtb_qfit_exposure_relation_vault" \
  2>&1 | tee "$RESULT_DIR/verilator_assert_run.log"

rm -rf "$BUILD_DIR/verilator_controller"
verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "$BUILD_DIR/verilator_controller" \
  --top-module tb_qfit_relation_memo_tile_controller \
  rtl_qfit/qfit_relation_memo_tile_controller.sv \
  verif_qfit/qfit_relation_memo_tile_controller_assertions.sv \
  tb_qfit/tb_qfit_relation_memo_tile_controller.sv \
  2>&1 | tee "$RESULT_DIR/verilator_controller_assert_compile.log"
"$BUILD_DIR/verilator_controller/Vtb_qfit_relation_memo_tile_controller" \
  2>&1 | tee "$RESULT_DIR/verilator_controller_assert_run.log"

verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module qfit_local5_relation_memo_tile_engine \
  "${ENGINE_RTL[@]}" \
  2>&1 | tee "$RESULT_DIR/verilator_tile_engine_lint.log"

yosys -q -l "$RESULT_DIR/yosys_tile_engine.log" -p "
  read_verilog -sv ${ENGINE_RTL[*]};
  hierarchy -check -top qfit_local5_relation_memo_tile_engine;
  proc; opt; memory -nomap; opt; check; stat
" 2>&1 | tee "$RESULT_DIR/yosys_tile_engine_console.log"

printf '%s\n' \
  'PASS Local5 exposure-aware relation memo fullflow' \
  '证据边界：功能仿真、SVA、lint、综合可读；不代表 ASIC PPA。' \
  | tee "$RESULT_DIR/SUMMARY.txt"
