#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:?VECTOR_DIR is required}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
BUILD="$ROOT/build_hitflow/checkpoint_atlif_dptme"
mkdir -p "$BUILD" "$RESULT_DIR"

VECTOR_DIR="$(realpath "$VECTOR_DIR")"
RESULT_DIR="$(realpath "$RESULT_DIR")"
cd "$VECTOR_DIR"

iverilog -g2012 -DSIMULATOR_ICARUS -DATLIF_TB_PROGRESS \
  -I"$VECTOR_DIR" -s tb_checkpoint_atlif_dptme \
  -o "$BUILD/tb_checkpoint_atlif_dptme.vvp" \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/tb_hitflow/tb_checkpoint_atlif_dptme.sv"
# Local5 rank-1 has 81 commands / 25920 events; 120s is enough for iverilog
# but keep headroom. Verilator sim below needs much longer.
ICARUS_TIMEOUT_S="${ICARUS_TIMEOUT_S:-600}"
timeout "${ICARUS_TIMEOUT_S}s" stdbuf -oL vvp "$BUILD/tb_checkpoint_atlif_dptme.vvp" \
  | tee "$RESULT_DIR/icarus.log"

rm -rf "$BUILD/verilator_obj"
verilator --binary --timing --assert \
  -DSIMULATOR_VERILATOR -DSVA_RUNTIME_ENABLED -DATLIF_TB_PROGRESS \
  -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-BLKSEQ -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  -I"$VECTOR_DIR" --Mdir "$BUILD/verilator_obj" \
  --top-module tb_checkpoint_atlif_dptme \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/verif_hitflow/hitflow_dptme_assertions.sv" \
  "$ROOT/verif_hitflow/bind_hitflow_dptme_assertions.sv" \
  "$ROOT/tb_hitflow/tb_checkpoint_atlif_dptme.sv"
# Verilator is far slower than iverilog on this TB; 120s only reaches command=0
# on the Local5 81-command corpus (exit 124). Allow multi-hour default.
VERILATOR_TIMEOUT_S="${VERILATOR_TIMEOUT_S:-7200}"
timeout "${VERILATOR_TIMEOUT_S}s" stdbuf -oL "$BUILD/verilator_obj/Vtb_checkpoint_atlif_dptme" \
  | tee "$RESULT_DIR/verilator.log"

# Bind the three directed protocol rejects into the same fail-closed evidence
# directory instead of relying on a separately observed regression.
iverilog -g2012 -DSIMULATOR_ICARUS -s tb_hitflow_dptme_array \
  -o "$BUILD/tb_dptme_protocol.vvp" \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/tb_hitflow/tb_hitflow_dptme_array.sv"
vvp "$BUILD/tb_dptme_protocol.vvp" | tee "$RESULT_DIR/directed_icarus.log"

DIRECTED_ASSERT_BUILD="$BUILD/verilator_directed_assertions"
rm -rf "$DIRECTED_ASSERT_BUILD"
verilator --binary --timing --assert -DSIMULATOR_VERILATOR \
  -DSVA_RUNTIME_ENABLED -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-BLKSEQ -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  --Mdir "$DIRECTED_ASSERT_BUILD" --top-module tb_hitflow_dptme_array \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  "$ROOT/verif_hitflow/hitflow_dptme_assertions.sv" \
  "$ROOT/verif_hitflow/bind_hitflow_dptme_assertions.sv" \
  "$ROOT/tb_hitflow/tb_hitflow_dptme_array.sv"
"$DIRECTED_ASSERT_BUILD/Vtb_hitflow_dptme_array" \
  | tee "$RESULT_DIR/directed_verilator.log"

verilator --lint-only -Wall -Wno-UNUSEDSIGNAL \
  --top-module hitflow_dptme_array \
  "$ROOT/rtl_hitflow/hitflow_dptme_array.sv" \
  >"$RESULT_DIR/verilator_lint.log" 2>&1

yosys -q -l "$RESULT_DIR/yosys.log" \
  -p "read_verilog -sv $ROOT/rtl_hitflow/hitflow_dptme_array.sv; hierarchy -check -top hitflow_dptme_array; proc; opt; check -assert; stat"

python "$ROOT/scripts/report_checkpoint_atlif_dptme_rtl.py" \
  --vector-dir "$VECTOR_DIR" \
  --result-dir "$RESULT_DIR"
