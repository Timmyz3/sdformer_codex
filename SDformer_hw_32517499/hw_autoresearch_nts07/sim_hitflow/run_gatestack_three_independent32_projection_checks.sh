#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_three_independent32_projection"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_resident_replay_joiner.sv
  rtl_hitflow/gatestack_ipd32w_replay_decoder.sv
  rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv
  rtl_hitflow/gatestack_adaptive_csr_replay_decoder.sv
  rtl_hitflow/gatestack_raw41_replay_decoder.sv
  rtl_hitflow/gatestack_raw_tail_retimer.sv
  rtl_hitflow/gatestack_raw_issue_adapter.sv
  rtl_hitflow/gatestack_replay_mux.sv
  rtl_hitflow/gatestack_term_fork.sv
  rtl_hitflow/gatestack_destination_bitmap_assembler.sv
  rtl_hitflow/gatestack_decoupled_product_engine.sv
  rtl_hitflow/gatestack_product_bitmap_join.sv
  rtl_hitflow/hitflow_segmented_multicast.sv
  rtl_hitflow/gatestack_tdr_multicast_backend.sv
  rtl_hitflow/hitflow_banked_accumulator.sv
  rtl_hitflow/gatestack_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_routed_multihead_tile_projection_top.sv
  rtl_hitflow/gatestack_multihead_decoder_projection_top.sv
  rtl_hitflow/gatestack_three_independent32_projection_top.sv
)
TB=tb_hitflow/tb_gatestack_three_independent32_projection_top.sv

iverilog -g2012 -Wall \
  -s tb_gatestack_three_independent32_projection_top \
  -o "$BUILD/tb.vvp" "${RTL[@]}" "$TB" \
  >"$BUILD/iverilog_build.log" 2>&1
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL -Wno-BLKSEQ \
  --top-module tb_gatestack_three_independent32_projection_top \
  -Mdir "$BUILD/verilator_obj" "${RTL[@]}" \
  verif_hitflow/gatestack_raw_tail_retimer_assertions.sv \
  verif_hitflow/bind_gatestack_raw_tail_retimer_assertions.sv \
  verif_hitflow/gatestack_replay_mux_assertions.sv \
  verif_hitflow/bind_gatestack_replay_mux_assertions.sv \
  verif_hitflow/gatestack_multihead_tile_projection_assertions.sv \
  verif_hitflow/bind_gatestack_multihead_decoder_projection_assertions.sv \
  "$TB" >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_three_independent32_projection_top" \
  | tee "$BUILD/verilator_sva.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_three_independent32_projection_top; proc; opt; check; stat"

python3 "$LINTER" \
  rtl_hitflow/gatestack_three_independent32_projection_top.sv \
  >"$BUILD/erie_rtl.log" 2>&1 || true
if grep -Eq '^\[ERROR\]|^\[MUST\].*ERROR|^ERROR' \
   "$BUILD/erie_rtl.log"; then
  echo "FAIL: Erie lint存在MUST error" >&2
  cat "$BUILD/erie_rtl.log" >&2
  exit 1
fi

grep -q 'status=PASS' "$BUILD/iverilog.log"
grep -q 'status=PASS' "$BUILD/verilator_sva.log"
echo "RESULT suite=three_independent32_checks status=PASS iverilog=PASS verilator_sva=PASS yosys=PASS erie=PASS product_lanes=96"
