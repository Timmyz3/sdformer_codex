#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_single_context_execution"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
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
  rtl_hitflow/gatestack_output_tile_scheduler.sv
  rtl_hitflow/gatestack_head_slot_sram_adapter.sv
  rtl_hitflow/gatestack_descriptor_residency_cache.sv
  rtl_hitflow/gatestack_replay_plan_builder.sv
  rtl_hitflow/gatestack_replay_atomic_commit.sv
  rtl_hitflow/gatestack_dualtag_replay_lifecycle_manager.sv
  rtl_hitflow/gatestack_replay_control_plane_top.sv
  rtl_hitflow/gatestack_slot_replay_word_router.sv
  rtl_hitflow/gatestack_backend_done_guard.sv
  rtl_hitflow/gatestack_context_abort_controller.sv
  rtl_hitflow/gatestack_ipd_cache_fill_adapter.sv
  rtl_hitflow/gatestack_single_context_execution_top.sv
)
SVA=(
  verif_hitflow/gatestack_output_tile_scheduler_assertions.sv
  verif_hitflow/bind_gatestack_output_tile_scheduler_assertions.sv
  verif_hitflow/gatestack_head_slot_sram_adapter_assertions.sv
  verif_hitflow/bind_gatestack_head_slot_sram_adapter_assertions.sv
  verif_hitflow/gatestack_descriptor_residency_cache_assertions.sv
  verif_hitflow/bind_gatestack_descriptor_residency_cache_assertions.sv
  verif_hitflow/gatestack_replay_plan_builder_assertions.sv
  verif_hitflow/bind_gatestack_replay_plan_builder_assertions.sv
  verif_hitflow/gatestack_replay_atomic_commit_assertions.sv
  verif_hitflow/bind_gatestack_replay_atomic_commit_assertions.sv
  verif_hitflow/gatestack_dualtag_replay_lifecycle_assertions.sv
  verif_hitflow/bind_gatestack_dualtag_replay_lifecycle_assertions.sv
  verif_hitflow/gatestack_replay_control_plane_assertions.sv
  verif_hitflow/bind_gatestack_replay_control_plane_assertions.sv
  verif_hitflow/gatestack_slot_replay_word_router_assertions.sv
  verif_hitflow/bind_gatestack_slot_replay_word_router_assertions.sv
  verif_hitflow/gatestack_backend_done_guard_assertions.sv
  verif_hitflow/bind_gatestack_backend_done_guard_assertions.sv
  verif_hitflow/gatestack_context_abort_controller_assertions.sv
  verif_hitflow/bind_gatestack_context_abort_controller_assertions.sv
  verif_hitflow/gatestack_ipd_cache_fill_adapter_assertions.sv
  verif_hitflow/bind_gatestack_ipd_cache_fill_adapter_assertions.sv
  verif_hitflow/gatestack_raw_tail_retimer_assertions.sv
  verif_hitflow/bind_gatestack_raw_tail_retimer_assertions.sv
  verif_hitflow/gatestack_replay_mux_assertions.sv
  verif_hitflow/bind_gatestack_replay_mux_assertions.sv
  verif_hitflow/gatestack_multihead_tile_projection_assertions.sv
  verif_hitflow/bind_gatestack_multihead_tile_projection_assertions.sv
  verif_hitflow/bind_gatestack_multihead_decoder_projection_assertions.sv
)

PYTHONPATH=scripts "$PYTHON" scripts/generate_gatestack_h67_stage3_trace.py \
  >"$BUILD/trace_generation.log"

iverilog -g2012 -Wall \
  -s tb_gatestack_single_context_execution_scale162_trace \
  -o "$BUILD/tb_scale162_trace.vvp" "${RTL[@]}" \
  tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv
vvp_args=()
if [[ "${DUMP_VCD:-0}" == "1" ]]; then
  vvp_args+=(+dump_vcd)
fi
vvp "$BUILD/tb_scale162_trace.vvp" "${vvp_args[@]}" | \
  tee "$BUILD/iverilog_scale162_trace.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_single_context_execution_scale162_trace \
  -Mdir "$BUILD/verilator_scale162_trace_obj" \
  "${RTL[@]}" "${SVA[@]}" \
  tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
  >"$BUILD/verilator_scale162_trace_build.log" 2>&1
"$BUILD/verilator_scale162_trace_obj/Vtb_gatestack_single_context_execution_scale162_trace" | \
  tee "$BUILD/verilator_scale162_trace.log"

if grep -Eq '%Warning|%Error' "$BUILD/verilator_scale162_trace_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: H67 stage3 trace-shaped默认规模full-top；Icarus/Verilator+SVA通过"
