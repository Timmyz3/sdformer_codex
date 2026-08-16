#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_real_trace_ablation"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
TRACE_MANIFEST="$ROOT/results/h67_real_bit_trace_20260717/manifest.json"
VECTOR_MANIFEST="$ROOT/results/gatestack_h67_real_trace_vectors_20260717/manifest.json"
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

PYTHONPATH=scripts "$PYTHON" scripts/generate_gatestack_real_trace_vectors.py \
  --manifest "$TRACE_MANIFEST" \
  --output-root tb_hitflow/vectors \
  --result "$VECTOR_MANIFEST" \
  >"$BUILD/vector_generation.log"

run_case() {
  local stage="$1"
  local heads="$2"
  local mode="$3"
  local vector_suffix="$4"
  local define="$5"
  local dir="$BUILD/s${stage}/$mode"
  local vector_dir="tb_hitflow/vectors/real_sample0_s${stage}_b0_${vector_suffix}"
  mkdir -p "$dir"
  local defines=(-DGATESTACK_REAL_TRACE)
  if [[ -n "$define" ]]; then
    defines+=("-D$define")
  fi
  iverilog -g2012 -Wall "${defines[@]}" \
    -Ptb_gatestack_single_context_execution_scale162_trace.HEADS="$heads" \
    -s tb_gatestack_single_context_execution_scale162_trace \
    -o "$dir/tb.vvp" "${RTL[@]}" \
    tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
    >"$dir/iverilog_build.log" 2>&1
  vvp "$dir/tb.vvp" "+vector_dir=$vector_dir" | tee "$dir/iverilog.log"
  verilator --binary --timing --assert -Wall "${defines[@]}" \
    -GHEADS="$heads" \
    --top-module tb_gatestack_single_context_execution_scale162_trace \
    -Mdir "$dir/verilator_obj" \
    "${RTL[@]}" "${SVA[@]}" \
    tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
    >"$dir/verilator_build.log" 2>&1
  "$dir/verilator_obj/Vtb_gatestack_single_context_execution_scale162_trace" \
    "+vector_dir=$vector_dir" | tee "$dir/verilator.log"
  if grep -Eq '%Warning|%Error' "$dir/verilator_build.log"; then
    echo "FAIL: S${stage}/$mode Verilator warning/error" >&2
    exit 1
  fi
}

for stage_heads in 0:3 1:6 2:12 3:24; do
  stage="${stage_heads%%:*}"
  heads="${stage_heads##*:}"
  run_case "$stage" "$heads" gatestack capacity ""
  run_case "$stage" "$heads" no_residency capacity GATESTACK_NO_RESIDENCY
  run_case "$stage" "$heads" raw_only rawonly ""
done

"$PYTHON" scripts/summarize_gatestack_real_trace_ablation.py \
  --vector-manifest "$VECTOR_MANIFEST" \
  --output-dir results/gatestack_real_trace_ablation_20260717
echo "PASS: H67四stage真实bit trace GateStack RTL消融完成"
