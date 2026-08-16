#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_fadc24_fulltop"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
TRACE_MANIFEST="$ROOT/results/h67_real_bit_trace_20260717/manifest.json"
VECTOR_MANIFEST="$ROOT/results/gatestack_fadc24_real_trace_vectors_20260718/manifest.json"
mkdir -p "$BUILD"
cd "$ROOT"

mapfile -t RTL < rtl_hitflow/filelist_single_context_execution.f
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
  verif_hitflow/gatestack_fadc24_streaming_replay_decoder_assertions.sv
  verif_hitflow/bind_gatestack_fadc24_streaming_replay_decoder_assertions.sv
)

PYTHONPATH=scripts "$PYTHON" scripts/generate_gatestack_fadc24_real_trace_vectors.py \
  --manifest "$TRACE_MANIFEST" \
  --source-vectors tb_hitflow/vectors \
  --output-root tb_hitflow/vectors \
  --result "$VECTOR_MANIFEST" \
  >"$BUILD/vector_generation.log"

run_stage() {
  local stage="$1"
  local heads="$2"
  local dir="$BUILD/s${stage}"
  local vector_dir="tb_hitflow/vectors/fadc24_real_sample0_s${stage}_b0"
  mkdir -p "$dir"

  iverilog -g2012 -Wall \
    -DGATESTACK_REAL_TRACE -DGATESTACK_NO_RESIDENCY -DGATESTACK_FADC24 \
    -Ptb_gatestack_single_context_execution_scale162_trace.HEADS="$heads" \
    -s tb_gatestack_single_context_execution_scale162_trace \
    -o "$dir/tb.vvp" "${RTL[@]}" \
    tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
    >"$dir/iverilog_build.log" 2>&1
  vvp "$dir/tb.vvp" "+vector_dir=$vector_dir" | tee "$dir/iverilog.log"

  verilator --binary --timing --assert -Wall \
    -DGATESTACK_REAL_TRACE -DGATESTACK_NO_RESIDENCY -DGATESTACK_FADC24 \
    -GHEADS="$heads" \
    --top-module tb_gatestack_single_context_execution_scale162_trace \
    -Mdir "$dir/verilator_obj" \
    "${RTL[@]}" "${SVA[@]}" \
    tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
    >"$dir/verilator_build.log" 2>&1
  "$dir/verilator_obj/Vtb_gatestack_single_context_execution_scale162_trace" \
    "+vector_dir=$vector_dir" | tee "$dir/verilator.log"
  if grep -Eq '%Warning|%Error' "$dir/verilator_build.log"; then
    echo "FAIL: S${stage} FADC24 Verilator warning/error" >&2
    exit 1
  fi
}

for stage_heads in 0:3 1:6 2:12 3:24; do
  run_stage "${stage_heads%%:*}" "${stage_heads##*:}"
done

"$PYTHON" scripts/summarize_gatestack_fadc24_fulltop.py \
  --vector-manifest "$VECTOR_MANIFEST" \
  --baseline-report results/gatestack_real_trace_ablation_20260717/report.json \
  --output-dir results/gatestack_fadc24_fulltop_20260718

echo "PASS: H67四stage真实trace FADC24同顶层RTL验证完成"
