#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${ENABLE_TYPED_RESIDENCY:-0}" == "1" ]]; then
  BUILD="$ROOT/build_hitflow/gatestack_typed_residency_fulltop"
  RESIDENCY_DEFINE=()
  OUTPUT_DIR="results/gatestack_typed_residency_fulltop_20260718"
else
  BUILD="$ROOT/build_hitflow/gatestack_adaptive_csr_fulltop"
  RESIDENCY_DEFINE=(-DGATESTACK_NO_RESIDENCY)
  OUTPUT_DIR="results/gatestack_adaptive_csr_fulltop_20260718"
fi
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
mkdir -p "$BUILD"
cd "$ROOT"

"$PYTHON" scripts/generate_gatestack_adaptive_mixed_vector.py \
  --ipd-dir tb_hitflow/vectors/real_sample0_s3_b0_capacity \
  --fadc-dir tb_hitflow/vectors/fadc24_real_sample0_s3_b0 \
  --output-dir tb_hitflow/vectors/adaptive_mixed_real_sample0_s3_b0 \
  >"$BUILD/mixed_vector_generation.log"
"$PYTHON" scripts/generate_gatestack_adaptive_mixed_vector.py \
  --ipd-dir tb_hitflow/vectors/real_sample0_s3_b0_capacity \
  --fadc-dir tb_hitflow/vectors/fadc24_real_sample0_s3_b0 \
  --output-dir tb_hitflow/vectors/adaptive_mixed_csr_real_sample0_s3_b0 \
  --replace-raw-with-fadc \
  >"$BUILD/mixed_csr_vector_generation.log"

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
  verif_hitflow/gatestack_ipd32w_replay_decoder_assertions.sv
  verif_hitflow/bind_gatestack_ipd32w_replay_decoder_assertions.sv
  verif_hitflow/gatestack_fadc24_streaming_replay_decoder_assertions.sv
  verif_hitflow/bind_gatestack_fadc24_streaming_replay_decoder_assertions.sv
  verif_hitflow/gatestack_adaptive_csr_replay_decoder_assertions.sv
  verif_hitflow/bind_gatestack_adaptive_csr_replay_decoder_assertions.sv
  verif_hitflow/gatestack_adaptive_csr_selector_assertions.sv
  verif_hitflow/bind_gatestack_adaptive_csr_selector_assertions.sv
)

run_stage() {
  local stage="$1"
  local heads="$2"
  local vector_dir="$3"
  local dir="$BUILD/s${stage}"
  mkdir -p "$dir"

  iverilog -g2012 -Wall \
    -DGATESTACK_REAL_TRACE "${RESIDENCY_DEFINE[@]}" -DGATESTACK_ADAPTIVE_CSR \
    -Ptb_gatestack_single_context_execution_scale162_trace.HEADS="$heads" \
    -s tb_gatestack_single_context_execution_scale162_trace \
    -o "$dir/tb.vvp" "${RTL[@]}" \
    tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
    >"$dir/iverilog_build.log" 2>&1
  vvp "$dir/tb.vvp" "+vector_dir=$vector_dir" | tee "$dir/iverilog.log"

  verilator --binary --timing --assert -Wall \
    -DGATESTACK_REAL_TRACE "${RESIDENCY_DEFINE[@]}" -DGATESTACK_ADAPTIVE_CSR \
    -GHEADS="$heads" \
    --top-module tb_gatestack_single_context_execution_scale162_trace \
    -Mdir "$dir/verilator_obj" \
    "${RTL[@]}" "${SVA[@]}" \
    tb_hitflow/tb_gatestack_single_context_execution_scale162_trace.sv \
    >"$dir/verilator_build.log" 2>&1
  "$dir/verilator_obj/Vtb_gatestack_single_context_execution_scale162_trace" \
    "+vector_dir=$vector_dir" | tee "$dir/verilator.log"
  if grep -Eq '%Warning|%Error' "$dir/verilator_build.log"; then
    echo "FAIL: S${stage} Adaptive CSR Verilator warning/error" >&2
    exit 1
  fi
}

run_stage 0 3  tb_hitflow/vectors/real_sample0_s0_b0_capacity
run_stage 1 6  tb_hitflow/vectors/real_sample0_s1_b0_capacity
run_stage 2 12 tb_hitflow/vectors/real_sample0_s2_b0_capacity
run_stage 3 24 tb_hitflow/vectors/fadc24_real_sample0_s3_b0
run_stage mixed 24 tb_hitflow/vectors/adaptive_mixed_real_sample0_s3_b0
run_stage mixedcsr 24 tb_hitflow/vectors/adaptive_mixed_csr_real_sample0_s3_b0

iverilog -g2012 -Wall \
  -s tb_gatestack_invalid_adaptive_residency \
  -o "$BUILD/invalid_adaptive_residency.vvp" "${RTL[@]}" \
  tb_hitflow/tb_gatestack_invalid_adaptive_residency.sv \
  >"$BUILD/invalid_adaptive_residency_build.log" 2>&1
vvp "$BUILD/invalid_adaptive_residency.vvp" \
  | tee "$BUILD/invalid_adaptive_residency.log"
grep -q 'Adaptive plus IPD-only residency configuration admitted' \
  "$BUILD/invalid_adaptive_residency.log"

yosys -ql "$BUILD/yosys_fair.log" -p \
  "read_verilog -sv rtl_hitflow/gatestack_ipd32w_replay_decoder.sv rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv rtl_hitflow/gatestack_adaptive_csr_replay_decoder.sv; hierarchy -check -top gatestack_adaptive_csr_replay_decoder; proc; opt; memory -nomap; stat"

if [[ "${ENABLE_TYPED_RESIDENCY:-0}" == "1" ]]; then
  "$PYTHON" scripts/summarize_gatestack_typed_residency_fulltop.py \
    --no-residency-report results/gatestack_adaptive_csr_fulltop_20260718/report.json \
    --output-dir "$OUTPUT_DIR"
else
  "$PYTHON" scripts/summarize_gatestack_adaptive_csr_fulltop.py \
    --baseline-report results/gatestack_real_trace_ablation_20260717/report.json \
    --fadc-report results/gatestack_fadc24_fulltop_20260718/report.json \
    --output-dir "$OUTPUT_DIR"
fi

echo "PASS: H67四stage统一Adaptive CSR真实trace同顶层RTL验证完成 residency=${ENABLE_TYPED_RESIDENCY:-0}"
