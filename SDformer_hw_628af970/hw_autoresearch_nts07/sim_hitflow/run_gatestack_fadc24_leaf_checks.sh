#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_fadc24_leaf"
VECTOR="$ROOT/tb_hitflow/vectors/fadc24_real_sample0_s3_b0_h4"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
mkdir -p "$BUILD"
cd "$ROOT"

PYTHONPATH=scripts "$PYTHON" scripts/generate_gatestack_fadc24_decoder_vector.py \
  --manifest results/h67_real_bit_trace_20260717/manifest.json \
  --stage 3 --head 4 --window 0 --output-dir "$VECTOR" \
  >"$BUILD/vector_generation.log"

run_variant() {
  local name="$1"
  local module_file="$2"
  local define="$3"
  local runtime_args="${4:-}"
  local dir="$BUILD/$name"
  local defines=()
  local assertions=()
  mkdir -p "$dir"
  if [[ -n "$define" ]]; then
    defines+=("-D$define")
  fi
  if [[ "$define" == "FADC24_STREAMING" ]]; then
    assertions+=(
      verif_hitflow/gatestack_fadc24_streaming_replay_decoder_assertions.sv
      verif_hitflow/bind_gatestack_fadc24_streaming_replay_decoder_assertions.sv
    )
  fi
  iverilog -g2012 -Wall "${defines[@]}" \
    -s tb_gatestack_fadc24_replay_decoder_real \
    -o "$dir/tb.vvp" \
    "$module_file" tb_hitflow/tb_gatestack_fadc24_replay_decoder_real.sv \
    >"$dir/iverilog_build.log" 2>&1
  # shellcheck disable=SC2086
  vvp "$dir/tb.vvp" "+vector_dir=$VECTOR" $runtime_args | tee "$dir/iverilog.log"

  verilator --binary --timing --assert -Wall "${defines[@]}" \
    --top-module tb_gatestack_fadc24_replay_decoder_real \
    -Mdir "$dir/verilator_obj" \
    "$module_file" "${assertions[@]}" \
    tb_hitflow/tb_gatestack_fadc24_replay_decoder_real.sv \
    >"$dir/verilator_build.log" 2>&1
  # shellcheck disable=SC2086
  "$dir/verilator_obj/Vtb_gatestack_fadc24_replay_decoder_real" \
    "+vector_dir=$VECTOR" $runtime_args | tee "$dir/verilator.log"

  if grep -Eq '%Warning|%Error' "$dir/verilator_build.log"; then
    echo "FAIL: FADC24 $name Verilator存在warning/error" >&2
    exit 1
  fi
}

run_variant buffered rtl_hitflow/gatestack_fadc24_replay_decoder.sv ""
run_variant streaming rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv \
  FADC24_STREAMING
run_variant streaming_no_backpressure \
  rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv \
  FADC24_STREAMING "+no_backpressure=1"
run_variant streaming_bad_padding \
  rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv \
  FADC24_STREAMING \
  "+payload_file=payload_words_bad_bitmap_padding.memh +expect_error=1"

python3 /root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py \
  rtl_hitflow/gatestack_fadc24_replay_decoder.sv \
  | tee "$BUILD/erie_buffered_lint.log"
python3 /root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py \
  rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv \
  | tee "$BUILD/erie_streaming_lint.log"

yosys -Q -p "read_verilog -sv rtl_hitflow/gatestack_fadc24_replay_decoder.sv; hierarchy -check -top gatestack_fadc24_replay_decoder; proc; memory_collect; check; stat" \
  >"$BUILD/yosys.log" 2>&1
yosys -Q -p "read_verilog -sv rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv; hierarchy -check -top gatestack_fadc24_streaming_replay_decoder; proc; memory_collect; check; stat" \
  >"$BUILD/yosys_streaming.log" 2>&1

echo "PASS: FADC24真实overflow-head leaf仿真、lint与综合可读检查完成"
