#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_ep35_real_weight_projection2_rtl_20260813}"
BASE="$ROOT/tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt"
TRACE="$ROOT/results/h67_fullres_ep35_postconvergence_t450_20260805_all12_bit_trace/manifest.json"
VEC_DIR="$ROOT/tb_h67/vectors/h67_ep35_real_weight_projection2_20260813"
REALW="$VEC_DIR/h67_real_weight_projection2.txt"
mkdir -p "$OUT/build" "$OUT/logs"

python3 "$ROOT/scripts/generate_h67_real_weight_projection2_vectors.py" \
  --trace-manifest "$TRACE" --base-vector "$BASE" --output-dir "$VEC_DIR"

RTL=(
  "$ROOT/rtl_ttx/ttx_ceil_log2_u32.sv"
  "$ROOT/rtl_ttx/ttx_exp2_lut_q8.sv"
  "$ROOT/rtl_ttx/ttx_gate_quant_q17.sv"
  "$ROOT/rtl_h67/h67_motionxor_score_q7.sv"
  "$ROOT/rtl_h67/h67_temporal_slot_encoder.sv"
  "$ROOT/rtl_h67/h67_sync_dual_bank_k_store.sv"
  "$ROOT/rtl_h67/h67_temporal_slot_fifo_2s.sv"
  "$ROOT/rtl_h67/h67_temporal_weighted_scs_directory_2s.sv"
  "$ROOT/rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv"
  "$ROOT/rtl_h67/h67_gated_k_projection2_acc.sv"
)
TB=(
  "$ROOT/verif_h67/h67_real_weight_projection2_bind.sv"
  "$ROOT/tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv"
  "$ROOT/tb_h67/tb_h67_temporal_slot_flow_real_weight_projection2.sv"
)
TOP=tb_h67_temporal_slot_flow_real_weight_projection2

iverilog -g2012 -Wall -s "$TOP" -o "$OUT/build/icarus.vvp" \
  "${RTL[@]}" "${TB[@]}" >"$OUT/logs/icarus_build.log" 2>&1
vvp "$OUT/build/icarus.vvp" "+VECTORS=$BASE" "+REALW=$REALW" \
  >"$OUT/logs/icarus_full.log" 2>&1

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-DECLFILENAME \
  --top-module "$TOP" --Mdir "$OUT/build/verilator_sva" \
  "${RTL[@]}" "$ROOT/verif_h67/h67_temporal_slot_flow_2s_assertions.sv" \
  "${TB[@]}" >"$OUT/logs/verilator_sva_build.log" 2>&1
"$OUT/build/verilator_sva/V$TOP" "+VECTORS=$BASE" "+REALW=$REALW" \
  >"$OUT/logs/verilator_sva_full.log" 2>&1

yosys -Q -p "read_verilog -sv $ROOT/rtl_h67/h67_gated_k_projection2_acc.sv; hierarchy -top h67_gated_k_projection2_acc; proc; opt; check" \
  >"$OUT/logs/yosys_projection2.log" 2>&1

python3 -m unittest \
  scripts.test_generate_h67_real_weight_projection2_vectors \
  scripts.test_report_h67_real_weight_projection2_rtl \
  >"$OUT/logs/python_tests.log" 2>&1

SOURCES=(
  "${RTL[@]}" "${TB[@]}"
  "$ROOT/scripts/generate_h67_real_weight_projection2_vectors.py"
  "$ROOT/scripts/report_h67_real_weight_projection2_rtl.py"
  "$ROOT/sim_h67/run_h67_real_weight_projection2_checks.sh"
)
ARGS=()
for source in "${SOURCES[@]}"; do ARGS+=(--source "$source"); done
python3 "$ROOT/scripts/report_h67_real_weight_projection2_rtl.py" \
  --result-dir "$OUT" --vector-manifest "$VEC_DIR/manifest.json" "${ARGS[@]}"

echo "PASS H67 ep35 real-weight projection2 full flow"
