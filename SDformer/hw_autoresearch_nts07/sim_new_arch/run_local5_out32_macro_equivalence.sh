#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_DIR="results/local5_out32_macro_equivalence_rtl_20260804"
VECTOR_DIR="tb_qfit/vectors/local5_active_projection_postg0_100_out32"
mkdir -p "$RESULT_DIR"

RTL=(
  tb_qfit/tb_qfit_local5_active_projection_postg0.sv
  rtl_qfit/qfit_local5_1rw_active_projection_tile.sv
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv
  rtl_qfit/qfit_dual_color_word_skipper_index.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_fakeram45_relation_bank_450.sv
  rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_local5_1rw_projection_backend.sv
  rtl_qfit/qfit_local5_color_map.sv
  rtl_qfit/qfit_direct_1rw_acc_bank.sv
  rtl_qfit/qfit_gasr2c_acc_bank.sv
  rtl_qfit/qfit_single_port_acc_memory.sv
  rtl_qfit/qfit_fakeram45_acc_memory_90x1024.sv
  tb_qfit/fakeram45_relation_models.sv
  tb_qfit/fakeram45_acc_models.sv
)
SVA=(
  verif_qfit/qfit_local5_1rw_active_projection_assertions.sv
  verif_qfit/qfit_gasr2c_acc_bank_assertions.sv
  verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv
  verif_qfit/qfit_single_port_acc_memory_assertions.sv
  verif_qfit/qfit_dual_color_word_skipper_assertions.sv
  verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv
  verif_qfit/qfit_source_multicast_assertions.sv
)

for impl in generic macro; do
  memory_impl=0
  if [[ "$impl" == "macro" ]]; then
    memory_impl=1
  fi
  for spec in direct:0:1 issue:1:0 ds:1:1; do
    IFS=: read -r mode_name mode sync_mode <<<"$spec"
    obj="/tmp/local5_out32_${impl}_${mode_name}_profile_obj"
    verilator --binary --timing -Wno-fatal \
      --top-module tb_qfit_local5_active_projection_postg0 \
      -Mdir "$obj" -GNEW_1RW_BACKEND=1 -GMODE="$mode" \
      -GGEOMETRY_SYNC_MODE="$sync_mode" -GOUT_DIM=32 \
      -GRELATION_MEMORY_IMPL="$memory_impl" \
      -GACC_MEMORY_IMPL="$memory_impl" -GGROUPS=100 -GRUN_GROUPS=100 \
      "${RTL[@]}" >"$RESULT_DIR/${impl}_${mode_name}_compile.log" 2>&1
    "$obj/Vtb_qfit_local5_active_projection_postg0" \
      +VECTOR_DIR="$VECTOR_DIR" \
      >"$RESULT_DIR/${impl}_${mode_name}_profile100.log" 2>&1
  done
done

for spec in direct:0:1 issue:1:0 ds:1:1; do
  IFS=: read -r mode_name mode sync_mode <<<"$spec"
  obj="/tmp/local5_out32_macro_${mode_name}_random_obj"
  verilator --binary --timing --assert -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    -Mdir "$obj" -GNEW_1RW_BACKEND=1 -GMODE="$mode" \
    -GGEOMETRY_SYNC_MODE="$sync_mode" -GOUT_DIM=32 \
    -GRELATION_MEMORY_IMPL=1 -GACC_MEMORY_IMPL=1 \
    -GGROUPS=100 -GRUN_GROUPS=100 \
    -GRANDOM_INPUT_GAPS=1 -GRANDOM_READ_GAPS=1 \
    "${RTL[@]}" "${SVA[@]}" \
    >"$RESULT_DIR/macro_${mode_name}_random_compile.log" 2>&1
  "$obj/Vtb_qfit_local5_active_projection_postg0" \
    +VECTOR_DIR="$VECTOR_DIR" \
    >"$RESULT_DIR/macro_${mode_name}_random_sva.log" 2>&1
done

{
  verilator --version
  git rev-parse HEAD
  sha256sum "$VECTOR_DIR/manifest.json"
} >"$RESULT_DIR/tool_and_vector_identity.txt"

python3 scripts/summarize_local5_out32_macro_equivalence.py \
  --result-dir "$RESULT_DIR" >"$RESULT_DIR/summary_stdout.json"

echo "PASS Local5 OUT_DIM32 accumulator macro equivalence"
