#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_DIR="results/local5_ds_gasr_sync_ablation_rtl_20260804"
VECTOR_DIR="tb_qfit/vectors/local5_active_projection_postg0_100"
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
)
SVA=(
  verif_qfit/qfit_local5_1rw_active_projection_assertions.sv
  verif_qfit/qfit_gasr2c_acc_bank_assertions.sv
  verif_qfit/qfit_single_port_acc_memory_assertions.sv
  verif_qfit/qfit_dual_color_word_skipper_assertions.sv
  verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv
  verif_qfit/qfit_source_multicast_assertions.sv
)

{
  verilator --version
  yosys -V
  git rev-parse HEAD
} >"$RESULT_DIR/tool_versions.txt"

for sync_mode in 0 1; do
  name="issue"
  if [[ "$sync_mode" == "1" ]]; then
    name="ds"
  fi
  obj="/tmp/local5_ds_gasr_ablate_${name}_obj"
  verilator --binary --timing --assert -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    -Mdir "$obj" -GNEW_1RW_BACKEND=1 -GMODE=1 \
    -GGEOMETRY_SYNC_MODE="$sync_mode" -GGROUPS=100 -GRUN_GROUPS=100 \
    "${RTL[@]}" "${SVA[@]}" >"$RESULT_DIR/${name}_compile.log" 2>&1
  "$obj/Vtb_qfit_local5_active_projection_postg0" \
    +VECTOR_DIR="$VECTOR_DIR" >"$RESULT_DIR/${name}_profile100.log" 2>&1

  obj_random="/tmp/local5_ds_gasr_ablate_${name}_random_obj"
  verilator --binary --timing --assert -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    -Mdir "$obj_random" -GNEW_1RW_BACKEND=1 -GMODE=1 \
    -GGEOMETRY_SYNC_MODE="$sync_mode" -GGROUPS=100 -GRUN_GROUPS=100 \
    -GRANDOM_INPUT_GAPS=1 -GRANDOM_READ_GAPS=1 \
    "${RTL[@]}" "${SVA[@]}" >"$RESULT_DIR/${name}_random_compile.log" 2>&1
  "$obj_random/Vtb_qfit_local5_active_projection_postg0" \
    +VECTOR_DIR="$VECTOR_DIR" >"$RESULT_DIR/${name}_random_sva.log" 2>&1
done

TRACKED_ARGS=()
for path in "${RTL[@]}" "${SVA[@]}"; do
  TRACKED_ARGS+=(--tracked-file "$path")
done
python3 scripts/summarize_local5_ds_gasr_sync_ablation.py \
  --result-dir "$RESULT_DIR" "${TRACKED_ARGS[@]}" \
  >"$RESULT_DIR/summary_stdout.json"

echo "PASS Local5 DS-GASR synchronization-point ablation"
