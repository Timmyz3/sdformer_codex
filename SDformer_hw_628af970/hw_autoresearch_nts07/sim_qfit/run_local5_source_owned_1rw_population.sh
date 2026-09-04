#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/local5_source_owned_1rw_population_20260814}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_qfit/local5_source_owned_1rw_population_20260814}"
TOP=tb_qfit_local5_score_projection_postg0
GROUP_EQUAL_GATES="${GROUP_EQUAL_GATES:-1}"
PRODUCT_CACHE_WAYS="${PRODUCT_CACHE_WAYS:-0}"
RUN_GROUPS="${RUN_GROUPS:-100}"

for path in "$RESULT_DIR" "$BUILD_DIR"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: refusing to overwrite existing path: $path" >&2
    exit 2
  fi
done
mkdir -p "$RESULT_DIR" "$BUILD_DIR"

RTL=(
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_qfit/qfit_tagged_compactor4.sv
  rtl_qfit/qfit_xorbank_compactor4.sv
  rtl_qfit/qfit_local5_score_leaf.sv
  rtl_qfit/qfit_local5_qsilent_score_leaf.sv
  rtl_qfit/qfit_sync_1r1w_bank.sv
  rtl_qfit/qfit_retirement_scheduler.sv
  rtl_qfit/qfit_relation_transpose_leaf.sv
  rtl_qfit/sidecar/qfit_dual_color_relation_frontier_sync.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_narrow_gate_weight_mul.sv
  rtl_qfit/qfit_sync_1rw_bank.sv
  rtl_qfit/qfit_lane_product_cache_leaf.sv
  rtl_qfit/qfit_single_port_acc_memory.sv
  rtl_qfit/qfit_direct_1rw_acc_bank.sv
  rtl_qfit/qfit_tcfm5_acc_bank.sv
  rtl_qfit/qfit_tcfm5_projection_top.sv
  rtl_qfit/qfit_cached_tcfm5_projection_top.sv
  rtl_qfit/qfit_linear5_projection_top.sv
  rtl_qfit/qfit_local5_active_projection_tile.sv
  rtl_qfit/qfit_local5_score_active_projection_tile.sv
)
ASSERTIONS=(
  verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv
  verif_qfit/qfit_local5_score_active_projection_assertions.sv
  verif_qfit/qfit_relation_transpose_assertions.sv
  verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv
  verif_qfit/qfit_source_multicast_assertions.sv
  verif_qfit/qfit_tcfm5_assertions.sv
  verif_qfit/qfit_local5_source_owned_term_conservation_assertions.sv
)

cd "$ROOT"
verilator \
  --binary --timing --assert -Wall -Wno-fatal -Wno-UNDRIVEN \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDPARAM \
  --top-module "$TOP" \
  -GBACKEND_KIND=0 -GACC_BACKEND_KIND=1 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=1 -GARCH_IDENTK=1 -GARCH_QSILENT_OVERLAP=1 \
  -GGROUP_EQUAL_GATES="$GROUP_EQUAL_GATES" \
  -GPRODUCT_CACHE_WAYS="$PRODUCT_CACHE_WAYS" \
  -GGROUPS=100 -GRUN_GROUPS="$RUN_GROUPS" -GOUT_DIM=2 \
  -DQFIT_ROLLING_SCHED_MODE=0 \
  --Mdir "$BUILD_DIR/obj" \
  "${RTL[@]}" \
  tb_qfit/tb_qfit_local5_score_projection_postg0.sv \
  "${ASSERTIONS[@]}" \
  >"$RESULT_DIR/verilator_build.log" 2>&1

"$BUILD_DIR/obj/V$TOP" "+VECTOR_DIR=$VECTOR_DIR" \
  >"$RESULT_DIR/verilator_assert.log" 2>&1

echo "PASS Local5 source-owned 1RW population groups=$RUN_GROUPS group_equal_gates=$GROUP_EQUAL_GATES product_cache_ways=$PRODUCT_CACHE_WAYS"
