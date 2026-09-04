#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_out32_v1_20260814}"
OUT2_VECTOR_DIR="${OUT2_VECTOR_DIR:-$ROOT/tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813}"
OUT2_RESULT_DIR="${OUT2_RESULT_DIR:-$ROOT/results/local5_qsilent_rolling_composition_20260814}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/local5_out32_population_sensitivity_20260814}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_qfit/local5_out32_population_sensitivity_20260814}"
TOP=tb_qfit_local5_score_projection_postg0

mkdir -p "$RESULT_DIR" "$BUILD_DIR"

python3 - "$VECTOR_DIR/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("schema") != "local5_score_projection_vectors_v1":
    raise SystemExit("OUT32 vector schema mismatch")
shape = manifest.get("shape", {})
selection = manifest.get("selection", {})
if shape.get("out_dim") != 32 or selection.get("groups") != 100:
    raise SystemExit("OUT32 vector shape/population mismatch")
if manifest.get("weight_mode") != "checkpoint_theta_folded_dyadic_int8_head_slice":
    raise SystemExit("OUT32 vectors are not theta-folded checkpoint weights")
PY

COMMON_RTL=(
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_qfit/qfit_tagged_compactor4.sv
  rtl_qfit/qfit_xorbank_compactor4.sv
  rtl_qfit/qfit_local5_score_leaf.sv
  rtl_qfit/qfit_local5_qsilent_score_leaf.sv
)
BACKEND_RTL=(
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_narrow_gate_weight_mul.sv
  rtl_qfit/qfit_sync_1rw_bank.sv
  rtl_qfit/qfit_lane_product_cache_leaf.sv
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
  verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv
  verif_qfit/qfit_source_multicast_assertions.sv
  verif_qfit/qfit_tcfm5_acc_bank_assertions.sv
  verif_qfit/qfit_tcfm5_assertions.sv
)
TB=tb_qfit/tb_qfit_local5_score_projection_postg0.sv
VERILATOR_FLAGS=(
  --binary --timing --assert -Wall -Wno-fatal
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDPARAM
  --top-module "$TOP"
  -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1
  -GARCH_QSILENT=1 -GARCH_IDENTK=1 -GARCH_QSILENT_OVERLAP=1
  -GGROUPS=100 -GRUN_GROUPS=100 -GOUT_DIM=32
)

cd "$ROOT"

iverilog -g2012 -Wall -s "$TOP" \
  -P"$TOP".BACKEND_KIND=0 -P"$TOP".RELATION_READ_LATENCY=1 \
  -P"$TOP".ARCH_QSILENT=1 -P"$TOP".ARCH_IDENTK=1 \
  -P"$TOP".ARCH_QSILENT_OVERLAP=1 \
  -P"$TOP".GROUPS=100 -P"$TOP".RUN_GROUPS=100 -P"$TOP".OUT_DIM=32 \
  -o "$BUILD_DIR/t450_qsilent_icarus.vvp" \
  "${COMMON_RTL[@]}" \
  rtl_qfit/qfit_dual_color_word_skipper_index.sv \
  rtl_qfit/qfit_sync_relation_bank.sv \
  rtl_qfit/qfit_fakeram45_relation_bank_450.sv \
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv \
  "${BACKEND_RTL[@]}" "$TB" \
  >"$RESULT_DIR/t450_qsilent_icarus_build.log" 2>&1
vvp "$BUILD_DIR/t450_qsilent_icarus.vvp" "+VECTOR_DIR=$VECTOR_DIR" \
  >"$RESULT_DIR/t450_qsilent_icarus.log" 2>&1

iverilog -g2012 -Wall -s "$TOP" -DQFIT_ROLLING_SCHED_MODE=0 \
  -P"$TOP".BACKEND_KIND=0 -P"$TOP".RELATION_READ_LATENCY=1 \
  -P"$TOP".ARCH_QSILENT=1 -P"$TOP".ARCH_IDENTK=1 \
  -P"$TOP".ARCH_QSILENT_OVERLAP=1 \
  -P"$TOP".GROUPS=100 -P"$TOP".RUN_GROUPS=100 -P"$TOP".OUT_DIM=32 \
  -o "$BUILD_DIR/rolling_qsilent_icarus.vvp" \
  "${COMMON_RTL[@]}" \
  rtl_qfit/qfit_sync_1r1w_bank.sv \
  rtl_qfit/qfit_retirement_scheduler.sv \
  rtl_qfit/qfit_relation_transpose_leaf.sv \
  rtl_qfit/sidecar/qfit_dual_color_relation_frontier_sync.sv \
  "${BACKEND_RTL[@]}" "$TB" \
  >"$RESULT_DIR/rolling_qsilent_icarus_build.log" 2>&1
vvp "$BUILD_DIR/rolling_qsilent_icarus.vvp" "+VECTOR_DIR=$VECTOR_DIR" \
  >"$RESULT_DIR/rolling_qsilent_icarus.log" 2>&1

verilator "${VERILATOR_FLAGS[@]}" \
  --Mdir "$BUILD_DIR/t450_qsilent" \
  "${COMMON_RTL[@]}" \
  rtl_qfit/qfit_dual_color_word_skipper_index.sv \
  rtl_qfit/qfit_sync_relation_bank.sv \
  rtl_qfit/qfit_fakeram45_relation_bank_450.sv \
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv \
  "${BACKEND_RTL[@]}" "$TB" "${ASSERTIONS[@]}" \
  >"$RESULT_DIR/t450_qsilent_build.log" 2>&1
"$BUILD_DIR/t450_qsilent/V$TOP" "+VECTOR_DIR=$VECTOR_DIR" \
  >"$RESULT_DIR/t450_qsilent_verilator_assert.log" 2>&1

verilator "${VERILATOR_FLAGS[@]}" -Wno-UNDRIVEN \
  -DQFIT_ROLLING_SCHED_MODE=0 \
  --Mdir "$BUILD_DIR/rolling_qsilent" \
  "${COMMON_RTL[@]}" \
  rtl_qfit/qfit_sync_1r1w_bank.sv \
  rtl_qfit/qfit_retirement_scheduler.sv \
  rtl_qfit/qfit_relation_transpose_leaf.sv \
  rtl_qfit/sidecar/qfit_dual_color_relation_frontier_sync.sv \
  "${BACKEND_RTL[@]}" "$TB" \
  verif_qfit/qfit_relation_transpose_assertions.sv \
  "${ASSERTIONS[@]}" \
  >"$RESULT_DIR/rolling_qsilent_build.log" 2>&1
"$BUILD_DIR/rolling_qsilent/V$TOP" "+VECTOR_DIR=$VECTOR_DIR" \
  >"$RESULT_DIR/rolling_qsilent_verilator_assert.log" 2>&1

python3 scripts/report_local5_out32_population_sensitivity.py \
  --vector-manifest "$VECTOR_DIR/manifest.json" \
  --out2-vector-manifest "$OUT2_VECTOR_DIR/manifest.json" \
  --t450-log "$RESULT_DIR/t450_qsilent_verilator_assert.log" \
  --rolling-log "$RESULT_DIR/rolling_qsilent_verilator_assert.log" \
  --icarus-t450-log "$RESULT_DIR/t450_qsilent_icarus.log" \
  --icarus-rolling-log "$RESULT_DIR/rolling_qsilent_icarus.log" \
  --out2-t450-log "$OUT2_RESULT_DIR/t450_q1_g100_verilator_assert.log" \
  --out2-rolling-log "$OUT2_RESULT_DIR/rolling_q1_g100_verilator_assert.log" \
  --output-dir "$RESULT_DIR"

echo "PASS Local5 OUT32 population sensitivity"
