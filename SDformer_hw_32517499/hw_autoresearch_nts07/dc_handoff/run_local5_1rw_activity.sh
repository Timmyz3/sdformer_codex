#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:-}"
RUN_GROUPS="${RUN_GROUPS:-100}"
DUMP_START_GROUP="${DUMP_START_GROUP:-0}"
DUMP_GROUPS="${DUMP_GROUPS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/local5_1rw_activity_population100_full}"
ACTIVITY_PURPOSE="${ACTIVITY_PURPOSE:-paper_power_with_io}"
DUMP_SCOPE="${DUMP_SCOPE:-full}"
DESIGN_NAME="local5_unified_out2_1rw_dc_top"

if ! command -v verilator >/dev/null 2>&1; then
  echo "未找到Verilator，不能生成1RW wrapper活动VCD。" >&2
  exit 3
fi
if [[ -z "$VECTOR_DIR" || ! -d "$VECTOR_DIR" ]]; then
  echo "必须通过VECTOR_DIR指定真实Local5 population向量目录。" >&2
  exit 4
fi
VECTOR_DIR="$(cd "$VECTOR_DIR" && pwd)"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "拒绝覆盖已有活动目录: $OUTPUT_DIR" >&2
  exit 5
fi
if (( RUN_GROUPS != 100 || DUMP_START_GROUP != 0 || DUMP_GROUPS != 100 )); then
  echo "1RW paper activity必须覆盖密封100-group population。" >&2
  exit 6
fi
if [[ "$DUMP_SCOPE" != "full" || "$ACTIVITY_PURPOSE" != "paper_power_with_io" ]]; then
  echo "1RW population activity仅准入full_load_compute_readback / paper_power_with_io。" >&2
  exit 7
fi

mkdir -p "$OUTPUT_DIR"
OBJ_DIR="$OUTPUT_DIR/obj"
VCD_FILE="$OUTPUT_DIR/${DESIGN_NAME}.vcd"
SIM_LOG="$OUTPUT_DIR/simulation.log"
CONTRACT="$OUTPUT_DIR/activity_contract.json"
mapfile -t RTL_FILES < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' \
  "$ROOT/filelists/date_local5_out2_1rw.f")
cd "$HW_ROOT"

verilator --binary --timing --assert --trace -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDPARAM -Wno-UNDRIVEN \
  --top-module tb_qfit_local5_score_projection_postg0 \
  -DQFIT_LOCAL5_1RW_DC_WRAPPER -DQFIT_ROLLING_SCHED_MODE=0 \
  -GBACKEND_KIND=0 -GACC_BACKEND_KIND=1 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=1 -GARCH_IDENTK=1 -GARCH_QSILENT_OVERLAP=1 \
  -GGROUPS=100 -GRUN_GROUPS=100 -GOUT_DIM=2 \
  --Mdir "$OBJ_DIR" \
  "${RTL_FILES[@]}" \
  "$HW_ROOT/tb_qfit/tb_qfit_local5_score_projection_postg0.sv" \
  "$HW_ROOT/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_local5_score_active_projection_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_source_multicast_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_tcfm5_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_local5_source_owned_term_conservation_assertions.sv" \
  >"$OUTPUT_DIR/compile.log" 2>&1

"$OBJ_DIR/Vtb_qfit_local5_score_projection_postg0" \
  +VECTOR_DIR="$VECTOR_DIR" \
  +DUMP_FILE="$VCD_FILE" \
  +DUMP_START_GROUP=0 \
  +DUMP_GROUPS=100 \
  +DUMP_SCOPE=full \
  >"$SIM_LOG" 2>&1

python3 "$ROOT/scripts/report_activity_vcd.py" \
  --design "$DESIGN_NAME" \
  --vcd "$VCD_FILE" \
  --log "$SIM_LOG" \
  --trace-root "$VECTOR_DIR" \
  --strip-path TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut \
  --purpose paper_power_with_io \
  --measurement-scope full_load_compute_readback \
  --output "$CONTRACT"

echo "$CONTRACT"
