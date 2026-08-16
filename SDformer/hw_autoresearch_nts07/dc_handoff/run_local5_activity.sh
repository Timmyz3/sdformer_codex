#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
VECTOR_DIR="${VECTOR_DIR:-}"
RUN_GROUPS="${RUN_GROUPS:-1}"
DUMP_START_GROUP="${DUMP_START_GROUP:-0}"
DUMP_GROUPS="${DUMP_GROUPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/local5_dc_activity}"
ACTIVITY_PURPOSE="${ACTIVITY_PURPOSE:-identity_smoke}"
DUMP_SCOPE="${DUMP_SCOPE:-full}"
DESIGN_NAME="local5_unified_out2_dc_top"

if ! command -v verilator >/dev/null 2>&1; then
  echo "未找到Verilator，不能生成wrapper活动VCD。" >&2
  exit 3
fi
if [[ -z "$VECTOR_DIR" || ! -d "$VECTOR_DIR" ]]; then
  echo "必须通过VECTOR_DIR指定真实Local5 population向量目录。" >&2
  exit 4
fi
VECTOR_DIR="$(cd "$VECTOR_DIR" && pwd)"
if (( RUN_GROUPS < 1 || RUN_GROUPS > 100 \
      || DUMP_START_GROUP < 0 || DUMP_GROUPS < 1 \
      || DUMP_START_GROUP + DUMP_GROUPS > RUN_GROUPS )); then
  echo "GROUP范围非法: RUN_GROUPS=$RUN_GROUPS START=$DUMP_START_GROUP COUNT=$DUMP_GROUPS" >&2
  exit 5
fi
case "$DUMP_SCOPE" in
  full) MEASUREMENT_SCOPE=full_load_compute_readback ;;
  busy) MEASUREMENT_SCOPE=busy_projection ;;
  *) echo "DUMP_SCOPE必须是full或busy。" >&2; exit 6 ;;
esac

mkdir -p "$OUTPUT_DIR"
OBJ_DIR="$OUTPUT_DIR/obj"
VCD_FILE="$OUTPUT_DIR/${DESIGN_NAME}.vcd"
SIM_LOG="$OUTPUT_DIR/simulation.log"
CONTRACT="$OUTPUT_DIR/activity_contract.json"
mapfile -t RTL_FILES < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' \
  "$ROOT/filelists/date_local5_out2.f")
cd "$HW_ROOT"

verilator --binary --timing --assert --trace -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDPARAM -Wno-UNDRIVEN \
  --top-module tb_qfit_local5_score_projection_postg0 \
  -DQFIT_LOCAL5_DC_WRAPPER -DQFIT_ROLLING_SCHED_MODE=0 \
  -GBACKEND_KIND=0 -GRELATION_READ_LATENCY=1 \
  -GARCH_QSILENT=1 -GARCH_IDENTK=1 -GARCH_QSILENT_OVERLAP=1 \
  -GGROUPS=100 -GRUN_GROUPS="$RUN_GROUPS" -GOUT_DIM=2 \
  --Mdir "$OBJ_DIR" \
  "${RTL_FILES[@]}" \
  "$HW_ROOT/tb_qfit/tb_qfit_local5_score_projection_postg0.sv" \
  "$HW_ROOT/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_local5_score_active_projection_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_relation_transpose_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_source_multicast_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_tcfm5_acc_bank_assertions.sv" \
  "$HW_ROOT/verif_qfit/qfit_tcfm5_assertions.sv" \
  2>&1 | tee "$OUTPUT_DIR/compile.log"

"$OBJ_DIR/Vtb_qfit_local5_score_projection_postg0" \
  +VECTOR_DIR="$VECTOR_DIR" \
  +DUMP_FILE="$VCD_FILE" \
  +DUMP_START_GROUP="$DUMP_START_GROUP" \
  +DUMP_GROUPS="$DUMP_GROUPS" \
  +DUMP_SCOPE="$DUMP_SCOPE" \
  | tee "$SIM_LOG"

python3 "$ROOT/scripts/report_activity_vcd.py" \
  --design "$DESIGN_NAME" \
  --vcd "$VCD_FILE" \
  --log "$SIM_LOG" \
  --trace-root "$VECTOR_DIR" \
  --strip-path TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut \
  --purpose "$ACTIVITY_PURPOSE" \
  --measurement-scope "$MEASUREMENT_SCOPE" \
  --output "$CONTRACT"

echo "$CONTRACT"
