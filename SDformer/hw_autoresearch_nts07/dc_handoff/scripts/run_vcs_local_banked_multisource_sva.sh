#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
ISSUE_WIDTH="${ISSUE_WIDTH:-4}"
OUT_LANES="${OUT_LANES:-16}"
MAX_CASES="${MAX_CASES:-20000}"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
VECTOR_DIR="${VECTOR_DIR:-$RUN_ROOT/dual_line_m2b_real_vectors_v2_20260821}"
if [[ "$OUT_LANES" == "16" ]]; then
  default_run_dir="$RUN_ROOT/local_banked_multisource_p${ISSUE_WIDTH}_vcs_sva_20260821"
else
  default_run_dir="$RUN_ROOT/local_banked_multisource_p${ISSUE_WIDTH}_l${OUT_LANES}_vcs_sva_20260821"
fi
RUN_DIR="${RUN_DIR:-$default_run_dir}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3.12}"
H67_TILES="${H67_TILES:-$ROOT/results/h67_ep35_real_tile_trace_s1_rows40_v2_20260821/real_tiles}"
LOCAL_TILES="${LOCAL_TILES:-$ROOT/results/local_ep44_real_tile_trace_s1_rows40_v2_20260821/real_tiles}"

case "$ISSUE_WIDTH" in
  1|2|4|8) ;;
  *) echo "ISSUE_WIDTH must be 1, 2, 4, or 8" >&2; exit 2 ;;
esac
case "$OUT_LANES" in
  16|96) ;;
  *) echo "OUT_LANES must be 16 or 96" >&2; exit 2 ;;
esac
if [[ "$MAX_CASES" -le 0 || "$MAX_CASES" -gt 20000 ]]; then
  echo "MAX_CASES must be in [1, 20000]" >&2
  exit 2
fi
command -v vcs >/dev/null 2>&1
test -x "$PYTHON_BIN"

mkdir -p "$VECTOR_DIR" "$RUN_DIR"
if [[ ! -s "$VECTOR_DIR/manifest.json" ]]; then
  "$PYTHON_BIN" "$ROOT/system_simulator/scripts/build_m2b_real_tile_vectors.py" \
    --identity H67 "$H67_TILES" --identity Local "$LOCAL_TILES" \
    --per-identity 10000 --output-dir "$VECTOR_DIR"
fi

cd "$RUN_DIR"
vcs -full64 -sverilog -assert svaext -debug_access+all \
  "+define+M2_ISSUE_WIDTH=$ISSUE_WIDTH" \
  "+define+M2_OUT_LANES=$OUT_LANES" \
  "$ROOT/rtl_qfit/qfit_local_banked_multisource_engine.sv" \
  "$ROOT/tb_qfit/tb_qfit_local_banked_multisource_engine.sv" \
  "$ROOT/verif_qfit/qfit_local_banked_multisource_engine_assertions.sv" \
  -top tb_qfit_local_banked_multisource_engine \
  -o simv 2>&1 | tee compile.log
./simv "+TRACE_FILE=$VECTOR_DIR/current_tiles.hex" "+MAX_CASES=$MAX_CASES" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log
grep -q "PASS M2B banked multi-source issue_width=$ISSUE_WIDTH" simulation.log
! grep -Eq "Assertion failed|Fatal:|Error-\[" simulation.log
sha256sum compile.log simulation.log simv "$VECTOR_DIR/manifest.json" > evidence.sha256
echo "PASS Synopsys VCS/SVA M2B ISSUE_WIDTH=$ISSUE_WIDTH OUT_LANES=$OUT_LANES"
