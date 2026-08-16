#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/local5_erep_t450_phase_calibration_v4_20260810}"
VECTOR_DIR="tb_qfit/vectors/local5_bb1e4_active_projection_postg0_all4800"
RUN_GROUPS=4
mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/verilator_obj"
rm -f "$OUT_DIR/complete.json" "$OUT_DIR/result_sha256.txt"

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
ASSERTIONS=(
  verif_qfit/qfit_local5_1rw_active_projection_assertions.sv
  verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv
  verif_qfit/qfit_single_port_acc_memory_assertions.sv
)

python3 - "$VECTOR_DIR/manifest.json" "$RUN_GROUPS" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
groups = int(sys.argv[2])
if (
    manifest.get("schema") != "local5_active_projection_postg0_vectors_v1"
    or manifest.get("shape") != {
        "height": 15, "width": 15, "planes": 2, "sources": 450,
        "head_dim": 32, "out_dim": 2,
    }
    or manifest.get("selection", {}).get("method") != "manifest_order_all_groups"
    or manifest.get("selection", {}).get("groups") != 4800
    or len(manifest.get("selection", {}).get("rows", [])) != 4800
    or len(manifest.get("selection", {}).get("rows", [])) < groups
):
    raise SystemExit("拒绝校准：固定T450/OUT_DIM2 vector合同不成立")
PY

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  verilator --version
  python3 --version
  uname -a
} >"$OUT_DIR/tool_versions.txt"

python3 -m py_compile sim_qfit/local5_erep_t450_phase_trace_v4.py
python3 -m unittest tests.test_local5_erep_t450_phase_trace_v4 \
  >"$OUT_DIR/parser_tests.log" 2>&1

verilator --binary --timing --assert -Wno-fatal \
  --top-module tb_qfit_local5_active_projection_postg0 \
  -Mdir "$OUT_DIR/verilator_obj" \
  -GNEW_1RW_BACKEND=1 -GMODE=0 -GGROUPS=4800 -GRUN_GROUPS="$RUN_GROUPS" \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  >"$OUT_DIR/verilator_compile.log" 2>&1

binary="$OUT_DIR/verilator_obj/Vtb_qfit_local5_active_projection_postg0"
"$binary" +VECTOR_DIR="$VECTOR_DIR" +EREP_PHASE_TRACE_V4 \
  >"$OUT_DIR/phase_trace.log" 2>&1
python3 sim_qfit/local5_erep_t450_phase_trace_v4.py \
  --trace "$OUT_DIR/phase_trace.log" \
  --vector-manifest "$VECTOR_DIR/manifest.json" \
  --output "$OUT_DIR/phase_evidence.json"

"$binary" +VECTOR_DIR="$VECTOR_DIR" >"$OUT_DIR/compatibility.log" 2>&1
grep -v '^EREP_PHASE_V4 ' "$OUT_DIR/phase_trace.log" \
  >"$OUT_DIR/phase_trace_without_probe.log"
diff -u "$OUT_DIR/phase_trace_without_probe.log" "$OUT_DIR/compatibility.log" \
  >"$OUT_DIR/compatibility.diff"
python3 - "$OUT_DIR/compatibility.log" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from summarize_local5_exact_backend_rtl_replay import parse_log

rows = parse_log(Path(sys.argv[1]))
if len(rows) != 4:
    raise SystemExit("旧汇总器兼容回归的group数错误")
PY

sha256sum \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  sim_qfit/local5_erep_t450_phase_trace_v4.py \
  sim_qfit/run_local5_erep_t450_phase_calibration_v4.sh \
  tests/test_local5_erep_t450_phase_trace_v4.py \
  scripts/summarize_local5_exact_backend_rtl_replay.py \
  "$VECTOR_DIR/manifest.json" \
  >"$OUT_DIR/source_sha256.txt"

{
  git rev-parse HEAD
  git status --short -- \
    "${RTL[@]}" "${ASSERTIONS[@]}" \
    sim_qfit/local5_erep_t450_phase_trace_v4.py \
    sim_qfit/run_local5_erep_t450_phase_calibration_v4.sh \
    tests/test_local5_erep_t450_phase_trace_v4.py
} >"$OUT_DIR/source_git_state.txt"

sha256sum \
  "$OUT_DIR"/{tool_versions.txt,parser_tests.log,verilator_compile.log,phase_trace.log,phase_trace_without_probe.log,phase_evidence.json,compatibility.log,compatibility.diff,source_sha256.txt,source_git_state.txt} \
  "$binary" \
  >"$OUT_DIR/result_sha256.txt"

python3 - "$OUT_DIR/complete.json" "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[2]).resolve()
hashes = root / "result_sha256.txt"
value = {
    "schema": "local5_erep_t450_phase_calibration_complete_v4",
    "status": "PASS_RTL_CALIBRATION_ONLY",
    "formal_adapter_status": "DENY",
    "run_groups": 4,
    "out_dim": 2,
    "result_sha256_file": str(hashes),
    "result_sha256_file_sha256": hashlib.sha256(hashes.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

echo "PASS Local5 EREP T450 phase calibration v4 groups=$RUN_GROUPS formal=DENY"
