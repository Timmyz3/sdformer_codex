#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/local5_erep_calibration_v4.XXXXXX")}"
mkdir -p "$OUT_DIR"

RTL=(
  verif_qfit/qfit_local5_erep_calibration_monitor_v4.sv
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
  rtl_qfit/qfit_tcfm5_projection_top.sv
  rtl_qfit/qfit_tcfm5_acc_bank.sv
  rtl_qfit/qfit_acc32_vector_serializer.sv
  tb_qfit/fakeram45_relation_models.sv
  tb_qfit/fakeram45_acc_models.sv
  tb_qfit/tb_qfit_local5_erep_calibration_v4.sv
)

ASSERTIONS=(
  verif_qfit/qfit_local5_1rw_active_projection_assertions.sv
  verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv
  verif_qfit/qfit_single_port_acc_memory_assertions.sv
  verif_qfit/qfit_tcfm5_assertions.sv
  verif_qfit/qfit_acc32_vector_serializer_assertions.sv
)

COMMON_VERILATOR_WARNINGS=(
  -Wno-fatal
  -Wno-DECLFILENAME
  -Wno-BLKSEQ
  -Wno-WIDTHEXPAND
  -Wno-WIDTHTRUNC
  -Wno-UNUSEDSIGNAL
)

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  python3 --version
} >"$OUT_DIR/tool_versions.txt"

python3 -m py_compile \
  scripts/local5_erep_command_schedule_v4.py \
  scripts/local5_erep_identity_service_v4.py \
  sim_qfit/local5_erep_calibration_trace_v4.py

python3 -m unittest discover -s tests \
  -p 'test_local5_erep_command_schedule_v4.py' \
  >"$OUT_DIR/strict_schedule_tests.log" 2>&1
python3 -m unittest discover -s tests \
  -p 'test_local5_erep_identity_service_v4.py' \
  >"$OUT_DIR/strict_identity_tests.log" 2>&1
python3 -m unittest discover -s tests \
  -p 'test_local5_erep_calibration_trace_v4.py' \
  >"$OUT_DIR/strict_trace_parser_tests.log" 2>&1

iverilog -g2012 \
  -s tb_qfit_local5_erep_calibration_v4 \
  -o "$OUT_DIR/iverilog.vvp" \
  "${RTL[@]}" >"$OUT_DIR/iverilog_build.log" 2>&1
vvp "$OUT_DIR/iverilog.vvp" +EREP_TRACE_V4 \
  >"$OUT_DIR/iverilog_trace.log" 2>&1
grep -q '^PASS Local5 EREP calibration v4 ' \
  "$OUT_DIR/iverilog_trace.log"
python3 sim_qfit/local5_erep_calibration_trace_v4.py \
  --trace "$OUT_DIR/iverilog_trace.log" \
  --output "$OUT_DIR/iverilog_evidence.json" \
  >"$OUT_DIR/iverilog_normalize.log" 2>&1
EREP_TRACE_PATH="$OUT_DIR/iverilog_trace.log" \
  python3 -m unittest tests.test_local5_erep_calibration_trace_v4 \
  >"$OUT_DIR/strict_trace_mutation_tests.log" 2>&1

verilator --lint-only --timing --assert -Wall \
  "${COMMON_VERILATOR_WARNINGS[@]}" \
  --top-module tb_qfit_local5_erep_calibration_v4 \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  >"$OUT_DIR/verilator_lint.log" 2>&1

verilator --binary --timing --assert -Wall \
  "${COMMON_VERILATOR_WARNINGS[@]}" \
  --top-module tb_qfit_local5_erep_calibration_v4 \
  -Mdir "$OUT_DIR/verilator_obj" \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  >"$OUT_DIR/verilator_build.log" 2>&1
"$OUT_DIR/verilator_obj/Vtb_qfit_local5_erep_calibration_v4" \
  +EREP_TRACE_V4 >"$OUT_DIR/verilator_trace.log" 2>&1
grep -q '^PASS Local5 EREP calibration v4 ' \
  "$OUT_DIR/verilator_trace.log"
python3 sim_qfit/local5_erep_calibration_trace_v4.py \
  --trace "$OUT_DIR/verilator_trace.log" \
  --output "$OUT_DIR/verilator_evidence.json" \
  >"$OUT_DIR/verilator_normalize.log" 2>&1

python3 - \
  "$OUT_DIR/iverilog_evidence.json" \
  "$OUT_DIR/verilator_evidence.json" \
  "$OUT_DIR/cross_sim_summary.json" <<'PY'
import json
import sys
from pathlib import Path

left = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
right = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
paths = (
    "status",
    "phase_boundaries",
    "common_event_contract",
    "snapshot_state_contract",
    "snapshot_event_contract",
    "measured_boundaries",
    "relation_fifo2",
    "bank_commands",
    "numeric_ledgers",
    "physical_rmw_ledgers",
    "backpressure_coverage",
    "c0_measured_tail_schedule_reconstruction",
    "identity_service.direct_online.ordered_ledger_digest",
    "identity_service.direct_online.unordered_multiset_digest",
    "identity_service.tcfm5_1rw.ordered_ledger_digest",
    "identity_service.tcfm5_1rw.unordered_multiset_digest",
)

def get(value, path):
    for key in path.split("."):
        value = value[key]
    return value

mismatches = [path for path in paths if get(left, path) != get(right, path)]
if mismatches:
    raise SystemExit(f"cross-simulator evidence mismatch: {mismatches}")
summary = {
    "schema": "local5_erep_calibration_cross_sim_v4",
    "status": "PASS",
    "matched_fields": list(paths),
    "measured_boundaries": left["measured_boundaries"],
    "relation_fifo2": left["relation_fifo2"],
    "bank_commands": left["bank_commands"],
    "c0": {
        key: left["c0_measured_tail_schedule_reconstruction"][key]
        for key in (
            "cycles",
            "measured_full_boundary_cycles",
            "measured_execute_tail_rule_reconstruction_cycles",
            "boundary_match",
            "relative_command_count",
            "event_ledger_sha256",
        )
    },
}
Path(sys.argv[3]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

sha256sum \
  "${RTL[@]}" \
  "${ASSERTIONS[@]}" \
  tests/test_local5_erep_calibration_trace_v4.py \
  tests/test_local5_erep_command_schedule_v4.py \
  tests/test_local5_erep_identity_service_v4.py \
  sim_qfit/local5_erep_calibration_trace_v4.py \
  sim_qfit/run_local5_erep_calibration_v4.sh \
  scripts/local5_erep_command_schedule_v4.py \
  scripts/local5_erep_identity_service_v4.py \
  >"$OUT_DIR/source_sha256.txt"

{
  git rev-parse HEAD
  git status --short -- \
    "${RTL[@]}" "${ASSERTIONS[@]}" \
    tests/test_local5_erep_calibration_trace_v4.py \
    tests/test_local5_erep_command_schedule_v4.py \
    tests/test_local5_erep_identity_service_v4.py \
    sim_qfit/local5_erep_calibration_trace_v4.py \
    sim_qfit/run_local5_erep_calibration_v4.sh \
    scripts/local5_erep_command_schedule_v4.py \
    scripts/local5_erep_identity_service_v4.py
} >"$OUT_DIR/source_git_state.txt"

sha256sum \
  "$OUT_DIR"/{tool_versions.txt,source_sha256.txt,source_git_state.txt,iverilog.vvp,iverilog_build.log,iverilog_trace.log,iverilog_evidence.json,iverilog_normalize.log,verilator_lint.log,verilator_build.log,verilator_trace.log,verilator_evidence.json,verilator_normalize.log,cross_sim_summary.json,strict_schedule_tests.log,strict_identity_tests.log,strict_trace_parser_tests.log,strict_trace_mutation_tests.log} \
  "$OUT_DIR/verilator_obj/Vtb_qfit_local5_erep_calibration_v4" \
  >"$OUT_DIR/result_sha256.txt"

python3 - "$OUT_DIR/complete.json" "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[2]).resolve()
hashes = output / "result_sha256.txt"
value = {
    "schema": "local5_erep_calibration_complete_v4",
    "status": "PASS_SYNTHETIC_RTL_CALIBRATION_ONLY",
    "formal_adapter_status": "DENY",
    "output_directory": str(output),
    "source_sha256_file": str(output / "source_sha256.txt"),
    "source_sha256_file_sha256": hashlib.sha256(
        (output / "source_sha256.txt").read_bytes()
    ).hexdigest(),
    "source_git_state_sha256": hashlib.sha256(
        (output / "source_git_state.txt").read_bytes()
    ).hexdigest(),
    "result_sha256_file": str(hashes),
    "result_sha256_file_sha256": hashlib.sha256(hashes.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

sha256sum \
  "$OUT_DIR/result_sha256.txt" \
  "$OUT_DIR/complete.json" \
  >"$OUT_DIR/receipt_sha256.txt"

printf 'PASS Local5 EREP calibration v4 output=%s\n' "$OUT_DIR"
