#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m4_stateful_real_vcs_sva_20260821}"
VECTOR_DIR="${VECTOR_DIR:-$RUN_ROOT/m4_stateful_real_vectors_s40x2_20260821}"
TRACE="$VECTOR_DIR/stateful_real_descriptors.txt"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3.12}"
PERF_MODE="${PERF_MODE:-0}"
STREAM_MODE="${STREAM_MODE:-0}"
STATE_QUEUE_DEPTH="${STATE_QUEUE_DEPTH:-4}"
USE_SHARED_WIDE_METADATA="${USE_SHARED_WIDE_METADATA:-1}"
mkdir -p "$RUN_DIR"

if ! [[ "$STATE_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "STATE_QUEUE_DEPTH must be a positive integer" >&2
  exit 2
fi
if [[ "$USE_SHARED_WIDE_METADATA" != "0" &&
      "$USE_SHARED_WIDE_METADATA" != "1" ]]; then
  echo "USE_SHARED_WIDE_METADATA must be 0 or 1" >&2
  exit 2
fi

"$PYTHON_BIN" - "$VECTOR_DIR/manifest.json" "$TRACE" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
trace = pathlib.Path(sys.argv[2])
expected = {
    "sequences": 80, "batches": 800, "descriptors": 12880,
    "outputs": 9360, "local_outputs": 7677, "motion_outputs": 1683,
}
if manifest.get("status") != "PASS_CHECKPOINT_BOUND_TEMPORAL_STATE_SEQUENCES":
    raise SystemExit("real state vector status mismatch")
if any(manifest["population"].get(key) != value for key, value in expected.items()):
    raise SystemExit("real state vector population mismatch")
if hashlib.sha256(trace.read_bytes()).hexdigest() != manifest["sha256"]["stateful_real_descriptors.txt"]:
    raise SystemExit("real state trace SHA mismatch")
if "not checkpoint-weight Acc32" not in manifest.get("claim_boundary", ""):
    raise SystemExit("real state claim boundary missing")
for label, identity in manifest.get("identities", {}).items():
    directory = pathlib.Path(identity["directory"])
    sources = {
        "tile_manifest_sha256": directory / "manifest.json",
        "tile_records_sha256": directory / "tile_records.csv",
        "packed_tiles_sha256": directory / "packed_tiles.npz",
    }
    for field, source in sources.items():
        if not source.is_file():
            raise SystemExit(f"{label} source package is missing: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != identity[field]:
            raise SystemExit(f"{label} source package SHA mismatch: {field}")
PY

cd "$RUN_DIR"
vcs -full64 -lca -sverilog -assert svaext -debug_access+all \
  -timescale=1ns/1ps +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  "-pvalue+tb_qfit_dual_line_descriptor_stateful_real.STATE_QUEUE_DEPTH=$STATE_QUEUE_DEPTH" \
  "-pvalue+tb_qfit_dual_line_descriptor_stateful_real.USE_SHARED_WIDE_METADATA=$USE_SHARED_WIDE_METADATA" \
  -top tb_qfit_dual_line_descriptor_stateful_real \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/rtl_qfit/qfit_wide_temporal_state_engine.sv" \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_stateful_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_wide_temporal_state_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_stateful_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_stateful_real.sv" \
  -o simv 2>&1 | tee compile.log

SIM_ARGS=("+REAL_STATE_TRACE=$TRACE")
if [[ "$PERF_MODE" == "1" ]]; then
  SIM_ARGS+=(+PERF_MODE)
elif [[ "$PERF_MODE" != "0" ]]; then
  echo "PERF_MODE must be 0 or 1" >&2
  exit 2
fi
if [[ "$STREAM_MODE" == "1" ]]; then
  SIM_ARGS+=(+STREAM_MODE)
elif [[ "$STREAM_MODE" != "0" ]]; then
  echo "STREAM_MODE must be 0 or 1" >&2
  exit 2
fi
./simv "${SIM_ARGS[@]}" \
  -assert report="$RUN_DIR/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "PASS_M4_STATEFUL_REAL sequences=80 batches=800 descriptors=12880 outputs=9360 local_outputs=7677 motion_outputs=1683" simulation.log
if [[ "$PERF_MODE" == "1" && "$STREAM_MODE" == "0" ]]; then
  grep -q "PASS_M4_STATEFUL_PERF pairs=40" simulation.log
fi
if [[ "$STREAM_MODE" == "1" ]]; then
  grep -q "PASS_M4_STATEFUL_STREAMING sequences=80 batches=800 outputs=9360 fifo_writes=9360 fifo_reads=9360" simulation.log
fi
if [[ "$PERF_MODE" == "1" && "$STREAM_MODE" == "1" ]]; then
  grep -q "PASS_M4_STATEFUL_STREAMING_PERF sequences=80 batches=800 outputs=9360 cycles=" simulation.log
fi
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" \
  simulation.log assertion_report.txt
if [[ "$PERF_MODE" == "0" ]]; then
  "$PYTHON_BIN" - assertion_report.txt <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
matches = [int(value) for value in re.findall(r"cp_rmw_backpressure.*?(\d+) match", text)]
if not matches or max(matches) <= 0:
    raise SystemExit("real state VCS missed RMW/output backpressure cover")
PY
fi
if [[ "$STREAM_MODE" == "1" ]]; then
  "$PYTHON_BIN" - assertion_report.txt <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
matches = [int(value) for value in re.findall(
    r"cp_next_batch_accepts_with_state_pending.*?(\d+) match", text)]
if not matches or max(matches) <= 0:
    raise SystemExit("streaming VCS missed next-batch/state-pending cover")
PY
fi
sha256sum \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/rtl_qfit/qfit_wide_temporal_state_engine.sv" \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_stateful_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_wide_temporal_state_engine_assertions.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_stateful_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_stateful_real.sv" \
  "$ROOT/system_simulator/scripts/build_m4_stateful_real_vectors.py" \
  "$ROOT/dc_handoff/scripts/run_vcs_m4_stateful_real_sva.sh" \
  "$VECTOR_DIR/manifest.json" "$TRACE" simv compile.log simulation.log \
  assertion_report.txt > evidence.sha256
echo "PASS Synopsys VCS/SVA M4 stateful H67+Local5 real temporal miter"
