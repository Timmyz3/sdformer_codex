#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m4_descriptor_resident_vcs_sva_b400_20260821}"
VECTOR_DIR="${VECTOR_DIR:-$RUN_ROOT/m4_descriptor_resident_real_vectors_b400_20260821}"
REAL_TRACE="$VECTOR_DIR/real_descriptors.txt"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3.12}"

command -v vcs >/dev/null 2>&1
test -x "$PYTHON_BIN"
test -s "$VECTOR_DIR/manifest.json"
test -s "$REAL_TRACE"
mkdir -p "$RUN_DIR/static" "$RUN_DIR/real" "$RUN_DIR/ideal"

"$PYTHON_BIN" - "$VECTOR_DIR/manifest.json" "$REAL_TRACE" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
trace_path = pathlib.Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
population = manifest.get("population", {})
if manifest.get("status") != "PASS_CHECKPOINT_BOUND_REAL_BITMAP_DESCRIPTOR_BATCHES":
    raise SystemExit("M4 vector manifest is not admitted")
if manifest.get("availability_mode") not in {"temporal_fenced", "layer_materialized_greedy"}:
    raise SystemExit("M4 B400 availability boundary is not explicit")
if population.get("batches") != 400 or population.get("descriptors", 0) <= 0:
    raise SystemExit("M4 real-vector population is not the frozen B400 contract")
if population.get("outputs", 0) <= 0 or population.get("negative_sources", 0) <= 0:
    raise SystemExit("M4 output/negative-source coverage is incomplete")
if population.get("compact_issue_cycles", 0) <= 0 or population.get("m4_wall_cycles", 0) <= 0:
    raise SystemExit("M4 executable scheduler/wall-cycle contract is empty")
digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
if digest != manifest.get("sha256", {}).get("real_descriptors.txt"):
    raise SystemExit("M4 real descriptor SHA256 mismatch")
if len(manifest.get("sample_batches", {})) != 40:
    raise SystemExit("M4 vectors do not cover both lines, identities, and all samples")
PY

cd "$RUN_DIR/static"
sha256sum \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_resident_engine.sv" \
  > compile_inputs.sha256
vcs -full64 -lca -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_resident_engine.sv" \
  -top tb_qfit_dual_line_descriptor_resident_engine -o simv \
  2>&1 | tee compile.log
./simv -assert report="$RUN_DIR/static/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

cd "$RUN_DIR/real"
sha256sum \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_resident_real.sv" \
  > compile_inputs.sha256
vcs -full64 -lca -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  "$ROOT/rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_descriptor_resident_real.sv" \
  -top tb_qfit_dual_line_descriptor_resident_real -o simv \
  2>&1 | tee compile.log
./simv "+REAL_TRACE=$REAL_TRACE" \
  +RANDOM_WEIGHT_BACKPRESSURE \
  -assert report="$RUN_DIR/real/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

cd "$RUN_DIR/ideal"
"$RUN_DIR/real/simv" "+REAL_TRACE=$REAL_TRACE" \
  +IDEAL_WALL_CYCLES \
  "+EXPECTED_WALL_CYCLES=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["population"]["m4_wall_cycles"])' "$VECTOR_DIR/manifest.json")" \
  -assert report="$RUN_DIR/ideal/assertion_report.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

grep -q "PASS_M4_DESCRIPTOR_RESIDENT outputs=4" \
  "$RUN_DIR/static/simulation.log"
"$PYTHON_BIN" - "$VECTOR_DIR/manifest.json" "$RUN_DIR/real/simulation.log" <<'PY'
import json
import pathlib
import re
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
text = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(
    r"PASS_M4_DESCRIPTOR_RESIDENT_REAL batches=(\d+) descriptors=(\d+) "
    r"outputs=(\d+) request_beats=(\d+) bank_reads=(\d+) output_stalls=(\d+) "
    r"request_stalls=(\d+) source_checks=(\d+)",
    text,
)
if len(matches) != 1:
    raise SystemExit("M4 real PASS record is missing or ambiguous")
actual = tuple(map(int, matches[0]))
population = manifest["population"]
expected = (
    population["batches"], population["descriptors"], population["outputs"],
    population["compact_issue_cycles"],
    population["lane_expanded_selected_sources"],
)
if actual[:5] != expected or any(value <= 0 for value in actual[5:]):
    raise SystemExit(f"M4 VCS/model mismatch actual={actual} expected={expected}")
PY
"$PYTHON_BIN" - "$VECTOR_DIR/manifest.json" "$RUN_DIR/ideal/simulation.log" <<'PY'
import json
import pathlib
import re
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
text = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
match = re.search(r"wall_cycles=(\d+) ideal=1", text)
expected = manifest["population"]["m4_wall_cycles"]
if match is None or int(match.group(1)) != expected:
    raise SystemExit(f"M4 ideal VCS wall-cycle mismatch expected={expected}")
PY
! grep -Eq "Assertion failed|failed at|Fatal:|^Error:" \
  "$RUN_DIR/static/simulation.log" "$RUN_DIR/real/simulation.log" \
  "$RUN_DIR/ideal/simulation.log" "$RUN_DIR/static/assertion_report.txt" \
  "$RUN_DIR/real/assertion_report.txt" "$RUN_DIR/ideal/assertion_report.txt"

cd "$RUN_DIR"
sha256sum \
  static/compile_inputs.sha256 static/compile.log static/simulation.log \
  static/assertion_report.txt static/simv \
  real/compile_inputs.sha256 real/compile.log real/simulation.log \
  real/assertion_report.txt real/simv \
  ideal/simulation.log ideal/assertion_report.txt \
  "$VECTOR_DIR/manifest.json" "$REAL_TRACE" > evidence.sha256
echo "PASS Synopsys VCS/SVA M4 descriptor-resident static plus B400 real regression"
