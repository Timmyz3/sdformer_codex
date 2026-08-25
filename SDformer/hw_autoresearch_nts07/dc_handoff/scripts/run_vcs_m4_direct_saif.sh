#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
VECTOR_DIR="${VECTOR_DIR:-$RUN_ROOT/m4_descriptor_resident_real_vectors_temporal_b400_20260821}"
VCS_DIR="${VCS_DIR:-$RUN_ROOT/m4_descriptor_resident_vcs_sva_temporal_b400_20260821}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m4_descriptor_resident_saif_temporal_b400_20260821}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3}"
REAL_TRACE="$VECTOR_DIR/real_descriptors.txt"
SIMV="$VCS_DIR/real/simv"
M4_SAIF_FILE="$RUN_DIR/qfit_dual_line_descriptor_resident_engine.saif"
ACTIVITY_MANIFEST="$RUN_DIR/activity_manifest.json"
STRIP_PATH="tb_qfit_dual_line_descriptor_resident_real/dut"

test -x "$SIMV"
test -s "$REAL_TRACE"
test -s "$VECTOR_DIR/manifest.json"
mkdir -p "$RUN_DIR"
export M4_SAIF_FILE

EXPECTED_WALL_CYCLES="$($PYTHON_BIN -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["population"]["m4_wall_cycles"])' \
  "$VECTOR_DIR/manifest.json")"

cd "$RUN_DIR"
"$SIMV" "+REAL_TRACE=$REAL_TRACE" +IDEAL_WALL_CYCLES \
  +UCLI_SAIF_STOP \
  "+EXPECTED_WALL_CYCLES=$EXPECTED_WALL_CYCLES" \
  -ucli -do "$ROOT/dc_handoff/scripts/m4_saif.ucli.tcl" \
  +ntb_random_seed=20260821 2>&1 | tee simulation.log

test -s "$M4_SAIF_FILE"
grep -q "wall_cycles=$EXPECTED_WALL_CYCLES ideal=1" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log
$PYTHON_BIN "$ROOT/dc_handoff/scripts/make_m4_direct_saif_manifest.py" \
  --vector-manifest "$VECTOR_DIR/manifest.json" \
  --trace "$REAL_TRACE" \
  --simulation-log "$RUN_DIR/simulation.log" \
  --saif "$M4_SAIF_FILE" \
  --simv "$SIMV" \
  --runner "$ROOT/dc_handoff/scripts/run_vcs_m4_direct_saif.sh" \
  --ucli-script "$ROOT/dc_handoff/scripts/m4_saif.ucli.tcl" \
  --strip-path "$STRIP_PATH" \
  --output "$ACTIVITY_MANIFEST"
$PYTHON_BIN "$ROOT/dc_handoff/scripts/audit_saif_manifest.py" \
  --design qfit_dual_line_descriptor_resident_engine \
  --saif "$M4_SAIF_FILE" \
  --strip-path "$STRIP_PATH" \
  --manifest "$ACTIVITY_MANIFEST"
sha256sum "$M4_SAIF_FILE" simulation.log "$ACTIVITY_MANIFEST" \
  "${ACTIVITY_MANIFEST%.json}_audit.json" "$VECTOR_DIR/manifest.json" \
  "$REAL_TRACE" "$ROOT/dc_handoff/scripts/m4_saif.ucli.tcl" \
  "$ROOT/dc_handoff/scripts/make_m4_direct_saif_manifest.py" > evidence.sha256
echo "PASS Synopsys VCS direct-SAIF temporal B400 activity"
