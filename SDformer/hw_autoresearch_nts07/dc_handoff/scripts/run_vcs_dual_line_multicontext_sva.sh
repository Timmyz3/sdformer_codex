#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m3_p16c4_vcs_sva_20260821}"
VECTOR_DIR="${VECTOR_DIR:-$RUN_ROOT/m3_p16c4_real_vectors_s20k_20260821}"
REAL_TRACE="${REAL_TRACE:-$VECTOR_DIR/real_commands.txt}"

command -v vcs >/dev/null 2>&1
test -s "$REAL_TRACE"
test -s "$VECTOR_DIR/manifest.json"
mkdir -p "$RUN_DIR"

python3 - "$VECTOR_DIR/manifest.json" "$REAL_TRACE" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
trace_path = pathlib.Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
if manifest.get("commands") != 20_000 or manifest.get("batches") != 5_000:
    raise SystemExit("M3 real-vector population is not the frozen 20k/5k contract")
if digest != manifest.get("sha256", {}).get("real_commands.txt"):
    raise SystemExit("M3 real-vector SHA256 does not match its manifest")
PY

cd "$RUN_DIR"
vcs -full64 -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
  "$ROOT/rtl_qfit/qfit_dual_line_multicontext_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_line_multicontext_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_line_multicontext_engine.sv" \
  -top tb_qfit_dual_line_multicontext_engine -o simv 2>&1 | tee compile.log

./simv -assert report="$RUN_DIR/assertion_report_random.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation_random.log
./simv "+REAL_TRACE=$REAL_TRACE" \
  -assert report="$RUN_DIR/assertion_report_real.txt" \
  +ntb_random_seed=20260821 2>&1 | tee simulation_real.log

grep -q "PASS_M3_P16C4_DUAL_LINE commands=805 outputs=805" simulation_random.log
grep -q "PASS_M3_P16C4_DUAL_LINE commands=20005 outputs=20005" simulation_real.log
grep -q "real_commands=20000" simulation_real.log
! grep -Eq "Assertion failed|failed at|Fatal:|^Error:" \
  simulation_random.log simulation_real.log \
  assertion_report_random.txt assertion_report_real.txt

sha256sum compile.log simulation_random.log simulation_real.log simv \
  assertion_report_random.txt assertion_report_real.txt \
  "$VECTOR_DIR/manifest.json" "$REAL_TRACE" > evidence.sha256
echo "PASS Synopsys VCS/SVA M3 P16C4 random plus 20k real-vector regression"
