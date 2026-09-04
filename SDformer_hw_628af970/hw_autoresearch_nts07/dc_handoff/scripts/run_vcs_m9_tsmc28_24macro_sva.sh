#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_MANIFEST="$ROOT/dc_handoff/macro_manifests/tsmc28_128x128_1rw_20260822.json"
ASSET_DIR="${TSMC28_SRAM_ASSET_DIR:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821}"
MODEL="${TSMC28_SRAM_MODEL:-$ASSET_DIR/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v}"
RUN_ROOT="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m9_tsmc28_24macro_vcs_sva_r4_twophase_bounded_20260822}"

command -v vcs >/dev/null
test -s "$MODEL"
test -s "$ASSET_DIR/SHA256SUMS"
test -s "$REPO_MANIFEST"
if [[ -e "$RUN_DIR/evidence.sha256" || -e "$RUN_DIR/simv" ]]; then
  echo "refusing to overwrite existing M9 24-macro VCS evidence: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

vcs -full64 -lca -sverilog -assert svaext \
  -timescale=1ns/1ps +define+UNIT_DELAY \
  +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
  +define+QFIT_TSMC28_FULL_DEPTH \
  -top tb_qfit_dual_granularity_temporal_state_engine \
  "$MODEL" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_granularity_temporal_state_engine.sv" \
  -o simv 2>&1 | tee compile.log
./simv -assert report="$RUN_DIR/assertion_report.txt" \
  +ntb_random_seed=20260822 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "M9_1_FULL_DEPTH_PROGRESS rows=128" simulation.log
grep -q "M9_1_FULL_DEPTH_PHASE phase=local rows=128" simulation.log
grep -q "M9_1_FULL_DEPTH_PHASE phase=motion rows=128" simulation.log
grep -q "M9_1_FULL_DEPTH rows=128 local_writes=128 motion_rmws=128 bank_lane_value_checks=24576" simulation.log
grep -q "M9_1_RESULT wide=12 narrow=14 wide_local=2 wide_motion=10 narrow_local=8 narrow_motion=6 abort=1 wide_errors=1 narrow_errors=1 rmw_stalls=3 reset_block_checks=3 domain_fault_checks=1" simulation.log
grep -q "PASS: Synopsys VCS M9.1 SRAM-realistic atomic Local/Motion shared temporal state exact" simulation.log
test -s assertion_report.txt.disablelog
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" \
  simulation.log assertion_report.txt assertion_report.txt.disablelog
mapfile -t disable_lines < <(sed '/^[[:space:]]*$/d' assertion_report.txt.disablelog)
if [[ "${#disable_lines[@]}" -ne 3 \
      || "${disable_lines[0]-}" != "Disabled Module Assertions (compiletime)" \
      || "${disable_lines[1]-}" != "Assertions disabled via '-assert hier' switch" \
      || "${disable_lines[2]-}" != "Dynamically disabled assertions at End-of-Simulation" ]]; then
  echo "M9 VCS disabled one or more concrete assertions" >&2
  sed -n '1,160p' assertion_report.txt.disablelog >&2
  exit 3
fi

# A separate expected-fail elaboration proves unsupported geometry cannot
# silently fall back to registers or an aliased/truncated vendor macro.
vcs -full64 -lca -sverilog \
  -timescale=1ns/1ps +define+UNIT_DELAY \
  -top tb_qfit_tsmc28_unsupported_geometry \
  "$MODEL" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  "$ROOT/tb_qfit/tb_qfit_tsmc28_unsupported_geometry.sv" \
  -o simv_unsupported 2>&1 | tee unsupported_compile.log
set +e
./simv_unsupported > unsupported_simulation.log 2>&1
unsupported_status=$?
set -e
# VCS reports an initial-block $fatal as a fatal transcript but can return
# process status zero after its internal $finish.  Treat the simulator status
# as diagnostic only and fail closed on the exact time-zero transcript.
grep -q "TSMC28 SRAM adapter requires DEPTH=128 DATA_W=512 ADDR_W=7" \
  unsupported_simulation.log
grep -q '^Fatal: .* at time 0 ps$' unsupported_simulation.log
grep -q '^\$finish at simulation time[[:space:]]*0$' unsupported_simulation.log
! grep -q "unsupported TSMC28 geometry failed to terminate" \
  unsupported_simulation.log
printf '%s\n' "$unsupported_status" > unsupported_exit_status.txt

python3 - "$REPO_MANIFEST" "$RUN_DIR/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

asset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mapping = asset["logical_mapping"]
if mapping != {
    "logical_banks": 6,
    "logical_depth": 128,
    "logical_word_bits": 512,
    "native_macros_per_logical_bank": 4,
    "native_macro_instances": 24,
    "total_data_bits": 393216,
}:
    raise SystemExit("private SRAM manifest geometry mismatch")
payload = {
    "schema": "qfit_m9_tsmc28_24macro_vcs_sva_v4",
    "status": "PASS_FULL_CONTROLLER_24_VENDOR_MACRO_FULL_DEPTH_FUNCTIONAL_SVA",
    "logical_geometry": mapping,
    "vendor_cell": asset["native_macro"]["cell"],
    "vendor_model_mode": "UNIT_DELAY",
    "test_scope": {
        "all_six_logical_banks_observed_by_wide_checks": True,
        "each_logical_bank_observed_by_narrow_checks": True,
        "all_128_physical_rows_local_write_motion_rmw_checked": True,
        "two_phase_cross_row_alias_sensitive_readback": True,
        "directed_prelude_isolated_by_reset_domain_fence": True,
        "full_depth_bank_lane_value_checks": 24576,
        "unsupported_geometry_expected_fail": True,
        "unsupported_geometry_vcs_fatal_transcript_required": True,
        "wide_local": 2,
        "wide_motion_rmw": 10,
        "narrow_local": 8,
        "narrow_motion_rmw": 6,
        "abort": 1,
        "protocol_error_injection": True,
        "reset_and_domain_fence": True,
        "sva_runtime_enabled": True,
    },
    "claim_boundary": (
        "Synopsys VCS integration of the M9 controller with six logical 128x512 "
        "banks realized by 24 private vendor functional-model instances. UNIT_DELAY "
        "functional/SVA evidence only; not SDF timing, DC/STA, PTPX, post-route, "
        "energy, FPS, or paper-PPA."
    ),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

sha256sum \
  "$MODEL" "$ASSET_DIR/SHA256SUMS" "$REPO_MANIFEST" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  "$ROOT/rtl_qfit/qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/verif_qfit/qfit_dual_granularity_temporal_state_engine_assertions.sv" \
  "$ROOT/tb_qfit/tb_qfit_dual_granularity_temporal_state_engine.sv" \
  "$ROOT/tb_qfit/tb_qfit_tsmc28_unsupported_geometry.sv" \
  "$ROOT/dc_handoff/scripts/run_vcs_m9_tsmc28_24macro_sva.sh" \
  run_manifest.json compile.log simulation.log assertion_report.txt \
  assertion_report.txt.disablelog \
  unsupported_compile.log unsupported_simulation.log unsupported_exit_status.txt \
  > evidence.sha256
sha256sum -c evidence.sha256
echo "PASS Synopsys VCS/SVA M9 full controller plus 24 TSMC28 SRAM macros"
