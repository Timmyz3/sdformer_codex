#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="${M498_DC_RUN:-$task_dc_root/runs/m498_segmented_enable_dc_3p000ns_r1_20260827}"
task_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
task_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
task_core="rtl_m498/m498_segmented_enable_parent_queue_pipeline.sv"
task_wrapper="rtl_m498/m498_segmented_enable_backpressure_safe_parent_queue_pipeline.sv"
task_filelist="dc_handoff/filelists/date_m498_segmented_enable_parent_queue_dc.f"
task_sdc="dc_handoff/constraints/date_m498_segmented_enable_parent_queue_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m498_segmented_enable_parent_queue_exact_sha.tcl"
task_contract="contracts/m498_segmented_enable_parent_queue_logic_only_dc_contract_r1_20260827.json"
task_vcs="results/m498_segmented_enable_vcs_r1_exact_20260827"

task_sha() { sha256sum "$1" | awk '{print $1}'; }
task_expect() {
    local task_path=$1 task_expected=$2
    [[ -f "$task_path" ]] || { echo "missing $task_path" >&2; exit 3; }
    [[ "$(task_sha "$task_path")" == "$task_expected" ]] || {
        echo "M498 SHA mismatch $task_path" >&2
        exit 3
    }
}

[[ ! -e "$task_run" ]] || { echo "M498 output exists $task_run" >&2; exit 5; }
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -f '/common_shell_exec -shell dc_shell ' >/dev/null; then
    echo "M498 refuses to collide with another Design Compiler run" >&2
    exit 4
fi
cd "$task_hw_root"
task_expect "$task_dc" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
task_expect "$task_slow" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
task_expect "$task_fast" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
task_expect "$task_core" 95967b386e31427e48b4a8cae81af244ee7b68b47316915d95fa6e1e92978fba
task_expect "$task_wrapper" 3f7c188df7325984dbf536faa6c09362a74d562ddbe5ffe13a80a042616954d7
task_expect "$task_filelist" 1e2f4c7f84632d5922adfc6ccbe11ec849843d701c7205d9f6ff4ebd6baf9614
task_expect "$task_sdc" b768d3d094b63d445fdf576be6ace5b134f861434a953862e30401098f740327
task_expect "$task_tcl" 974b67f00c451fc79d3effca14e38713721fced57d50f8d7cc0ef88334ca2265
task_expect "$task_contract" 87d77361232c637ac2b92d9ce75dfc9d1c632fbd0ba0a37f9e5719473cdc6600
task_expect "$task_vcs/SHA256SUMS.seal.sha256" 066099ce5e57f736ec4fdf6891cff48c8e15453066fb1cbe743c98591a6d4113
task_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
(cd "$task_vcs" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
grep -Fq PASS_M498_SEGMENTED_ENABLE_EXACT_VCS "$task_vcs/RUN_COMPLETE.txt"

mkdir -p "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
sha256sum "$task_core" "$task_wrapper" "$task_filelist" "$task_sdc" \
    "$task_tcl" "$task_contract" "$task_vcs/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md "$task_dc" "$task_slow" \
    "$task_fast" > "$task_run/input_sha256.txt"
cp "$task_contract" "$task_run/contract.json"

export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_hw_root/$task_filelist"
export LIB_DB="$task_slow"
export MIN_LIB_DB="$task_fast"
export SDC_FILE="$task_hw_root/$task_sdc"
export OUTPUT_DIR="$task_run"
export CLOCK_PERIOD_NS=3.000
export OPERATING_CONDITION=ssg0p9v125c

set +e
"$task_dc" -f "$task_hw_root/$task_tcl" > "$task_run/dc.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/dc.rc"
[[ "$task_rc" -eq 0 ]] || exit 20
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$task_run/dc.log" || exit 21
grep -Fq "Current design is 'm498_segmented_enable_backpressure_safe_parent_queue_pipeline'." "$task_run/dc.log" || exit 22
! grep -Fq "Current design is 'm479_lane_local_backpressure_safe_parent_queue_pipeline'." "$task_run/dc.log" || exit 22
! grep -Fq "Current design is 'm476r2_backpressure_safe_parent_queue_pipeline'." "$task_run/dc.log" || exit 22
grep -Fq 'Thank you...' "$task_run/dc.log" || exit 23
for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt hierarchy_postcompile.rpt \
        resources_postcompile.rpt references_postcompile.rpt ports.rpt \
        port_count.txt; do
    [[ -s "$task_run/reports/$task_report" ]] || exit 30
done
task_mapped="$task_run/netlist/m498_segmented_enable_backpressure_safe_parent_queue_pipeline_mapped.v"
[[ -s "$task_mapped" ]] || exit 31
grep -Fq 'module m498_segmented_enable_backpressure_safe_parent_queue_pipeline' "$task_mapped" || exit 31
! grep -Fq 'module m479_lane_local_backpressure_safe_parent_queue_pipeline' "$task_mapped" || exit 31
! grep -Fq 'slack (VIOLATED)' "$task_run/reports/timing_setup.rpt" "$task_run/reports/timing_hold.rpt" || exit 32
[[ "$(grep -Fc 'This design has no violated constraints.' "$task_run/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33

task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_combo="$(awk '/Number of combinational cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_buffer_count="$(grep -c 'BUFFD1BWP35P140' "$task_mapped" || true)"
for task_value in "$task_area" "$task_cells" "$task_seq" "$task_combo" \
        "$task_levels" "$task_setup" "$task_hold" "$task_buffer_count"; do
    [[ -n "$task_value" ]] || exit 34
done
python3 - "$task_area" "$task_setup" "$task_hold" "$task_seq" \
    "$task_buffer_count" <<'PY'
import sys
area, setup, hold = map(float, sys.argv[1:4])
seq, buffers = map(int, sys.argv[4:6])
assert area <= 44779.2, area
assert setup >= 0 and hold >= 0, (setup, hold)
assert seq == 5508, seq
assert buffers >= 204, buffers
PY

python3 - "$task_run" "$task_area" "$task_cells" "$task_seq" \
    "$task_combo" "$task_levels" "$task_setup" "$task_hold" \
    "$task_buffer_count" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
area, cells, seq, combo, levels, setup, hold, buffers = (
    float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
    float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9]))
receipt = {
    "schema": "m498_segmented_enable_parent_queue_logic_only_dc_receipt_v1",
    "status": "PASS_M498_SEGMENTED_ENABLE_LOGIC_ONLY_DC_3NS_CLEAN",
    "tool": "Synopsys Design Compiler V-2023.12-SP3",
    "design_identity": "m498_segmented_enable_backpressure_safe_parent_queue_pipeline",
    "technology": "TSMC28 HPC+ standard cells",
    "operating_condition": "ssg0p9v125c",
    "clock_period_ns": 3.0,
    "measured": {
        "cell_area_um2": area, "cell_count": cells,
        "sequential_cells": seq, "combinational_cells": combo,
        "logic_levels": levels, "setup_worst_slack_ns": setup,
        "hold_worst_slack_ns": hold,
        "mapped_buffd1_occurrences": buffers,
    },
    "physical_tree_gate": {
        "expected_explicit_buffers_minimum": 204,
        "mapped_buffd1_occurrences_at_least_expected": buffers >= 204,
        "registered_enable_staging_added": False,
        "five_design_rule_reports_clean": True,
        "area_gate_um2": 44779.2,
        "area_gate_pass": area <= 44779.2,
    },
    "macro_count": 0,
    "admission": {
        "m498_logic_only_dc_sta": True,
        "zero_cycle_segmented_enable_tree": True,
        "formality": False, "physical_timing": False,
        "scratch_or_psum_macro_ppa": False, "power": False,
        "energy": False, "performance_admitted": False,
        "paper_ppa_ready": False, "full_network": False,
        "system_speedup": False, "date_headline": False,
    },
    "required_next_gate": "Independent receipt-blind hammer must verify exact identity, buffer topology, five clean electrical classes, area, and no sequential staging.",
}
(run / "m498_segmented_enable_parent_queue_logic_only_dc_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(run / "RUN_COMPLETE.txt").write_text(
    "PASS_M498_SEGMENTED_ENABLE_LOGIC_ONLY_DC_3NS_CLEAN\n"
    "cell_area_um2={}\nsetup_worst_slack_ns={}\n"
    "hold_worst_slack_ns={}\nsequential_cells={}\n"
    "mapped_buffd1_occurrences={}\nmacro_count=0\n"
    "performance_admitted=false\npaper_ppa_ready=false\n".format(
        area, setup, hold, seq, buffers))
files = [p for p in sorted(run.rglob("*")) if p.is_file() and p.name not in {
    "evidence_manifest.sha256", "evidence_manifest.seal.sha256"}]
(run / "evidence_manifest.sha256").write_text("".join(
    "{}  {}\n".format(hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(run))
    for p in files))
(run / "evidence_manifest.seal.sha256").write_text(
    hashlib.sha256((run / "evidence_manifest.sha256").read_bytes()).hexdigest()
    + "  evidence_manifest.sha256\n")
PY
(cd "$task_run" && sha256sum -c evidence_manifest.sha256 >/dev/null && \
    sha256sum -c evidence_manifest.seal.sha256 >/dev/null)
task_complete=1
rm -f "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"
echo "PASS M498 exact DC sealed at $task_run"
