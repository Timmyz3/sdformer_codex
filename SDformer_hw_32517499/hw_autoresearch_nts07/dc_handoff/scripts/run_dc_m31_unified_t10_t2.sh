#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_design=qfit_atlif_unified_t10_t2_stream_core
task_filelist="${task_dc_root}/filelists/date_m31_unified_t10_t2_dc.f"
task_sdc="${task_dc_root}/constraints/date_m31_unified_t10_t2.sdc"
task_period="${CLOCK_PERIOD_NS:-3.000}"
task_period_tag="${task_period//./p}ns"
task_output="${OUTPUT_DIR:-${task_dc_root}/runs/m31_unified_t10_t2_dc_${task_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"
task_python="${PYTHON_BIN:-python3}"

task_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$task_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$task_fast_default}"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M31 DC requires Synopsys dc_shell" >&2
    exit 2
fi
if ! command -v "$task_python" >/dev/null 2>&1; then
    echo "M31 DC requires the configured Python interpreter" >&2
    exit 14
fi
if [[ ! -f "$LIB_DB" || ! -f "$MIN_LIB_DB" ]]; then
    echo "M31 DC slow/fast library is missing" >&2
    exit 3
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
    || pgrep -f 'common_shell_exec.*-shell[[:space:]]+dc_shell|/dc_shell([[:space:]]|$)' >/dev/null; then
    echo "refusing M31 DC because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$task_output/dc.log" ]]; then
    echo "refusing to overwrite M31 DC evidence: $task_output" >&2
    exit 5
fi

mkdir -p "$task_output"
export DESIGN_NAME="$task_design"
export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_filelist"
export SDC_FILE="$task_sdc"
export OUTPUT_DIR="$task_output"
export CLOCK_PERIOD_NS="$task_period"
export LIB_DB MIN_LIB_DB
export OPERATING_CONDITION="${OPERATING_CONDITION:-ssg0p9v125c}"

dc_shell -f "${task_dc_root}/scripts/run_dc_m31_unified_t10_t2.tcl" \
    2>&1 | tee "$task_output/dc.log"

if grep -q '^Error:' "$task_output/dc.log"; then
    echo "M31 DC log contains a Tcl/DC error" >&2
    exit 9
fi

for task_report in \
    reports/m31_resource_audit_precompile.rpt \
    reports/m31_resource_audit_postcompile.rpt \
    reports/resources_precompile.rpt reports/resources_postcompile.rpt \
    reports/references_postcompile.rpt reports/qor.rpt reports/area.rpt \
    reports/clocks.rpt \
    reports/timing_setup.rpt reports/timing_hold.rpt \
    reports/check_design_postcompile.rpt \
    reports/check_timing_postcompile.rpt \
    netlist/${task_design}_mapped.v netlist/${task_design}.ddc; do
    if [[ ! -s "$task_output/$task_report" ]]; then
        echo "M31 DC missing evidence: $task_report" >&2
        exit 6
    fi
done
if [[ "$(grep -c '^status=PASS_EXACT_ONE_POOL_96_LEAVES$' \
        "$task_output/reports/m31_resource_audit_postcompile.rpt")" -ne 1 ]]; then
    echo "M31 postcompile exact-resource audit did not pass" >&2
    exit 7
fi
if grep -Eiq '(^|[^A-Za-z])(GTECH|DW_[A-Za-z0-9_]*mult|unresolved)([^A-Za-z]|$)' \
        "$task_output/reports/references_postcompile.rpt"; then
    echo "M31 postcompile references contain unresolved arithmetic" >&2
    exit 8
fi

task_lint_report="$task_output/reports/check_design_postcompile.rpt"
task_warning_count="$(grep -c '^Warning:' "$task_lint_report" || true)"
task_lint33_count="$(grep -c '^Warning:.*same net.*u_mul_pool.*(LINT-33)' \
    "$task_lint_report" || true)"
task_operand_b_count="$(grep -c "^   Net '.*operand_b" "$task_lint_report" || true)"
if [[ "$task_warning_count" -ne 256 || "$task_lint33_count" -ne 256 \
        || "$task_operand_b_count" -ne 256 ]]; then
    echo "M31 check_design warning population is outside the exact operand_b whitelist" >&2
    exit 10
fi
if grep '^Warning:' "$task_lint_report" \
        | grep -Ev '^Warning:.*same net.*u_mul_pool.*\(LINT-33\)$' >/dev/null; then
    echo "M31 check_design contains a non-whitelisted warning" >&2
    exit 11
fi
if grep '^   Net ' "$task_lint_report" | grep -v 'operand_b' >/dev/null; then
    echo "M31 LINT-33 is not confined to multiplier operand_b reuse" >&2
    exit 12
fi
{
    echo "status=PASS_EXACT_LINT33_OPERAND_B_WEIGHT_REUSE_ONLY"
    echo "warning_count=$task_warning_count"
    echo "lint33_count=$task_lint33_count"
    echo "operand_b_detail_count=$task_operand_b_count"
} > "$task_output/reports/m31_lint33_audit.rpt"

task_timing_status=NOT_MET
if grep -q 'slack (MET)' "$task_output/reports/timing_setup.rpt" \
    && grep -q 'slack (MET)' "$task_output/reports/timing_hold.rpt" \
    && ! grep -q 'slack (VIOLATED)' "$task_output/reports/timing_setup.rpt" \
    && ! grep -q 'slack (VIOLATED)' "$task_output/reports/timing_hold.rpt"; then
    task_timing_status=MET
fi
if [[ "$task_timing_status" != MET ]]; then
    echo "M31 setup/hold timing is not met" >&2
    exit 13
fi

task_machine_audit="$task_output/reports/m31_r4_dc_machine_audit.json"
"$task_python" \
    "${task_dc_root}/scripts/audit_m31_r4_dc_reports.py" \
    --run-dir "$task_output" --period "$task_period" \
    --output "$task_machine_audit"
"$task_python" - "$task_machine_audit" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], "r"))
if result.get("status") != (
        "PASS_M31_R4_EXACT96_ZERO_WIRE_IDEAL_CLOCK_3NS_LOGIC_ONLY"):
    raise SystemExit("M31 DC machine-audit status drift")
cells = result.get("cell_accounting", {})
if cells.get("total_cell_instances_including_hierarchy") != (
        cells.get("hierarchical_cell_instances", -1)
        + cells.get("leaf_mapped_cell_instances", -2)):
    raise SystemExit("M31 DC machine-audit cell accounting drift")
PY
{
    echo "status=PASS_EXACT96_PREMACRO_LOGIC_ONLY"
    echo "paper_ppa_ready=false"
    echo "clock_period_ns=$task_period"
    echo "timing_status=$task_timing_status"
    echo "pool_count=1"
    echo "multiplier_leaf_count=96"
    echo "cell_count_fields=SEE_m31_r4_dc_machine_audit_total_hierarchical_leaf"
    echo "interconnect_model=ZERO_WIRE_LOAD"
    echo "clock_network_model=IDEAL_UNPROPAGATED"
    echo "macro_db=NONE"
    echo "library_slow=$LIB_DB"
    echo "library_fast=$MIN_LIB_DB"
} > "$task_output/admission.txt"

sha256sum \
    "$task_hw_root/rtl_m31/qfit_signed_int8_mul96_pool.sv" \
    "$task_hw_root/rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv" \
    "$task_filelist" "$task_sdc" \
    "${task_dc_root}/scripts/run_dc_m31_unified_t10_t2.tcl" \
    "${task_dc_root}/scripts/run_dc_m31_unified_t10_t2.sh" \
    "${task_dc_root}/scripts/audit_m31_r4_dc_reports.py" \
    "$LIB_DB" "$MIN_LIB_DB" "$task_output/dc.log" \
    "$task_output/admission.txt" \
    "$task_machine_audit" \
    "$task_output/reports/m31_resource_audit_precompile.rpt" \
    "$task_output/reports/m31_resource_audit_postcompile.rpt" \
    "$task_output/reports/m31_lint33_audit.rpt" \
    "$task_output/reports/qor.rpt" "$task_output/reports/area.rpt" \
    "$task_output/reports/clocks.rpt" \
    "$task_output/reports/references_postcompile.rpt" \
    "$task_output/reports/check_design_postcompile.rpt" \
    "$task_output/reports/check_timing_postcompile.rpt" \
    "$task_output/reports/timing_setup.rpt" \
    "$task_output/reports/timing_hold.rpt" \
    "$task_output/netlist/${task_design}_mapped.v" \
    "$task_output/netlist/${task_design}.svf" \
    "$task_output/netlist/${task_design}.ddc" \
    > "$task_output/evidence.sha256"
echo "M31_DC_PASS exact96 timing=$task_timing_status run=$task_output"
