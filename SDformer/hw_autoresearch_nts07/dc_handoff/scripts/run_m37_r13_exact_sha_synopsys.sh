#!/usr/bin/env bash
set -euo pipefail

# M37-r13 is an exact-input, non-overwriting, foreground-only Synopsys run.
# Tool stdout/stderr is written directly to raw logs: no tee pipeline, no
# background shell, and therefore no reconstructed pipeline status or orphan.

m37_repo=/home/zhumd/work/sdformer_codex/SDformer
m37_hw="$m37_repo/hw_autoresearch_nts07"
m37_dc="$m37_hw/dc_handoff"
m37_run=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_r13_exact_sha_synopsys_r1_20260823
m37_design=qfit_atlif_csd_reconstruct_t10
m37_rtl="$m37_hw/rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv"
m37_receipt="$m37_hw/contracts/m37_r12_exact_sha_vcs_receipt_r1_20260823.json"
m37_review="$m37_hw/results/m37_r12_independent_hammer_review_20260823/m37_r12_independent_hammer_review.json"
m37_validator="$m37_dc/scripts/validate_m37_r12_independent_hammer_review.py"
m37_contract="$m37_hw/contracts/m37_r13_exact_sha_synopsys_contract_r1_20260823.json"
m37_filelist="$m37_dc/filelists/date_m37_r13_csd_reconstruct_t10_dc.f"
m37_sdc="$m37_dc/constraints/date_m37_csd_reconstruct_t10.sdc"
m37_dc_tcl="$m37_dc/scripts/run_dc_m37_r13_exact_sha.tcl"
m37_sta_tcl="$m37_dc/scripts/run_sta_m37_r13_exact_sha.tcl"
m37_fm_tcl="$m37_dc/scripts/run_formality_m37_r13_exact_sha.tcl"
m37_auditor="$m37_dc/scripts/audit_m37_dc_evidence.py"
m37_runner="$m37_dc/scripts/run_m37_r13_exact_sha_synopsys.sh"
m37_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m37_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m37_r8=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_dc_3p000ns_r2_20260822
m37_r9=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_dc_3p000ns_r9_20260822
m37_r8_area="$m37_r8/reports/area.rpt"
m37_r8_qor="$m37_r8/reports/qor.rpt"
m37_r9_area="$m37_r9/reports/area.rpt"
m37_r9_orphan="$m37_r9/ORPHAN_RUN_DO_NOT_CITE.txt"
m37_finalized=0

m37_sha() {
    sha256sum "$1" | awk '{print $1}'
}

m37_require_sha() {
    local m37_path="$1"
    local m37_expected="$2"
    if [[ ! -s "$m37_path" ]]; then
        echo "missing or empty exact input: $m37_path" >&2
        exit 3
    fi
    local m37_observed
    m37_observed="$(m37_sha "$m37_path")"
    if [[ "$m37_observed" != "$m37_expected" ]]; then
        echo "exact-input SHA mismatch: $m37_path expected=$m37_expected observed=$m37_observed" >&2
        exit 3
    fi
}

m37_partial_seal() {
    local m37_rc=$?
    if [[ -d "$m37_run" && "$m37_finalized" -eq 0 ]]; then
        set +e
        {
            echo "status=FAIL_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$m37_rc"
            echo "candidate_changed=false"
            echo "headline_admitted=false"
            echo "power_energy_ptpx_system_claim_admitted=false"
            date -u +"sealed_utc=%Y-%m-%dT%H:%M:%SZ"
        } > "$m37_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
        (
            cd "$m37_run" || exit 1
            find . -type f \
                ! -name PARTIAL_EVIDENCE.sha256 \
                ! -name PARTIAL_EVIDENCE_CHECK.raw.log \
                -print0 | sort -z | xargs -0 sha256sum \
                > PARTIAL_EVIDENCE.sha256
            sha256sum --strict -c PARTIAL_EVIDENCE.sha256 \
                > PARTIAL_EVIDENCE_CHECK.raw.log 2>&1
        )
        find "$m37_run" -type f -exec chmod 0444 {} +
        find "$m37_run" -type d -exec chmod 0555 {} +
        set -e
    fi
    exit "$m37_rc"
}
trap m37_partial_seal EXIT

if [[ -e "$m37_run" ]]; then
    echo "refusing to overwrite fixed M37-r13 evidence directory: $m37_run" >&2
    exit 5
fi
for m37_tool in dc_shell fm_shell /usr/bin/python3.6 sha256sum awk grep find xargs; do
    if ! command -v "$m37_tool" >/dev/null 2>&1; then
        echo "required tool missing: $m37_tool" >&2
        exit 2
    fi
done
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -x fm_shell >/dev/null || pgrep -x fm_shell_exec >/dev/null; then
    echo "refusing M37-r13 because a DC/Formality shell is already active" >&2
    exit 4
fi

m37_require_sha "$m37_rtl" f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd
m37_require_sha "$m37_receipt" 5d23131f4ec721d7028cec5d363000798f749f7689e3db18c269b08e3cefb265
m37_require_sha "$m37_review" 8828cac5642ba5648ccaa4734340e92b551e9adffaad0a92b570a0c43b456319
m37_require_sha "$m37_validator" 5e748caa3d295e873fbfaffa9b06234c5339e5f97b424c35405976aec785fd72
m37_require_sha "$m37_contract" 5a3ba83be2bf75b90b9b0a4229a32bc0dd03339536151fdfc01340d677dbef0f
m37_require_sha "$m37_filelist" 7a8df4c793963a17de642194820c812c6a3e7bd4d521cb803f2065446dcfc2af
m37_require_sha "$m37_sdc" 6c76b6207d2bd9b4e9a14887fb534f208f9bcdff8ee23b4f925144b1da07eff4
m37_require_sha "$m37_dc_tcl" 1ab5eb1d19ac79405fabe7449d565b3d8b84ad8a906c6a0f7170bef3cc7edc9b
m37_require_sha "$m37_sta_tcl" 6cf7e597b7171b26e032f6f34b1bfac0e04101ca5ddbff6099587a2efff03d19
m37_require_sha "$m37_fm_tcl" cdd8e8551a00cd96593c5f8277fe37f50373c537466699f2f05feea55327b4c9
m37_require_sha "$m37_auditor" e4222ca0515cb41d0739124d0b8d0b117531466873905f48aebdd0acc0f28341
m37_require_sha "$m37_slow" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m37_require_sha "$m37_fast" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m37_require_sha "$m37_r8_area" 262f179fbc839908a1ab7cf4f9a0539bdc6369f8362e524f8c38ec1d1f7bd24d
m37_require_sha "$m37_r8_qor" b6abee05c23625273ca10c0c4e3102d1e879e57bd4cf5cfdb4838e4127351037
m37_require_sha "$m37_r9_area" 25e6b8c2c376ac6f0f4c1323894e372d642943f02492e279db1c6f5de8bf3999
m37_require_sha "$m37_r9_orphan" 41c091fc9883b0bb0230547241ebe3ba21051d473356c0f2b493ce05107d94da

mkdir -p "$m37_run/reports" "$m37_run/netlist" \
    "$m37_run/work/dc" "$m37_run/work/sta" "$m37_run/work/formality"
{
    echo "status=RUNNING_NOT_CITABLE"
    echo "candidate_sha256=f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd"
    echo "clock_period_ns=3.000"
    echo "tool_invocation=foreground_raw_log_no_tee_no_background"
} > "$m37_run/RUN_IN_PROGRESS.txt"

{
    sha256sum "$m37_rtl" "$m37_receipt" "$m37_review" "$m37_validator" \
        "$m37_contract" "$m37_filelist" "$m37_sdc" "$m37_dc_tcl" \
        "$m37_sta_tcl" "$m37_fm_tcl" "$m37_auditor" "$m37_runner" \
        "$m37_slow" "$m37_fast" "$m37_r8_area" "$m37_r8_qor" \
        "$m37_r9_area" "$m37_r9_orphan"
} > "$m37_run/input_sha256.txt"
set +e
sha256sum --strict -c "$m37_run/input_sha256.txt" \
    > "$m37_run/input_manifest_check.raw.log" 2>&1
m37_input_rc=$?
set -e
echo "$m37_input_rc" > "$m37_run/input_manifest_check.rc"
[[ "$m37_input_rc" -eq 0 ]]

# Re-run the frozen independent hammer.  Its canonical JSON stdout must be
# byte-identical to the pinned review, not merely return zero.
set +e
(
    cd "$m37_repo"
    /usr/bin/python3.6 "$m37_validator"
) > "$m37_run/r12_review_validation.raw.log" \
    2> "$m37_run/r12_review_validation.stderr.raw.log"
m37_validation_rc=$?
set -e
echo "$m37_validation_rc" > "$m37_run/r12_review_validation.rc"
[[ "$m37_validation_rc" -eq 0 ]]
[[ ! -s "$m37_run/r12_review_validation.stderr.raw.log" ]]
cmp -s "$m37_run/r12_review_validation.raw.log" "$m37_review"
[[ "$(m37_sha "$m37_run/r12_review_validation.raw.log")" == \
    8828cac5642ba5648ccaa4734340e92b551e9adffaad0a92b570a0c43b456319 ]]

export HW_ROOT="$m37_hw"
export RTL_FILELIST="$m37_filelist"
export SDC_FILE="$m37_sdc"
export OUTPUT_DIR="$m37_run"
export CLOCK_PERIOD_NS=3.000
export LIB_DB="$m37_slow"
export MIN_LIB_DB="$m37_fast"
export OPERATING_CONDITION=ssg0p9v125c

echo "dc_shell -f $m37_dc_tcl" > "$m37_run/dc.command.txt"
set +e
(
    cd "$m37_run/work/dc"
    dc_shell -f "$m37_dc_tcl"
) > "$m37_run/dc.raw.log" 2>&1
m37_dc_rc=$?
set -e
echo "$m37_dc_rc" > "$m37_run/dc.rc"
[[ "$m37_dc_rc" -eq 0 ]]
[[ "$(grep -xc 'M37_R13_DC_INTERNAL_COMPLETE=PASS' "$m37_run/DC_INTERNAL_COMPLETE.txt")" -eq 1 ]]
if grep -Eq '^(Error|Fatal):' "$m37_run/dc.raw.log"; then
    echo "DC raw log contains Error/Fatal" >&2
    exit 9
fi

m37_required_dc=(
    reports/constraint_contract_precompile.rpt
    reports/constraint_contract_postcompile.rpt
    reports/hierarchy_precompile.rpt
    reports/hierarchy_postcompile.rpt
    reports/resources_precompile.rpt
    reports/resources_postcompile.rpt
    reports/references_precompile.rpt
    reports/references_postcompile.rpt
    reports/qor.rpt
    reports/area.rpt
    reports/timing_setup.rpt
    reports/timing_hold.rpt
    reports/check_design_postcompile.rpt
    reports/check_timing_postcompile.rpt
    reports/constraint_violators.rpt
    reports/clocks.rpt
    "netlist/${m37_design}_mapped.v"
    "netlist/${m37_design}_mapped.sdc"
    "netlist/${m37_design}.ddc"
    "netlist/${m37_design}.svf"
)
for m37_rel in "${m37_required_dc[@]}"; do
    [[ -s "$m37_run/$m37_rel" ]] || {
        echo "DC missing required output: $m37_rel" >&2
        exit 10
    }
done
for m37_contract_report in \
        "$m37_run/reports/constraint_contract_precompile.rpt" \
        "$m37_run/reports/constraint_contract_postcompile.rpt"; do
    grep -qx 'physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO' \
        "$m37_contract_report"
    grep -q '^Name[[:space:]]*:[[:space:]]*ZeroWireload$' \
        "$m37_contract_report"
done
/usr/bin/python3.6 - "$m37_run/reports/clocks.rpt" <<'PY'
from __future__ import print_function
import pathlib
import re
import sys
data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(r"^core_clk\s+([0-9.]+)\s+\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$", data, re.MULTILINE)
if not match or abs(float(match.group(1)) - 3.0) > 1e-9 or "p" in match.group(2):
    raise SystemExit("3ns ideal/unpropagated core_clk contract failed")
PY

set +e
/usr/bin/python3.6 "$m37_auditor" \
    --dc-log "$m37_run/dc.raw.log" \
    --resources-pre "$m37_run/reports/resources_precompile.rpt" \
    --resources-post "$m37_run/reports/resources_postcompile.rpt" \
    --references-pre "$m37_run/reports/references_precompile.rpt" \
    --references-post "$m37_run/reports/references_postcompile.rpt" \
    --mapped-netlist "$m37_run/netlist/${m37_design}_mapped.v" \
    --report "$m37_run/reports/m37_strict_zero_multiplier_link_audit.rpt" \
    > "$m37_run/structural_audit.raw.log" 2> "$m37_run/structural_audit.stderr.raw.log"
m37_audit_rc=$?
set -e
echo "$m37_audit_rc" > "$m37_run/structural_audit.rc"
[[ "$m37_audit_rc" -eq 0 ]]
grep -qx 'physical_multiplier_hit_total=0' \
    "$m37_run/reports/m37_strict_zero_multiplier_link_audit.rpt"

export DDC_FILE="$m37_run/netlist/${m37_design}.ddc"
echo "dc_shell -f $m37_sta_tcl" > "$m37_run/sta.command.txt"
set +e
(
    cd "$m37_run/work/sta"
    dc_shell -f "$m37_sta_tcl"
) > "$m37_run/sta.raw.log" 2>&1
m37_sta_rc=$?
set -e
echo "$m37_sta_rc" > "$m37_run/sta.rc"
[[ "$m37_sta_rc" -eq 0 ]]
[[ "$(grep -xc 'M37_R13_STA_INTERNAL_COMPLETE=PASS' "$m37_run/STA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
if grep -Eq '^(Error|Fatal):' "$m37_run/sta.raw.log"; then
    echo "fresh STA raw log contains Error/Fatal" >&2
    exit 12
fi
for m37_rel in reports/sta_qor.rpt reports/sta_area.rpt \
        reports/sta_setup.rpt reports/sta_hold.rpt reports/sta_check_timing.rpt; do
    [[ -s "$m37_run/$m37_rel" ]] || {
        echo "STA missing required output: $m37_rel" >&2
        exit 12
    }
done

/usr/bin/python3.6 - "$m37_run" <<'PY'
from __future__ import print_function
import collections
import json
import pathlib
import re
import sys

run = pathlib.Path(sys.argv[1])

def text(rel):
    return (run / rel).read_text(encoding="utf-8", errors="replace")

area_text = text("reports/sta_area.rpt")
def number(pattern, source, cast=float):
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        raise SystemExit("missing metric pattern: " + pattern)
    return cast(match.group(1))

def min_slack(rel):
    values = [float(value) for value in re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)", text(rel))]
    if not values:
        raise SystemExit("no slack in " + rel)
    return min(values)

cells = number(r"^Number of cells:\s+([0-9]+)", area_text, int)
comb_cells = number(r"^Number of combinational cells:\s+([0-9]+)", area_text, int)
seq_cells = number(r"^Number of sequential cells:\s+([0-9]+)", area_text, int)
macro_cells = number(r"^Number of macros/black boxes:\s+([0-9]+)", area_text, int)
comb_area = number(r"^Combinational area:\s+([0-9.]+)", area_text)
noncomb_area = number(r"^Noncombinational area:\s+([0-9.]+)", area_text)
total_area = number(r"^Total cell area:\s+([0-9.]+)", area_text)
setup = min_slack("reports/sta_setup.rpt")
hold = min_slack("reports/sta_hold.rpt")

registers = {}
for name, width in re.findall(
        r"^\|\s*([^|]+?)\s*\|\s*Flip-flop\s*\|\s*([0-9]+)\s*\|",
        text("dc.raw.log"), re.MULTILINE):
    name = name.strip()
    width = int(width)
    if name in registers and registers[name] != width:
        raise SystemExit("register width drift for " + name)
    registers[name] = width
architectural_bits = sum(registers.values())
if architectural_bits != 5979:
    raise SystemExit("unexpected architectural register sum: {}".format(architectural_bits))

warnings = []
codes = collections.Counter()
for line in text("dc.raw.log").splitlines():
    if line.startswith("Warning:"):
        warnings.append(line)
        match = re.search(r"\(([A-Z][A-Z0-9_-]*-[0-9]+)\)\s*$", line)
        codes[match.group(1) if match else "UNCLASSIFIED"] += 1

audit = text("reports/m37_strict_zero_multiplier_link_audit.rpt")
multiplier = number(r"^physical_multiplier_hit_total=([0-9]+)$", audit, int)
r8 = 63671.579642
r9 = 185820.892828
gate = 70038.737606
metrics = {
    "schema": "m37_r13_dc_sta_metrics_v1",
    "candidate_sha256": "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd",
    "scope": "standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro",
    "clock_period_ns": 3.0,
    "cell_count": cells,
    "combinational_cell_count": comb_cells,
    "sequential_cell_count": seq_cells,
    "macro_or_blackbox_cell_count": macro_cells,
    "architectural_register_bits_from_dc_elaboration": architectural_bits,
    "architectural_registers": dict(sorted(registers.items())),
    "combinational_area_um2": comb_area,
    "noncombinational_area_um2": noncomb_area,
    "total_cell_area_um2": total_area,
    "setup_wns_ns": setup,
    "hold_wns_ns": hold,
    "physical_multiplier_count": multiplier,
    "dc_warning_count": len(warnings),
    "dc_warning_codes": dict(sorted(codes.items())),
    "ab": {
        "r8_area_um2": r8,
        "r9_orphan_diagnostic_area_um2": r9,
        "r13_over_r8": total_area / r8,
        "r13_reduction_vs_r8_percent": (r8 - total_area) / r8 * 100.0,
        "r13_over_r9_diagnostic": total_area / r9,
        "r13_reduction_vs_r9_diagnostic_percent": (r9 - total_area) / r9 * 100.0,
    },
    "gates": {
        "area_limit_um2": gate,
        "area_pass": total_area <= gate,
        "setup_pass": setup >= 0.0,
        "hold_pass": hold >= 0.0,
        "zero_physical_multiplier_pass": multiplier == 0,
        "zero_macro_or_blackbox_pass": macro_cells == 0,
    },
}
metrics["gates"]["all_dc_sta_gates_pass"] = all([
    metrics["gates"]["area_pass"], metrics["gates"]["setup_pass"],
    metrics["gates"]["hold_pass"], metrics["gates"]["zero_physical_multiplier_pass"],
    metrics["gates"]["zero_macro_or_blackbox_pass"]])
(run / "dc_sta_metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(run / "reports/m37_dc_warning_ledger.rpt").write_text(
    "warning_count={}\nwarning_codes={}\nwarning_sequence_begin\n{}\nwarning_sequence_end\n".format(
        len(warnings), json.dumps(dict(sorted(codes.items())), sort_keys=True),
        "\n".join(warnings)), encoding="utf-8")
print(json.dumps(metrics, sort_keys=True))
PY

export MAPPED_NETLIST="$m37_run/netlist/${m37_design}_mapped.v"
export SVF_FILE="$m37_run/netlist/${m37_design}.svf"
echo "fm_shell -f $m37_fm_tcl" > "$m37_run/formality.command.txt"
set +e
(
    cd "$m37_run/work/formality"
    fm_shell -f "$m37_fm_tcl"
) > "$m37_run/formality.raw.log" 2>&1
m37_fm_rc=$?
set -e
echo "$m37_fm_rc" > "$m37_run/formality.rc"
[[ "$m37_fm_rc" -eq 0 ]]
[[ "$(grep -xc 'M37_R13_FORMALITY_INTERNAL_COMPLETE=PASS' "$m37_run/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
if grep -Eq '^(Error|Fatal):' "$m37_run/formality.raw.log"; then
    echo "Formality raw log contains Error/Fatal" >&2
    exit 14
fi

/usr/bin/python3.6 - "$m37_run" <<'PY'
from __future__ import print_function
import json
import pathlib
import re
import sys

run = pathlib.Path(sys.argv[1])
log = (run / "formality.raw.log").read_text(encoding="utf-8", errors="replace")
unmatched_path = run / "reports/formality_unmatched.rpt"
if not unmatched_path.is_file() or unmatched_path.stat().st_size == 0:
    raise SystemExit("missing Formality unmatched report")
unmatched = unmatched_path.read_text(encoding="utf-8", errors="replace")

succeeded = len(re.findall(r"^Verification SUCCEEDED$", log, re.MULTILINE))
passing_values = [int(v) for v in re.findall(r"^\s*([0-9]+) Passing compare points\s*$", log, re.MULTILINE)]
failing_rows = re.findall(r"^Failing \(not equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$", log, re.MULTILINE)
if succeeded != 1 or len(passing_values) != 1 or not failing_rows:
    raise SystemExit("Formality terminal result is missing or ambiguous")
failing_columns = [int(v) for v in failing_rows[-1].split()]
failing = failing_columns[-1]

def unmatched_count(label):
    values = [int(v) for v in re.findall(
        r"^\s*([0-9]+)\([0-9]+\) Unmatched reference\(implementation\) " + label + r"\s*$",
        unmatched + "\n" + log, re.MULTILINE)]
    if not values:
        if "No unmatched points." in unmatched:
            return 0
        raise SystemExit("missing unmatched evidence: " + label)
    return values[-1]

unmatched_compare = unmatched_count(r"compare points")
unmatched_primary = unmatched_count(r"primary inputs, black-box outputs")
unmatched_unread = unmatched_count(r"unread points")

def report_nonempty(rel):
    path = run / rel
    if not path.exists() or path.stat().st_size == 0:
        return 0
    data = path.read_text(encoding="utf-8", errors="replace")
    # Empty reports contain headings/prompts but no object-name rows.  Count
    # only explicit point summary or list rows, never guidance prose.
    values = [int(v) for v in re.findall(
        r"^\s*([0-9]+)\s+(?:Aborted|Unverified|Failing) (?:compare )?points\s*$",
        data, re.MULTILINE | re.IGNORECASE)]
    return max(values) if values else 0

aborted = report_nonempty("reports/formality_aborted.rpt")
unverified = report_nonempty("reports/formality_unverified.rpt")
fmr147 = len(re.findall(r"FMR_ELAB-147", log))
metrics = {
    "schema": "m37_r13_formality_metrics_v1",
    "verification_succeeded_terminal_count": succeeded,
    "passing_compare_points": passing_values[-1],
    "failing_compare_points": failing,
    "failing_result_columns": failing_columns,
    "aborted_compare_points": aborted,
    "unverified_compare_points": unverified,
    "unmatched_compare_points": unmatched_compare,
    "unmatched_primary_or_blackbox_points": unmatched_primary,
    "unmatched_unread_points_diagnostic_only": unmatched_unread,
    "fmr_elab_147_count": fmr147,
    "message_filters_used": False,
}
metrics["all_formality_gates_pass"] = all([
    succeeded == 1, passing_values[-1] > 0, failing == 0, aborted == 0,
    unverified == 0, unmatched_compare == 0, unmatched_primary == 0,
    fmr147 == 0])
(run / "formality_metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not metrics["all_formality_gates_pass"]:
    raise SystemExit("strict Formality gates failed: " + json.dumps(metrics, sort_keys=True))
print(json.dumps(metrics, sort_keys=True))
PY

/usr/bin/python3.6 - "$m37_run" <<'PY'
from __future__ import print_function
import json
import pathlib
import sys
run = pathlib.Path(sys.argv[1])
dc = json.loads((run / "dc_sta_metrics.json").read_text())
fm = json.loads((run / "formality_metrics.json").read_text())
passed = bool(dc["gates"]["all_dc_sta_gates_pass"] and fm["all_formality_gates_pass"])
status = {
    "schema": "m37_r13_exact_sha_synopsys_runner_status_v1",
    "status": "PASS_EXACT_SHA_DC_STA_FORMALITY" if passed else "FAIL_GATE_DO_NOT_CITE",
    "candidate_sha256": "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd",
    "candidate_changed": False,
    "dc_rc": 0,
    "sta_rc": 0,
    "formality_rc": 0,
    "dc_sta_gates_pass": bool(dc["gates"]["all_dc_sta_gates_pass"]),
    "formality_gates_pass": bool(fm["all_formality_gates_pass"]),
    "claim_boundary": {
        "scope": "standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro",
        "headline_paper_ppa_admitted": False,
        "power_energy_ptpx_system_speedup_admitted": False,
    },
}
(run / "runner_status.json").write_text(
    json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not passed:
    raise SystemExit("M37-r13 DC/STA/Formality gate failure")
PY

(
    cd "$m37_run"
    find . -type f \
        ! -path './work/*' \
        ! -name output_sha256.txt \
        ! -name output_manifest_check.raw.log \
        ! -name output_manifest_check.rc \
        ! -name run_local_seal.sha256 \
        ! -name run_local_seal_check.raw.log \
        ! -name completion_seal.sha256 \
        ! -name completion_seal_check.raw.log \
        ! -name RUN_IN_PROGRESS.txt \
        ! -name RUN_COMPLETE.txt \
        -print0 | sort -z | xargs -0 sha256sum > output_sha256.txt
    set +e
    sha256sum --strict -c output_sha256.txt > output_manifest_check.raw.log 2>&1
    m37_output_rc=$?
    set -e
    echo "$m37_output_rc" > output_manifest_check.rc
    [[ "$m37_output_rc" -eq 0 ]]
    sha256sum input_sha256.txt output_sha256.txt runner_status.json \
        dc_sta_metrics.json formality_metrics.json > run_local_seal.sha256
    sha256sum --strict -c run_local_seal.sha256 > run_local_seal_check.raw.log 2>&1
    mv RUN_IN_PROGRESS.txt RUN_COMPLETE.txt
    {
        echo "status=PASS_EXACT_SHA_DC_STA_FORMALITY"
        echo "candidate_sha256=f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd"
        echo "candidate_changed=false"
        echo "clock_period_ns=3.000"
        echo "scope=standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro"
        echo "headline_paper_ppa_admitted=false"
        echo "power_energy_ptpx_system_speedup_admitted=false"
    } > RUN_COMPLETE.txt
    sha256sum input_sha256.txt output_sha256.txt run_local_seal.sha256 \
        RUN_COMPLETE.txt > completion_seal.sha256
    sha256sum --strict -c completion_seal.sha256 > completion_seal_check.raw.log 2>&1
)

m37_finalized=1
trap - EXIT
find "$m37_run" -type f -exec chmod 0444 {} +
find "$m37_run" -type d -exec chmod 0555 {} +
echo "M37_R13_EXACT_SHA_SYNOPSYS=PASS run=$m37_run"
