#!/usr/bin/env bash
set -euo pipefail

m37_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m37_hw_root="$(cd "${m37_dc_root}/.." && pwd)"
m37_design=qfit_atlif_csd_reconstruct_t10
m37_filelist="${m37_dc_root}/filelists/date_m37_csd_reconstruct_t10_dc.f"
m37_sdc="${m37_dc_root}/constraints/date_m37_csd_reconstruct_t10.sdc"
m37_tcl="${m37_dc_root}/scripts/run_dc_m37_csd_reconstruct_t10.tcl"
m37_runner="${m37_dc_root}/scripts/run_dc_m37_csd_reconstruct_t10.sh"
m37_source_auditor="${m37_dc_root}/scripts/audit_m37_r8_source_intent.py"
m37_dc_auditor="${m37_dc_root}/scripts/audit_m37_dc_evidence.py"
m37_fm_tcl="${m37_dc_root}/scripts/run_formality_m37_csd_reconstruct_t10.tcl"
m37_snapshot_runner="${m37_dc_root}/scripts/seal_m37_r8_dc_formality_snapshot.sh"
m37_receipt="${m37_hw_root}/contracts/m37_output_receipt_r3_20260822.json"
m37_vcs_contract="${m37_hw_root}/contracts/m37_csd_reconstruct_t10_vcs_contract_r3_20260822.json"
m37_rtl="${m37_hw_root}/rtl_m37/qfit_atlif_csd_reconstruct_t10.sv"
m37_r8_default=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_vcs_r8_20260822
m37_r8="${M37_VCS_R8_DIR:-$m37_r8_default}"
m37_old_dc_default=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m37_csd_reconstruct_t10_dc_3p000ns_r1_20260822
m37_old_dc="${M37_OLD_DC_R1_DIR:-$m37_old_dc_default}"
m37_period="${CLOCK_PERIOD_NS:-3.000}"
m37_period_tag="${m37_period//./p}ns"
m37_output="${OUTPUT_DIR:-${m37_dc_root}/runs/m37_csd_reconstruct_t10_dc_${m37_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"

m37_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m37_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$m37_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$m37_fast_default}"

if [[ "$m37_period" != "3.000" ]]; then
    echo "M37 milestone is frozen to 3.000ns; refusing period $m37_period" >&2
    exit 2
fi
for m37_tool in dc_shell fm_shell python3 sha256sum; do
    if ! command -v "$m37_tool" >/dev/null 2>&1; then
        echo "M37 requires tool: $m37_tool" >&2
        exit 2
    fi
done
m37_required_files=(
    "$LIB_DB"
    "$MIN_LIB_DB"
    "$m37_rtl"
    "$m37_receipt"
    "$m37_vcs_contract"
    "$m37_filelist"
    "$m37_sdc"
    "$m37_tcl"
    "$m37_runner"
    "$m37_source_auditor"
    "$m37_dc_auditor"
    "$m37_fm_tcl"
    "$m37_snapshot_runner"
    "$m37_r8/input_sha256.txt"
    "$m37_r8/output_sha256.txt"
    "$m37_r8/run_local_seal.sha256"
    "$m37_old_dc/FAILED_RESOURCE_AUDIT_DO_NOT_CITE.txt"
    "$m37_old_dc/reports/m37_multiplier_identifier_hits.rpt"
)
for m37_required in "${m37_required_files[@]}"; do
    if [[ ! -s "$m37_required" ]]; then
        echo "M37 required input is missing or empty: $m37_required" >&2
        exit 3
    fi
done
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -x fm_shell >/dev/null || pgrep -x fm_shell_exec >/dev/null; then
    echo "refusing M37 run because a Synopsys synthesis/formality shell is active" >&2
    exit 4
fi
if [[ -e "$m37_output" ]]; then
    echo "refusing to overwrite any M37 evidence path: $m37_output" >&2
    exit 5
fi

# The VCS manifests have different reference roots by construction.
(cd "$m37_hw_root" && sha256sum --strict -c "$m37_r8/input_sha256.txt")
(cd "$m37_r8" && sha256sum --strict -c output_sha256.txt)
(cd "$m37_r8" && sha256sum --strict -c run_local_seal.sha256)
python3 - "$m37_receipt" "$m37_vcs_contract" "$m37_r8" "$m37_rtl" \
        "$m37_r8/input_sha256.txt" "$m37_r8/output_sha256.txt" \
        "$m37_r8/run_local_seal.sha256" \
        "$m37_old_dc/FAILED_RESOURCE_AUDIT_DO_NOT_CITE.txt" \
        "$m37_old_dc/reports/m37_multiplier_identifier_hits.rpt" <<'PY'
import hashlib
import json
import pathlib
import sys

(receipt_path, contract_path, r8_dir, rtl_path, input_manifest,
 output_manifest, local_seal, old_marker, old_report) = map(pathlib.Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
expected = {
    "receipt": "363fb61d2838b6379a065dd8eb23b6219441cfb8ed70164766f07d8469e95d97",
    "contract": "91155bebc0ac7526be6082130c4b05ece5e15d977fa05f64b055cf611ddcd214",
    "rtl": "ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd",
    "input_manifest": "b81d30b3e25795abb53d3e151a9a5c4ec9d6a520e4269dedc889c694c8e092a7",
    "output_manifest": "4a272bd92ab6776d17b0a5d20d18bd259c264b721e7d7daec4156f0de7517d3a",
    "local_seal": "0528c4b311d70c4be88566a7112f36818c757a0c9a7cc482e479722da397c630",
    "old_fail_marker": "9f448e04de0256f3b2f2a741a57810f1836b1ec0707d5b3bd90ec2d00f5c263d",
    "old_multiplier_report": "39285d1604024a9e7570e302fb1d4122cb731dbb17b4cbf793324f61e64e46f7",
}
observed = {
    "receipt": sha(receipt_path),
    "contract": sha(contract_path),
    "rtl": sha(rtl_path),
    "input_manifest": sha(input_manifest),
    "output_manifest": sha(output_manifest),
    "local_seal": sha(local_seal),
    "old_fail_marker": sha(old_marker),
    "old_multiplier_report": sha(old_report),
}
if observed != expected:
    raise SystemExit("M37 frozen evidence SHA mismatch: {}".format(observed))
receipt = json.loads(receipt_path.read_text())
contract = json.loads(contract_path.read_text())
if receipt.get("schema") != "m37_output_receipt_v3":
    raise SystemExit("M37 receipt schema is not r3")
if contract.get("contract") != "m37_csd_reconstruct_t10_vcs_contract_r3":
    raise SystemExit("M37 VCS contract identity is not r3")
if pathlib.Path(receipt["vcs_run"]["directory"]).resolve() != r8_dir.resolve():
    raise SystemExit("M37 receipt does not bind the selected r8 VCS directory")
if receipt["contract"]["sha256"] != expected["contract"]:
    raise SystemExit("M37 receipt contract SHA mismatch")
if receipt["files"]["rtl"][1] != expected["rtl"]:
    raise SystemExit("M37 receipt RTL SHA mismatch")
if receipt["vcs_run"]["input_sha256_manifest"] != expected["input_manifest"]:
    raise SystemExit("M37 receipt input-manifest SHA mismatch")
if receipt["vcs_run"]["output_sha256_manifest"] != expected["output_manifest"]:
    raise SystemExit("M37 receipt output-manifest SHA mismatch")
if receipt["vcs_run"]["run_local_seal"] != expected["local_seal"]:
    raise SystemExit("M37 receipt local-seal SHA mismatch")
if receipt["r8_delta"]["old_dc_3ns_r1_state"] != "FAIL_RESOURCE_AUDIT_DO_NOT_CITE":
    raise SystemExit("M37 old DC failure boundary is absent")
if receipt["r8_delta"]["new_physical_zero_multiplier_resource_claim"] \
        != "NOT_YET_ADMITTED_REQUIRES_FRESH_POST_R8_DC_AND_INDEPENDENT_REVIEW":
    raise SystemExit("M37 receipt physical claim boundary drift")
if contract["r8_delta"]["required_dc_action_after_independent_vcs_review"] \
        != "rerun_from_source_and_require_zero_DW_or_mul_resources":
    raise SystemExit("M37 contract does not require fresh strict-zero DC")
if old_marker.read_text().splitlines()[0] != "status=FAIL_RESOURCE_AUDIT_DO_NOT_CITE":
    raise SystemExit("M37 old DC marker content drift")
print("M37_R8_RECEIPT_R3_AND_OLD_FAIL_BINDING=PASS")
PY

mkdir -p "$m37_output/reports" "$m37_output/netlist"
{
    cd "$m37_hw_root"
    sha256sum \
        rtl_m37/qfit_atlif_csd_reconstruct_t10.sv \
        contracts/m37_output_receipt_r3_20260822.json \
        contracts/m37_csd_reconstruct_t10_vcs_contract_r3_20260822.json \
        dc_handoff/filelists/date_m37_csd_reconstruct_t10_dc.f \
        dc_handoff/constraints/date_m37_csd_reconstruct_t10.sdc \
        dc_handoff/scripts/audit_m37_r8_source_intent.py \
        dc_handoff/scripts/audit_m37_dc_evidence.py \
        dc_handoff/scripts/run_dc_m37_csd_reconstruct_t10.tcl \
        dc_handoff/scripts/run_formality_m37_csd_reconstruct_t10.tcl \
        dc_handoff/scripts/seal_m37_r8_dc_formality_snapshot.sh \
        dc_handoff/scripts/run_dc_m37_csd_reconstruct_t10.sh
    sha256sum "$LIB_DB" "$MIN_LIB_DB" \
        "$m37_r8/input_sha256.txt" "$m37_r8/output_sha256.txt" \
        "$m37_r8/run_local_seal.sha256" \
        "$m37_old_dc/FAILED_RESOURCE_AUDIT_DO_NOT_CITE.txt" \
        "$m37_old_dc/reports/m37_multiplier_identifier_hits.rpt"
} > "$m37_output/input_sha256.txt"
(cd "$m37_hw_root" && sha256sum --strict -c "$m37_output/input_sha256.txt")

python3 "$m37_source_auditor" "$m37_rtl" \
    "$m37_output/reports/m37_source_star_token_ledger.rpt"

export DESIGN_NAME="$m37_design"
export HW_ROOT="$m37_hw_root"
export RTL_FILELIST="$m37_filelist"
export SDC_FILE="$m37_sdc"
export OUTPUT_DIR="$m37_output"
export CLOCK_PERIOD_NS="$m37_period"
export LIB_DB MIN_LIB_DB
export OPERATING_CONDITION="${OPERATING_CONDITION:-ssg0p9v125c}"

set +e
dc_shell -f "$m37_tcl" 2>&1 | tee "$m37_output/dc.log"
m37_dc_rc=${PIPESTATUS[0]}
set -e
echo "$m37_dc_rc" > "$m37_output/dc.exit_status"
if [[ "$m37_dc_rc" -ne 0 ]] \
        || grep -Eq '^(Error|Fatal):' "$m37_output/dc.log"; then
    {
        echo "status=FAIL_DC_TOOL_OR_TCL_DO_NOT_CITE"
        echo "dc_exit_code=$m37_dc_rc"
        echo "formality_ran=false"
    } > "$m37_output/FAILED_DC_TOOL_OR_TCL_DO_NOT_CITE.txt"
    sha256sum "$m37_output/dc.log" "$m37_output/dc.exit_status" \
        "$m37_output/input_sha256.txt" \
        "$m37_output/reports/m37_source_star_token_ledger.rpt" \
        "$m37_output/FAILED_DC_TOOL_OR_TCL_DO_NOT_CITE.txt" \
        > "$m37_output/failed_dc_evidence.sha256"
    exit 9
fi

m37_required_outputs=(
    reports/constraint_contract_precompile.rpt
    reports/constraint_contract_postcompile.rpt
    reports/references_precompile.rpt
    reports/resources_precompile.rpt
    reports/references_postcompile.rpt
    reports/resources_postcompile.rpt
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
for m37_report in "${m37_required_outputs[@]}"; do
    if [[ ! -s "$m37_output/$m37_report" ]]; then
        echo "M37 DC missing evidence: $m37_report" >&2
        exit 6
    fi
done

# This is the first post-DC adjudication.  Any multiplier hit stops the runner
# before timing/area admission and before Formality.
if ! python3 "$m37_dc_auditor" \
        --dc-log "$m37_output/dc.log" \
        --resources-pre "$m37_output/reports/resources_precompile.rpt" \
        --resources-post "$m37_output/reports/resources_postcompile.rpt" \
        --references-pre "$m37_output/reports/references_precompile.rpt" \
        --references-post "$m37_output/reports/references_postcompile.rpt" \
        --mapped-netlist "$m37_output/netlist/${m37_design}_mapped.v" \
        --report "$m37_output/reports/m37_strict_zero_multiplier_link_audit.rpt"; then
    {
        echo "status=FAIL_RESOURCE_AUDIT_DO_NOT_CITE"
        echo "rtl_sha256=ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd"
        echo "fresh_r8_dc=true"
        echo "timing_area_admitted=false"
        echo "formality_ran=false"
    } > "$m37_output/FAILED_RESOURCE_AUDIT_DO_NOT_CITE.txt"
    sha256sum "$m37_output/dc.log" "$m37_output/dc.exit_status" \
        "$m37_output/input_sha256.txt" \
        "$m37_output/reports/m37_source_star_token_ledger.rpt" \
        "$m37_output/reports/m37_strict_zero_multiplier_link_audit.rpt" \
        "$m37_output/FAILED_RESOURCE_AUDIT_DO_NOT_CITE.txt" \
        > "$m37_output/failed_resource_evidence.sha256"
    echo "M37 fresh DC failed strict-zero resource audit; stopping before Formality" >&2
    exit 7
fi

if ! grep -q 'ZeroWireload' \
        "$m37_output/reports/constraint_contract_precompile.rpt" \
        "$m37_output/reports/constraint_contract_postcompile.rpt"; then
    echo "M37 DC did not report the required ZeroWireload contract" >&2
    exit 10
fi
if awk '$1 == "core_clk" {print}' "$m37_output/reports/clocks.rpt" \
        | grep -Eq '[[:space:]][dfpGg]*p[dfpGg]*[[:space:]]+\{clk_core\}$'; then
    echo "M37 clock report marks core_clk propagated; ideal-clock contract failed" >&2
    exit 10
fi
if grep -q '^Warning:' "$m37_output/reports/check_design_postcompile.rpt"; then
    echo "M37 check_design contains warnings" >&2
    grep '^Warning:' "$m37_output/reports/check_design_postcompile.rpt" >&2
    exit 11
fi

# Seal the exact warning sequence observed in this fresh run.  Counts are not
# inherited from r1; only narrowly safe message families are accepted.
python3 - "$m37_output/dc.log" "$m37_rtl" \
        "$m37_output/reports/m37_warning_set.rpt" <<'PY'
import collections
import pathlib
import re
import sys

log_path, rtl_path, report_path = map(pathlib.Path, sys.argv[1:])
warnings = [
    line for line in log_path.read_text(errors="replace").splitlines()
    if line.startswith("Warning:")
]
safe_codes = collections.Counter()
normalized = []
unknown = []
uisn = re.compile(r"^Warning: DesignWare synthetic library dw_foundation\.sldb is added to the synthetic_library in the current command\. \(UISN-40\)$")
tim = re.compile(r"^Warning: Design 'qfit_atlif_csd_reconstruct_t10' contains [0-9]+ high-fanout nets\. A fanout number of 1000 will be used for delay calculations involving these nets\. \(TIM-134\)$")
ver = re.compile(r"^Warning:\s+(.+\.sv):([0-9]+): .+ \(VER-318\)$")
for line in warnings:
    if uisn.match(line):
        safe_codes["UISN-40"] += 1
        normalized.append(line)
        continue
    if tim.match(line):
        safe_codes["TIM-134"] += 1
        normalized.append(line)
        continue
    match = ver.match(line)
    if match and pathlib.Path(match.group(1)).resolve() == rtl_path.resolve():
        safe_codes["VER-318"] += 1
        normalized.append(line.replace(str(rtl_path.resolve()), "RTL"))
        continue
    unknown.append(line)
lines = [
    "status={}".format("PASS_ACTUAL_WARNING_SET_SEALED" if not unknown else "FAIL_UNKNOWN_WARNING_DO_NOT_CITE"),
    "warning_count={}".format(len(warnings)),
    "allowed_message_families=UISN-40,TIM-134,VER-318_ON_FROZEN_RTL",
]
for code in ("UISN-40", "TIM-134", "VER-318"):
    lines.append("{}_actual_count={}".format(code, safe_codes[code]))
lines.append("actual_warning_sequence_begin")
lines.extend(normalized)
lines.append("actual_warning_sequence_end")
lines.append("unknown_warning_sequence_begin")
lines.extend(unknown)
lines.append("unknown_warning_sequence_end")
report_path.write_text("\n".join(lines) + "\n")
if unknown:
    raise SystemExit("M37 DC warning family outside strict safe policy")
PY

m37_timing_status=NOT_MET
if grep -q 'slack (MET)' "$m37_output/reports/timing_setup.rpt" \
        && grep -q 'slack (MET)' "$m37_output/reports/timing_hold.rpt" \
        && ! grep -q 'slack (VIOLATED)' "$m37_output/reports/timing_setup.rpt" \
        && ! grep -q 'slack (VIOLATED)' "$m37_output/reports/timing_hold.rpt"; then
    m37_timing_status=MET
fi
m37_setup_wns="$(awk '/slack \((MET|VIOLATED)\)/ {value=$NF+0; if (!seen || value < minimum) minimum=value; seen=1} END {if (!seen) exit 1; printf "%.4f", minimum}' "$m37_output/reports/timing_setup.rpt")"
m37_hold_wns="$(awk '/slack \((MET|VIOLATED)\)/ {value=$NF+0; if (!seen || value < minimum) minimum=value; seen=1} END {if (!seen) exit 1; printf "%.4f", minimum}' "$m37_output/reports/timing_hold.rpt")"
m37_cell_count="$(awk '/^Number of cells:/ {print $4; found=1} END {if (!found) exit 1}' "$m37_output/reports/area.rpt")"
m37_total_area="$(awk '/^Total cell area:/ {print $4; found=1} END {if (!found) exit 1}' "$m37_output/reports/area.rpt")"
m37_comb_area="$(awk '/^Combinational area:/ {print $3; found=1} END {if (!found) exit 1}' "$m37_output/reports/area.rpt")"
m37_seq_area="$(awk '/^Noncombinational area:/ {print $3; found=1} END {if (!found) exit 1}' "$m37_output/reports/area.rpt")"

{
    if [[ "$m37_timing_status" == MET ]]; then
        echo "status=PASS_FRESH_STANDALONE_M37_R8_DC_STA_STRICT_ZERO_MULTIPLIER_PENDING_INDEPENDENT_REVIEW"
    else
        echo "status=FAIL_TIMING_STANDALONE_M37_R8_DC_STA_PRESERVED_DO_NOT_CITE"
    fi
    echo "review_required=true"
    echo "headline_admitted=false"
    echo "scope=STANDALONE_M37_T10_CSD_RECONSTRUCTION_ONLY"
    echo "vcs_anchor=m37_r8_receipt_r3"
    echo "vcs_r8_receipt_sha256=363fb61d2838b6379a065dd8eb23b6219441cfb8ed70164766f07d8469e95d97"
    echo "vcs_r8_contract_sha256=91155bebc0ac7526be6082130c4b05ece5e15d977fa05f64b055cf611ddcd214"
    echo "rtl_sha256=ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd"
    echo "old_dc_r1_fail_marker_sha256=9f448e04de0256f3b2f2a741a57810f1836b1ec0707d5b3bd90ec2d00f5c263d"
    echo "old_dc_r1_multiplier_report_sha256=39285d1604024a9e7570e302fb1d4122cb731dbb17b4cbf793324f61e64e46f7"
    echo "old_dc_r1_citable=false"
    echo "clock_period_ns=$m37_period"
    echo "timing_status=$m37_timing_status"
    echo "setup_wns_reported_ns=$m37_setup_wns"
    echo "hold_wns_reported_ns=$m37_hold_wns"
    echo "mapped_cell_count=$m37_cell_count"
    echo "total_cell_area_um2=$m37_total_area"
    echo "combinational_area_um2=$m37_comb_area"
    echo "noncombinational_area_um2=$m37_seq_area"
    echo "precompile_physical_multiplier_hit_count=0"
    echo "postcompile_physical_multiplier_hit_count=0"
    echo "reference_physical_multiplier_hit_count=0"
    echo "mapped_netlist_physical_multiplier_hit_count=0"
    echo "source_structure_proof_from_uses_integer_multiplier_signal=false"
    echo "wire_load_model=ZeroWireload"
    echo "clock_network=IDEAL_UNPROPAGATED"
    echo "macro_db=NONE"
    echo "sram_macro=NONE"
    echo "paper_ppa_ready=false"
    echo "postroute_timing_admitted=false"
    echo "power_energy_admitted=false"
    echo "system_speedup_admitted=false"
    echo "integrated_m31_claim_admitted=false"
    echo "library_slow=$LIB_DB"
    echo "library_fast=$MIN_LIB_DB"
} > "$m37_output/admission.txt"

(
    cd "$m37_output"
    sha256sum dc.log dc.exit_status admission.txt input_sha256.txt \
        reports/check_design_precompile.rpt \
        reports/check_timing_precompile.rpt \
        reports/constraint_contract_precompile.rpt \
        reports/hierarchy_precompile.rpt \
        reports/references_precompile.rpt \
        reports/resources_precompile.rpt \
        reports/check_design_postcompile.rpt \
        reports/check_timing_postcompile.rpt \
        reports/constraint_contract_postcompile.rpt \
        reports/hierarchy_postcompile.rpt \
        reports/references_postcompile.rpt \
        reports/resources_postcompile.rpt \
        reports/m37_source_star_token_ledger.rpt \
        reports/m37_strict_zero_multiplier_link_audit.rpt \
        reports/m37_warning_set.rpt \
        reports/qor.rpt reports/area.rpt reports/clocks.rpt \
        reports/ports.rpt reports/timing_setup.rpt \
        reports/timing_hold.rpt reports/constraint_violators.rpt \
        "netlist/${m37_design}_mapped.v" \
        "netlist/${m37_design}_mapped.sdc" \
        "netlist/${m37_design}.ddc" "netlist/${m37_design}.svf" \
        > dc_output_sha256.txt
    sha256sum --strict -c dc_output_sha256.txt
)
{
    echo "runner_post_dc_audit_status=COMPLETE"
    echo "dc_exit_code=$m37_dc_rc"
    echo "timing_status=$m37_timing_status"
    echo "strict_zero_physical_multiplier=true"
    echo "formality_status=PENDING"
    echo "standalone_only=true"
    echo "review_required=true"
} > "$m37_output/dc_runner_status.txt"
(
    cd "$m37_output"
    sha256sum input_sha256.txt dc_output_sha256.txt dc_runner_status.txt \
        > dc_live_seal.sha256
    sha256sum --strict -c dc_live_seal.sha256
)
echo "M37_DC_LIVE_EVIDENCE_SEALED timing=$m37_timing_status run=$m37_output"
if [[ "$m37_timing_status" != MET ]]; then
    echo "M37 timing is not met; stopping before Formality admission" >&2
    exit 14
fi

if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -x fm_shell >/dev/null || pgrep -x fm_shell_exec >/dev/null; then
    echo "refusing M37 Formality because a Synopsys shell is active" >&2
    exit 15
fi
export MAPPED_NETLIST="$m37_output/netlist/${m37_design}_mapped.v"
export SVF_FILE="$m37_output/netlist/${m37_design}.svf"
set +e
fm_shell -f "$m37_fm_tcl" 2>&1 | tee "$m37_output/formality.log"
m37_fm_rc=${PIPESTATUS[0]}
set -e
echo "$m37_fm_rc" > "$m37_output/formality.exit_status"
if [[ "$m37_fm_rc" -ne 0 ]]; then
    {
        echo "status=FAIL_FORMALITY_TOOL_OR_VERIFY_DO_NOT_CITE"
        echo "formality_exit_code=$m37_fm_rc"
    } > "$m37_output/FAILED_FORMALITY_DO_NOT_CITE.txt"
    sha256sum "$m37_output/formality.log" \
        "$m37_output/formality.exit_status" \
        "$m37_output/dc_live_seal.sha256" \
        "$m37_output/FAILED_FORMALITY_DO_NOT_CITE.txt" \
        > "$m37_output/failed_formality_evidence.sha256"
    exit 16
fi
for m37_formal_file in reports/formality_status.txt \
        reports/formality_unmatched.rpt reports/formality_verify.rpt; do
    if [[ ! -s "$m37_output/$m37_formal_file" ]]; then
        echo "M37 Formality missing evidence: $m37_formal_file" >&2
        exit 17
    fi
done

python3 - "$m37_output/formality.log" \
        "$m37_output/reports/formality_status.txt" \
        "$m37_output/formality_admission.txt" \
        "$m37_output/reports/formality_warning_set.rpt" <<'PY'
import pathlib
import re
import sys

log_path, status_path, admission_path, warning_path = map(pathlib.Path, sys.argv[1:])
log_lines = log_path.read_text(errors="replace").splitlines()
if status_path.read_text().strip() != "PASS":
    raise SystemExit("M37 Formality status is not PASS")
if sum(line.strip() == "Verification SUCCEEDED" for line in log_lines) != 1:
    raise SystemExit("M37 Formality success marker count is not exactly one")
passing = [int(match.group(1)) for line in log_lines for match in [re.match(r"^\s*([0-9]+) Passing compare points\s*$", line)] if match]
if len(passing) != 1 or passing[0] <= 0:
    raise SystemExit("M37 Formality passing compare-point summary is invalid")
failing_rows = []
for line in log_lines:
    match = re.match(r"^Failing \(not equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$", line)
    if match:
        failing_rows.append([int(value) for value in match.group(1).split()])
if len(failing_rows) != 1 or any(failing_rows[0]):
    raise SystemExit("M37 Formality has failing or ambiguous compare points")
unmatched = []
for line in log_lines:
    match = re.match(r"^\s*([0-9]+)\(([0-9]+)\) Unmatched .+$", line)
    if match:
        unmatched.append((int(match.group(1)), int(match.group(2)), line.strip()))
if not unmatched or any(reference or implementation for reference, implementation, _ in unmatched):
    raise SystemExit("M37 Formality has nonzero or absent unmatched-point closure")
errors = [line for line in log_lines if line.startswith(("Error:", "Fatal:"))]
if errors:
    raise SystemExit("M37 Formality log contains Error/Fatal")
warnings = [line for line in log_lines if line.startswith("Warning:")]
warning_path.write_text("\n".join([
    "status=ACTUAL_FORMALITY_WARNING_SET_SEALED_NOT_USED_AS_EQUIVALENCE_PROOF",
    "warning_count={}".format(len(warnings)),
    "actual_warning_sequence_begin",
] + warnings + ["actual_warning_sequence_end"]) + "\n")
admission_path.write_text("\n".join([
    "status=PASS_RTL_TO_FRESH_M37_R8_MAPPED_NETLIST_FORMALITY_PENDING_INDEPENDENT_REVIEW",
    "passing_compare_points={}".format(passing[0]),
    "failing_compare_points=0",
    "unmatched_reference_points=0",
    "unmatched_implementation_points=0",
    "mapped_netlist_is_dc_live_sealed=true",
    "standalone_only=true",
    "headline_admitted=false",
]) + "\n")
print("M37_FORMALITY_AUDIT=PASS passing={} failing=0 unmatched=0".format(passing[0]))
PY

(
    cd "$m37_output"
    sha256sum formality.log formality.exit_status formality_admission.txt \
        reports/formality_status.txt reports/formality_unmatched.rpt \
        reports/formality_verify.rpt reports/formality_warning_set.rpt \
        dc_live_seal.sha256 \
        "$m37_fm_tcl" "$m37_runner" "$m37_snapshot_runner" \
        > formality_evidence.sha256
    sha256sum --strict -c formality_evidence.sha256
    sha256sum dc_live_seal.sha256 formality_evidence.sha256 \
        > formality_live_seal.sha256
    sha256sum --strict -c formality_live_seal.sha256
)
echo "M37_FORMALITY_LIVE_EVIDENCE_SEALED run=$m37_output"

bash "$m37_snapshot_runner" "$m37_output" "$m37_hw_root" "$m37_r8" \
    "$m37_old_dc" m37_r8_dc3p000ns_formality_r1_20260822
echo "M37_FRESH_DC_FORMALITY_COMPLETE period_ns=$m37_period run=$m37_output"
