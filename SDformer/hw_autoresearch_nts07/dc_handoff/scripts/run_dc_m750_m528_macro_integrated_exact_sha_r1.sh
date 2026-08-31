#!/usr/bin/env bash
set -euo pipefail

m750_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m750_hw_root="$(cd "${m750_dc_root}/.." && pwd)"
m750_runner="$(realpath "${BASH_SOURCE[0]}")"
m750_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m750_dc_actual=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m750_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m750_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m750_macro=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db
m750_forbidden_macro_v=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v
m750_filelist=dc_handoff/filelists/date_m750_m528_macro_integrated_dc.f
m750_sdc=dc_handoff/constraints/date_m750_m528_macro_integrated_3ns.sdc
m750_tcl=dc_handoff/scripts/run_dc_m750_m528_macro_integrated.tcl
m750_top=rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv
m750_adapter=rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv
m750_binding=rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json
m750_source_contract=contracts/m750_m528_macro_integrated_dc_source_only_contract_r1_20260828.json
m750_candidate=contracts/m750_m528_macro_integrated_dc_launch_admission_candidate_r1_20260828.json
m750_release=contracts/m750_m528_macro_integrated_dc_launch_release_r1_20260828.json
m750_result_rel=dc_handoff/runs/m750_m528_macro_integrated_dc_3p000ns_r1_20260828
m750_result="${m750_hw_root}/${m750_result_rel}"
m750_attempt="${m750_dc_root}/runs/.m750_m528_macro_integrated_dc_attempt_consumed"
m750_work="${m750_dc_root}/runs/.m750_m528_macro_integrated_dc_work.$$"
m750_quarantine="${m750_result}.failed_or_incomplete.$$.quarantine"
m750_lock="${m750_dc_root}/runs/.m750_m528_macro_integrated_dc_launch_lock"
m750_uid="$(id -u)"
m750_attempt_consumed=0

m750_sha() { sha256sum "$1" | awk '{print $1}'; }
m750_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m750_sha "${path}")" == "${expected}" ]] || {
        echo "M750 identity mismatch: ${path}" >&2
        exit 3
    }
}
m750_verify_file_seal() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"
    base="$(basename "${payload}")"
    [[ -f "${payload}.sha256" && -f "${payload}.sha256.seal.sha256" ]] || exit 3
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}
m750_verify_dir_seal() {
    local evidence_dir=$1
    [[ -d "${evidence_dir}" && -f "${evidence_dir}/SHA256SUMS" && \
       -f "${evidence_dir}/SHA256SUMS.seal.sha256" ]] || exit 3
    (cd "${evidence_dir}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
}
m750_seal_dir() {
    local evidence_dir=$1
    (
        cd "${evidence_dir}"
        find . -type f ! -path './SHA256SUMS' \
            ! -path './SHA256SUMS.seal.sha256' -print0 | sort -z | \
            xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}
m750_cleanup() {
    local rc=$?
    set +e
    if [[ ${rc} -ne 0 && ${m750_attempt_consumed} -eq 1 && -d "${m750_work}" ]]; then
        printf 'status=FAILED_DO_NOT_CITE\nexit_code=%s\nmacro_integrated_ppa=false\npaper_ppa_ready=false\n' \
            "${rc}" >"${m750_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m750_seal_dir "${m750_work}"
        mv "${m750_work}" "${m750_quarantine}"
    fi
    rmdir "${m750_lock}" 2>/dev/null || true
    return "${rc}"
}
trap m750_cleanup EXIT

bash -n "${m750_runner}"
[[ -n "${M750_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m750_sha "${m750_runner}")" == "${M750_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M750 caller must pin the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M750_EXPECTED_DC_LAUNCH_RELEASE_SHA256:-}" ]] || {
    echo "M750 source-only package has no implicit launch authorization" >&2
    exit 3
}
[[ -z "${OUTPUT_DIR:-}${CLOCK_PERIOD_NS:-}${LIB_DB:-}${MIN_LIB_DB:-}${MACRO_DB:-}${OPERATING_CONDITION:-}" ]] || {
    echo "M750 forbids path, library, clock and corner overrides" >&2
    exit 3
}
[[ ! -e "${m750_result}" && ! -e "${m750_work}" && ! -e "${m750_attempt}" ]] || {
    echo "M750 refuses a consumed or colliding result identity" >&2
    exit 5
}

cd "${m750_hw_root}"
m750_expect "${m750_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m750_expect "${m750_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m750_expect "${m750_macro}" cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf
m750_expect "${m750_filelist}" 802424663328c911e5042e2223718773a2a85a8cc92e67edddaa64571b753963
m750_expect "${m750_sdc}" 75537c06e7a77df85b073484ceaf5b8709a30dfd959a22db9b6ee520d5d8d799
m750_expect "${m750_tcl}" 675549087f90107c7d675821a962e64f0ffe8058a045049cec7edd9caadbc715
m750_expect "${m750_top}" 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1
m750_expect "${m750_adapter}" 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783
m750_expect "${m750_binding}" db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983
m750_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
[[ -f "${m750_forbidden_macro_v}" ]] || exit 3
! rg -n '\.v($|[[:space:]])|ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c\.v' \
    "${m750_filelist}" >/dev/null || {
    echo "M750 DC filelist illegally names a behavioral Verilog macro view" >&2
    exit 3
}
python3 - "${m750_adapter}" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = re.sub(r"//.*?$|/\*.*?\*/", "", s, flags=re.M | re.S)
if len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\b", s)) != 1:
    raise SystemExit("adapter must contain one generated macro instantiation template")
if re.search(r"\b(?:reg|logic)\b[^;]*\[[^]]+\]\s*\[[^]]+\]", s):
    raise SystemExit("adapter contains an unpacked register-array fallback")
if "{1'b0, address}" not in s or "slice*128 +: 128" not in s:
    raise SystemExit("adapter address/slice binding changed")
PY

m750_expect "${m750_release}" "${M750_EXPECTED_DC_LAUNCH_RELEASE_SHA256}"
m750_verify_file_seal "${m750_release}"
m750_expect "${m750_source_contract}" \
    "$(jq -er '.source_package.source_contract_sha256' "${m750_release}")"
m750_verify_file_seal "${m750_source_contract}"
m750_expect "${m750_candidate}" \
    "$(jq -er '.source_package.candidate_sha256' "${m750_release}")"
m750_verify_file_seal "${m750_candidate}"

jq -e --arg runner_sha "$(m750_sha "${m750_runner}")" \
       --arg macro_sha "$(m750_sha "${m750_macro}")" \
       --arg result "${m750_result_rel}" \
       '.schema == "m750_m528_macro_integrated_dc_launch_release_v1"
        and .status == "AUTHORIZED_ONE_M750_MACRO_INTEGRATED_DC_ATTEMPT_R1"
        and .launch_now == true
        and .authorization.run_dc == true
        and .authorization.max_attempts == 1
        and .authorization.run_vcs == false
        and .authorization.run_formality == false
        and .authorization.run_pt == false
        and .authorization.run_ptpx == false
        and .authorization.run_remote == false
        and .source_package.runner_sha256 == $runner_sha
        and .source_package.macro_db_sha256 == $macro_sha
        and .unique_attempt.result_path == $result
        and .claim_boundary.paper_ppa_ready == false
        and .claim_boundary.ptpx == false
        and .claim_boundary.saif == false' "${m750_release}" >/dev/null || exit 3
jq -e '.launch_now == false
       and .status == "SOURCE_CANDIDATE_ONLY__M746_PASS_AND_FRESH_HAMMERS_REQUIRED"
       and .authorization.run_dc == false
       and .claim_boundary.source_only == true' "${m750_candidate}" >/dev/null || exit 3

m750_vcs_result="$(jq -er '.m746_prerequisite.vcs_result_path' "${m750_release}")"
m750_vcs_review="$(jq -er '.m746_prerequisite.independent_result_review_path' "${m750_release}")"
[[ "${m750_vcs_result}" == results/m746_m533_m528_dead_write_only_1rw_unit_delay_vcs_r12_20260828 && \
   "${m750_vcs_review}" == reviews/*/review.json ]] || exit 3
m750_verify_dir_seal "${m750_vcs_result}"
m750_expect "${m750_vcs_result}/RUN_COMPLETE.txt" \
    "$(jq -er '.m746_prerequisite.vcs_run_complete_sha256' "${m750_release}")"
m750_expect "${m750_vcs_review}" \
    "$(jq -er '.m746_prerequisite.independent_result_review_sha256' "${m750_release}")"
m750_verify_dir_seal "$(dirname "${m750_vcs_review}")"
jq -e '.verdict | startswith("PASS")' "${m750_vcs_review}" >/dev/null || exit 3

m750_source_review="$(jq -er '.fresh_hammers.source_candidate_review_path' "${m750_release}")"
m750_final_review="$(jq -er '.fresh_hammers.final_release_review_path' "${m750_release}")"
for m750_review in "${m750_source_review}" "${m750_final_review}"; do
    [[ "${m750_review}" == reviews/*/review.json ]] || exit 3
    m750_expect "${m750_review}" \
        "$(jq -er --arg p "${m750_review}" '.fresh_hammers.review_sha256_by_path[$p]' "${m750_release}")"
    m750_verify_dir_seal "$(dirname "${m750_review}")"
    jq -e '.verdict | startswith("PASS")' "${m750_review}" >/dev/null || exit 3
done

[[ -x "${m750_dc}" && -x "${m750_dc_actual}" ]] || exit 4
if ps -u "${m750_uid}" -o args= | rg -q \
    '(^|/)(dc_shell|dc_shell-t)([[:space:]]|$)|common_shell_exec -shell dc_shell'; then
    echo "M750 refuses a same-UID DC collision" >&2
    exit 4
fi
mkdir "${m750_lock}" || exit 4

for m750_sample in 1 2 3; do
    m750_mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    m750_swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    m750_commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    m750_committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    m750_headroom=$((m750_commit_limit - m750_committed))
    [[ ${m750_mem_available} -ge 134217728 && ${m750_swap_free} -ge 33554432 && \
       ${m750_headroom} -ge 67108864 ]] || {
        echo "M750 resource preflight failed at sample ${m750_sample}" >&2
        exit 4
    }
    sleep 2
done
if ps -u "${m750_uid}" -o args= | rg -q \
    '(^|/)(dc_shell|dc_shell-t)([[:space:]]|$)|common_shell_exec -shell dc_shell'; then
    echo "M750 same-UID DC appeared during preflight" >&2
    exit 4
fi

mkdir "${m750_attempt}"
m750_attempt_consumed=1
mkdir "${m750_work}"
{
    echo status=M750_DC_ATTEMPT_CONSUMED
    echo paper_ppa_ready=false
    echo macro_integrated_dc_setup_area_candidate=true
    echo ptpx=false
    echo saif=false
    echo formality=false
    echo physical_routing=false
    echo macro_cell=TS1N28HPCPHVTB128X128M4S
    echo expected_macro_count=9
    echo clock_period_ns=3.000
    echo setup_corner=ssg0p9v125c
    echo hold_stdcell_corner=ffg1p05vm40c
    echo hold_macro_corner=slow_db_only__not_signoff
    echo wireload=ZeroWireload
    sha256sum "${m750_runner}" "${m750_release}" "${m750_source_contract}" \
        "${m750_candidate}" "${m750_slow}" "${m750_fast}" "${m750_macro}" \
        "${m750_filelist}" "${m750_sdc}" "${m750_tcl}" "${m750_top}" \
        "${m750_adapter}" "${m750_binding}" "${m750_vcs_result}/RUN_COMPLETE.txt" \
        "${m750_vcs_review}" "${m750_source_review}" "${m750_final_review}"
} >"${m750_work}/admission.txt"

export DESIGN_NAME=m528_dead_write_only_1rw_product_capture_island_r2
export HW_ROOT="${m750_hw_root}"
export RTL_FILELIST="${m750_hw_root}/${m750_filelist}"
export SDC_FILE="${m750_hw_root}/${m750_sdc}"
export OUTPUT_DIR="${m750_work}"
export CLOCK_PERIOD_NS=3.000
export LIB_DB="${m750_slow}"
export MIN_LIB_DB="${m750_fast}"
export MACRO_DB="${m750_macro}"
export OPERATING_CONDITION=ssg0p9v125c

"${m750_dc}" -f "${m750_hw_root}/${m750_tcl}" \
    2>&1 | tee "${m750_work}/dc.log"

if rg -ni '(^|[^A-Za-z])(Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+)' \
        "${m750_work}/dc.log" >/dev/null; then
    echo "M750 DC log contains an error or unresolved reference" >&2
    exit 9
fi
for m750_required in \
    reports/macro_binding_audit.txt \
    reports/check_design_precompile.rpt reports/check_design_postcompile.rpt \
    reports/check_timing_precompile.rpt reports/check_timing_postcompile.rpt \
    reports/resources_precompile.rpt reports/resources_postcompile.rpt \
    reports/references_precompile.rpt reports/references_postcompile.rpt \
    reports/qor.rpt reports/area.rpt reports/timing_setup.rpt \
    reports/timing_hold.rpt reports/constraint_violators.rpt \
    netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v \
    netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.sdc \
    netlist/m528_dead_write_only_1rw_product_capture_island_r2.ddc; do
    [[ -s "${m750_work}/${m750_required}" ]] || {
        echo "M750 missing evidence: ${m750_required}" >&2
        exit 6
    }
done
grep -qx 'status=M750_MACRO_BINDING_STRUCTURAL_PASS' \
    "${m750_work}/reports/macro_binding_audit.txt"
grep -qx 'macro_count_pre=9' "${m750_work}/reports/macro_binding_audit.txt"
grep -qx 'macro_count_post=9' "${m750_work}/reports/macro_binding_audit.txt"
grep -qx 'behavioral_macro_verilog_read_by_dc=false' \
    "${m750_work}/reports/macro_binding_audit.txt"
m750_netlist="${m750_work}/netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v"
[[ "$(rg -o 'TS1N28HPCPHVTB128X128M4S' "${m750_netlist}" | wc -l)" -eq 9 ]] || {
    echo "M750 mapped netlist does not contain exactly nine macro references" >&2
    exit 9
}
! rg -ni 'unresolved|black[ -]?box|register.array fallback|inferred.*parent|parent.*inferred' \
    "${m750_work}/reports" "${m750_work}/dc.log" >/dev/null || {
    echo "M750 unresolved or inferred-parent marker found" >&2
    exit 9
}
for m750_timing in reports/timing_setup.rpt reports/timing_hold.rpt; do
    rg -q 'slack \(MET\)' "${m750_work}/${m750_timing}" || {
        echo "M750 timing report lacks a MET path: ${m750_timing}" >&2
        exit 9
    }
    ! rg -q 'slack \(VIOLATED\)' "${m750_work}/${m750_timing}" || {
        echo "M750 timing violation: ${m750_timing}" >&2
        exit 9
    }
done
rg -q 'Total cell area:[[:space:]]*[1-9][0-9]*(\.[0-9]+)?' \
    "${m750_work}/reports/area.rpt" || {
    echo "M750 area report lacks a positive total cell area" >&2
    exit 9
}

python3 - "${m750_work}" <<'PY'
import json, pathlib, re, sys
r = pathlib.Path(sys.argv[1])
area = (r / "reports/area.rpt").read_text(errors="replace")
setup = (r / "reports/timing_setup.rpt").read_text(errors="replace")
hold = (r / "reports/timing_hold.rpt").read_text(errors="replace")
def slack(text):
    vals = [float(v) for v in re.findall(r"slack \(MET\)\s+([-+]?\d+(?:\.\d+)?)", text)]
    if not vals:
        raise SystemExit("missing MET slack")
    return min(vals)
m = re.search(r"Total cell area:\s*([0-9.]+)", area)
if not m or float(m.group(1)) <= 0:
    raise SystemExit("invalid area")
receipt = {
    "schema": "m750_m528_macro_integrated_dc_receipt_v1",
    "status": "PASS_DC_SETUP_HOLD_AREA_AND_NINE_MACRO_STRUCTURE__NOT_PAPER_PPA_READY",
    "macro_cell": "TS1N28HPCPHVTB128X128M4S",
    "macro_count": 9,
    "clock_period_ns": 3.0,
    "setup_slack_min_reported_ns": slack(setup),
    "hold_slack_min_reported_ns": slack(hold),
    "total_cell_area_um2_dc_reported": float(m.group(1)),
    "claim_boundary": {
        "macro_integrated_dc_setup_area": True,
        "macro_integrated_ppa": False,
        "paper_ppa_ready": False,
        "physical_routing": False,
        "extracted_parasitics": False,
        "macro_fast_hold_view": False,
        "saif": False,
        "ptpx": False,
        "system_speedup": False,
    },
}
(r / "m750_dc_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
printf 'PASS M750 macro-integrated DC setup/hold/area; macros=9; paper_ppa_ready=false\n' \
    >"${m750_work}/RUN_COMPLETE.txt"
m750_seal_dir "${m750_work}"
mv "${m750_work}" "${m750_result}"
m750_attempt_consumed=0
rmdir "${m750_attempt}"
trap - EXIT
rmdir "${m750_lock}"
echo "M750 DC completed at ${m750_result}"
