#!/usr/bin/env bash
set -euo pipefail

m756_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m756_hw_root="$(cd "${m756_dc_root}/.." && pwd)"
m756_runner="$(realpath "${BASH_SOURCE[0]}")"
m756_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m756_dc_actual=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m756_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m756_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m756_macro=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db
m756_forbidden_macro_v=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v
m756_filelist=dc_handoff/filelists/date_m750_m528_macro_integrated_dc.f
m756_sdc=dc_handoff/constraints/date_m750_m528_macro_integrated_3ns.sdc
m756_tcl=dc_handoff/scripts/run_dc_m750_m528_macro_integrated.tcl
m756_top=rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv
m756_adapter=rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv
m756_binding=rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json
m756_source_contract=contracts/m756_m528_macro_integrated_dc_source_only_contract_r2_20260828.json
m756_candidate=contracts/m756_m528_macro_integrated_dc_launch_admission_candidate_r2_20260828.json
m756_release=contracts/m756_m528_macro_integrated_dc_launch_release_r2_20260828.json
m756_final_review=reviews/m756_m528_macro_integrated_dc_final_launch_release_hammer_r1_20260828/review.json
m756_m750_audit=reviews/m756_m750_macro_dc_hash_cycle_self_audit_r1_20260828/review.json
m756_result_rel=dc_handoff/runs/m756_m528_macro_integrated_dc_3p000ns_r2_20260828
m756_result="${m756_hw_root}/${m756_result_rel}"
m756_attempt="${m756_dc_root}/runs/.m756_m528_macro_integrated_dc_attempt_consumed"
m756_work="${m756_dc_root}/runs/.m756_m528_macro_integrated_dc_work.$$"
m756_quarantine="${m756_result}.failed_or_incomplete.$$.quarantine"
m756_lock="${m756_dc_root}/runs/.m756_m528_macro_integrated_dc_launch_lock"
m756_uid="$(id -u)"
m756_attempt_consumed=0

m756_sha() { sha256sum "$1" | awk '{print $1}'; }
m756_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m756_sha "${path}")" == "${expected}" ]] || {
        echo "M756 identity mismatch: ${path}" >&2
        exit 3
    }
}
m756_verify_file_seal() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"
    base="$(basename "${payload}")"
    [[ -f "${payload}.sha256" && -f "${payload}.sha256.seal.sha256" ]] || exit 3
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}
m756_verify_dir_seal() {
    local evidence_dir=$1
    [[ -d "${evidence_dir}" && -f "${evidence_dir}/SHA256SUMS" && \
       -f "${evidence_dir}/SHA256SUMS.seal.sha256" ]] || exit 3
    (cd "${evidence_dir}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
}
m756_seal_dir() {
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
m756_cleanup() {
    local rc=$?
    set +e
    if [[ ${rc} -ne 0 && ${m756_attempt_consumed} -eq 1 && -d "${m756_work}" ]]; then
        printf 'status=FAILED_DO_NOT_CITE\nexit_code=%s\nmacro_integrated_ppa=false\npaper_ppa_ready=false\n' \
            "${rc}" >"${m756_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m756_seal_dir "${m756_work}"
        mv "${m756_work}" "${m756_quarantine}"
    fi
    rmdir "${m756_lock}" 2>/dev/null || true
    return "${rc}"
}
trap m756_cleanup EXIT

bash -n "${m756_runner}"
[[ -n "${M756_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m756_sha "${m756_runner}")" == "${M756_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M756 caller must pin the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M756_EXPECTED_DC_LAUNCH_RELEASE_SHA256:-}" ]] || {
    echo "M756 source-only package has no implicit launch authorization" >&2
    exit 3
}
[[ -n "${M756_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256:-}" ]] || {
    echo "M756 caller must independently pin the final-release review payload SHA" >&2
    exit 3
}
[[ -z "${OUTPUT_DIR:-}${CLOCK_PERIOD_NS:-}${LIB_DB:-}${MIN_LIB_DB:-}${MACRO_DB:-}${OPERATING_CONDITION:-}" ]] || {
    echo "M756 forbids path, library, clock and corner overrides" >&2
    exit 3
}
[[ ! -e "${m756_result}" && ! -e "${m756_work}" && ! -e "${m756_attempt}" ]] || {
    echo "M756 refuses a consumed or colliding result identity" >&2
    exit 5
}

cd "${m756_hw_root}"
m756_expect "${m756_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m756_expect "${m756_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m756_expect "${m756_macro}" cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf
m756_expect "${m756_filelist}" 802424663328c911e5042e2223718773a2a85a8cc92e67edddaa64571b753963
m756_expect "${m756_sdc}" 75537c06e7a77df85b073484ceaf5b8709a30dfd959a22db9b6ee520d5d8d799
m756_expect "${m756_tcl}" 675549087f90107c7d675821a962e64f0ffe8058a045049cec7edd9caadbc715
m756_expect "${m756_top}" 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1
m756_expect "${m756_adapter}" 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783
m756_expect "${m756_binding}" db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983
m756_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
m756_expect "${m756_m750_audit}" 2da8f9e12ae1a76d25cd57ba28115f8247d5f8a30e6abeb88d437f9106d1a54c
m756_verify_dir_seal "$(dirname "${m756_m750_audit}")"
jq -e '.status == "NO_GO_M750_RELEASE_IDENTITY_HAS_FUTURE_FINAL_REVIEW_HASH_CYCLE__ADDITIVE_M756_REQUIRED"
       and .verdict == "NO_GO"
       and .severity_counts.p0 == 1
       and .finding.finite_acyclic_authoring_order_exists == false
       and .immutability.m750_release_creation_forbidden == true
       and .immutability.only_additive_m756_r2_allowed == true' \
    "${m756_m750_audit}" >/dev/null || exit 3
[[ -f "${m756_forbidden_macro_v}" ]] || exit 3
! rg -n '\.v($|[[:space:]])|ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c\.v' \
    "${m756_filelist}" >/dev/null || {
    echo "M756 DC filelist illegally names a behavioral Verilog macro view" >&2
    exit 3
}
python3 - "${m756_adapter}" <<'PY'
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

m756_expect "${m756_release}" "${M756_EXPECTED_DC_LAUNCH_RELEASE_SHA256}"
m756_verify_file_seal "${m756_release}"
m756_expect "${m756_source_contract}" \
    "$(jq -er '.source_package.source_contract_sha256' "${m756_release}")"
m756_verify_file_seal "${m756_source_contract}"
m756_expect "${m756_candidate}" \
    "$(jq -er '.source_package.candidate_sha256' "${m756_release}")"
m756_verify_file_seal "${m756_candidate}"

jq -e --arg runner_sha "$(m756_sha "${m756_runner}")" \
       --arg macro_sha "$(m756_sha "${m756_macro}")" \
       --arg result "${m756_result_rel}" \
       '.schema == "m756_m528_macro_integrated_dc_launch_release_v2"
        and .status == "AUTHORIZED_ONE_M756_R2_MACRO_INTEGRATED_DC_ATTEMPT"
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
        and .release_chain.final_review_sha_must_be_caller_pinned == true
        and .release_chain.final_review_sha_embedded_in_release == false
        and (.fresh_hammers | has("final_release_review_sha256") | not)
        and (.fresh_hammers | has("review_sha256_by_path") | not)
        and .claim_boundary.paper_ppa_ready == false
        and .claim_boundary.ptpx == false
        and .claim_boundary.saif == false' "${m756_release}" >/dev/null || exit 3
jq -e '.schema == "m756_m528_macro_integrated_dc_launch_admission_candidate_v2"
       and .launch_now == false
       and .status == "SOURCE_CANDIDATE_ONLY__ACYCLIC_RELEASE_CHAIN_AND_FRESH_HAMMERS_REQUIRED"
       and .authorization.run_dc == false
       and .claim_boundary.source_only == true' "${m756_candidate}" >/dev/null || exit 3

m756_vcs_result="$(jq -er '.m746_prerequisite.vcs_result_path' "${m756_release}")"
m756_vcs_review="$(jq -er '.m746_prerequisite.independent_result_review_path' "${m756_release}")"
[[ "${m756_vcs_result}" == results/m746_m533_m528_dead_write_only_1rw_unit_delay_vcs_r12_20260828 && \
   "${m756_vcs_review}" == reviews/*/review.json ]] || exit 3
m756_verify_dir_seal "${m756_vcs_result}"
m756_expect "${m756_vcs_result}/RUN_COMPLETE.txt" \
    "$(jq -er '.m746_prerequisite.vcs_run_complete_sha256' "${m756_release}")"
m756_expect "${m756_vcs_review}" \
    "$(jq -er '.m746_prerequisite.independent_result_review_sha256' "${m756_release}")"
m756_verify_dir_seal "$(dirname "${m756_vcs_review}")"
jq -e '.verdict | startswith("PASS")' "${m756_vcs_review}" >/dev/null || exit 3

m756_source_review="$(jq -er '.fresh_hammers.source_candidate_review_path' "${m756_release}")"
[[ "${m756_source_review}" == reviews/*/review.json ]] || exit 3
m756_expect "${m756_source_review}" \
    "$(jq -er '.fresh_hammers.source_candidate_review_sha256' "${m756_release}")"
m756_verify_dir_seal "$(dirname "${m756_source_review}")"
jq -e '.verdict == "PASS" and .score_100 == 100
       and .severity_counts.p0 == 0 and .severity_counts.p1 == 0
       and .severity_counts.p2 == 0' "${m756_source_review}" >/dev/null || exit 3

# The final review is deliberately not hashed by the release.  The immutable
# runner fixes its path, the caller independently pins its payload SHA, and the
# review binds the already-authored release SHA.  This is the acyclic edge that
# M750 lacked: release -> final review -> caller pin.
m756_expect "${m756_final_review}" "${M756_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256}"
m756_verify_dir_seal "$(dirname "${m756_final_review}")"
python3 - "${m756_final_review}" "$(m756_sha "${m756_release}")" \
    "$(m756_sha "${m756_runner}")" "$(m756_sha "${m756_candidate}")" \
    "$(m756_sha "${m756_source_review}")" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h:
    r = json.load(h)
def req(c, m):
    if not c:
        raise SystemExit(m)
req(r.get("schema") == "m756_m528_macro_integrated_dc_final_launch_release_hammer_v1", "final schema")
req(r.get("status") == "PASS_M756_M528_MACRO_INTEGRATED_DC_FINAL_LAUNCH_RELEASE_HAMMER", "final status")
req(r.get("verdict") == "PASS" and r.get("score_100") == 100, "final score")
s = r.get("severity_counts", {})
req([s.get(k) for k in ("p0", "p1", "p2")] == [0, 0, 0], "final findings")
i = r.get("identity", {})
req(i.get("final_release_sha256") == sys.argv[2], "final release binding")
req(i.get("runner_sha256") == sys.argv[3], "final runner binding")
req(i.get("candidate_sha256") == sys.argv[4], "final candidate binding")
req(i.get("source_candidate_review_sha256") == sys.argv[5], "final source-review binding")
d = r.get("decision", {})
req(d.get("exactly_one_dc_attempt_authorized_now") is True, "final dc authorization")
req(d.get("all_other_runs_authorized") is False, "final closed authorization")
PY

[[ -x "${m756_dc}" && -x "${m756_dc_actual}" ]] || exit 4
if ps -u "${m756_uid}" -o args= | rg -q \
    '(^|/)(dc_shell|dc_shell-t)([[:space:]]|$)|common_shell_exec -shell dc_shell'; then
    echo "M756 refuses a same-UID DC collision" >&2
    exit 4
fi
mkdir "${m756_lock}" || exit 4

for m756_sample in 1 2 3; do
    m756_mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    m756_swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    m756_commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    m756_committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    m756_headroom=$((m756_commit_limit - m756_committed))
    [[ ${m756_mem_available} -ge 134217728 && ${m756_swap_free} -ge 33554432 && \
       ${m756_headroom} -ge 67108864 ]] || {
        echo "M756 resource preflight failed at sample ${m756_sample}" >&2
        exit 4
    }
    sleep 2
done
if ps -u "${m756_uid}" -o args= | rg -q \
    '(^|/)(dc_shell|dc_shell-t)([[:space:]]|$)|common_shell_exec -shell dc_shell'; then
    echo "M756 same-UID DC appeared during preflight" >&2
    exit 4
fi

mkdir "${m756_attempt}"
m756_attempt_consumed=1
mkdir "${m756_work}"
{
    echo status=M756_R2_DC_ATTEMPT_CONSUMED
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
    sha256sum "${m756_runner}" "${m756_release}" "${m756_source_contract}" \
        "${m756_candidate}" "${m756_slow}" "${m756_fast}" "${m756_macro}" \
        "${m756_filelist}" "${m756_sdc}" "${m756_tcl}" "${m756_top}" \
        "${m756_adapter}" "${m756_binding}" "${m756_vcs_result}/RUN_COMPLETE.txt" \
        "${m756_vcs_review}" "${m756_source_review}" "${m756_final_review}"
} >"${m756_work}/admission.txt"

export DESIGN_NAME=m528_dead_write_only_1rw_product_capture_island_r2
export HW_ROOT="${m756_hw_root}"
export RTL_FILELIST="${m756_hw_root}/${m756_filelist}"
export SDC_FILE="${m756_hw_root}/${m756_sdc}"
export OUTPUT_DIR="${m756_work}"
export CLOCK_PERIOD_NS=3.000
export LIB_DB="${m756_slow}"
export MIN_LIB_DB="${m756_fast}"
export MACRO_DB="${m756_macro}"
export OPERATING_CONDITION=ssg0p9v125c

"${m756_dc}" -f "${m756_hw_root}/${m756_tcl}" \
    2>&1 | tee "${m756_work}/dc.log"

if rg -ni '(^|[^A-Za-z])(Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+)' \
        "${m756_work}/dc.log" >/dev/null; then
    echo "M756 DC log contains an error or unresolved reference" >&2
    exit 9
fi
for m756_required in \
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
    [[ -s "${m756_work}/${m756_required}" ]] || {
        echo "M756 missing evidence: ${m756_required}" >&2
        exit 6
    }
done
grep -qx 'status=M750_MACRO_BINDING_STRUCTURAL_PASS' \
    "${m756_work}/reports/macro_binding_audit.txt"
grep -qx 'macro_count_pre=9' "${m756_work}/reports/macro_binding_audit.txt"
grep -qx 'macro_count_post=9' "${m756_work}/reports/macro_binding_audit.txt"
grep -qx 'behavioral_macro_verilog_read_by_dc=false' \
    "${m756_work}/reports/macro_binding_audit.txt"
m756_netlist="${m756_work}/netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v"
[[ "$(rg -o 'TS1N28HPCPHVTB128X128M4S' "${m756_netlist}" | wc -l)" -eq 9 ]] || {
    echo "M756 mapped netlist does not contain exactly nine macro references" >&2
    exit 9
}
! rg -ni 'unresolved|black[ -]?box|register.array fallback|inferred.*parent|parent.*inferred' \
    "${m756_work}/reports" "${m756_work}/dc.log" >/dev/null || {
    echo "M756 unresolved or inferred-parent marker found" >&2
    exit 9
}
for m756_timing in reports/timing_setup.rpt reports/timing_hold.rpt; do
    rg -q 'slack \(MET\)' "${m756_work}/${m756_timing}" || {
        echo "M756 timing report lacks a MET path: ${m756_timing}" >&2
        exit 9
    }
    ! rg -q 'slack \(VIOLATED\)' "${m756_work}/${m756_timing}" || {
        echo "M756 timing violation: ${m756_timing}" >&2
        exit 9
    }
done
rg -q 'Total cell area:[[:space:]]*[1-9][0-9]*(\.[0-9]+)?' \
    "${m756_work}/reports/area.rpt" || {
    echo "M756 area report lacks a positive total cell area" >&2
    exit 9
}

python3 - "${m756_work}" <<'PY'
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
    "schema": "m756_m528_macro_integrated_dc_receipt_v1",
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
(r / "m756_dc_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
printf 'PASS M756 macro-integrated DC setup/hold/area; macros=9; paper_ppa_ready=false\n' \
    >"${m756_work}/RUN_COMPLETE.txt"
m756_seal_dir "${m756_work}"
mv "${m756_work}" "${m756_result}"
m756_attempt_consumed=0
rmdir "${m756_attempt}"
trap - EXIT
rmdir "${m756_lock}"
echo "M756 DC completed at ${m756_result}"
