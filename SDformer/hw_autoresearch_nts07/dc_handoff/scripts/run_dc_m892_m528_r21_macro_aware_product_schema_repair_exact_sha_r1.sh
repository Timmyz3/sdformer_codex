#!/usr/bin/env bash
set -euo pipefail

m892_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m892_hw_root="$(cd "${m892_dc_root}/.." && pwd)"
m892_runner="$(realpath "${BASH_SOURCE[0]}")"
m892_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m892_dc_wrapper=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
m892_dc_actual=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m892_lmutil=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
m892_license=/opt/synopsys/Synopsys.dat
m892_std_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m892_std_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m892_macro_root=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821
m892_macro_manifest="${m892_macro_root}/SHA256SUMS"
m892_macro_slow="${m892_macro_root}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
m892_macro_fast="${m892_macro_root}/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"
m892_forbidden_macro_v="${m892_macro_root}/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
m892_filelist=dc_handoff/filelists/date_m884_m528_r21_macro_aware_product_dc.f
m892_sdc=dc_handoff/constraints/date_m884_m528_r21_macro_aware_product_3ns.sdc
m892_tcl=dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl
m892_top=rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv
m892_sva=verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv
m892_adapter=rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv
m892_binding=rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json
m892_contract=contracts/m892_m528_r21_macro_aware_product_dc_schema_repair_source_only_contract_r1_20260829.json
m892_candidate=contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_candidate_source_only_r1_20260829.json
m892_release=contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_release_r1_20260829.json
m892_final_review=reviews/m892_m528_r21_macro_aware_product_schema_repair_dc_final_launch_hammer_r1_20260829/review.json
m892_m881=reviews/m881_c1_m528_m533_physical_evidence_first_principles_audit_r1_20260829
m892_m879=reviews/m879_m863_c1_r21_unit_delay_vcs_result_hammer_r1_20260829
m892_m863=results/m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_r21_20260829
m892_m623=reviews/m623_m617_m597_m593_parent_scratch_energy_r5_result_hammer_r1_20260828
m892_m885=reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829
m892_m891=reviews/m891_m884_macro_dc_release_author_preflight_failure_audit_r1_20260829
m892_result_rel=dc_handoff/runs/m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829
m892_result="${m892_hw_root}/${m892_result_rel}"
m892_attempt="${m892_dc_root}/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed"
m892_work="${m892_dc_root}/runs/.m892_m528_r21_macro_aware_product_dc_work.$$"
m892_quarantine="${m892_result}.failed_or_incomplete.$$.quarantine"
m892_lock="${m892_dc_root}/runs/.m892_m528_r21_macro_aware_product_dc_launch_lock"
m892_attempt_consumed=0
m892_completed=0

m892_sha() { sha256sum "$1" | awk '{print $1}'; }
m892_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && ! -L "${path}" && "$(m892_sha "${path}")" == "${expected}" ]] || {
        echo "M892 identity mismatch: ${path}" >&2
        exit 3
    }
}
m892_expect_linkable_tool() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m892_sha "${path}")" == "${expected}" ]] || {
        echo "M892 tool identity mismatch: ${path}" >&2
        exit 3
    }
}
m892_strict_json() {
    /usr/libexec/platform-python3.6 - "$1" <<'PY'
import json, sys
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: %s" % key)
        out[key] = value
    return out
def nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)
with open(sys.argv[1], "rb") as handle:
    json.loads(handle.read().decode("utf-8"), object_pairs_hook=unique,
               parse_constant=nonfinite)
PY
}
m892_closed_keys() {
    local file=$1 expression=$2 expected=$3 actual
    actual="$(jq -er "${expression} | keys[]" "${file}" | LC_ALL=C sort | paste -sd, -)"
    [[ "${actual}" == "${expected}" ]] || {
        echo "M892 unknown or missing JSON key at ${expression}: ${actual}" >&2
        exit 3
    }
}
m892_verify_file_seal() {
    local payload=$1 dir base
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    [[ -f "${payload}.sha256" && -f "${payload}.sha256.seal.sha256" ]] || exit 3
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}
m892_verify_dir_seal() {
    local evidence=$1
    [[ -d "${evidence}" && ! -L "${evidence}" && \
       -f "${evidence}/SHA256SUMS" && -f "${evidence}/SHA256SUMS.seal.sha256" ]] || exit 3
    (cd "${evidence}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
}
m892_seal_dir() {
    local evidence=$1
    (
        cd "${evidence}"
        find . -type l -print -quit | grep -q . && exit 1
        find . -type f ! -path './SHA256SUMS' ! -path './SHA256SUMS.seal.sha256' \
            -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}
m892_cleanup() {
    local rc=$?
    set +e
    if [[ "${rc}" -ne 0 && "${m892_attempt_consumed}" -eq 1 && \
          -d "${m892_work}" && ! -e "${m892_quarantine}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nfair_K_zero_bit=false\npaper_ppa_ready=false\n' \
            "${rc}" >"${m892_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m892_seal_dir "${m892_work}" || true
        mv -T -- "${m892_work}" "${m892_quarantine}" || true
    fi
    rmdir "${m892_lock}" 2>/dev/null || true
    return "${rc}"
}
trap m892_cleanup EXIT INT TERM

bash -n "${m892_runner}"
[[ -n "${M892_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m892_sha "${m892_runner}")" == "${M892_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M892 caller must pin independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M892_EXPECTED_DC_ADMISSION_SHA256:-}" ]] || {
    echo "M892 source-only package has no implicit launch authorization" >&2
    exit 3
}
[[ -z "${OUTPUT_DIR:-}${CLOCK_PERIOD_NS:-}${LIB_DB:-}${MIN_LIB_DB:-}${MACRO_DB:-}${OPERATING_CONDITION:-}" ]] || {
    echo "M892 forbids path, library, clock and corner overrides" >&2
    exit 3
}
[[ ! -e "${m892_result}" && ! -e "${m892_attempt}" && ! -e "${m892_work}" && \
   ! -e "${m892_quarantine}" ]] || {
    echo "M892 refuses consumed or colliding result identity" >&2
    exit 5
}

m892_admission=${m892_release}
m892_expected_schema=m892_m528_r21_macro_aware_product_dc_launch_release_v1
m892_expected_status=AUTHORIZED_ONE_M892_M528_R21_MACRO_AWARE_PRODUCT_DC_ATTEMPT
m892_expected_launch=true
m892_expected_run_dc=true
m892_no_eda=0
m892_production_schema_selftest=0
if [[ -n "${M892_NO_EDA_FULL_PATH_SELFTEST:-}" ]]; then
    [[ "${M892_NO_EDA_FULL_PATH_SELFTEST}" == 1 && \
       -n "${M892_NO_EDA_SELFTEST_ROOT:-}" && \
       "${M892_NO_EDA_SELFTEST_ROOT}" == /* && \
       -d "${M892_NO_EDA_SELFTEST_ROOT}" ]] || exit 87
    m892_admission=${m892_candidate}
    m892_expected_schema=m892_m528_r21_macro_aware_product_dc_launch_candidate_source_only_v1
    m892_expected_status=READY_FOR_FRESH_M892_SOURCE_HAMMER__NO_EDA_AUTHORIZED
    m892_expected_launch=false
    m892_expected_run_dc=false
    m892_no_eda=1
    if [[ -n "${M892_NO_EDA_PRODUCTION_SCHEMA_SELFTEST:-}" ]]; then
        [[ "${M892_NO_EDA_PRODUCTION_SCHEMA_SELFTEST}" == 1 ]] || exit 87
        m892_production_schema_selftest=1
    fi
fi

cd "${m892_hw_root}"
m892_expect "${m892_admission}" "${M892_EXPECTED_DC_ADMISSION_SHA256}"
m892_verify_file_seal "${m892_admission}"
m892_strict_json "${m892_admission}"
m892_closed_keys "${m892_admission}" '.' \
    'authorization,claim_boundary,date,docs359_sha256,fairness,frozen_authorities,future_release_chain,identity,launch_now,prospective_attempt,schema,status'
m892_closed_keys "${m892_admission}" '.authorization' \
    'max_attempts,run_dc,run_formality,run_pt,run_ptpx,run_remote,run_saif,run_vcs'
m892_closed_keys "${m892_admission}" '.claim_boundary' \
    'candidate_only,energy,fair_K_zero_bit,headline,hold_signoff,macro_linked_dc_result,paper_ppa_ready,physical_route,power,ppa,speedup,system,system_speedup,throughput_per_mm2'
jq -e --arg schema "${m892_expected_schema}" \
       --arg status "${m892_expected_status}" \
       --argjson launch "${m892_expected_launch}" \
       --argjson run_dc "${m892_expected_run_dc}" \
       '.schema == $schema and .status == $status and .launch_now == $launch
        and .authorization.run_dc == $run_dc
        and .authorization.max_attempts == (if $run_dc then 1 else 0 end)
        and .authorization.run_vcs == false
        and .authorization.run_formality == false
        and .authorization.run_pt == false
        and .authorization.run_ptpx == false
        and .authorization.run_saif == false
        and .authorization.run_remote == false
        and .fairness.fair_K_zero_bit == false
        and .fairness.zero_rtl_baseline_present == false
        and .fairness.bit_rtl_baseline_present == false
        and .claim_boundary.fair_K_zero_bit == false
        and .claim_boundary.throughput_per_mm2 == false
        and .claim_boundary.speedup == false
        and .claim_boundary.system_speedup == false
        and .claim_boundary.paper_ppa_ready == false' \
    "${m892_admission}" >/dev/null || exit 3

m892_contract_sha="$(jq -er '.identity.source_contract_sha256' "${m892_admission}")"
m892_expect "${m892_contract}" "${m892_contract_sha}"
m892_verify_file_seal "${m892_contract}"
m892_strict_json "${m892_contract}"
m892_closed_keys "${m892_contract}" '.' \
    'authorization,claim_boundary,date,docs359_sha256,exact_files,fairness,foundry_views,frozen_authorities,future_release_chain,physical_point,schema,status,tool_identity'
m892_closed_keys "${m892_contract}" '.authorization' \
    'author_ran_eda,run_dc_now,run_formality_now,run_pt_now,run_ptpx_now,run_remote_now,run_saif_now,run_vcs_now'
jq -e '.schema == "m892_m528_r21_macro_aware_product_dc_source_only_contract_v1"
       and .status == "SOURCE_ONLY_M892_M528_R21_MACRO_AWARE_PRODUCT_DC__FRESH_HAMMER_REQUIRED__NO_EDA_AUTHORIZED"
       and .authorization == {"author_ran_eda":false,"run_dc_now":false,
            "run_formality_now":false,"run_pt_now":false,"run_ptpx_now":false,
            "run_remote_now":false,"run_saif_now":false,"run_vcs_now":false}
       and .fairness.fair_K_zero_bit == false
       and .claim_boundary.throughput_per_mm2 == false
       and .claim_boundary.speedup == false
       and .claim_boundary.system_speedup == false' \
    "${m892_contract}" >/dev/null || exit 3

m892_expected_paths=(
    dc_handoff/scripts/run_dc_m892_m528_r21_macro_aware_product_schema_repair_exact_sha_r1.sh
    dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl
    dc_handoff/filelists/date_m884_m528_r21_macro_aware_product_dc.f
    dc_handoff/constraints/date_m884_m528_r21_macro_aware_product_3ns.sdc
    rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv
    verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv
    rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv
    rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json
    verif_m528_dw1rw/test_m892_m528_r21_macro_dc_schema_repair_source_closure.py
    docs/359_DATE终局冻结_20260813.md
)
m892_actual_paths="$(jq -r '.exact_files | keys[]' "${m892_contract}" | LC_ALL=C sort | paste -sd, -)"
m892_expected_paths_csv="$(printf '%s\n' "${m892_expected_paths[@]}" | LC_ALL=C sort | paste -sd, -)"
[[ "${m892_actual_paths}" == "${m892_expected_paths_csv}" ]] || exit 3
while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m892_expect "${path}" "${expected}"
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "${m892_contract}")

m892_expect_linkable_tool "${m892_dc}" "$(jq -er '.tool_identity.dc_shell_sha256' "${m892_contract}")"
m892_expect_linkable_tool "${m892_dc_wrapper}" "$(jq -er '.tool_identity.dc_wrapper_sha256' "${m892_contract}")"
m892_expect_linkable_tool "${m892_dc_actual}" "$(jq -er '.tool_identity.dc_actual_sha256' "${m892_contract}")"
m892_expect "${m892_lmutil}" "$(jq -er '.tool_identity.lmutil_sha256' "${m892_contract}")"
m892_expect "${m892_license}" "$(jq -er '.tool_identity.license_file_sha256' "${m892_contract}")"
m892_expect "${m892_std_slow}" "$(jq -er '.foundry_views.std_slow_sha256' "${m892_contract}")"
m892_expect "${m892_std_fast}" "$(jq -er '.foundry_views.std_fast_sha256' "${m892_contract}")"
m892_expect "${m892_macro_slow}" "$(jq -er '.foundry_views.macro_slow_sha256' "${m892_contract}")"
m892_expect "${m892_macro_fast}" "$(jq -er '.foundry_views.macro_fast_sha256' "${m892_contract}")"
m892_expect "${m892_macro_manifest}" "$(jq -er '.foundry_views.macro_manifest_sha256' "${m892_contract}")"
(cd "${m892_macro_root}" && sha256sum -c SHA256SUMS >/dev/null) || exit 3
[[ -f "${m892_forbidden_macro_v}" ]] || exit 3
! rg -n '\.v($|[[:space:]])|ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c\.v' \
    "${m892_filelist}" >/dev/null || exit 3
python3 - "${m892_adapter}" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.M | re.S)
if len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\b", text)) != 1:
    raise SystemExit("adapter macro template count is not one")
if re.search(r"\b(?:reg|logic)\b[^;]*\[[^]]+\]\s*\[[^]]+\]", text):
    raise SystemExit("adapter contains an unpacked register-array fallback")
if "{1'b0, address}" not in text or "slice*128 +: 128" not in text:
    raise SystemExit("adapter address/slice binding drift")
PY

for evidence in "${m892_m881}" "${m892_m879}" "${m892_m863}" "${m892_m623}" \
        "${m892_m885}" "${m892_m891}"; do
    m892_verify_dir_seal "${evidence}"
done
for json_file in "${m892_m881}/review.json" "${m892_m879}/review.json" \
        "${m892_m863}/RUN_COMPLETE.json" "${m892_m623}/review.json" \
        "${m892_m885}/review.json" "${m892_m891}/review.json" "${m892_binding}"; do
    m892_strict_json "${json_file}"
done
m892_expect "${m892_m881}/review.json" \
    "$(jq -er '.frozen_authorities.m881_review_sha256' "${m892_contract}")"
m892_expect "${m892_m879}/review.json" \
    "$(jq -er '.frozen_authorities.m879_review_sha256' "${m892_contract}")"
m892_expect "${m892_m863}/RUN_COMPLETE.json" \
    "$(jq -er '.frozen_authorities.m863_run_complete_sha256' "${m892_contract}")"
m892_expect "${m892_m623}/review.json" \
    "$(jq -er '.frozen_authorities.m623_review_sha256' "${m892_contract}")"
m892_expect "${m892_m885}/review.json" \
    "$(jq -er '.frozen_authorities.m885_source_review_sha256' "${m892_contract}")"
m892_expect "${m892_m885}/SHA256SUMS" \
    "$(jq -er '.frozen_authorities.m885_source_manifest_file_sha256' "${m892_contract}")"
m892_expect "${m892_m885}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.frozen_authorities.m885_source_outer_seal_file_sha256' "${m892_contract}")"
m892_expect "${m892_m891}/review.json" \
    "$(jq -er '.frozen_authorities.m891_failure_review_sha256' "${m892_contract}")"
m892_expect "${m892_m891}/SHA256SUMS" \
    "$(jq -er '.frozen_authorities.m891_failure_manifest_file_sha256' "${m892_contract}")"
m892_expect "${m892_m891}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.frozen_authorities.m891_failure_outer_seal_file_sha256' "${m892_contract}")"
jq -e '.status == "PASS_AUDIT__NO_CURRENT_M528_DC_STA_FORMALITY__FRESH_R21_MACRO_AWARE_SUCCESSOR_REQUIRED"
       and .verdict == "PASS_AUDIT" and .score_100 == 100
       and .macro_binding_semantics.fast_macro_db_available == true
       and .claim_boundary.current_m528_dc_sta == false' \
    "${m892_m881}/review.json" >/dev/null || exit 3
jq -e '.status == "PASS100_M863_C1_R21_SYNOPSYS_VCS_E3_FUNCTIONAL_RESULT_ADMITTED"
       and .verdict == "PASS" and .score_out_of_100 == 100
       and [.p0_count,.p1_count,.p2_count] == [0,0,0]
       and .claim_boundary.directed_component_synopsys_vcs_e3_functional_citable == true
       and .claim_boundary.timing_verified == false' \
    "${m892_m879}/review.json" >/dev/null || exit 3
jq -e '.claim_boundary.functional_vcs_only == true
       and .claim_boundary.speedup == false
       and .claim_boundary.ppa == false
       and .claim_boundary.timing_verified == false' \
    "${m892_m863}/RUN_COMPLETE.json" >/dev/null || exit 3
jq -e '.status == "PASS_M623_M617_R5_BOUNDED_GENERATED_MACRO_COMPONENT_RESULT"
       and .score_0_to_100 == 99 and [.p0_count,.p1_count,.p2_count] == [0,0,0]
       and .claim_boundary.c1_total_energy == false
       and .claim_boundary.rtl_integrated_macro_ppa == false' \
    "${m892_m623}/review.json" >/dev/null || exit 3
jq -e '.verdict == "PASS" and .score_out_of_100 == 100
       and [.p0_count,.p1_count,.p2_count] == [0,0,0]' \
    "${m892_m885}/review.json" >/dev/null || exit 3
jq -e '.status == "PASS_FAILURE_AUDIT__M884_RELEASE_NOT_AUTHORED__SOURCE_REVIEW_SCHEMA_MISMATCH__ADDITIVE_RUNNER_SOURCE_REPAIR_REQUIRED"
       and .verdict == "PASS_FAILURE_AUDIT" and .score_out_of_100 == 100
       and .decision.m884_known_defective_command_must_not_execute == true' \
    "${m892_m891}/review.json" >/dev/null || exit 3
jq -e '.cell == "TS1N28HPCPHVTB128X128M4S" and .instance_count == 9
       and .rtl_adapter.synthesizable_register_array_fallback == false
       and .claim_boundary.dc_sta == false' "${m892_binding}" >/dev/null || exit 3

[[ "$(jq -er '.identity.runner_sha256' "${m892_admission}")" == \
   "${M892_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.identity.result_path' "${m892_admission}")" == "${m892_result_rel}" ]] || exit 3
[[ "$(jq -er '.identity.attempt_path' "${m892_admission}")" == \
   'dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed' ]] || exit 3

if [[ "${m892_no_eda}" -eq 1 && "${m892_production_schema_selftest}" -eq 0 ]]; then
    printf '%s\n' \
        'status=PASS_M892_FULL_ADMISSION_CONTRACT_PATH_NO_EDA' \
        'admission_launch_now=false' \
        'candidate_only=true' \
        'fair_K_zero_bit=false' \
        'attempt_consumed=false' \
        'license_query_started=false' \
        'dc_shell_started=false' \
        'strict_duplicate_nonfinite_unknown_checks=delegated_to_source_closure' \
        >"${M892_NO_EDA_SELFTEST_ROOT}/FULL_PATH_PASS.txt"
    trap - EXIT INT TERM
    exit 0
fi

# Production source-review schema admission. A no-EDA test-only branch exercises
# this exact production predicate and exits before final review, resource, license,
# attempt, or dc_shell actions.
if [[ "${m892_production_schema_selftest}" -eq 1 ]]; then
    if [[ -n "${M892_NO_EDA_SOURCE_REVIEW_FIXTURE:-}" ]]; then
        [[ "${M892_NO_EDA_SOURCE_REVIEW_FIXTURE}" == /* && \
           -n "${M892_EXPECTED_NO_EDA_SOURCE_REVIEW_SHA256:-}" ]] || exit 87
        m892_source_review="${M892_NO_EDA_SOURCE_REVIEW_FIXTURE}"
        m892_source_review_sha="${M892_EXPECTED_NO_EDA_SOURCE_REVIEW_SHA256}"
        m892_expect "${m892_source_review}" "${m892_source_review_sha}"
    else
        m892_source_review="${m892_m885}/review.json"
        m892_source_review_sha="$(jq -er '.frozen_authorities.m885_source_review_sha256' "${m892_contract}")"
        m892_expect "${m892_source_review}" "${m892_source_review_sha}"
        m892_verify_dir_seal "${m892_m885}"
    fi
else
    m892_closed_keys "${m892_admission}" '.future_release_chain' \
        'final_review_path,final_review_sha_caller_pinned,release_binds_candidate_sha,release_binds_source_hammer_sha,source_hammer_review_path,source_hammer_review_sha256'
    m892_source_review="$(jq -er '.future_release_chain.source_hammer_review_path' "${m892_admission}")"
    m892_source_review_sha="$(jq -er '.future_release_chain.source_hammer_review_sha256' "${m892_admission}")"
    m892_expect "${m892_source_review}" "${m892_source_review_sha}"
    m892_verify_dir_seal "$(dirname "${m892_source_review}")"
fi
m892_strict_json "${m892_source_review}"
jq -e '.verdict == "PASS" and .score_out_of_100 == 100
       and [.p0_count,.p1_count,.p2_count] == [0,0,0]' \
    "${m892_source_review}" >/dev/null || exit 3
if [[ "${m892_production_schema_selftest}" -eq 1 ]]; then
    printf '%s\n' \
        'status=PASS_M892_PRODUCTION_M885_SCHEMA_PATH_NO_EDA' \
        'source_review_schema=score_out_of_100_plus_p0_p1_p2' \
        'attempt_consumed=false' \
        'license_query_started=false' \
        'dc_shell_started=false' \
        >"${M892_NO_EDA_SELFTEST_ROOT}/PRODUCTION_SCHEMA_PASS.txt"
    trap - EXIT INT TERM
    exit 0
fi
[[ -n "${M892_EXPECTED_DC_FINAL_REVIEW_SHA256:-}" ]] || exit 3
m892_expect "${m892_final_review}" "${M892_EXPECTED_DC_FINAL_REVIEW_SHA256}"
m892_verify_dir_seal "$(dirname "${m892_final_review}")"
m892_strict_json "${m892_final_review}"
jq -e --arg release_sha "$(m892_sha "${m892_release}")" \
       --arg runner_sha "$(m892_sha "${m892_runner}")" \
       '.schema == "m892_m528_r21_macro_aware_product_dc_final_launch_hammer_v1"
        and .status == "PASS100_M892_FINAL_LAUNCH_HAMMER"
        and .verdict == "PASS" and .score_100 == 100
        and .severity_counts == {"p0":0,"p1":0,"p2":0}
        and .identity.release_sha256 == $release_sha
        and .identity.runner_sha256 == $runner_sha
        and .decision.exactly_one_dc_attempt_authorized == true' \
    "${m892_final_review}" >/dev/null || exit 3

[[ -x "${m892_dc}" && -x "${m892_dc_actual}" && -x "${m892_lmutil}" ]] || exit 4
if ps -u "$(id -u)" -o args= | rg -q \
    '(^|/)(dc_shell|dc_shell-t)([[:space:]]|$)|common_shell_exec -shell dc_shell'; then
    echo "M892 refuses a same-UID DC collision" >&2
    exit 4
fi
mkdir "${m892_lock}" || exit 4
for sample in 1 2 3; do
    mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    headroom=$((commit_limit - committed))
    [[ "${mem_available}" -ge 134217728 && "${swap_free}" -ge 33554432 && \
       "${headroom}" -ge 67108864 ]] || exit 4
    sleep 2
done
"${m892_lmutil}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler >/dev/null
"${m892_lmutil}" lmstat -c 27030@ic.ismd-nemo -f DC-Ultra-Opt >/dev/null

mkdir "${m892_attempt}"
m892_attempt_consumed=1
printf 'status=M892_ATTEMPT_CONSUMED\nmax_attempts=1\nretry=false\n' \
    >"${m892_attempt}/ATTEMPT_CONSUMED.txt"
m892_seal_dir "${m892_attempt}"
mkdir "${m892_work}"
{
    echo status=M892_DC_ATTEMPT_ADMITTED
    echo fair_K_zero_bit=false
    echo clock_period_ns=3.000
    echo macro_count=9
    echo std_setup_corner=ssg0p9v125c
    echo std_hold_corner=ffg1p05vm40c
    echo macro_setup_corner=ssg0p9v125c
    echo macro_hold_corner=ffg1p05vm40c
    sha256sum "${m892_runner}" "${m892_release}" "${m892_contract}" \
        "${m892_source_review}" "${m892_final_review}" "${m892_m879}/review.json"
} >"${m892_work}/admission.txt"

set +e
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
    SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${m892_license}" \
    DESIGN_NAME=m528_dead_write_only_1rw_product_capture_island_r2 \
    HW_ROOT="${m892_hw_root}" RTL_FILELIST="${m892_hw_root}/${m892_filelist}" \
    SDC_FILE="${m892_hw_root}/${m892_sdc}" OUTPUT_DIR="${m892_work}" \
    STD_SLOW_DB="${m892_std_slow}" STD_FAST_DB="${m892_std_fast}" \
    MACRO_SLOW_DB="${m892_macro_slow}" MACRO_FAST_DB="${m892_macro_fast}" \
    "${m892_dc}" -f "${m892_hw_root}/${m892_tcl}" \
    >"${m892_work}/dc.log" 2>&1
m892_dc_rc=$?
set -e
printf '%s\n' "${m892_dc_rc}" >"${m892_work}/dc.rc"
[[ "${m892_dc_rc}" -eq 0 ]] || exit "${m892_dc_rc}"

if rg -ni '(^|[^A-Za-z])(Error:|Fatal:|unresolved reference|unable to resolve reference|LINK-[0-9]+)|\((TIM-209|OPT-150)\)' \
        "${m892_work}/dc.log" >/dev/null; then
    echo "M892 DC log contains fatal/link/TIM-209/OPT-150 evidence" >&2
    exit 9
fi
required=(
    reports/link.rpt reports/macro_binding_audit.txt
    reports/check_design_precompile.rpt reports/check_design_postcompile.rpt
    reports/check_timing_precompile.rpt reports/check_timing_postcompile.rpt
    reports/resources_precompile.rpt reports/resources_postcompile.rpt
    reports/references_precompile.rpt reports/references_postcompile.rpt
    reports/hierarchy_postcompile.rpt reports/qor.rpt reports/area_hierarchy.rpt
    reports/timing_setup.rpt reports/timing_hold_diagnostic.rpt
    reports/constraint_setup.rpt reports/constraint_hold_diagnostic.rpt
    reports/constraint_max_capacitance.rpt reports/constraint_max_transition.rpt
    reports/constraint_max_fanout.rpt reports/flow_contract.rpt
    reports/precompile_loop_gate.rpt netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v
    netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.sdc
    netlist/m528_dead_write_only_1rw_product_capture_island_r2.ddc
    netlist/m528_dead_write_only_1rw_product_capture_island_r2.svf
    TCL_PASS_TERMINAL.txt
)
for artifact in "${required[@]}"; do
    [[ -s "${m892_work}/${artifact}" && ! -L "${m892_work}/${artifact}" ]] || exit 6
done
grep -Fxq 'status=PASS_M892_RESOLVED_LIBRARY_MACRO_STRUCTURE' \
    "${m892_work}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_pre=9' "${m892_work}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_count_post=9' "${m892_work}/reports/macro_binding_audit.txt"
grep -Fxq 'macro_slow_fast_min_pair=true' "${m892_work}/reports/macro_binding_audit.txt"
grep -Fxq 'TIM-209=0' "${m892_work}/reports/precompile_loop_gate.rpt"
grep -Fxq 'OPT-150=0' "${m892_work}/reports/precompile_loop_gate.rpt"
grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' "${m892_work}/reports/precompile_loop_gate.rpt"
! rg -q 'slack \(VIOLATED\)' "${m892_work}/reports/timing_setup.rpt" || exit 9
rg -q 'slack \(MET\)' "${m892_work}/reports/timing_setup.rpt" || exit 9
for report in constraint_setup.rpt constraint_max_capacitance.rpt \
        constraint_max_transition.rpt constraint_max_fanout.rpt; do
    grep -Fq 'This design has no violated constraints.' \
        "${m892_work}/reports/${report}" || exit 9
done
m892_netlist="${m892_work}/netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v"
[[ "$(rg -o 'TS1N28HPCPHVTB128X128M4S' "${m892_netlist}" | wc -l)" -eq 9 ]] || exit 9
! rg -ni 'unresolved reference|unable to resolve reference|inferred.*parent|parent.*inferred|register.array fallback' \
    "${m892_work}/reports" "${m892_work}/dc.log" >/dev/null || exit 9

python3 - "${m892_work}" <<'PY'
import json, math, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
area_text = (root / "reports/area_hierarchy.rpt").read_text(errors="replace")
setup_text = (root / "reports/timing_setup.rpt").read_text(errors="replace")
area_match = re.search(r"Total cell area:\s*([0-9.]+)", area_text)
slacks = [float(v) for v in re.findall(r"slack \(MET\)\s+([-+]?\d+(?:\.\d+)?)", setup_text)]
if not area_match or not slacks:
    raise SystemExit("missing area/setup metric")
area = float(area_match.group(1))
setup = min(slacks)
if not math.isfinite(area) or area <= 0 or not math.isfinite(setup):
    raise SystemExit("invalid area/setup metric")
receipt = {
    "claim_boundary": {
        "candidate_only": True, "energy": False, "fair_K_zero_bit": False,
        "headline": False, "hold_signoff": False, "paper_ppa_ready": False,
        "physical_route": False, "power": False, "ppa": False,
        "speedup": False, "system": False, "system_speedup": False,
        "throughput_per_mm2": False,
    },
    "clock_period_ns": 3.0,
    "fair_K_zero_bit": False,
    "hold": {"diagnostic_only": True, "macro_fast_min_view_bound": True},
    "macro_cell": "TS1N28HPCPHVTB128X128M4S",
    "macro_count": 9,
    "schema": "m892_m528_r21_macro_aware_product_dc_receipt_v1",
    "setup_slack_min_reported_ns": setup,
    "status": "PASS_RAW_MACRO_LINKED_PRODUCT_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
    "total_cell_area_um2_dc_reported": area,
}
(root / "m892_dc_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n")
PY
m892_strict_json "${m892_work}/m892_dc_receipt.json"
m892_closed_keys "${m892_work}/m892_dc_receipt.json" '.' \
    'claim_boundary,clock_period_ns,fair_K_zero_bit,hold,macro_cell,macro_count,schema,setup_slack_min_reported_ns,status,total_cell_area_um2_dc_reported'
printf 'status=PASS_M892_RAW_MACRO_LINKED_PRODUCT_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER\nfair_K_zero_bit=false\nthroughput_per_mm2=false\nspeedup=false\nsystem_speedup=false\nhold_diagnostic_only=true\npaper_ppa_ready=false\n' \
    >"${m892_work}/RUN_COMPLETE.txt"
m892_seal_dir "${m892_work}"
mv -T -- "${m892_work}" "${m892_result}"
m892_completed=1
trap - EXIT INT TERM
rmdir "${m892_lock}"
echo "M892 raw macro-linked product candidate completed at ${m892_result}"
