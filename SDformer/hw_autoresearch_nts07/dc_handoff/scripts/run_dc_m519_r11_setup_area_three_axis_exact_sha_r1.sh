#!/usr/bin/env bash
set -euo pipefail

m519_r11_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m519_r11_hw_root="$(cd "${m519_r11_dc_root}/.." && pwd)"
m519_r11_runner="$(realpath "${BASH_SOURCE[0]}")"
m519_r11_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m519_r11_dc_wrapper=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
m519_r11_dc_actual_exe=/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec
m519_r11_dc_install_root=/opt/synopsys/syn/V-2023.12-SP3
m519_r11_slow=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
m519_r11_fast=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
m519_r11_filelist=dc_handoff/filelists/date_m519_r5_channel_local_fault_three_axis_logic_only_dc.f
m519_r11_sdc=dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
m519_r11_tcl=dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl
m519_r11_contract=contracts/m519_r11_setup_area_three_axis_recovery_contract_r1_20260828.json
m519_r11_final_admission=contracts/m519_r11_setup_area_three_axis_dc_launch_admission_r1_20260828.json
m519_r11_candidate=contracts/m742_m519_r11_setup_area_three_axis_dc_launch_admission_candidate_r1_20260828.json
m519_r11_admission=${m519_r11_final_admission}
m519_r11_expected_admission_status=AUTHORIZED_ONE_M519_R11_THREE_AXIS_SETUP_AREA_DC_ATTEMPT_R1
m519_r11_expected_launch_now=true
m519_r11_r5_static=reviews/m519_r5_channel_local_fault_static_hammer_r1_20260827
m519_r11_r5_vcs=results/m519_r5_channel_local_fault_vcs_r1_20260827
m519_r11_r5_vcs_review=reviews/m519_r5_channel_local_fault_vcs_receipt_blind_hammer_r1_20260827
m519_r11_r5_failure=reviews/m519_r5_final_failure_receipt_hammer_r1_20260827
m519_r11_r5_quarantine=dc_handoff/runs/m519_r5_channel_local_fault_three_axis_logic_only_dc_3p000ns_r1_20260827.failed_or_incomplete.4165439.quarantine
m519_r11_r6_failed_review=reviews/m538_m519_r6_setup_area_flow_static_hammer_r1_20260827
m519_r11_r7_disqualified_review=reviews/m540_m519_r7_setup_area_flow_static_hammer_r1_20260827
m519_r11_m694=reviews/m694_m519_r9_three_axis_dc_release_fresh_hammer_r1_20260828
m519_r11_m701=reviews/m701_m519_r9_pre_eda_shell_failure_receipt_r1_20260828
m519_r11_r10_failure=dc_handoff/runs/m519_r10_pre_attempt_shell_failure.693765.receipt
m519_r11_m740=reviews/m740_m519_r10_pre_eda_shell_failure_fresh_hammer_r1_20260828
m519_r11_canonical="${m519_r11_dc_root}/runs/m519_r11_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260828"
m519_r11_work="${m519_r11_dc_root}/runs/.m519_r11_channel_local_fault_dc_work.$$"
m519_r11_attempt="${m519_r11_dc_root}/runs/.m519_r11_channel_local_fault_dc_attempt_consumed"
m519_r11_quarantine="${m519_r11_canonical}.failed_or_incomplete.$$.quarantine"
m519_r11_preflight_staging="${m519_r11_dc_root}/runs/.m519_r11_preflight.$$.staging"
m519_r11_preflight_reject="${m519_r11_canonical}.preflight_rejected.$$.quarantine"
m519_r11_uid="$(id -u)"
m519_r11_attempt_consumed=0
if [[ -n "${M519_R11_NO_EDA_SELF_TEST:-}" && \
      -n "${M519_R11_SELF_TEST_ROOT:-}" ]]; then
    m519_r11_pre_attempt_receipt_root="${M519_R11_SELF_TEST_ROOT:-}"
elif [[ -n "${M519_R11_NO_EDA_FULL_PATH_SELF_TEST:-}" && \
        -n "${M519_R11_FULL_PATH_SELF_TEST_ROOT:-}" ]]; then
    m519_r11_pre_attempt_receipt_root="${M519_R11_FULL_PATH_SELF_TEST_ROOT:-}"
else
    m519_r11_pre_attempt_receipt_root="${m519_r11_dc_root}/runs"
fi
m519_r11_pre_attempt_receipt="${m519_r11_pre_attempt_receipt_root}/m519_r11_pre_attempt_shell_failure.$$.receipt"

# This minimal trap is installed before any admission/helper call.  It uses no
# compound local declaration and guarantees a fresh, noncanonical, double-
# sealed receipt for every runtime shell failure before the attempt is consumed
# or the full post-work cleanup trap takes ownership.
m519_r11_pre_attempt_seal_dir() {
    local seal_dir
    seal_dir=$1
    (
        cd "${seal_dir}"
        find . -type f ! -path './SHA256SUMS' \
            ! -path './SHA256SUMS.seal.sha256' -print0 | sort -z | \
            xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}
m519_r11_pre_attempt_failure_cleanup() {
    m519_r11_pre_attempt_saved_rc=$?
    local rc
    rc=${m519_r11_pre_attempt_saved_rc}
    set +e
    if [[ "${rc}" -ne 0 && "${m519_r11_attempt_consumed}" -eq 0 && \
          -n "${m519_r11_pre_attempt_receipt_root}" && \
          ! -e "${m519_r11_pre_attempt_receipt}" ]]; then
        mkdir -p "${m519_r11_pre_attempt_receipt}"
        printf 'status=PRE_ATTEMPT_SHELL_FAILURE__NO_EDA_RESULT_ADMITTED\nexit_code=%s\nattempt_consumed=false\nrunner=%s\n' \
            "${rc}" "${m519_r11_runner}" \
            >"${m519_r11_pre_attempt_receipt}/FAILURE.txt"
        m519_r11_pre_attempt_seal_dir "${m519_r11_pre_attempt_receipt}"
    fi
    return "${rc}"
}
trap m519_r11_pre_attempt_failure_cleanup EXIT

# All memory units are KiB, matching /proc/meminfo.
m519_r11_preflight_commit_kib=67108864
m519_r11_runtime_commit_kib=33554432
m519_r11_mem_available_kib=134217728
m519_r11_swap_free_kib=33554432

m519_r11_sha() { sha256sum "$1" | awk '{print $1}'; }
m519_r11_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m519_r11_sha "${path}")" == "${expected}" ]] || {
        echo "M519 R11 identity mismatch: ${path}" >&2
        exit 3
    }
}
m519_r11_closed_keys() {
    local file=$1 expression=$2 expected=$3 actual
    actual="$(jq -er "${expression} | keys[]" "${file}" | LC_ALL=C sort | paste -sd, -)"
    [[ "${actual}" == "${expected}" ]] || {
        echo "M519 R11 unknown or missing JSON key at ${expression}: ${actual}" >&2
        exit 3
    }
}
m519_r11_json_equal() {
    local left_file=$1 left_expr=$2 right_file=$3 right_expr=$4
    [[ "$(jq -er "${left_expr}" "${left_file}")" == \
       "$(jq -er "${right_expr}" "${right_file}")" ]] || {
        echo "M519 R11 admission/contract identity disagreement: ${left_expr}" >&2
        exit 3
    }
}
m519_r11_verify_double_seal_file() {
    local payload
    local sidecar
    local outer
    local dir
    local base
    payload=$1
    sidecar="${payload}.sha256"
    outer="${payload}.sha256.seal.sha256"
    [[ -f "${sidecar}" && -f "${outer}" ]] || exit 3
    dir="$(dirname "${payload}")"; base="$(basename "${payload}")"
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) || exit 3
}

# The current full runner is syntax-checked before any admission, resource,
# attempt or tool path.  The injectable self-test exercises the exact double-
# seal helper and optionally the early failure trap, then exits before EDA.
bash -n "${m519_r11_runner}"
if [[ -n "${M519_R11_NO_EDA_SELF_TEST:-}" ]]; then
    [[ "${M519_R11_NO_EDA_SELF_TEST}" == 1 && \
       -n "${M519_R11_SELF_TEST_ROOT:-}" && \
       "${M519_R11_SELF_TEST_ROOT}" == /* && \
       -d "${M519_R11_SELF_TEST_ROOT}" ]] || exit 83
    m519_r11_self_payload="${M519_R11_SELF_TEST_ROOT}/payload.txt"
    printf 'm519-r11-no-eda-self-test\n' >"${m519_r11_self_payload}"
    (
        cd "${M519_R11_SELF_TEST_ROOT}"
        sha256sum payload.txt >payload.txt.sha256
        sha256sum payload.txt.sha256 >payload.txt.sha256.seal.sha256
    )
    m519_r11_verify_double_seal_file "${m519_r11_self_payload}"
    [[ -z "${M519_R11_SELF_TEST_INJECT_PRE_ATTEMPT_FAILURE:-}" ]] || exit 86
    trap - EXIT
    exit 0
fi

# Unlike the early helper test above, this mode traverses the complete sealed
# candidate-admission and recovery-contract validation path.  It switches only
# the admission identity and expected launch bit, then exits at the explicit
# marker below before resource preflight, attempt publication, or any tool.
if [[ -n "${M519_R11_NO_EDA_FULL_PATH_SELF_TEST:-}" ]]; then
    [[ "${M519_R11_NO_EDA_FULL_PATH_SELF_TEST}" == 1 && \
       -z "${M519_R11_NO_EDA_SELF_TEST:-}" && \
       -n "${M519_R11_FULL_PATH_SELF_TEST_ROOT:-}" && \
       "${M519_R11_FULL_PATH_SELF_TEST_ROOT}" == /* && \
       -d "${M519_R11_FULL_PATH_SELF_TEST_ROOT}" ]] || exit 87
    m519_r11_admission=${m519_r11_candidate}
    m519_r11_expected_admission_status=READY_FOR_FRESH_INDEPENDENT_STATIC_HAMMER__NO_EDA_AUTHORIZED
    m519_r11_expected_launch_now=false
fi

[[ -n "${M519_R11_EXPECTED_DC_RUNNER_SHA256:-}" && \
   "$(m519_r11_sha "${m519_r11_runner}")" == \
   "${M519_R11_EXPECTED_DC_RUNNER_SHA256}" ]] || {
    echo "M519 R11 caller must pin independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M519_R11_EXPECTED_DC_LAUNCH_ADMISSION_SHA256:-}" ]] || {
    echo "M519 R11 source-only package has no implicit launch authorization" >&2
    exit 3
}
[[ ! -e "${m519_r11_canonical}" && ! -e "${m519_r11_work}" && \
   ! -e "${m519_r11_attempt}" && ! -e "${m519_r11_quarantine}" && \
   ! -e "${m519_r11_preflight_staging}" ]] || {
    echo "M519 R11 refuses consumed or colliding result identity" >&2
    exit 5
}
[[ -z "${M519_R11_DC_RUN:-}" ]] || {
    echo "M519 R11 canonical path override is forbidden" >&2
    exit 5
}

cd "${m519_r11_hw_root}"
m519_r11_expect "${m519_r11_admission}" \
    "${M519_R11_EXPECTED_DC_LAUNCH_ADMISSION_SHA256}"
m519_r11_verify_double_seal_file "${m519_r11_admission}"
jq -e --arg expected_status "${m519_r11_expected_admission_status}" \
       --argjson expected_launch_now "${m519_r11_expected_launch_now}" \
       '.status == $expected_status
       and .launch_now == $expected_launch_now
       and .authorization.run_dc == true
       and .authorization.max_attempts == 1
       and .authorization.run_vcs == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false
       and .authorization.run_formality == false
       and .authorization.run_remote == false' \
    "${m519_r11_admission}" >/dev/null || exit 3
m519_r11_closed_keys "${m519_r11_admission}" '.authorization' \
    'max_attempts,run_dc,run_formality,run_pt,run_ptpx,run_remote,run_vcs'
m519_r11_closed_keys "${m519_r11_admission}" '.r10_repair_provenance' \
    'm694_manifest_file_sha256,m694_outer_seal_file_sha256,m694_review_path,m694_review_sha256,m694_status,m701_manifest_file_sha256,m701_no_eda_started,m701_outer_seal_file_sha256,m701_review_path,m701_review_sha256,m701_status,r10_is_additive,r9_attempt_remains_absent,r9_result_remains_absent'
m519_r11_closed_keys "${m519_r11_admission}" '.r11_repair_provenance' \
    'm740_manifest_file_sha256,m740_outer_seal_file_sha256,m740_review_path,m740_review_sha256,m740_status,r10_attempt_consumed,r10_canonical_absent,r10_failure_manifest_file_sha256,r10_failure_outer_seal_file_sha256,r10_failure_path,r10_failure_payload_sha256,r11_is_additive'
jq -e '.r10_repair_provenance.r10_is_additive == true
       and .r10_repair_provenance.m701_no_eda_started == true
       and .r10_repair_provenance.r9_result_remains_absent == true
       and .r10_repair_provenance.r9_attempt_remains_absent == true' \
    "${m519_r11_admission}" >/dev/null || exit 3
[[ "$(jq -er '.r10_repair_provenance.m694_review_path' "${m519_r11_admission}")" == \
   "${m519_r11_m694}/review.json" ]] || exit 3
[[ "$(jq -er '.r10_repair_provenance.m701_review_path' "${m519_r11_admission}")" == \
   "${m519_r11_m701}/review.json" ]] || exit 3

# R10 exists only because the independently sealed R9 GO was consumed by the
# independently sealed pre-EDA shell failure.  Bind both exact receipts and
# their exact statuses before any resource preflight or attempt consumption.
for sealed in "${m519_r11_m694}" "${m519_r11_m701}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
for item in m694 m701; do
    if [[ "${item}" == m694 ]]; then
        basis=${m519_r11_m694}
    else
        basis=${m519_r11_m701}
    fi
    m519_r11_expect "${basis}/review.json" \
        "$(jq -er ".r10_repair_provenance.${item}_review_sha256" "${m519_r11_admission}")"
    m519_r11_expect "${basis}/SHA256SUMS" \
        "$(jq -er ".r10_repair_provenance.${item}_manifest_file_sha256" "${m519_r11_admission}")"
    m519_r11_expect "${basis}/SHA256SUMS.seal.sha256" \
        "$(jq -er ".r10_repair_provenance.${item}_outer_seal_file_sha256" "${m519_r11_admission}")"
done
[[ "$(jq -er '.status' "${m519_r11_m694}/review.json")" == \
   "$(jq -er '.r10_repair_provenance.m694_status' "${m519_r11_admission}")" ]] || exit 3
[[ "$(jq -er '.status' "${m519_r11_m701}/review.json")" == \
   "$(jq -er '.r10_repair_provenance.m701_status' "${m519_r11_admission}")" ]] || exit 3
jq -e '.status == "GO_ONE_M519_R9_DC_ONLY_ATTEMPT__FINAL_LIVE_RECHECK_REQUIRED"
       and .severity_counts.p0 == 0 and .severity_counts.p1 == 0
       and .authorization.max_attempts == 1
       and .authorization.run_dc == true
       and .authorization.run_vcs == false
       and .authorization.run_formality == false
       and .authorization.run_pt == false
       and .authorization.run_ptpx == false' \
    "${m519_r11_m694}/review.json" >/dev/null || exit 3
jq -e '.status == "PRE_EDA_SHELL_FAILURE__NO_DC_STARTED__M519_R9_NOT_CITABLE__ADDITIVE_R10_REQUIRED"
       and .failure.exit_code == 1
       and .failure.failure_stage == "shell function definition before admission verification, preflight, attempt consumption, or EDA launch"
       and .observed_absence_after_failure.m519_r9_canonical_result_absent == true
       and .observed_absence_after_failure.m519_r9_attempt_sentinel_absent == true
       and .claim_boundary.dc_started == false
       and (.required_next_step | contains("additive M519 R10"))' \
    "${m519_r11_m701}/review.json" >/dev/null || exit 3

# R11 is an additive successor to the double-sealed R10 pre-EDA failure.  Bind
# both that receipt and M740's independent causal audit before evaluating the
# remaining inherited admission and contract predicates.
for sealed in "${m519_r11_r10_failure}" "${m519_r11_m740}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m519_r11_expect "${m519_r11_r10_failure}/FAILURE.txt" \
    "$(jq -er '.r11_repair_provenance.r10_failure_payload_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_r10_failure}/SHA256SUMS" \
    "$(jq -er '.r11_repair_provenance.r10_failure_manifest_file_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_r10_failure}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r11_repair_provenance.r10_failure_outer_seal_file_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_m740}/review.json" \
    "$(jq -er '.r11_repair_provenance.m740_review_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_m740}/SHA256SUMS" \
    "$(jq -er '.r11_repair_provenance.m740_manifest_file_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_m740}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.r11_repair_provenance.m740_outer_seal_file_sha256' "${m519_r11_admission}")"
[[ "$(jq -er '.r11_repair_provenance.r10_failure_path' "${m519_r11_admission}")" == \
   "${m519_r11_r10_failure}" ]] || exit 3
[[ "$(jq -er '.r11_repair_provenance.m740_review_path' "${m519_r11_admission}")" == \
   "${m519_r11_m740}/review.json" ]] || exit 3
jq -e '.r11_repair_provenance.r11_is_additive == true
       and .r11_repair_provenance.r10_attempt_consumed == false
       and .r11_repair_provenance.r10_canonical_absent == true' \
    "${m519_r11_admission}" >/dev/null || exit 3
jq -e '.status == "PASS_FAILURE_AUDIT__R10_BLOCKED__PRE_EDA_JQ_ESCAPE__ADDITIVE_R11_REQUIRED"
       and .verdict == "PASS"
       and .score_out_of_100 == 100
       and .finding.exact_invalid_program_replay_rc == 3
       and .finding.same_predicate_without_literal_backslash_rc == 0
       and .authorization.run_r11_now == false
       and .authorization.run_dc == false' \
    "${m519_r11_m740}/review.json" >/dev/null || exit 3
grep -Fxq 'status=PRE_ATTEMPT_SHELL_FAILURE__NO_EDA_RESULT_ADMITTED' \
    "${m519_r11_r10_failure}/FAILURE.txt"
grep -Fxq 'exit_code=3' "${m519_r11_r10_failure}/FAILURE.txt"
grep -Fxq 'attempt_consumed=false' "${m519_r11_r10_failure}/FAILURE.txt"
# Bind the exact independently sealed M576 status; a paraphrase is not admissible.
m519_r11_m576=reviews/m576_m519_r8_dc_launch_admission_candidate_hammer_r1_20260828
(cd "${m519_r11_m576}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
m519_r11_expect "${m519_r11_m576}/review.json" \
    "$(jq -er '.fresh_successor_provenance.candidate_hammer_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_m576}/SHA256SUMS" \
    "$(jq -er '.fresh_successor_provenance.candidate_hammer_manifest_file_sha256' "${m519_r11_admission}")"
m519_r11_expect "${m519_r11_m576}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.fresh_successor_provenance.candidate_hammer_outer_seal_file_sha256' "${m519_r11_admission}")"
[[ "$(jq -er '.fresh_successor_provenance.candidate_hammer_status' "${m519_r11_admission}")" == \
   "$(jq -er '.status' "${m519_r11_m576}/review.json")" ]] || exit 3
jq -e '.verdict == "PASS" and .score_out_of_100 == 100
       and .severity_counts == {"p0":0,"p1":0,"p2":0}' \
    "${m519_r11_m576}/review.json" >/dev/null || exit 3
m519_r11_closed_keys "${m519_r11_admission}" '.identity' \
    'dc_actual_exec_path,dc_actual_exec_sha256,dc_filelist_path,dc_filelist_sha256,dc_runner_path,dc_runner_sha256,dc_shell_path,dc_shell_sha256,dc_tcl_path,dc_tcl_sha256,dc_wrapper_path,dc_wrapper_sha256,docs359_path,docs359_sha256,fast_lib_path,fast_lib_sha256,r5_final_failure_review_outer_seal_file_sha256,r5_final_failure_review_path,r5_quarantine_outer_seal_file_sha256,r5_quarantine_path,r5_static_review_outer_seal_file_sha256,r5_static_review_path,r5_vcs_result_outer_seal_file_sha256,r5_vcs_result_path,r5_vcs_review_outer_seal_file_sha256,r5_vcs_review_path,r6_static_review_outer_seal_file_sha256,r6_static_review_path,r7_disqualified_review_outer_seal_file_sha256,r7_disqualified_review_path,recovery_contract_path,recovery_contract_sha256,sdc_path,sdc_sha256,slow_lib_path,slow_lib_sha256'
for key in $(jq -r '.identity | keys[]' "${m519_r11_admission}"); do
    value="$(jq -er ".identity.${key}" "${m519_r11_admission}")"
    case "${key}" in
        *_sha256) [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3 ;;
        *_path) [[ -n "${value}" && "${value}" != *$'\n'* && \
                    "${value}" != *$'\t'* ]] || exit 3 ;;
        *) exit 3 ;;
    esac
done
[[ "$(jq -er '.identity.recovery_contract_path' "${m519_r11_admission}")" == \
   "${m519_r11_contract}" ]] || exit 3
m519_r11_expect "${m519_r11_contract}" \
    "$(jq -er '.identity.recovery_contract_sha256' "${m519_r11_admission}")"
m519_r11_verify_double_seal_file "${m519_r11_contract}"
m519_r11_json_equal "${m519_r11_admission}" '.r10_repair_provenance' \
    "${m519_r11_contract}" '.r10_repair_provenance'
m519_r11_json_equal "${m519_r11_admission}" '.r11_repair_provenance' \
    "${m519_r11_contract}" '.r11_repair_provenance'

jq -e '.status == "AUTHOR_SOURCE_ONLY_COMPLETE__FRESH_INDEPENDENT_STATIC_HAMMER_REQUIRED__NO_EDA_AUTHORIZED"
       and .authorization.author_ran_eda == false
       and .authorization.run_dc_now == false
       and .authorization.run_vcs_now == false
       and .authorization.run_pt_now == false
       and .authorization.run_ptpx_now == false
       and .authorization.run_formality_now == false
       and .authorization.run_remote_now == false' \
    "${m519_r11_contract}" >/dev/null || exit 3

m519_r11_expected_exact_paths=(
    dc_handoff/scripts/run_dc_m519_r11_setup_area_three_axis_exact_sha_r1.sh
    dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl
    dc_handoff/filelists/date_m519_r5_channel_local_fault_three_axis_logic_only_dc.f
    dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc
    rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv
    rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv
    rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv
    rtl_m218/m218_fc2_tagged_slice_service_island.sv
    rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv
    rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv
    rtl_m519/m519_fc2_k1_registered_release_service_island.sv
    rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv
    rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv
    rtl_m519/m519_fc2_k8_registered_release_8bank_raw4_acc24.sv
    rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv
    rtl_m519/m519_fc2_registered_release_matched_8bank_raw4_acc24.sv
    docs/359_DATE终局冻结_20260813.md
)
m519_r11_actual_exact_paths="$(jq -r '.exact_files | keys[]' \
    "${m519_r11_contract}" | LC_ALL=C sort | paste -sd, -)"
m519_r11_expected_exact_csv="$(printf '%s\n' "${m519_r11_expected_exact_paths[@]}" | \
    LC_ALL=C sort | paste -sd, -)"
[[ "${m519_r11_actual_exact_paths}" == "${m519_r11_expected_exact_csv}" ]] || {
    echo "M519 R11 contract exact_files has unknown or missing path" >&2
    exit 3
}
: > /tmp/m519_r11_exact_verified.$$.tsv
while IFS=$'\t' read -r path expected; do
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || exit 3
    m519_r11_expect "${path}" "${expected}"
    printf '%s\t%s\n' "${path}" "${expected}" \
        >>/tmp/m519_r11_exact_verified.$$.tsv
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' \
    "${m519_r11_contract}")

# Cross-check every future admission path and SHA against the frozen contract.
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_runner_path' \
    "${m519_r11_contract}" '.setup_area_flow.runner'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_runner_sha256' \
    "${m519_r11_contract}" '.setup_area_flow.runner_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_tcl_path' \
    "${m519_r11_contract}" '.setup_area_flow.tcl'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_tcl_sha256' \
    "${m519_r11_contract}" '.setup_area_flow.tcl_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_filelist_path' \
    "${m519_r11_contract}" '.setup_area_flow.filelist'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_filelist_sha256' \
    "${m519_r11_contract}" '.setup_area_flow.filelist_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.sdc_path' \
    "${m519_r11_contract}" '.setup_area_flow.sdc'
m519_r11_json_equal "${m519_r11_admission}" '.identity.sdc_sha256' \
    "${m519_r11_contract}" '.setup_area_flow.sdc_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_shell_path' \
    "${m519_r11_contract}" '.tool_identity.dc_shell'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_shell_sha256' \
    "${m519_r11_contract}" '.tool_identity.dc_shell_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_wrapper_path' \
    "${m519_r11_contract}" '.tool_identity.dc_shell_wrapper'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_wrapper_sha256' \
    "${m519_r11_contract}" '.tool_identity.dc_shell_wrapper_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_actual_exec_path' \
    "${m519_r11_contract}" '.tool_identity.dc_shell_actual_executable'
m519_r11_json_equal "${m519_r11_admission}" '.identity.dc_actual_exec_sha256' \
    "${m519_r11_contract}" '.tool_identity.dc_shell_actual_executable_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.slow_lib_path' \
    "${m519_r11_contract}" '.tool_identity.slow_library'
m519_r11_json_equal "${m519_r11_admission}" '.identity.slow_lib_sha256' \
    "${m519_r11_contract}" '.tool_identity.slow_library_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.fast_lib_path' \
    "${m519_r11_contract}" '.tool_identity.fast_library'
m519_r11_json_equal "${m519_r11_admission}" '.identity.fast_lib_sha256' \
    "${m519_r11_contract}" '.tool_identity.fast_library_sha256'
m519_r11_json_equal "${m519_r11_admission}" '.identity.docs359_path' \
    "${m519_r11_contract}" '.frozen_docs.path'
m519_r11_json_equal "${m519_r11_admission}" '.identity.docs359_sha256' \
    "${m519_r11_contract}" '.docs359_sha256'
for stem in r5_static_review r5_vcs_result r5_vcs_review \
        r5_final_failure_review r5_quarantine; do
    m519_r11_json_equal "${m519_r11_admission}" ".identity.${stem}_path" \
        "${m519_r11_contract}" ".sealed_basis.${stem}"
    m519_r11_json_equal "${m519_r11_admission}" \
        ".identity.${stem}_outer_seal_file_sha256" \
        "${m519_r11_contract}" \
        ".sealed_basis.${stem}_outer_seal_file_sha256"
done
for stem in r6_static_review r7_disqualified_review; do
    m519_r11_json_equal "${m519_r11_admission}" ".identity.${stem}_path" \
        "${m519_r11_contract}" ".sealed_basis.${stem}"
    m519_r11_json_equal "${m519_r11_admission}" \
        ".identity.${stem}_outer_seal_file_sha256" \
        "${m519_r11_contract}" \
        ".sealed_basis.${stem}_outer_seal_file_sha256"
done
[[ "$(jq -er '.identity.dc_runner_sha256' "${m519_r11_admission}")" == \
   "${M519_R11_EXPECTED_DC_RUNNER_SHA256}" ]] || exit 3
[[ "$(jq -er '.identity.dc_runner_path' "${m519_r11_admission}")" == \
   dc_handoff/scripts/run_dc_m519_r11_setup_area_three_axis_exact_sha_r1.sh ]] || exit 3
[[ "$(jq -er '.identity.dc_tcl_path' "${m519_r11_admission}")" == \
   "${m519_r11_tcl}" && \
   "$(jq -er '.identity.dc_filelist_path' "${m519_r11_admission}")" == \
   "${m519_r11_filelist}" && \
   "$(jq -er '.identity.sdc_path' "${m519_r11_admission}")" == \
   "${m519_r11_sdc}" && \
   "$(jq -er '.identity.dc_shell_path' "${m519_r11_admission}")" == \
   "${m519_r11_dc}" && \
   "$(jq -er '.identity.dc_wrapper_path' "${m519_r11_admission}")" == \
   "${m519_r11_dc_wrapper}" && \
   "$(jq -er '.identity.dc_actual_exec_path' "${m519_r11_admission}")" == \
   "${m519_r11_dc_actual_exe}" && \
   "$(jq -er '.identity.slow_lib_path' "${m519_r11_admission}")" == \
   "${m519_r11_slow}" && \
   "$(jq -er '.identity.fast_lib_path' "${m519_r11_admission}")" == \
   "${m519_r11_fast}" ]] || exit 3

# Launch-time byte closure covers the symlinked entry, its wrapper, the actual
# long-lived common_shell executable, both timing libraries, and all workspace
# exact_files checked above.  Contract/admission string equality is never used
# as a substitute for checking the current bytes.
[[ "$(realpath "${m519_r11_dc}")" == "${m519_r11_dc_wrapper}" ]] || exit 3
m519_r11_expect "${m519_r11_dc}" \
    "$(jq -er '.tool_identity.dc_shell_sha256' "${m519_r11_contract}")"
m519_r11_expect "${m519_r11_dc_wrapper}" \
    "$(jq -er '.tool_identity.dc_shell_wrapper_sha256' "${m519_r11_contract}")"
m519_r11_expect "${m519_r11_dc_actual_exe}" \
    "$(jq -er '.tool_identity.dc_shell_actual_executable_sha256' "${m519_r11_contract}")"
m519_r11_expect "${m519_r11_slow}" \
    "$(jq -er '.tool_identity.slow_library_sha256' "${m519_r11_contract}")"
m519_r11_expect "${m519_r11_fast}" \
    "$(jq -er '.tool_identity.fast_library_sha256' "${m519_r11_contract}")"

for sealed in "${m519_r11_r5_static}" "${m519_r11_r5_vcs}" \
        "${m519_r11_r5_vcs_review}" "${m519_r11_r5_failure}" \
        "${m519_r11_r5_quarantine}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done

# Each sealed basis must both self-verify and equal the outer-seal file SHA
# independently frozen in the contract and future admission.
for stem in r5_static_review r5_vcs_result r5_vcs_review \
        r5_final_failure_review r5_quarantine; do
    basis_path="$(jq -er ".sealed_basis.${stem}" "${m519_r11_contract}")"
    basis_sha="$(jq -er ".sealed_basis.${stem}_outer_seal_file_sha256" \
        "${m519_r11_contract}")"
    m519_r11_expect "${basis_path}/SHA256SUMS.seal.sha256" "${basis_sha}"
done
for sealed in "${m519_r11_r6_failed_review}" \
        "${m519_r11_r7_disqualified_review}"; do
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
done
m519_r11_expect "${m519_r11_r6_failed_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.sealed_basis.r6_static_review_outer_seal_file_sha256' \
        "${m519_r11_contract}")"
m519_r11_expect "${m519_r11_r7_disqualified_review}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.sealed_basis.r7_disqualified_review_outer_seal_file_sha256' \
        "${m519_r11_contract}")"
jq -e '.status == "FAIL_STATIC_HAMMER__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION"
       and .severity_counts.p0 == 0 and .severity_counts.p1 == 3' \
    "${m519_r11_r6_failed_review}/m538_m519_r6_setup_area_flow_static_hammer_verdict_r1.json" \
    >/dev/null || exit 3
jq -e '.status == "DISQUALIFIED_REVIEWER_TOOL_INVOCATION__R7_SOURCE_BLOCKED__NO_LAUNCH_ADMISSION"
       and .severity_counts.p0 == 2 and .severity_counts.p1 == 2
       and .review_protocol.reviewer_eligible_for_launch_admission == false
       and .review_protocol.accidental_dc_executable_invocations == 1' \
    "${m519_r11_r7_disqualified_review}/review.json" >/dev/null || exit 3

if [[ -n "${M519_R11_NO_EDA_FULL_PATH_SELF_TEST:-}" ]]; then
    printf '%s\n' \
        'status=PASS_M519_R11_FULL_ADMISSION_CONTRACT_PATH_NO_EDA' \
        'admission_launch_now=false' \
        'preflight_started=false' \
        'attempt_consumed=false' \
        'dc_shell_started=false' \
        >"${M519_R11_FULL_PATH_SELF_TEST_ROOT}/FULL_PATH_PASS.txt"
    rm -f /tmp/m519_r11_exact_verified.$$.tsv
    trap - EXIT
    exit 0
fi

m519_r11_proc_identity() {
    local pid=$1 raw rest uid exe
    local -a fields
    [[ "${pid}" =~ ^[0-9]+$ && -r "/proc/${pid}/stat" ]] || return 1
    IFS= read -r raw <"/proc/${pid}/stat" || return 1
    [[ "${raw}" == *") "* ]] || return 1
    rest="${raw##*) }"
    read -r -a fields <<<"${rest}"
    [[ "${#fields[@]}" -ge 20 ]] || return 1
    uid="$(stat -Lc %u "/proc/${pid}" 2>/dev/null)" || return 1
    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null)" || exe=UNREADABLE
    M519_R11_PROC_PID=${pid}
    M519_R11_PROC_STATE=${fields[0]}
    M519_R11_PROC_PPID=${fields[1]}
    M519_R11_PROC_STARTTIME=${fields[19]}
    M519_R11_PROC_UID=${uid}
    M519_R11_PROC_EXE=${exe}
    M519_R11_PROC_COMM_HEX="$(od -An -tx1 -v "/proc/${pid}/comm" \
        2>/dev/null | tr -d ' \n')"
    M519_R11_PROC_EXE_HEX="$(printf '%s' "${exe}" | od -An -tx1 -v | tr -d ' \n')"
    M519_R11_PROC_CMDLINE_NUL_HEX="$(od -An -tx1 -v "/proc/${pid}/cmdline" \
        2>/dev/null | tr -d ' \n')"
    return 0
}

# Return 0 only for the exact live tuple, 1 if absent/completed zombie, and 2
# for PID reuse or any birth identity mismatch.  Optional parent and complete
# NUL-safe cmdline pins extend the tuple.  Callers never signal return-2.
m519_r11_root_state() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=${5:-} cmdline_hex=${6:-}
    [[ -e "/proc/${pid}" ]] || return 1
    m519_r11_proc_identity "${pid}" || return 2
    [[ "${M519_R11_PROC_STARTTIME}" == "${start}" && \
       "${M519_R11_PROC_UID}" == "${uid}" ]] || return 2
    [[ -z "${parent}" || "${M519_R11_PROC_PPID}" == "${parent}" ]] || return 2
    [[ "${M519_R11_PROC_STATE}" != Z ]] || return 1
    [[ "${M519_R11_PROC_EXE}" == "${exe}" ]] || return 2
    [[ -z "${cmdline_hex}" || \
       "${M519_R11_PROC_CMDLINE_NUL_HEX}" == "${cmdline_hex}" ]] || return 2
    return 0
}

# Every ancestor is represented by a (pid,starttime) pair, then reread before
# accepting the chain.  This closes intermediate as well as root PID reuse.
m519_r11_pid_is_descendant() {
    local pid=$1 candidate_start=$2 root=$3 root_start=$4
    local guard=0 index current_start parent
    local -a chain_pid=() chain_start=()
    while [[ "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 && \
             "${guard}" -lt 64 ]]; do
        m519_r11_proc_identity "${pid}" || return 2
        current_start=${M519_R11_PROC_STARTTIME}
        [[ "${guard}" -ne 0 || "${current_start}" == "${candidate_start}" ]] \
            || return 2
        chain_pid+=("${pid}"); chain_start+=("${current_start}")
        if [[ "${pid}" -eq "${root}" ]]; then
            [[ "${current_start}" == "${root_start}" ]] || return 2
            for index in "${!chain_pid[@]}"; do
                m519_r11_proc_identity "${chain_pid[${index}]}" || return 2
                [[ "${M519_R11_PROC_STARTTIME}" == \
                   "${chain_start[${index}]}" ]] || return 2
            done
            return 0
        fi
        parent=${M519_R11_PROC_PPID}
        [[ "${parent}" =~ ^[0-9]+$ && "${parent}" -ne "${pid}" ]] || return 2
        pid=${parent}; guard=$((guard + 1))
    done
    return 1
}

m519_r11_external_eda_pids() {
    local root=${1:-} root_start=${2:-} root_uid=${3:-} root_exe=${4:-}
    local root_parent=${5:-} root_cmdline=${6:-} collision_log=$7 label=$8
    local proc pid comm exe_base first=1 state=1 candidate_start rc kind
    local saved_ppid saved_uid saved_start saved_state saved_comm_hex
    local saved_exe_hex saved_cmdline_hex
    if [[ ! -e "${collision_log}" ]]; then
        printf 'timestamp\tlabel\tkind\tpid\tppid\tuid\tstarttime\tstate\tcomm_hex\texe_hex\tcmdline_nul_hex\n' \
            >"${collision_log}"
    fi
    if [[ -n "${root}" ]]; then
        set +e; m519_r11_root_state "${root}" "${root_start}" \
            "${root_uid}" "${root_exe}" "${root_parent}" \
            "${root_cmdline}"; state=$?; set -e
        if [[ "${state}" -eq 2 ]]; then
            printf 'campaign_root_identity_mismatch:%s' "${root}"
            first=0
            if m519_r11_proc_identity "${root}"; then
                printf '%s\t%s\tcampaign_root_identity_mismatch\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date --iso-8601=seconds)" "${label}" \
                    "${M519_R11_PROC_PID}" "${M519_R11_PROC_PPID}" \
                    "${M519_R11_PROC_UID}" "${M519_R11_PROC_STARTTIME}" \
                    "${M519_R11_PROC_STATE}" "${M519_R11_PROC_COMM_HEX}" \
                    "${M519_R11_PROC_EXE_HEX}" \
                    "${M519_R11_PROC_CMDLINE_NUL_HEX}" >>"${collision_log}"
            fi
        fi
    fi
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m519_r11_proc_identity "${pid}" || continue
        [[ "${M519_R11_PROC_UID}" == "${m519_r11_uid}" && \
           "${M519_R11_PROC_STATE}" != Z ]] || continue
        IFS= read -r comm <"/proc/${pid}/comm" 2>/dev/null || continue
        exe_base=${M519_R11_PROC_EXE##*/}
        case "${comm}:${exe_base}" in
            dc_shell:*|dc_shell-t:*|fm_shell:*|pt_shell:*|vcs:*|vcs1:*|vlogan:*|simv:*|common_shell_ex*:common_shell_exec)
                ;;
            *) continue ;;
        esac
        candidate_start=${M519_R11_PROC_STARTTIME}
        saved_ppid=${M519_R11_PROC_PPID}
        saved_uid=${M519_R11_PROC_UID}
        saved_start=${M519_R11_PROC_STARTTIME}
        saved_state=${M519_R11_PROC_STATE}
        saved_comm_hex=${M519_R11_PROC_COMM_HEX}
        saved_exe_hex=${M519_R11_PROC_EXE_HEX}
        saved_cmdline_hex=${M519_R11_PROC_CMDLINE_NUL_HEX}
        kind=external_eda_collision
        if [[ "${state}" -eq 0 ]]; then
            set +e
            m519_r11_pid_is_descendant "${pid}" "${candidate_start}" \
                "${root}" "${root_start}"
            rc=$?
            set -e
            [[ "${rc}" -ne 0 ]] || continue
            [[ "${rc}" -ne 2 ]] || kind=ancestry_identity_mismatch
        fi
        # Reread immediately before emitting the independently reconstructable
        # collision tuple; PID reuse becomes explicit mismatch evidence.
        if ! m519_r11_proc_identity "${pid}" || \
                [[ "${M519_R11_PROC_STARTTIME}" != "${candidate_start}" ]]; then
            kind=collision_identity_changed_before_record
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date --iso-8601=seconds)" "${label}" "${kind}" "${pid}" \
                "${saved_ppid}" "${saved_uid}" "${saved_start}" \
                "${saved_state}" "${saved_comm_hex}" "${saved_exe_hex}" \
                "${saved_cmdline_hex}" >>"${collision_log}"
            [[ "${first}" -eq 1 ]] || printf ','
            printf '%s:%s' "${kind}" "${pid}"
            first=0
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${kind}" \
            "${M519_R11_PROC_PID}" "${M519_R11_PROC_PPID}" \
            "${M519_R11_PROC_UID}" "${M519_R11_PROC_STARTTIME}" \
            "${M519_R11_PROC_STATE}" "${M519_R11_PROC_COMM_HEX}" \
            "${M519_R11_PROC_EXE_HEX}" "${M519_R11_PROC_CMDLINE_NUL_HEX}" \
            >>"${collision_log}"
        [[ "${first}" -eq 1 ]] || printf ','
        printf '%s:%s:%s' "${kind}" "${pid}" "${candidate_start}"
        first=0
    done
}

m519_r11_read_cgroup() {
    M519_R11_CGROUP_FAILCNT="$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)"
    M519_R11_CGROUP_UNDER_OOM="$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
    M519_R11_CGROUP_OOM_KILL="$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)"
}

m519_r11_resource_snapshot() {
    local label=$1 log=$2 h0=${3:-NA} root=${4:-}
    local root_start=${5:-} root_uid=${6:-} root_exe=${7:-}
    local root_parent=${8:-} root_cmdline=${9:-}
    local limit committed delta
    limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
    committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
    M519_R11_HEADROOM_KIB=$((limit - committed))
    M519_R11_MEM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    M519_R11_SWAP_FREE_KIB="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    m519_r11_read_cgroup
    M519_R11_COLLISION="$(m519_r11_external_eda_pids "${root}" \
        "${root_start}" "${root_uid}" "${root_exe}" "${root_parent}" \
        "${root_cmdline}" "${log%.log}_external_collisions.tsv" "${label}")"
    M519_R11_IDENTITY_MISMATCH=0
    [[ "${M519_R11_COLLISION}" != *identity_mismatch* ]] || \
        M519_R11_IDENTITY_MISMATCH=1
    if [[ "${h0}" =~ ^[0-9]+$ ]]; then
        delta=$((h0 - M519_R11_HEADROOM_KIB))
    else
        delta=NA
    fi
    printf 'timestamp=%s label=%s h0_commit_headroom_kib=%s commit_headroom_kib=%s h0_minus_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s external_eda_collision=%s campaign_identity_mismatch=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${h0}" \
        "${M519_R11_HEADROOM_KIB}" "${delta}" \
        "${M519_R11_MEM_AVAILABLE_KIB}" "${M519_R11_SWAP_FREE_KIB}" \
        "${M519_R11_CGROUP_FAILCNT}" "${M519_R11_CGROUP_UNDER_OOM}" \
        "${M519_R11_CGROUP_OOM_KILL}" "${M519_R11_COLLISION:-none}" \
        "${M519_R11_IDENTITY_MISMATCH}" >>"${log}"
}

m519_r11_pid_tree_snapshot() {
    local label=$1 log=$2 proc pid
    printf 'timestamp=%s label=%s\n' "$(date --iso-8601=seconds)" \
        "${label}" >>"${log}"
    printf 'pid\tppid\tuid\tstarttime\tstate\tcomm_hex\texe_hex\tcmdline_nul_hex\n' \
        >>"${log}"
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m519_r11_proc_identity "${pid}" || continue
        [[ "${M519_R11_PROC_UID}" == "${m519_r11_uid}" ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${M519_R11_PROC_PID}" "${M519_R11_PROC_PPID}" \
            "${M519_R11_PROC_UID}" "${M519_R11_PROC_STARTTIME}" \
            "${M519_R11_PROC_STATE}" "${M519_R11_PROC_COMM_HEX}" \
            "${M519_R11_PROC_EXE_HEX}" \
            "${M519_R11_PROC_CMDLINE_NUL_HEX}" >>"${log}"
    done
}

m519_r11_seal_dir() {
    local dir=$1
    (
        cd "${dir}"
        find . -type f ! -path './SHA256SUMS' \
            ! -path './SHA256SUMS.seal.sha256' -print0 | sort -z | \
            xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}

m519_r11_axis_preflight() {
    local axis=$1 dir=$2 sample pass=1 h0=0
    mkdir -p "${dir}"
    : >"${dir}/resource_preflight.log"
    : >"${dir}/pid_tree_preflight.log"
    for sample in 1 2 3; do
        m519_r11_resource_snapshot "${axis}_preflight_${sample}" \
            "${dir}/resource_preflight.log" NA
        m519_r11_pid_tree_snapshot "${axis}_preflight_${sample}" \
            "${dir}/pid_tree_preflight.log"
        if [[ "${sample}" -eq 1 || "${M519_R11_HEADROOM_KIB}" -lt "${h0}" ]]; then
            h0=${M519_R11_HEADROOM_KIB}
        fi
        if [[ "${M519_R11_HEADROOM_KIB}" -lt "${m519_r11_preflight_commit_kib}" || \
              "${M519_R11_MEM_AVAILABLE_KIB}" -lt "${m519_r11_mem_available_kib}" || \
              "${M519_R11_SWAP_FREE_KIB}" -lt "${m519_r11_swap_free_kib}" || \
              "${M519_R11_CGROUP_FAILCNT}" -ne 0 || \
              "${M519_R11_CGROUP_UNDER_OOM}" -ne 0 || \
              "${M519_R11_CGROUP_OOM_KILL}" -ne 0 || \
              -n "${M519_R11_COLLISION}" ]]; then
            pass=0
        fi
        [[ "${sample}" -eq 3 ]] || sleep 10
    done
    printf 'axis=%s\nh0_commit_headroom_kib=%s\nsamples=3\nsample_interval_seconds=10\ncommit_headroom_gate_kib=%s\nmem_available_gate_kib=%s\nswap_free_gate_kib=%s\ncgroup_required_zero=true\nsame_uid_external_eda_required_none=true\nstatus=%s\n' \
        "${axis}" "${h0}" "${m519_r11_preflight_commit_kib}" \
        "${m519_r11_mem_available_kib}" "${m519_r11_swap_free_kib}" \
        "$([[ "${pass}" -eq 1 ]] && echo PASS || echo FAIL)" \
        >"${dir}/preflight_receipt.txt"
    m519_r11_seal_dir "${dir}"
    [[ "${pass}" -eq 1 ]]
}

if ! m519_r11_axis_preflight k1 "${m519_r11_preflight_staging}"; then
    printf 'status=PREFLIGHT_REJECTED_NO_DC_ATTEMPT_CONSUMED\n' \
        >"${m519_r11_preflight_staging}/PREFLIGHT_REJECTED.txt"
    m519_r11_seal_dir "${m519_r11_preflight_staging}"
    mv -T "${m519_r11_preflight_staging}" "${m519_r11_preflight_reject}"
    rm -f /tmp/m519_r11_exact_verified.$$.tsv
    exit 40
fi

mkdir "${m519_r11_work}"
mkdir "${m519_r11_work}/preflight"
mv -T "${m519_r11_preflight_staging}" "${m519_r11_work}/preflight/k1"
mv -T /tmp/m519_r11_exact_verified.$$.tsv \
    "${m519_r11_work}/contract_exact_files_verified.tsv"
m519_r11_run_created=1
m519_r11_complete=0
m519_r11_child_pid=""
m519_r11_child_start=""
m519_r11_child_uid=""
m519_r11_child_exe=""
m519_r11_child_parent=""
m519_r11_child_cmdline=""
m519_r11_monitor_pid=""
m519_r11_monitor_start=""
m519_r11_child_rc=not_started
m519_r11_monitor_rc=not_started
m519_r11_signal=none
m519_r11_runtime_latch=0
m519_r11_runtime_latch_reason=none

m519_r11_term_exact() {
    local pid=$1 start=$2 uid=$3 exe=$4 parent=$5 cmdline=$6 signal_name=$7 state
    set +e
    m519_r11_root_state "${pid}" "${start}" "${uid}" "${exe}" \
        "${parent}" "${cmdline}"
    state=$?
    set -e
    if [[ "${state}" -eq 0 ]]; then
        kill -s "${signal_name}" "${pid}" 2>/dev/null || return 1
        return 0
    elif [[ "${state}" -eq 1 ]]; then
        return 0
    fi
    return 2
}

m519_r11_signal_handler() {
    local signal_name=$1 term_rc=0
    m519_r11_signal="${signal_name}"
    if [[ -n "${m519_r11_child_pid}" ]]; then
        set +e
        m519_r11_term_exact "${m519_r11_child_pid}" "${m519_r11_child_start}" \
            "${m519_r11_child_uid}" "${m519_r11_child_exe}" \
            "${m519_r11_child_parent}" "${m519_r11_child_cmdline}" \
            "${signal_name}"
        term_rc=$?
        set -e
    fi
    printf 'timestamp=%s signal=%s child_pid=%s child_starttime=%s exact_term_rc=%s monitor_pid=%s monitor_starttime=%s\n' \
        "$(date --iso-8601=seconds)" "${signal_name}" \
        "${m519_r11_child_pid:-none}" "${m519_r11_child_start:-none}" \
        "${term_rc}" "${m519_r11_monitor_pid:-none}" \
        "${m519_r11_monitor_start:-none}" \
        >>"${m519_r11_work}/signal_provenance.txt"
}
trap 'm519_r11_signal_handler INT' INT
trap 'm519_r11_signal_handler TERM' TERM

m519_r11_failure_cleanup() {
    local rc=$? state term_rc=0
    set +e
    if [[ -n "${m519_r11_child_pid}" ]]; then
        m519_r11_root_state "${m519_r11_child_pid}" "${m519_r11_child_start}" \
            "${m519_r11_child_uid}" "${m519_r11_child_exe}" \
            "${m519_r11_child_parent}" "${m519_r11_child_cmdline}"
        state=$?
        if [[ "${state}" -eq 0 ]]; then
            m519_r11_pid_tree_snapshot failure_before_term \
                "${m519_r11_work}/failure_pid_tree.log"
            m519_r11_term_exact "${m519_r11_child_pid}" "${m519_r11_child_start}" \
                "${m519_r11_child_uid}" "${m519_r11_child_exe}" \
                "${m519_r11_child_parent}" "${m519_r11_child_cmdline}" TERM
            term_rc=$?
            wait "${m519_r11_child_pid}"
            m519_r11_child_rc=$?
        elif [[ "${state}" -eq 2 ]]; then
            term_rc=2
        fi
    fi
    if [[ -n "${m519_r11_monitor_pid}" ]]; then
        wait "${m519_r11_monitor_pid}" 2>/dev/null
        m519_r11_monitor_rc=$?
    fi
    if [[ "${m519_r11_run_created}" -eq 1 && \
          "${m519_r11_complete}" -ne 1 && -d "${m519_r11_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\nchild_exit_code=%s\nmonitor_exit_code=%s\nsignal=%s\nruntime_resource_latch=%s\nruntime_latch_reason=%s\nexact_term_rc=%s\n' \
            "${rc}" "${m519_r11_child_rc}" "${m519_r11_monitor_rc}" \
            "${m519_r11_signal}" "${m519_r11_runtime_latch}" \
            "${m519_r11_runtime_latch_reason}" "${term_rc}" \
            >"${m519_r11_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        m519_r11_seal_dir "${m519_r11_work}"
        mv -T "${m519_r11_work}" "${m519_r11_quarantine}"
        m519_r11_run_created=0
    fi
    return "${rc}"
}
trap m519_r11_failure_cleanup EXIT

mkdir "${m519_r11_work}/.attempt_staging"
printf 'status=CONSUMED_AT_FIRST_DC_LAUNCH\ntimestamp=%s\ncanonical=%s\npreflight_k1_outer_seal_sha256=%s\n' \
    "$(date --iso-8601=seconds)" "${m519_r11_canonical}" \
    "$(m519_r11_sha "${m519_r11_work}/preflight/k1/SHA256SUMS.seal.sha256")" \
    >"${m519_r11_work}/.attempt_staging/ATTEMPT_CONSUMED.txt"
sha256sum "${m519_r11_runner}" "${m519_r11_contract}" \
    "${m519_r11_admission}" >"${m519_r11_work}/.attempt_staging/identity.sha256"
m519_r11_seal_dir "${m519_r11_work}/.attempt_staging"
mv -T "${m519_r11_work}/.attempt_staging" "${m519_r11_attempt}"
m519_r11_attempt_consumed=1

sha256sum "${m519_r11_runner}" "${m519_r11_contract}" \
    "${m519_r11_admission}" "${m519_r11_tcl}" "${m519_r11_filelist}" \
    "${m519_r11_sdc}" "${m519_r11_dc}" "${m519_r11_dc_wrapper}" \
    "${m519_r11_dc_actual_exe}" "${m519_r11_slow}" "${m519_r11_fast}" \
    "${m519_r11_r5_static}/SHA256SUMS.seal.sha256" \
    "${m519_r11_r5_vcs}/SHA256SUMS.seal.sha256" \
    "${m519_r11_r5_vcs_review}/SHA256SUMS.seal.sha256" \
    "${m519_r11_r5_failure}/SHA256SUMS.seal.sha256" \
    "${m519_r11_r5_quarantine}/SHA256SUMS.seal.sha256" \
    "${m519_r11_r6_failed_review}/SHA256SUMS.seal.sha256" \
    "${m519_r11_r7_disqualified_review}/SHA256SUMS.seal.sha256" \
    "${m519_r11_m694}/SHA256SUMS.seal.sha256" \
    "${m519_r11_m701}/SHA256SUMS.seal.sha256" \
    docs/359_DATE终局冻结_20260813.md >"${m519_r11_work}/input_sha256.txt"
cp "${m519_r11_contract}" "${m519_r11_work}/contract.json"

export HW_ROOT="${m519_r11_hw_root}"
export LIB_DB="${m519_r11_slow}"
export MIN_LIB_DB="${m519_r11_fast}"
export SDC_FILE="${m519_r11_hw_root}/${m519_r11_sdc}"
export OPERATING_CONDITION=ssg0p9v125c
export CLOCK_PERIOD_NS=3.000

m519_r11_gate_current_snapshot() {
    local label=$1 point=$2 sample=$3 current_reason=none
    if [[ "${M519_R11_HEADROOM_KIB}" -lt "${m519_r11_runtime_commit_kib}" ]]; then
        M519_R11_COMMIT_BAD_COUNT=$((M519_R11_COMMIT_BAD_COUNT + 1))
    else
        M519_R11_COMMIT_BAD_COUNT=0
    fi
    if [[ "${M519_R11_IDENTITY_MISMATCH}" -ne 0 ]]; then
        current_reason=campaign_pid_identity_mismatch
    elif [[ "${M519_R11_COMMIT_BAD_COUNT}" -ge 3 ]]; then
        current_reason=commit_headroom_below_32gib_for_three_consecutive_samples
    elif [[ "${M519_R11_MEM_AVAILABLE_KIB}" -lt "${m519_r11_mem_available_kib}" ]]; then
        current_reason=mem_available_below_128gib
    elif [[ "${M519_R11_SWAP_FREE_KIB}" -lt "${m519_r11_swap_free_kib}" ]]; then
        current_reason=swap_free_below_32gib
    elif [[ "${M519_R11_CGROUP_FAILCNT}" -ne 0 || \
            "${M519_R11_CGROUP_UNDER_OOM}" -ne 0 || \
            "${M519_R11_CGROUP_OOM_KILL}" -ne 0 ]]; then
        current_reason=cgroup_or_oom_counter_nonzero
    elif [[ -n "${M519_R11_COLLISION}" ]]; then
        current_reason=new_external_same_uid_eda_collision
    fi
    printf 'timestamp=%s label=%s sample=%s commit_bad_consecutive=%s gate_reason=%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${sample}" \
        "${M519_R11_COMMIT_BAD_COUNT}" "${current_reason}" \
        >>"${point}/runtime_gate_every_snapshot.log"
    if [[ "${current_reason}" != none ]]; then
        M519_R11_RUNTIME_FAILED=1
        [[ "${M519_R11_RUNTIME_REASON}" != none ]] || \
            M519_R11_RUNTIME_REASON=${current_reason}
        printf 'timestamp=%s status=RUNTIME_RESOURCE_LATCH reason=%s label=%s sample=%s commit_bad_consecutive=%s\n' \
            "$(date --iso-8601=seconds)" "${current_reason}" "${label}" \
            "${sample}" "${M519_R11_COMMIT_BAD_COUNT}" \
            >>"${point}/runtime_latch.txt"
        return 1
    fi
    return 0
}

m519_r11_record_descendants() {
    local child=$1 child_start=$2 sample=$3 point=$4 proc pid rc key candidate_start
    local vmpeak vmsize vmrss vmswap
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        m519_r11_proc_identity "${pid}" || continue
        candidate_start=${M519_R11_PROC_STARTTIME}
        set +e
        m519_r11_pid_is_descendant "${pid}" "${candidate_start}" \
            "${child}" "${child_start}"
        rc=$?
        set -e
        [[ "${rc}" -eq 0 ]] || {
            if [[ "${rc}" -eq 2 ]]; then
                printf 'timestamp=%s sample=%s pid=%s status=ANCESTRY_IDENTITY_MISMATCH\n' \
                    "$(date --iso-8601=seconds)" "${sample}" "${pid}" \
                    >>"${point}/descendant_identity_faults.log"
                M519_R11_DESCENDANT_IDENTITY_FAULT=1
            fi
            continue
        }
        # The ancestry checker deliberately overwrites its scratch identity
        # while walking the chain.  Reread and revalidate the candidate tuple
        # before recording its own provenance and memory counters.
        if ! m519_r11_proc_identity "${pid}" || \
                [[ "${M519_R11_PROC_STARTTIME}" != "${candidate_start}" ]]; then
            printf 'timestamp=%s sample=%s pid=%s status=CANDIDATE_IDENTITY_CHANGED_BEFORE_RECORD\n' \
                "$(date --iso-8601=seconds)" "${sample}" "${pid}" \
                >>"${point}/descendant_identity_faults.log"
            M519_R11_DESCENDANT_IDENTITY_FAULT=1
            continue
        fi
        vmpeak="$(awk '/^VmPeak:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmpeak=${vmpeak:-0}
        vmsize="$(awk '/^VmSize:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmsize=${vmsize:-0}
        vmrss="$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmrss=${vmrss:-0}
        vmswap="$(awk '/^VmSwap:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"; vmswap=${vmswap:-0}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${sample}" "${M519_R11_PROC_PID}" \
            "${M519_R11_PROC_PPID}" "${M519_R11_PROC_UID}" \
            "${M519_R11_PROC_STARTTIME}" "${M519_R11_PROC_COMM_HEX}" \
            "${M519_R11_PROC_EXE_HEX}" "${M519_R11_PROC_CMDLINE_NUL_HEX}" \
            "${vmpeak}" "${vmsize}" "${vmrss}" "${vmswap}" \
            >>"${point}/descendant_memory_runtime.tsv"
        key="${pid}_${M519_R11_PROC_STARTTIME}"
        M519_R11_HIGH_PID[${key}]=${pid}
        M519_R11_HIGH_START[${key}]=${M519_R11_PROC_STARTTIME}
        M519_R11_HIGH_COMM[${key}]=${M519_R11_PROC_COMM_HEX}
        M519_R11_HIGH_EXE[${key}]=${M519_R11_PROC_EXE_HEX}
        M519_R11_HIGH_CMD[${key}]=${M519_R11_PROC_CMDLINE_NUL_HEX}
        [[ "${vmpeak}" -le "${M519_R11_HIGH_PEAK[${key}]:-0}" ]] || M519_R11_HIGH_PEAK[${key}]=${vmpeak}
        [[ "${vmsize}" -le "${M519_R11_HIGH_SIZE[${key}]:-0}" ]] || M519_R11_HIGH_SIZE[${key}]=${vmsize}
        [[ "${vmrss}" -le "${M519_R11_HIGH_RSS[${key}]:-0}" ]] || M519_R11_HIGH_RSS[${key}]=${vmrss}
        [[ "${vmswap}" -le "${M519_R11_HIGH_SWAP[${key}]:-0}" ]] || M519_R11_HIGH_SWAP[${key}]=${vmswap}
    done
}

m519_r11_runtime_monitor() {
    local child=$1 child_start=$2 child_uid=$3 child_exe=$4
    local child_parent=$5 child_cmdline=$6 h0=$7 point=$8
    local state sample=0 gate_rc=0 key
    M519_R11_RUNTIME_FAILED=0
    M519_R11_RUNTIME_REASON=none
    M519_R11_COMMIT_BAD_COUNT=0
    M519_R11_DESCENDANT_IDENTITY_FAULT=0
    declare -Ag M519_R11_HIGH_PID=() M519_R11_HIGH_START=()
    declare -Ag M519_R11_HIGH_COMM=() M519_R11_HIGH_EXE=() M519_R11_HIGH_CMD=()
    declare -Ag M519_R11_HIGH_PEAK=() M519_R11_HIGH_SIZE=()
    declare -Ag M519_R11_HIGH_RSS=() M519_R11_HIGH_SWAP=()
    : >"${point}/resource_runtime.log"
    : >"${point}/resource_runtime_external_collisions.tsv"
    printf 'timestamp\tlabel\tkind\tpid\tppid\tuid\tstarttime\tstate\tcomm_hex\texe_hex\tcmdline_nul_hex\n' \
        >"${point}/resource_runtime_external_collisions.tsv"
    : >"${point}/runtime_gate_every_snapshot.log"
    : >"${point}/runtime_latch.txt"
    : >"${point}/descendant_identity_faults.log"
    printf 'timestamp\tsample\tpid\tppid\tuid\tstarttime\tcomm_hex\texe_hex\tcmdline_nul_hex\tVmPeak_kib\tVmSize_kib\tVmRSS_kib\tVmSwap_kib\n' \
        >"${point}/descendant_memory_runtime.tsv"

    while true; do
        set +e; m519_r11_root_state "${child}" "${child_start}" \
            "${child_uid}" "${child_exe}" "${child_parent}" \
            "${child_cmdline}"; state=$?; set -e
        [[ "${state}" -eq 0 ]] || break
        sample=$((sample + 1))
        m519_r11_resource_snapshot "runtime_${sample}" \
            "${point}/resource_runtime.log" "${h0}" "${child}" \
            "${child_start}" "${child_uid}" "${child_exe}" \
            "${child_parent}" "${child_cmdline}"
        m519_r11_record_descendants "${child}" "${child_start}" \
            "${sample}" "${point}"
        [[ "${M519_R11_DESCENDANT_IDENTITY_FAULT}" -eq 0 ]] || \
            M519_R11_IDENTITY_MISMATCH=1
        set +e
        m519_r11_gate_current_snapshot "runtime_${sample}" "${point}" "${sample}"
        gate_rc=$?
        set -e
        if [[ "${gate_rc}" -ne 0 ]]; then
            set +e
            m519_r11_term_exact "${child}" "${child_start}" "${child_uid}" \
                "${child_exe}" "${child_parent}" "${child_cmdline}" TERM
            set -e
            break
        fi
        sleep 10
    done

    if [[ "${state}" -eq 2 ]]; then
        M519_R11_RUNTIME_FAILED=1
        M519_R11_RUNTIME_REASON=campaign_pid_identity_mismatch
    fi
    # A latched child must be gone before the synchronous final sample.  The
    # exact tuple is polled; a reused PID is never signalled and is a failure.
    while [[ "${state}" -eq 0 && "${M519_R11_RUNTIME_FAILED}" -ne 0 ]]; do
        sleep 1
        set +e; m519_r11_root_state "${child}" "${child_start}" \
            "${child_uid}" "${child_exe}" "${child_parent}" \
            "${child_cmdline}"; state=$?; set -e
    done
    [[ "${state}" -ne 2 ]] || {
        M519_R11_RUNTIME_FAILED=1
        M519_R11_RUNTIME_REASON=campaign_pid_identity_mismatch
    }

    sample=$((sample + 1))
    m519_r11_resource_snapshot runtime_final \
        "${point}/resource_runtime.log" "${h0}"
    [[ "${state}" -ne 2 ]] || M519_R11_IDENTITY_MISMATCH=1
    set +e
    m519_r11_gate_current_snapshot runtime_final "${point}" "${sample}"
    gate_rc=$?
    set -e

    printf 'pid\tstarttime\tcomm_hex\texe_hex\tcmdline_nul_hex\tVmPeak_kib\tVmSize_kib\tVmRSS_kib\tVmSwap_kib\n' \
        >"${point}/descendant_memory_highwater.tsv"
    for key in "${!M519_R11_HIGH_PID[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${M519_R11_HIGH_PID[${key}]}" "${M519_R11_HIGH_START[${key}]}" \
            "${M519_R11_HIGH_COMM[${key}]}" "${M519_R11_HIGH_EXE[${key}]}" \
            "${M519_R11_HIGH_CMD[${key}]}" "${M519_R11_HIGH_PEAK[${key}]:-0}" \
            "${M519_R11_HIGH_SIZE[${key}]:-0}" "${M519_R11_HIGH_RSS[${key}]:-0}" \
            "${M519_R11_HIGH_SWAP[${key}]:-0}"
    done | sort -n >>"${point}/descendant_memory_highwater.tsv"
    printf 'timestamp=%s final_sample_label=runtime_final final_gate_applied=true child_exact_state=%s commit_below_32gib_consecutive_final=%s runtime_resource_latch=%s reason=%s status=%s\n' \
        "$(date --iso-8601=seconds)" "${state}" \
        "${M519_R11_COMMIT_BAD_COUNT}" "${M519_R11_RUNTIME_FAILED}" \
        "${M519_R11_RUNTIME_REASON}" \
        "$([[ "${M519_R11_RUNTIME_FAILED}" -eq 0 && "${gate_rc}" -eq 0 ]] && \
            echo PASS_FINAL_GATE_ACK || echo FAIL_FINAL_GATE_ACK)" \
        >"${point}/runtime_final_gate_ack.txt"
    printf 'runtime_resource_latch=%s\nreason=%s\ncommit_below_32gib_consecutive_final=%s\nfinal_gate_ack_present=true\n' \
        "${M519_R11_RUNTIME_FAILED}" "${M519_R11_RUNTIME_REASON}" \
        "${M519_R11_COMMIT_BAD_COUNT}" >>"${point}/resource_runtime.log"
    [[ "${M519_R11_RUNTIME_FAILED}" -eq 0 && "${gate_rc}" -eq 0 ]]
}

m519_r11_dc_cmdline_matches() {
    local pid
    local exact_tcl
    local -a argv=()
    pid=$1
    exact_tcl="${m519_r11_hw_root}/${m519_r11_tcl}"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    mapfile -d '' -t argv <"/proc/${pid}/cmdline" || return 1
    [[ "${#argv[@]}" -eq 7 && \
       "${argv[0]}" == "${m519_r11_dc_actual_exe}" && \
       "${argv[1]}" == -shell && "${argv[2]}" == dc_shell && \
       "${argv[3]}" == -r && "${argv[4]}" == "${m519_r11_dc_install_root}" && \
       "${argv[5]}" == -f && "${argv[6]}" == "${exact_tcl}" ]]
}

# Capture succeeds only after the stable wrapper PID has exec'd into the
# frozen common_shell executable and exposes the exact dc_shell selector,
# install root and R8 Tcl argv.  PID birth, UID and parent must remain unchanged
# throughout the wrapper-to-exec transition.
m519_r11_capture_dc_identity() {
    local pid=$1 tries birth_start= birth_uid= birth_parent=
    for tries in $(seq 1 200); do
        m519_r11_proc_identity "${pid}" || return 1
        if [[ -z "${birth_start}" ]]; then
            birth_start=${M519_R11_PROC_STARTTIME}
            birth_uid=${M519_R11_PROC_UID}
            birth_parent=${M519_R11_PROC_PPID}
            m519_r11_child_start=${birth_start}
            m519_r11_child_uid=${birth_uid}
            m519_r11_child_parent=${birth_parent}
        fi
        [[ "${M519_R11_PROC_STARTTIME}" == "${birth_start}" && \
           "${M519_R11_PROC_UID}" == "${birth_uid}" && \
           "${M519_R11_PROC_PPID}" == "${birth_parent}" && \
           "${birth_uid}" == "${m519_r11_uid}" && \
           "${birth_parent}" == "$$" && \
           "${M519_R11_PROC_STATE}" != Z ]] || return 1
        m519_r11_child_exe=${M519_R11_PROC_EXE}
        m519_r11_child_cmdline=${M519_R11_PROC_CMDLINE_NUL_HEX}
        if [[ "${M519_R11_PROC_EXE}" == "${m519_r11_dc_actual_exe}" ]]; then
            m519_r11_dc_cmdline_matches "${pid}" || return 1
            # Reread after argv parsing to close a transition/reuse race.
            m519_r11_proc_identity "${pid}" || return 1
            [[ "${M519_R11_PROC_STARTTIME}" == "${birth_start}" && \
               "${M519_R11_PROC_UID}" == "${birth_uid}" && \
               "${M519_R11_PROC_PPID}" == "${birth_parent}" && \
               "${M519_R11_PROC_EXE}" == "${m519_r11_dc_actual_exe}" ]] || return 1
            m519_r11_dc_cmdline_matches "${pid}" || return 1
            m519_r11_child_exe=${M519_R11_PROC_EXE}
            m519_r11_child_cmdline=${M519_R11_PROC_CMDLINE_NUL_HEX}
            return 0
        fi
        sleep 0.01
    done
    return 1
}

# If stable common_shell capture fails, no runtime monitor is allowed to be
# skipped.  TERM is issued immediately only to the exact fork birth tuple;
# after a bounded grace period KILL is permitted only for that same tuple.
m519_r11_fail_closed_capture() {
    local pid=$1 point=$2 state=1 tries signal_sent=none
    printf 'timestamp=%s status=FAIL_DC_IDENTITY_CAPTURE child_pid=%s frozen_starttime=%s frozen_uid=%s frozen_parent=%s last_exe=%s last_cmdline_nul_hex=%s\n' \
        "$(date --iso-8601=seconds)" "${pid}" \
        "${m519_r11_child_start:-unavailable}" \
        "${m519_r11_child_uid:-unavailable}" \
        "${m519_r11_child_parent:-unavailable}" \
        "${m519_r11_child_exe:-unavailable}" \
        "${m519_r11_child_cmdline:-unavailable}" \
        >"${point}/dc_identity_capture_failure.txt"
    if [[ -n "${m519_r11_child_start}" && -n "${m519_r11_child_uid}" && \
          -n "${m519_r11_child_parent}" ]] && m519_r11_proc_identity "${pid}" && \
            [[ "${M519_R11_PROC_STARTTIME}" == "${m519_r11_child_start}" && \
               "${M519_R11_PROC_UID}" == "${m519_r11_child_uid}" && \
               "${M519_R11_PROC_PPID}" == "${m519_r11_child_parent}" ]]; then
        printf 'term_tuple_exe=%s\nterm_tuple_cmdline_nul_hex=%s\n' \
            "${M519_R11_PROC_EXE}" "${M519_R11_PROC_CMDLINE_NUL_HEX}" \
            >>"${point}/dc_identity_capture_failure.txt"
        kill -TERM "${pid}" 2>/dev/null || true
        signal_sent=TERM
        for tries in $(seq 1 50); do
            [[ -e "/proc/${pid}" ]] || break
            m519_r11_proc_identity "${pid}" || break
            [[ "${M519_R11_PROC_STARTTIME}" == "${m519_r11_child_start}" && \
               "${M519_R11_PROC_UID}" == "${m519_r11_child_uid}" && \
               "${M519_R11_PROC_PPID}" == "${m519_r11_child_parent}" && \
               "${M519_R11_PROC_STATE}" != Z ]] || break
            sleep 0.1
        done
        if m519_r11_proc_identity "${pid}" && \
                [[ "${M519_R11_PROC_STARTTIME}" == "${m519_r11_child_start}" && \
                   "${M519_R11_PROC_UID}" == "${m519_r11_child_uid}" && \
                   "${M519_R11_PROC_PPID}" == "${m519_r11_child_parent}" && \
                   "${M519_R11_PROC_STATE}" != Z ]]; then
            kill -KILL "${pid}" 2>/dev/null || true
            signal_sent=TERM_THEN_KILL
        fi
    fi
    set +e
    wait "${pid}"
    m519_r11_child_rc=$?
    set -e
    printf 'signal_sent=%s\nwait_exit_code=%s\nstatus=QUARANTINE_REQUIRED_NO_MONITOR_BYPASS\n' \
        "${signal_sent}" "${m519_r11_child_rc}" \
        >>"${point}/dc_identity_capture_failure.txt"
}

m519_r11_run_point() {
    local id
    local mode
    local point
    local h0
    local state
    id=$1
    mode=$2
    point="${m519_r11_work}/${id}"
    h0="$(awk -F= '/^h0_commit_headroom_kib=/ {print $2}' \
        "${m519_r11_work}/preflight/${id}/preflight_receipt.txt")"
    mkdir "${point}"
    export DESIGN_NAME=m519_fc2_registered_release_matched_8bank_raw4_acc24
    export RTL_FILELIST="${m519_r11_hw_root}/${m519_r11_filelist}"
    export OUTPUT_DIR="${point}"
    export ELAB_PARAMETERS="ARCH_MODE=${mode}"
    m519_r11_child_pid=""; m519_r11_child_start=""; m519_r11_child_uid=""
    m519_r11_child_exe=""; m519_r11_child_parent=""; m519_r11_child_cmdline=""
    m519_r11_monitor_pid=""; m519_r11_monitor_start=""
    m519_r11_child_rc=running; m519_r11_monitor_rc=running
    set +e
    "${m519_r11_dc}" -f "${m519_r11_hw_root}/${m519_r11_tcl}" \
        >"${point}/dc.log" 2>&1 &
    m519_r11_child_pid=$!
    m519_r11_capture_dc_identity "${m519_r11_child_pid}"
    state=$?
    if [[ "${state}" -ne 0 ]]; then
        m519_r11_fail_closed_capture "${m519_r11_child_pid}" "${point}"
        return 47
    fi
    printf 'timestamp=%s axis=%s child_pid=%s child_starttime=%s child_uid=%s child_parent=%s child_exe=%s child_cmdline_nul_hex=%s runner_pid=%s h0_commit_headroom_kib=%s\n' \
        "$(date --iso-8601=seconds)" "${id}" "${m519_r11_child_pid}" \
        "${m519_r11_child_start}" "${m519_r11_child_uid}" \
        "${m519_r11_child_parent}" "${m519_r11_child_exe}" \
        "${m519_r11_child_cmdline}" "$$" "${h0}" \
        >"${point}/launch_pid_tree_root.txt"
    m519_r11_runtime_monitor "${m519_r11_child_pid}" "${m519_r11_child_start}" \
        "${m519_r11_child_uid}" "${m519_r11_child_exe}" \
        "${m519_r11_child_parent}" "${m519_r11_child_cmdline}" \
        "${h0}" "${point}" &
    m519_r11_monitor_pid=$!
    if m519_r11_proc_identity "${m519_r11_monitor_pid}"; then
        m519_r11_monitor_start=${M519_R11_PROC_STARTTIME}
    else
        m519_r11_monitor_start=unavailable
    fi
    printf 'monitor_pid=%s\nmonitor_starttime=%s\nmonitor_launch_liveness=%s\n' \
        "${m519_r11_monitor_pid}" "${m519_r11_monitor_start}" \
        "$([[ -e "/proc/${m519_r11_monitor_pid}" ]] && echo ALIVE || echo EXITED_EARLY)" \
        >>"${point}/launch_pid_tree_root.txt"
    wait "${m519_r11_child_pid}"
    m519_r11_child_rc=$?
    wait "${m519_r11_monitor_pid}"
    m519_r11_monitor_rc=$?
    set -e
    printf '%s\n' "${m519_r11_child_rc}" >"${point}/dc.rc"
    printf '%s\n' "${m519_r11_monitor_rc}" >"${point}/runtime_monitor.rc"
    m519_r11_child_pid=""; m519_r11_child_start=""; m519_r11_child_uid=""
    m519_r11_child_exe=""; m519_r11_child_parent=""; m519_r11_child_cmdline=""
    m519_r11_monitor_pid=""; m519_r11_monitor_start=""

    [[ "${m519_r11_signal}" == none ]] || return 130
    [[ -s "${point}/runtime_final_gate_ack.txt" ]] || return 42
    grep -Fq 'final_gate_applied=true' "${point}/runtime_final_gate_ack.txt" || return 42
    grep -Fq 'status=PASS_FINAL_GATE_ACK' "${point}/runtime_final_gate_ack.txt" || return 42
    [[ "${m519_r11_monitor_rc}" -eq 0 ]] || {
        m519_r11_runtime_latch=1
        m519_r11_runtime_latch_reason="$(awk -F= '/^reason=/ {print $2}' \
            "${point}/resource_runtime.log" | tail -1)"
        return 42
    }
    [[ "${m519_r11_child_rc}" -eq 0 ]] || return "${m519_r11_child_rc}"
    [[ -s "${point}/TCL_PASS_TERMINAL.txt" ]] || return 43
    grep -Fxq 'status=PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL' \
        "${point}/TCL_PASS_TERMINAL.txt" || return 43
    grep -Fxq 'compile_ultra_count=1' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'incremental_compile_count=0' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'hold_optimization_count=0' "${point}/TCL_PASS_TERMINAL.txt"
    grep -Fxq 'hold_not_closed_at_dc=true' "${point}/TCL_PASS_TERMINAL.txt"
    [[ ! -e "${point}/TCL_EXPLICIT_FAILURE.txt" ]] || return 43
    grep -Fxq 'TIM-209=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'OPT-150=0' "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
        "${point}/reports/precompile_loop_gate.rpt"
    grep -Fxq 'flow=m519_r8_setup_area_only' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'compile_ultra_count=1' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'incremental_compile_count=0' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'hold_fix_command_count=0' "${point}/reports/flow_contract.rpt"
    grep -Fxq 'hold_only_optimization_count=0' "${point}/reports/flow_contract.rpt"
    ! grep -Eq '^(Warning|Information):.*\((TIM-209|OPT-150)\)|^Error:|^Fatal:' \
        "${point}/dc.log" || return 44
    for report in area.rpt qor.rpt timing_setup.rpt \
            timing_hold_diagnostic.rpt constraint_setup.rpt \
            constraint_hold_diagnostic.rpt constraint_max_capacitance.rpt \
            constraint_max_transition.rpt constraint_max_fanout.rpt \
            check_design_postcompile.rpt check_timing_postcompile.rpt \
            flow_contract.rpt compile_receipt.rpt; do
        [[ -s "${point}/reports/${report}" ]] || return 45
    done
    [[ -s "${point}/netlist/m519_fc2_registered_release_matched_8bank_raw4_acc24_mapped.v" ]] || return 45
    ! grep -Fq 'slack (VIOLATED)' "${point}/reports/timing_setup.rpt" || return 46
    for report in constraint_setup.rpt constraint_max_capacitance.rpt \
            constraint_max_transition.rpt constraint_max_fanout.rpt; do
        grep -Fq 'This design has no violated constraints.' \
            "${point}/reports/${report}" || return 46
    done
    printf 'status=PASS_M519_R11_%s_SETUP_AREA_LOGIC_ONLY_DC_3NS_PENDING_RECEIPT_REVIEW\nmacro_count=0\nhold_not_closed_at_dc=true\npaper_ppa_ready=false\nsystem_speedup=false\nheadline=false\n' \
        "${id^^}" >"${point}/RUN_COMPLETE.txt"
}

m519_r11_run_point k1 0
m519_r11_axis_preflight k8 "${m519_r11_work}/preflight/k8" || exit 40
m519_r11_run_point k8 1
m519_r11_axis_preflight k1x8 "${m519_r11_work}/preflight/k1x8" || exit 40
m519_r11_run_point k1x8 2
m519_r11_axis_preflight post_k1x8_recovery \
    "${m519_r11_work}/preflight/post_k1x8_recovery" || exit 40

printf 'status=PASS_M519_R11_THREE_AXIS_SETUP_AREA_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RECEIPT_REVIEW\nhold_not_closed_at_dc=true\npaper_ppa_ready=false\nsystem_speedup=false\nheadline=false\n' \
    >"${m519_r11_work}/RUN_COMPLETE.txt"
m519_r11_seal_dir "${m519_r11_work}"
mv -T "${m519_r11_work}" "${m519_r11_canonical}"
m519_r11_run_created=0
m519_r11_complete=1
trap - EXIT INT TERM
echo "PASS M519 R11 raw setup/area DC result sealed at ${m519_r11_canonical}"
