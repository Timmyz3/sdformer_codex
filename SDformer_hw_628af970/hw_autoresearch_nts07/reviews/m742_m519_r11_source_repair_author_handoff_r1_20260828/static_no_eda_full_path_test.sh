#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
runner="${hw_root}/dc_handoff/scripts/run_dc_m519_r11_setup_area_three_axis_exact_sha_r1.sh"
contract="${hw_root}/contracts/m519_r11_setup_area_three_axis_recovery_contract_r1_20260828.json"
candidate="${hw_root}/contracts/m742_m519_r11_setup_area_three_axis_dc_launch_admission_candidate_r1_20260828.json"
r10_failure="${hw_root}/dc_handoff/runs/m519_r10_pre_attempt_shell_failure.693765.receipt"
m740="${hw_root}/reviews/m740_m519_r10_pre_eda_shell_failure_fresh_hammer_r1_20260828"
runs="${hw_root}/dc_handoff/runs"

runner_sha=7c588b1a95a0afb075de97d148b5a07bad9dc2040ab890c7eb00f6c507ff6692
contract_sha=6d9f30852e4afec80384417fa8bd01d561101846a6b88079cff6ea8088e11334
candidate_sha=9e6b5de45d26a133a08b05caa60889a10c34aa497af426d8bc3bd35580e1da1b

expect_sha() {
    local expected=$1
    local path=$2
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

verify_dir_seal() {
    local sealed=$1
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

verify_file_seal() {
    local payload=$1
    local dir base
    dir="$(dirname "${payload}")"
    base="$(basename "${payload}")"
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}

assert_r11_unconsumed() {
    [[ ! -e "${runs}/m519_r11_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260828" ]]
    [[ ! -e "${runs}/.m519_r11_channel_local_fault_dc_attempt_consumed" ]]
    [[ -z "$(find "${runs}" -maxdepth 1 -name '.m519_r11_channel_local_fault_dc_work.*' -print -quit)" ]]
    [[ -z "$(find "${runs}" -maxdepth 1 -name '.m519_r11_preflight.*.staging' -print -quit)" ]]
    [[ -z "$(find "${runs}" -maxdepth 1 -name 'm519_r11_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r1_20260828.preflight_rejected.*' -print -quit)" ]]
}

bash -n "${runner}"
expect_sha "${runner_sha}" "${runner}"
expect_sha "${contract_sha}" "${contract}"
expect_sha "${candidate_sha}" "${candidate}"
verify_file_seal "${contract}"
verify_file_seal "${candidate}"
verify_dir_seal "${r10_failure}"
verify_dir_seal "${m740}"

jq -e '.schema == "m519_r11_setup_area_three_axis_recovery_contract_r1_v1"
       and .status == "AUTHOR_SOURCE_ONLY_COMPLETE__FRESH_INDEPENDENT_STATIC_HAMMER_REQUIRED__NO_EDA_AUTHORIZED"
       and .setup_area_flow.runner_sha256 == $runner
       and .exact_files[.setup_area_flow.runner] == $runner
       and .r11_repair_provenance.r11_is_additive == true
       and .authorization.run_dc_now == false' \
    --arg runner "${runner_sha}" "${contract}" >/dev/null
jq -e '.schema == "m742_m519_r11_setup_area_three_axis_dc_launch_admission_candidate_r1_v1"
       and .status == "READY_FOR_FRESH_INDEPENDENT_STATIC_HAMMER__NO_EDA_AUTHORIZED"
       and .launch_now == false
       and .identity.dc_runner_sha256 == $runner
       and .identity.recovery_contract_sha256 == $contract
       and .source_only_authorization.run_dc_now == false
       and .source_only_authorization.create_true_release_now == false' \
    --arg runner "${runner_sha}" --arg contract "${contract_sha}" \
    "${candidate}" >/dev/null
[[ "$(jq -cS '.r10_repair_provenance' "${contract}")" == \
   "$(jq -cS '.r10_repair_provenance' "${candidate}")" ]]
[[ "$(jq -cS '.r11_repair_provenance' "${contract}")" == \
   "$(jq -cS '.r11_repair_provenance' "${candidate}")" ]]

expect_sha d96f2a17cd77aab31db8828fb4f729a6ce0825ae4cff0349b655d636af1f58d3 \
    "${r10_failure}/FAILURE.txt"
expect_sha 34e3049ebb6dc29dba5daf7ca8102ef27e82f7b4debfa821014e326ace52d97e \
    "${r10_failure}/SHA256SUMS"
expect_sha da19a42e10b299c22d700ac41c31842a883d33be141c883aad0e05799b9139d0 \
    "${r10_failure}/SHA256SUMS.seal.sha256"
expect_sha 90eb7821899e288095c9e3ace7da3eb7e044f14915f8cfe7b4161a9874d17b9c \
    "${m740}/review.json"
expect_sha e775f74cf6215d47f1a97fa177abd109188c2cd8a773810401539a7c47dc6284 \
    "${m740}/SHA256SUMS"
expect_sha 9d31519366b2d34cbd80e6e6901bcbe17fa824bc331c9c5170d55b269a12a3ef \
    "${m740}/SHA256SUMS.seal.sha256"
grep -Fxq 'exit_code=3' "${r10_failure}/FAILURE.txt"
grep -Fxq 'attempt_consumed=false' "${r10_failure}/FAILURE.txt"
jq -e '.status == "PASS_FAILURE_AUDIT__R10_BLOCKED__PRE_EDA_JQ_ESCAPE__ADDITIVE_R11_REQUIRED"
       and .authorization.run_r11_now == false
       and .authorization.run_dc == false' "${m740}/review.json" >/dev/null

# Regression for the exact R10 bug: the repaired predicate is parsed and true.
jq -e '.verdict == "PASS" and .score_out_of_100 == 100
       and .severity_counts == {"p0":0,"p1":0,"p2":0}' \
    "${hw_root}/reviews/m576_m519_r8_dc_launch_admission_candidate_hammer_r1_20260828/review.json" \
    >/dev/null
! awk 'BEGIN { in_jq=0; bad=0 }
       /jq -e '\''/ { in_jq=1 }
       in_jq && /\\$/ { bad=1 }
       in_jq && /'\'' \\$/ { in_jq=0 }
       END { exit bad ? 0 : 1 }' "${runner}"

[[ ! -e "${hw_root}/contracts/m519_r11_setup_area_three_axis_dc_launch_admission_r1_20260828.json" ]]
assert_r11_unconsumed

tmp_root="$(mktemp -d /tmp/m519_r11_full_path_selftest.XXXXXX)"
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    M519_R11_NO_EDA_FULL_PATH_SELF_TEST=1 \
    M519_R11_FULL_PATH_SELF_TEST_ROOT="${tmp_root}" \
    M519_R11_EXPECTED_DC_RUNNER_SHA256="${runner_sha}" \
    M519_R11_EXPECTED_DC_LAUNCH_ADMISSION_SHA256="${candidate_sha}" \
    bash "${runner}"
grep -Fxq 'status=PASS_M519_R11_FULL_ADMISSION_CONTRACT_PATH_NO_EDA' \
    "${tmp_root}/FULL_PATH_PASS.txt"
grep -Fxq 'preflight_started=false' "${tmp_root}/FULL_PATH_PASS.txt"
grep -Fxq 'attempt_consumed=false' "${tmp_root}/FULL_PATH_PASS.txt"
grep -Fxq 'dc_shell_started=false' "${tmp_root}/FULL_PATH_PASS.txt"
assert_r11_unconsumed

find "${tmp_root}" -type f -delete
find "${tmp_root}" -depth -type d -empty -delete

expect_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
    "${hw_root}/docs/359_DATE终局冻结_20260813.md"

printf '%s\n' \
    'PASS bash_n_and_exact_source_hashes' \
    'PASS contract_candidate_double_seals_and_closed_provenance' \
    'PASS r10_failure_and_m740_exact_hash_status_double_seals' \
    'PASS repaired_m576_jq_predicate_rc0' \
    'PASS full_admission_contract_path_no_eda_before_preflight' \
    'PASS r11_canonical_attempt_work_preflight_absent' \
    'PASS final_launch_admission_absent' \
    'PASS docs359_immutable' \
    'PASS no_dc_vcs_pt_ptpx_formality_or_remote_run'
