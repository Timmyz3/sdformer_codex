#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
runner="${hw_root}/dc_handoff/scripts/run_dc_m519_r10_setup_area_three_axis_exact_sha_r3.sh"
contract="${hw_root}/contracts/m519_r10_setup_area_three_axis_recovery_contract_r3_20260828.json"
admission="${hw_root}/contracts/m519_r10_setup_area_three_axis_dc_launch_admission_r3_20260828.json"
m694="${hw_root}/reviews/m694_m519_r9_three_axis_dc_release_fresh_hammer_r1_20260828"
m701="${hw_root}/reviews/m701_m519_r9_pre_eda_shell_failure_receipt_r1_20260828"
runs="${hw_root}/dc_handoff/runs"

expect_sha() {
    local expected
    local path
    expected=$1
    path=$2
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

verify_dir_seal() {
    local sealed
    sealed=$1
    (cd "${sealed}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

verify_file_seal() {
    local payload
    local base
    local dir
    payload=$1
    base="$(basename "${payload}")"
    dir="$(dirname "${payload}")"
    (cd "${dir}" && sha256sum -c "${base}.sha256" >/dev/null && \
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null)
}

assert_r10_unconsumed() {
    [[ ! -e "${runs}/m519_r10_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r3_20260828" ]]
    [[ ! -e "${runs}/.m519_r10_channel_local_fault_dc_attempt_consumed" ]]
    [[ -z "$(find "${runs}" -maxdepth 1 -name '.m519_r10_channel_local_fault_dc_work.*' -print -quit)" ]]
    [[ -z "$(find "${runs}" -maxdepth 1 -name '.m519_r10_preflight.*.staging' -print -quit)" ]]
}

bash -n "${runner}"
expect_sha 7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f "${runner}"
expect_sha 2ba563ed4c3ddb2c89d0a13855bb4b11be7522aef505cfe1ef374a33b5501a4e "${contract}"
expect_sha f4bccc501dea216396d2755ef6b1f627209efe18346701cd5d448367cf4a3424 "${admission}"
verify_file_seal "${contract}"
verify_file_seal "${admission}"

jq -e '.status == "AUTHOR_SOURCE_ONLY_COMPLETE__FRESH_INDEPENDENT_STATIC_HAMMER_REQUIRED__NO_EDA_AUTHORIZED"
       and .setup_area_flow.runner_sha256 == "7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f"
       and .exact_files[.setup_area_flow.runner] == .setup_area_flow.runner_sha256
       and .claim_boundary.r10_pre_eda_shell_repair_authored == true
       and .authorization.author_ran_eda == false
       and .authorization.run_dc_now == false' "${contract}" >/dev/null
jq -e '.status == "AUTHORIZED_ONE_M519_R10_THREE_AXIS_SETUP_AREA_DC_ATTEMPT_R3"
       and .authorization == {"max_attempts":1,"run_dc":true,"run_formality":false,"run_pt":false,"run_ptpx":false,"run_remote":false,"run_vcs":false}
       and .identity.dc_runner_sha256 == "7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f"
       and .identity.recovery_contract_sha256 == "2ba563ed4c3ddb2c89d0a13855bb4b11be7522aef505cfe1ef374a33b5501a4e"
       and .unique_attempt.release_creation_consumes_attempt == false' "${admission}" >/dev/null
[[ "$(jq -cS '.r10_repair_provenance' "${contract}")" == \
   "$(jq -cS '.r10_repair_provenance' "${admission}")" ]]

verify_dir_seal "${m694}"
verify_dir_seal "${m701}"
expect_sha 8026ceb19d39c7065204366d58a3f823de0d677940d8c191a00ebb2afa1dac0b "${m694}/review.json"
expect_sha cbd6789599917782a2e7156b2ccc60a5bae17ceda14ffcdf6105766d947444b6 "${m694}/SHA256SUMS"
expect_sha c6903561cf29fc3682da0d079058798380b50354e012d33f50463f8e6d015918 "${m694}/SHA256SUMS.seal.sha256"
expect_sha 8bf29fa5c5cf7e7b9993ff86c576ca7a3b22a0b58a6b49363b8be3e945ad33a4 "${m701}/review.json"
expect_sha 7033315764fbb96c34f0f936d604aaf86c49b48d58d29e7bbfd17177336ce87f "${m701}/SHA256SUMS"
expect_sha ff8a1161ecff00b043bda44749a33c993ff6c1817e70ceb8f866d0996d26d0f4 "${m701}/SHA256SUMS.seal.sha256"
[[ "$(jq -r '.status' "${m694}/review.json")" == \
   "GO_ONE_M519_R9_DC_ONLY_ATTEMPT__FINAL_LIVE_RECHECK_REQUIRED" ]]
[[ "$(jq -r '.status' "${m701}/review.json")" == \
   "PRE_EDA_SHELL_FAILURE__NO_DC_STARTED__M519_R9_NOT_CITABLE__ADDITIVE_R10_REQUIRED" ]]

while IFS=$'\t' read -r relative expected; do
    expect_sha "${expected}" "${hw_root}/${relative}"
done < <(jq -r '.exact_files | to_entries[] | [.key,.value] | @tsv' "${contract}")

expect_sha 0a1e1b0d2b391e45c43e0ec337a0b1114a407fc94be0d3d0ce37e103986e909c \
    "${hw_root}/dc_handoff/scripts/run_dc_m519_r9_setup_area_three_axis_exact_sha.sh"
expect_sha 74b13288e9bd13aa07feb68abc9f1f95b5255962bc80a6a1f759103f2608bf41 \
    "${hw_root}/contracts/m519_r9_setup_area_three_axis_recovery_contract_r2_20260828.json"
expect_sha 608a4afb0fe5a706a0f90700b2967231a5aeb3ef3ee6714a99f43d260d6242d3 \
    "${hw_root}/contracts/m519_r9_setup_area_three_axis_dc_launch_admission_r2_20260828.json"
expect_sha 426acd92672037dcab072c98fa3183bbb953cc35924adc26499cf82b1ba439ba \
    "${hw_root}/contracts/m519_r8_setup_area_three_axis_dc_launch_admission_r1_20260827.json"
expect_sha c52b3c34d0cf98ab5f8c526e2ca0a2c869ebc700115e5664dc3f9a90f84e021e \
    "${hw_root}/reviews/m580_m519_r8_dc_final_launch_release_hammer_r1_20260828/review.json"
expect_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
    "${hw_root}/docs/359_DATE终局冻结_20260813.md"
[[ ! -e "${runs}/m519_r9_channel_local_fault_three_axis_setup_area_logic_only_dc_3p000ns_r2_20260828" ]]
[[ ! -e "${runs}/.m519_r9_channel_local_fault_dc_attempt_consumed" ]]
assert_r10_unconsumed

# Static regression for both same-command local dependencies found during R9
# and R10 authoring.  The historical R9 source is immutable; only R10 is gated.
! grep -F 'local payload=$1 sidecar=${payload}.sha256' "${runner}" >/dev/null
! grep -F 'local id=$1 mode=$2 point="${m519_r10_work}/${id}"' "${runner}" >/dev/null

tmp_root="$(mktemp -d /tmp/m519_r10_static_no_eda.XXXXXX)"
ok_root="${tmp_root}/success"
fail_root="${tmp_root}/injected_failure"
mkdir "${ok_root}" "${fail_root}"
env -i PATH="${PATH}" M519_R10_NO_EDA_SELF_TEST=1 \
    M519_R10_SELF_TEST_ROOT="${ok_root}" bash "${runner}"
set +e
env -i PATH="${PATH}" M519_R10_NO_EDA_SELF_TEST=1 \
    M519_R10_SELF_TEST_ROOT="${fail_root}" \
    M519_R10_SELF_TEST_INJECT_PRE_ATTEMPT_FAILURE=1 bash "${runner}"
injected_rc=$?
set -e
[[ "${injected_rc}" -eq 86 ]]
receipt="$(find "${fail_root}" -mindepth 1 -maxdepth 1 -type d \
    -name 'm519_r10_pre_attempt_shell_failure.*.receipt' -print -quit)"
[[ -n "${receipt}" ]]
verify_dir_seal "${receipt}"
grep -Fx 'status=PRE_ATTEMPT_SHELL_FAILURE__NO_EDA_RESULT_ADMITTED' \
    "${receipt}/FAILURE.txt" >/dev/null
grep -Fx 'exit_code=86' "${receipt}/FAILURE.txt" >/dev/null
grep -Fx 'attempt_consumed=false' "${receipt}/FAILURE.txt" >/dev/null
assert_r10_unconsumed
[[ "${tmp_root}" == /tmp/m519_r10_static_no_eda.* ]]
rm -rf -- "${tmp_root}"

printf '%s\n' \
    'PASS bash_n' \
    'PASS runner_contract_admission_exact_sha_and_double_seals' \
    'PASS contract_exact_files_and_closed_repair_provenance' \
    'PASS m694_go_and_m701_pre_eda_failure_exact_sha_status_double_seals' \
    'PASS immutable_r9_r8_m580_docs359_hashes' \
    'PASS r9_and_r10_canonical_attempt_absent' \
    'PASS set_u_compound_local_regressions' \
    'PASS no_eda_selftest_success' \
    'PASS injected_pre_attempt_failure_rc86_double_sealed' \
    'PASS no_dc_vcs_pt_fm_invoked_by_author_test'
