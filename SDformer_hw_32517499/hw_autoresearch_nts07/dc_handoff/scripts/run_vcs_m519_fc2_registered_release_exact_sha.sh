#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${task_hw_root}/results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827"
task_negative_run="${task_hw_root}/results/m519_fc2_registered_release_vcs_r3_negative_preflight_r1_20260827"
task_failed_run="${task_hw_root}/results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r1_20260827"
task_vcs="/opt/synopsys/vcs/V-2023.12-SP1"
task_contract_rel="contracts/m519_fc2_registered_release_three_axis_recovery_contract_r3_20260827.json"
task_contract="${task_hw_root}/${task_contract_rel}"
task_static_review_dir="${task_hw_root}/reviews/m519_registered_release_static_hammer_r3_20260827"
task_static_review_seal="${task_static_review_dir}/SHA256SUMS.seal.sha256"
task_static_review_verdict="${task_static_review_dir}/m519_registered_release_static_hammer_verdict_r3.json"
task_static_review_identity="${task_static_review_dir}/evidence_identity.sha256"
task_failure_review_dir="${task_hw_root}/reviews/m519_registered_release_vcs_failure_hammer_r1_20260827"
task_failure_review_seal="${task_failure_review_dir}/SHA256SUMS.seal.sha256"
task_failure_review_verdict="${task_failure_review_dir}/m519_registered_release_vcs_failure_hammer_verdict_r1.json"
task_failure_review_identity="${task_failure_review_dir}/evidence_identity.sha256"
task_mode="${M519_VCS_MODE:-positive}"

[[ -n "${M519_EXPECTED_VCS_RUNNER_SHA256:-}" && \
   "$(sha256sum "${task_runner}" | awk '{print $1}')" == \
   "${M519_EXPECTED_VCS_RUNNER_SHA256}" ]] || {
    echo "M519 caller must pin the independently reviewed VCS runner SHA" >&2
    exit 3
}
[[ -n "${M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256:-}" ]] || {
    echo "M519 r3 P0=0 static-hammer seal is not caller-pinned" >&2
    exit 3
}
[[ -f "${task_static_review_seal}" && -f "${task_static_review_verdict}" && \
   -f "${task_static_review_identity}" ]] || {
    echo "M519 sealed r3 static hammer is absent; VCS remains unauthorized" >&2
    exit 3
}
[[ "$(sha256sum "${task_static_review_seal}" | awk '{print $1}')" == \
   "${M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256}" ]] || {
    echo "M519 r3 static-hammer outer-seal file SHA mismatch" >&2
    exit 3
}
(cd "${task_static_review_dir}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || {
    echo "M519 r3 static-hammer double-seal verification failed" >&2
    exit 3
}
jq -e '.p0_count == 0
       and .authorization.run_vcs == true
       and .authorization.run_dc == false' \
    "${task_static_review_verdict}" >/dev/null || {
    echo "M519 r3 static hammer does not authorize one VCS while blocking DC" >&2
    exit 3
}
task_contract_expected_sha="$(awk -v path="${task_contract_rel}" \
    '$2 == path {print $1}' "${task_static_review_identity}")"
[[ "${task_contract_expected_sha}" =~ ^[0-9a-f]{64}$ && \
   -f "${task_contract}" && \
   "$(sha256sum "${task_contract}" | awk '{print $1}')" == \
   "${task_contract_expected_sha}" ]] || {
    echo "M519 r3 static hammer did not bind the superseding contract" >&2
    exit 3
}
grep -Fqx \
    "${M519_EXPECTED_VCS_RUNNER_SHA256}  dc_handoff/scripts/run_vcs_m519_fc2_registered_release_exact_sha.sh" \
    "${task_static_review_identity}" || {
    echo "M519 r3 static hammer did not bind this VCS runner identity" >&2
    exit 3
}
jq -e --arg runner_sha "${M519_EXPECTED_VCS_RUNNER_SHA256}" '
       .authorization.run_vcs_now == false
       and .authorization.run_dc == false
       and .authorization.one_r3_vcs_transition.required_p0_count == 0
       and .vcs_runner.path == "dc_handoff/scripts/run_vcs_m519_fc2_registered_release_exact_sha.sh"
       and .vcs_runner.sha256 == $runner_sha' "${task_contract}" >/dev/null || {
    echo "M519 r3 contract does not bind this runner and VCS-only transition" >&2
    exit 3
}

[[ -f "${task_failure_review_seal}" && \
   "$(sha256sum "${task_failure_review_seal}" | awk '{print $1}')" == \
   "0e74bdb5cf1672d9237eae588e5e01ef2a77452b949754344f52c1bd3e6884e6" ]] || {
    echo "M519 sealed r2 VCS failure-hammer identity mismatch" >&2
    exit 3
}
(cd "${task_failure_review_dir}" && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || {
    echo "M519 r2 VCS failure-hammer double-seal verification failed" >&2
    exit 3
}
jq -e '.verdict == "PASS_FAILURE_ATTRIBUTION__TWO_RUNNER_COVER_DOMAIN_MISMATCHES__PRIMARY_SIM_DIAGNOSTIC_ONLY__R3_REAUTH_CONDITIONAL"
       and .authorization.run_vcs == false
       and .authorization.run_dc == false
       and .r3_minimum_repair.remove_candidate_service_cp_protocol_fault_rise_gate == true
       and .r3_minimum_repair.remove_m499_cp_retire_then_slot_reuse_gate == true' \
    "${task_failure_review_verdict}" >/dev/null || exit 3
(cd "${task_hw_root}" && \
    awk '$2 ~ /^results\/m519_fc2_registered_release_k1_vs_k1x8_vcs_r1_20260827\//' \
        "${task_failure_review_identity}" | sha256sum -c - >/dev/null) || {
    echo "M519 r2 diagnostic selected-file identity drift" >&2
    exit 3
}
task_failed_tree_fingerprint="$(cd "${task_failed_run}" && \
    find . -type f -print0 | sort -z | xargs -0 sha256sum | \
    sha256sum | awk '{print $1}')"
[[ "${task_failed_tree_fingerprint}" == \
   "8e66bed323b58ea5d89cea5d2a3697c19a0004521ba588442dd1cc9ae686b03f" ]] || {
    echo "M519 r2 diagnostic failure tree was modified" >&2
    exit 3
}

[[ ! -e "${task_run}" ]] || {
    echo "M519 refuses to overwrite ${task_run}" >&2
    exit 2
}

if [[ "${task_mode}" == negative_preflight ]]; then
    [[ ! -e "${task_negative_run}" ]] || {
        echo "M519 refuses to overwrite ${task_negative_run}" >&2
        exit 2
    }
    mkdir "${task_negative_run}"
    task_negative_complete=0
    task_negative_cleanup() {
        local task_rc=$?
        if [[ "${task_negative_complete}" -ne 1 ]]; then
            printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
                "${task_rc}" >"${task_negative_run}/RUN_FAILED_OR_INCOMPLETE.txt"
        fi
        return "${task_rc}"
    }
    trap task_negative_cleanup EXIT
    task_wrong_sha=0000000000000000000000000000000000000000000000000000000000000000
    set +e
    M519_EXPECTED_VCS_RUNNER_SHA256="${task_wrong_sha}" \
    M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256="${M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256}" \
    M519_VCS_MODE=positive "${task_runner}" \
        >"${task_negative_run}/child.stdout" \
        2>"${task_negative_run}/child.stderr"
    task_negative_rc=$?
    set -e
    [[ "${task_negative_rc}" -eq 3 && ! -e "${task_run}" ]] || exit 6
    python3 - "${task_negative_run}" \
        "${M519_EXPECTED_VCS_RUNNER_SHA256}" "${task_wrong_sha}" \
        "${task_contract_expected_sha}" \
        "${M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256}" <<'PYNEG'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
receipt = {
    "schema": "m519_fc2_registered_release_vcs_negative_preflight_receipt_v1",
    "status": "PASS_WRONG_RUNNER_SHA_EXIT3_NO_CANONICAL_RESULT_NO_VCS",
    "expected_runner_sha256": sys.argv[2],
    "supplied_wrong_runner_sha256": sys.argv[3],
    "recovery_contract_r3_sha256": sys.argv[4],
    "static_hammer_r3_outer_seal_file_sha256": sys.argv[5],
    "child_exit_code": 3,
    "positive_canonical_result_absent": True,
    "vcs_invocations": 0,
    "dc_authorized": False,
}
(root / "m519_vcs_negative_preflight_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PYNEG
    printf 'PASS_WRONG_RUNNER_SHA_EXIT3_NO_CANONICAL_RESULT_NO_VCS\n' \
        >"${task_negative_run}/RUN_COMPLETE.txt"
    (
        cd "${task_negative_run}"
        find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
            -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
        sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
    task_negative_complete=1
    trap - EXIT
    echo "PASS M519 r3 negative preflight sealed at ${task_negative_run}"
    exit 0
fi
[[ "${task_mode}" == positive ]] || {
    echo "M519_VCS_MODE must be negative_preflight or positive" >&2
    exit 3
}
[[ -f "${task_negative_run}/m519_vcs_negative_preflight_receipt_r1.json" && \
   -f "${task_negative_run}/SHA256SUMS.seal.sha256" ]] || {
    echo "M519 sealed r3 wrong-SHA negative preflight is required" >&2
    exit 3
}
(cd "${task_negative_run}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
jq -e --arg runner_sha "${M519_EXPECTED_VCS_RUNNER_SHA256}" \
      --arg contract_sha "${task_contract_expected_sha}" \
      --arg review_sha "${M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256}" '
      .status == "PASS_WRONG_RUNNER_SHA_EXIT3_NO_CANONICAL_RESULT_NO_VCS"
      and .expected_runner_sha256 == $runner_sha
      and .supplied_wrong_runner_sha256 == "0000000000000000000000000000000000000000000000000000000000000000"
      and .supplied_wrong_runner_sha256 != $runner_sha
      and .recovery_contract_r3_sha256 == $contract_sha
      and .static_hammer_r3_outer_seal_file_sha256 == $review_sha
      and .child_exit_code == 3
      and .positive_canonical_result_absent == true
      and .vcs_invocations == 0
      and .dc_authorized == false' \
    "${task_negative_run}/m519_vcs_negative_preflight_receipt_r1.json" \
    >/dev/null || exit 3

mkdir "${task_run}"
task_complete=0
task_cleanup() {
    local task_rc=$?
    if [[ "${task_complete}" -ne 1 ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
    return "${task_rc}"
}
trap task_cleanup EXIT
cd "${task_hw_root}"

declare -A task_expected=(
 ["rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"]="e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5"
 ["rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"]="8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0"
 ["rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"]="529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267"
 ["rtl_m218/m218_fc2_tagged_slice_service_island.sv"]="f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1"
 ["rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv"]="597e4d9e9a606afa58111d01be8e8304e4fb5d4656cabdd4da9fca4b8393f43b"
 ["rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv"]="e5f3022e23736216f61482e1e33638d84c9a39dfb807c1c2fc53a14c90696456"
 ["rtl_m519/m519_fc2_k1_registered_release_service_island.sv"]="3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871"
 ["rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv"]="010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b"
 ["rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv"]="6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815"
 ["rtl_m519/m519_fc2_k8_registered_release_8bank_raw4_acc24.sv"]="5a4b05af5dcecd9c104aef00b4e0f818bc26e48e7c061424699a5ab00cefc96b"
 ["rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv"]="11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff"
 ["verif_m216/m216_fc2_raw4_to_source_cap_frontend_assertions.sv"]="1c8afec4c8035f60237156b93e9af05c4565eaa9eaa4c2527c35356e841689f0"
 ["verif_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter_assertions.sv"]="28b137431102c6a45a98eadba7b06a1bd94105f9e406df87fd02819f133cc8a0"
 ["verif_m218/m218_fc2_tagged_slice_service_assertions.sv"]="030f3cde04488a3d08e42bb074289ea96d022cbc4fc6c0446fc2fac711a16f45"
 ["verif_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter_assertions.sv"]="36a359435f1b004a0057cd6402f92152e6c87ee8a9b201886960c68f59d53a75"
 ["verif_m519/m519_fc2_k1_registered_release_service_assertions.sv"]="f7f228752bb89bd7ee374d513a31da311cf86a39b4e43c47fc9afd0d182ff153"
 ["verif_m519/m519_fc2_k1x8_registered_release_assertions.sv"]="45efc517a3eee305a25c9edf266ea80ab05f2af514475e684d574933c16229ee"
 ["tb_m349/m349_fc2_scalar_bank_memory_model.sv"]="4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa"
 ["tb_m519/tb_m519_fc2_registered_release_k1_vs_k1x8_raw4_acc24.sv"]="04d343795d9aaf9289f1b724ab2ee4c9ae25e403639b7a87f36a45deb796127b"
 ["tb_m519/tb_m519_fc2_registered_release_k8_vs_k1x8_raw4_acc24.sv"]="6fdb0601ac43a858b8c912afd0c0e9371348f60531ebc350c1fe0bd7a7bf8f01"
 ["dc_handoff/filelists/date_m519_fc2_registered_release_k1_vs_k1x8_vcs.f"]="b41a01d95d308c9573d77054d28fc0d8ca8e9e90b9330c7a09c75ad138e064ab"
 ["dc_handoff/filelists/date_m519_fc2_registered_release_k8_vs_k1x8_vcs.f"]="f10a42ec390432f34bea308a0d1a09794fb586b381c4f3eb05cf1751c3e0bcd3"
 ["reviews/m519_registered_release_vcs_failure_hammer_r1_20260827/SHA256SUMS.seal.sha256"]="0e74bdb5cf1672d9237eae588e5e01ef2a77452b949754344f52c1bd3e6884e6"
 ["reviews/m519_registered_release_vcs_failure_hammer_r1_20260827/m519_registered_release_vcs_failure_hammer_verdict_r1.json"]="e46277cbdbe16907f25b0ecab87890cb4f3a36ce043a30191383ae8378d34243"
 ["reviews/m496_r3_internal_loop_failure_hammer_r1_20260827/SHA256SUMS.seal.sha256"]="c8e49b3aeb1406c103604d6fec23e48ff27682f58eaed0e9abdd5b2cae6b3b79"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "${task_path}" \
        "${task_expected[$task_path]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[$task_path]}" ]] || exit 10
done
task_contract_observed_sha="$(sha256sum "${task_contract}" | awk '{print $1}')"
printf 'path=%s expected=%s observed=%s\n' "${task_contract_rel}" \
    "${task_contract_expected_sha}" "${task_contract_observed_sha}" \
    >>"${task_run}/preflight_sha_checks.txt"
[[ "${task_contract_observed_sha}" == "${task_contract_expected_sha}" ]] || exit 10
(cd reviews/m496_r3_internal_loop_failure_hammer_r1_20260827 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
sha256sum "${task_contract}" \
    >>"${task_run}/input_sha256.txt"
sha256sum "${task_static_review_identity}" "${task_static_review_verdict}" \
    "${task_static_review_seal}" \
    >>"${task_run}/input_sha256.txt"
sha256sum "${task_failure_review_identity}" \
    "${task_failure_review_verdict}" "${task_failure_review_seal}" \
    >>"${task_run}/input_sha256.txt"
find "${task_negative_run}" -type f -print0 | sort -z | \
    xargs -0 sha256sum >>"${task_run}/input_sha256.txt"
sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"

export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m519_fc2_registered_release_k1_vs_k1x8_vcs.f \
    -top tb_m519_fc2_registered_release_k1_vs_k1x8_raw4_acc24 \
    -o "${task_run}/simv" >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
echo "${task_rc}" >"${task_run}/compile.rc"
[[ "${task_rc}" -eq 0 && -x "${task_run}/simv" ]] || exit 20
! grep -Eiq 'Error-\[|^Error|^Fatal' "${task_run}/compile.log" || exit 21

set +e
"${task_run}/simv" +ntb_random_seed=519027 -no_save \
    -assert report="${task_run}/assert.report" -cm assert \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
echo "${task_rc}" >"${task_run}/sim.rc"
[[ "${task_rc}" -eq 0 ]] || exit 22
! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${task_run}/sim.log" "${task_run}/assert.report" || exit 23
grep -Eq '^PASS M519 registered-release K1 versus K1x8 FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 .*numeric_mismatches=0 .*tuple_mismatches=0 .*weight_mismatches=0 .*same_edge_release_violations=0 .*request_stalls=[1-9][0-9]* .*response_injection_stalls=[1-9][0-9]* .*result_stalls=[1-9][0-9]* .*raw_stalls=[1-9][0-9]* .*distinct_same_edge_req_rsp=[1-9][0-9]* .*next_cycle_slot_reuse=[1-9][0-9]* .*next_cycle_context_reuse=[1-9][0-9]*' \
    "${task_run}/sim.log" || exit 30

for task_cover in cp_same_cycle_distinct_release \
        cp_release_then_slot_reissue cp_release_then_context_reissue \
        cp_result_stall cp_done; do
    grep -Eq "candidate\.core\.g_k1\.service\.m519_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 40
done
for task_bank in 0 1 2 3 4 5 6 7; do
    for task_cover in cp_same_cycle_distinct_release \
            cp_release_then_slot_reissue cp_release_then_context_reissue \
            cp_result_stall cp_done; do
        grep -Eq "baseline\.g_lane\[${task_bank}\]\.service\.m519_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
            "${task_run}/assert.report" || exit 41
    done
done
for task_cover in cp_pending_request_stall cp_out_of_order_bundle_response \
        cp_cutthrough_bundle_response cp_protocol_attack; do
    grep -Eq "candidate\.memory_adapter\.m499_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 42
done

# The second exact-SHA test measures the unchanged shared-state K8 path against
# the new registered-release K1x8 endpoint.  Together the two tests give one
# coherent M519 K1/K8/K1x8 cycle identity; no M492/M497 row is reused.
mkdir "${task_run}/equalbw"
set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/equalbw/csrc" \
    -f dc_handoff/filelists/date_m519_fc2_registered_release_k8_vs_k1x8_vcs.f \
    -top tb_m519_fc2_registered_release_k8_vs_k1x8_raw4_acc24 \
    -o "${task_run}/equalbw/simv" \
    >"${task_run}/equalbw/compile.log" 2>&1
task_rc=$?
set -e
echo "${task_rc}" >"${task_run}/equalbw/compile.rc"
[[ "${task_rc}" -eq 0 && -x "${task_run}/equalbw/simv" ]] || exit 50
! grep -Eiq 'Error-\[|^Error|^Fatal' \
    "${task_run}/equalbw/compile.log" || exit 51

set +e
"${task_run}/equalbw/simv" +ntb_random_seed=519028 -no_save \
    -assert report="${task_run}/equalbw/assert.report" -cm assert \
    >"${task_run}/equalbw/sim.log" 2>&1
task_rc=$?
set -e
echo "${task_rc}" >"${task_run}/equalbw/sim.rc"
[[ "${task_rc}" -eq 0 ]] || exit 52
! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${task_run}/equalbw/sim.log" \
    "${task_run}/equalbw/assert.report" || exit 53
grep -Eq '^PASS M519EQ cutthrough-8bank equal-bandwidth FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 .*numeric_mismatches=0 .*tuple_mismatches=0 .*weight_mismatches=0 .*request_stalls=[1-9][0-9]* .*result_stalls=[1-9][0-9]* .*raw_stalls=[1-9][0-9]*' \
    "${task_run}/equalbw/sim.log" || exit 54

for task_cover in cp_b1 cp_b2 cp_b4 cp_b8 cp_all_eight_lane_group \
        cp_eight_requests_same_cycle cp_request_backpressure \
        cp_result_stall cp_done cp_protocol_fault; do
    grep -Eq "baseline\.m519_top_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/equalbw/assert.report" || exit 55
done
for task_cover in cp_k8_request cp_same_cycle_replace cp_result_stall cp_done; do
    grep -Eq "candidate\.core\.g_k8\.service\.m519_bound_k8_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/equalbw/assert.report" || exit 56
done
for task_bank in 0 1 2 3 4 5 6 7; do
    for task_cover in cp_same_cycle_distinct_release \
            cp_release_then_slot_reissue cp_release_then_context_reissue \
            cp_result_stall cp_done; do
        grep -Eq "baseline\.g_lane\[${task_bank}\]\.service\.m519_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
            "${task_run}/equalbw/assert.report" || exit 57
    done
done
for task_cover in cp_full_eight_bank_request cp_eight_responses_same_cycle \
        cp_out_of_order_bundle_response cp_retire_then_slot_reuse \
        cp_same_cycle_slot_reuse cp_cutthrough_bundle_response \
        cp_protocol_attack; do
    grep -Eq "candidate\.memory_adapter\.m490_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/equalbw/assert.report" || exit 58
done

python3 - "${task_run}" \
    "${task_contract_expected_sha}" \
    "${M519_EXPECTED_STATIC_HAMMER_R3_OUTER_SEAL_FILE_SHA256}" \
    "0e74bdb5cf1672d9237eae588e5e01ef2a77452b949754344f52c1bd3e6884e6" \
    "$(sha256sum "${task_negative_run}/SHA256SUMS.seal.sha256" | awk '{print $1}')" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
recovery_contract_sha256 = sys.argv[2]
authorizing_static_review_outer_seal_file_sha256 = sys.argv[3]
failure_hammer_outer_seal_file_sha256 = sys.argv[4]
negative_preflight_outer_seal_file_sha256 = sys.argv[5]
text = (root / "sim.log").read_text()
equal_text = (root / "equalbw" / "sim.log").read_text()
pattern = re.compile(
    r"M519 canonical K1 versus K1x8 B=(\d+) events=(\d+) "
    r"k1_cycles=(\d+) k1x8_cycles=(\d+) k1x8_speedup_vs_k1=([0-9.]+) "
    r"tuple_mismatches=(\d+) weight_mismatches=(\d+)"
)
rows = []
observed = [tuple(match.groups()) for match in pattern.finditer(text)]
expected_shape = [(1, 20), (2, 41), (4, 90), (8, 110), (1, 0)]
if len(observed) != len(expected_shape):
    raise SystemExit("M519 expected five cycle rows, got {!r}".format(observed))
for raw, shape in zip(observed, expected_shape):
    blocks, events, k1, k1x8, printed_ratio, tuple_mm, weight_mm = raw
    if (int(blocks), int(events)) != shape:
        raise SystemExit("M519 row identity mismatch: {!r}".format(raw))
    if int(k1) <= 0 or int(k1x8) <= 0 or int(tuple_mm) or int(weight_mm):
        raise SystemExit("M519 row failed conservation: {!r}".format(raw))
    exact_ratio = int(k1) / int(k1x8)
    if abs(float(printed_ratio) - exact_ratio) > 1e-6:
        raise SystemExit("M519 printed ratio mismatch: {!r}".format(raw))
    rows.append({
        "output_blocks": int(blocks),
        "events": int(events),
        "registered_release_k1_cycles": int(k1),
        "registered_release_k1x8_cycles": int(k1x8),
        "k1x8_over_k1": exact_ratio,
    })
equal_pattern = re.compile(
    r"M519EQ cutthrough equalbw B=(\d+) events=(\d+) k8_cycles=(\d+) "
    r"k1x8_cycles=(\d+) speedup=([0-9.]+) tuple_mismatches=(\d+) "
    r"weight_mismatches=(\d+)"
)
equal_observed = [tuple(match.groups())
                  for match in equal_pattern.finditer(equal_text)]
if len(equal_observed) != len(expected_shape):
    raise SystemExit("M519 expected five K8/K1x8 rows, got {!r}".format(
        equal_observed))
equal_rows = []
for raw, shape in zip(equal_observed, expected_shape):
    blocks, events, k8, k1x8, printed_ratio, tuple_mm, weight_mm = raw
    if (int(blocks), int(events)) != shape:
        raise SystemExit("M519 equal-bw row identity mismatch: {!r}".format(raw))
    if int(k8) <= 0 or int(k1x8) <= 0 or int(tuple_mm) or int(weight_mm):
        raise SystemExit("M519 equal-bw conservation failed: {!r}".format(raw))
    exact_ratio = int(k1x8) / int(k8)
    if abs(float(printed_ratio) - exact_ratio) > 1e-6:
        raise SystemExit("M519 equal-bw printed ratio mismatch: {!r}".format(raw))
    equal_rows.append({
        "output_blocks": int(blocks),
        "events": int(events),
        "registered_release_k8_cycles": int(k8),
        "registered_release_k1x8_cycles": int(k1x8),
        "k8_over_k1x8_speedup": exact_ratio,
    })
for k1_row, equal_row in zip(rows, equal_rows):
    if (k1_row["output_blocks"], k1_row["events"]) != (
            equal_row["output_blocks"], equal_row["events"]):
        raise SystemExit("M519 cross-test row mismatch")
    if k1_row["registered_release_k1x8_cycles"] != \
            equal_row["registered_release_k1x8_cycles"]:
        raise SystemExit("M519 K1x8 cycle identity differs across tests")
receipt = {
    "schema": "m519_fc2_registered_release_vcs_receipt_v2",
    "status": "PASS_M519_REGISTERED_RELEASE_EXACT_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seeds": {"primary": 519027, "equal_bandwidth": 519028},
    "authorization_identity": {
        "recovery_contract_r3_sha256": recovery_contract_sha256,
        "static_hammer_r3_outer_seal_file_sha256":
            authorizing_static_review_outer_seal_file_sha256,
        "r2_vcs_failure_hammer_outer_seal_file_sha256":
            failure_hammer_outer_seal_file_sha256,
        "wrong_sha_negative_preflight_outer_seal_file_sha256":
            negative_preflight_outer_seal_file_sha256,
        "authorized_vcs_invocations": 1,
        "dc_authorized": False,
    },
    "k1_vs_k1x8_cycle_rows": rows,
    "k8_vs_k1x8_cycle_rows": equal_rows,
    "all_three_axes_same_m519_identity": True,
    "old_cycle_ratios_reused": False,
    "r2_diagnostic_outputs_reused": False,
    "checks": {
        "transaction_multiset": "exact",
        "numeric": "bit_exact",
        "done": "exact",
        "protocol_attacks": 4,
        "request_response_result_raw_stalls": True,
        "same_edge_release_reissue_violations": 0,
        "next_cycle_release_reissue_covered": True,
    },
    "claim_boundary": {
        "rtl_functional": True,
        "combinational_loop_free": False,
        "dc": False,
        "power": False,
        "complete_fc2": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
    "required_next_gate": "Independent receipt-blind r2 VCS hammer with P0=0, followed by a separately sealed post-r3-VCS DC launch-admission; DC remains forbidden now."
}
(root / "m519_fc2_registered_release_vcs_receipt_r2.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

printf 'PASS_M519_REGISTERED_RELEASE_EXACT_VCS\n' \
    >"${task_run}/RUN_COMPLETE.txt"
(
    cd "${task_run}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
task_complete=1
echo "PASS M519 exact VCS sealed at ${task_run}"
