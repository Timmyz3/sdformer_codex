#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
hw_root="${repo_root}/hw_autoresearch_nts07"
runner="${hw_root}/dc_handoff/scripts/run_dc_m756_m528_macro_integrated_exact_sha_r2.sh"
contract="${hw_root}/contracts/m756_m528_macro_integrated_dc_source_only_contract_r2_20260828.json"
candidate="${hw_root}/contracts/m756_m528_macro_integrated_dc_launch_admission_candidate_r2_20260828.json"
audit_dir="${hw_root}/reviews/m756_m750_macro_dc_hash_cycle_self_audit_r1_20260828"
release="${hw_root}/contracts/m756_m528_macro_integrated_dc_launch_release_r2_20260828.json"
final_review="${hw_root}/reviews/m756_m528_macro_integrated_dc_final_launch_release_hammer_r1_20260828/review.json"
result="${hw_root}/dc_handoff/runs/m756_m528_macro_integrated_dc_3p000ns_r2_20260828"
attempt="${hw_root}/dc_handoff/runs/.m756_m528_macro_integrated_dc_attempt_consumed"

expect_sha() {
    local expected=$1 path=$2
    [[ -f "${path}" && ! -L "${path}" ]]
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

# Static source checks only.  This script never invokes the production runner,
# dc_shell, VCS, PT, Formality, a simulator, GPU code or a remote command.
bash -n "${runner}"
expect_sha 8ef1f24ae7b67768feeb7fa18363589db23d939f5a722e67f2b7c23ab6cac722 "${runner}"
expect_sha 75ff6be742e844bd7582d67f908158589e949b2c7639d08bc608b3d98355a4ea "${contract}"
expect_sha 2da8f9e12ae1a76d25cd57ba28115f8247d5f8a30e6abeb88d437f9106d1a54c "${audit_dir}/review.json"
(cd "${audit_dir}" && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

expect_sha 7f08071dee1941d8db2fcf4e9eeba40b9beb3a6f6f17644fe94098cdac3778fa \
    "${hw_root}/dc_handoff/scripts/run_dc_m750_m528_macro_integrated_exact_sha_r1.sh"
expect_sha 73208f7ce9ef2491f81028c153532e152aa8c86dfcbe04c3a5e41ff1afdc5f89 \
    "${hw_root}/contracts/m750_m528_macro_integrated_dc_source_only_contract_r1_20260828.json"
expect_sha 661baa6e7618e41aa83f6d290000cf7c93797fa14a8dd5cb5b9f0dfab507a550 \
    "${hw_root}/contracts/m750_m528_macro_integrated_dc_launch_admission_candidate_r1_20260828.json"

rg -q '^m756_final_review=reviews/m756_m528_macro_integrated_dc_final_launch_release_hammer_r1_20260828/review\.json$' "${runner}"
rg -q 'M756_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256' "${runner}"
rg -q 'm756_expect "\$\{m756_final_review\}" "\$\{M756_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256\}"' "${runner}"
rg -q 'final_release_sha256.*sys\.argv\[2\]' "${runner}"
! rg -q 'fresh_hammers\.review_sha256_by_path' "${runner}"

jq -e '.schema == "m756_m528_macro_integrated_dc_source_only_contract_v2"
       and .authorization.runner_executions == 0
       and .authorization.dc_runs == 0
       and .acyclic_release_chain.runner_embeds_future_final_review_sha == false
       and .acyclic_release_chain.release_embeds_final_review_sha == false
       and .acyclic_release_chain.final_review_binds_release_sha == true
       and .m750_permanent_no_go.old_files_modified == false
       and .claim_boundary.source_only == true' "${contract}" >/dev/null
jq -e '.schema == "m756_m528_macro_integrated_dc_launch_admission_candidate_v2"
       and .launch_now == false
       and .authorization.run_dc == false
       and .pending_gates.m756_source_candidate_hammer_present == false
       and .pending_gates.m756_launch_now_true_release_present == false
       and .pending_gates.m756_final_release_hammer_present == false' "${candidate}" >/dev/null

[[ ! -e "${release}" && ! -e "${final_review}" && ! -e "${result}" && ! -e "${attempt}" ]]
expect_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
    "${hw_root}/docs/359_DATE终局冻结_20260813.md"

printf 'PASS M756 R2 static NO_EDA selftest; runner_executions=0; dc_runs=0; hash_cycle_removed=true\n'
