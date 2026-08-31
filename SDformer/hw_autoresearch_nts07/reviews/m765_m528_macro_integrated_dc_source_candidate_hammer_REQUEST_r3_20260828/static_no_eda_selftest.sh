#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
hw_root="${repo_root}/hw_autoresearch_nts07"
runner="${hw_root}/dc_handoff/scripts/run_dc_m765_m528_macro_integrated_exact_sha_r3.sh"
contract="${hw_root}/contracts/m765_m528_macro_integrated_dc_source_only_contract_r3_20260828.json"
candidate="${hw_root}/contracts/m765_m528_macro_integrated_dc_launch_admission_candidate_r3_20260828.json"
audit_dir="${hw_root}/reviews/m756_m750_macro_dc_hash_cycle_self_audit_r1_20260828"
release="${hw_root}/contracts/m765_m528_macro_integrated_dc_launch_release_r3_20260828.json"
final_review="${hw_root}/reviews/m765_m528_macro_integrated_dc_final_launch_release_hammer_r1_20260828/review.json"
result="${hw_root}/dc_handoff/runs/m765_m528_macro_integrated_dc_3p000ns_r3_20260828"
attempt="${hw_root}/dc_handoff/runs/.m765_m528_macro_integrated_dc_attempt_consumed"
m758_result="${hw_root}/results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828"
m766_result_review="${hw_root}/reviews/m766_m758_m533_r13_unit_delay_vcs_result_hammer_r1_20260828/review.json"

expect_sha() {
    local expected=$1 path=$2
    [[ -f "${path}" && ! -L "${path}" ]]
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

# Static source checks only.  This script never invokes the production runner,
# dc_shell, VCS, PT, Formality, a simulator, GPU code or a remote command.
bash -n "${runner}"
expect_sha bbd14ef6cf2aa29b3c6dbb9b52dbd545f7bd20bd676366e14c95e43608e0fe15 "${runner}"
expect_sha 131e4bc482bdb7f8ccaacec4505b56750c6f08fcd29da255bfdaa74d9bdfce75 "${contract}"
expect_sha 2da8f9e12ae1a76d25cd57ba28115f8247d5f8a30e6abeb88d437f9106d1a54c "${audit_dir}/review.json"
(cd "${audit_dir}" && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

expect_sha 7f08071dee1941d8db2fcf4e9eeba40b9beb3a6f6f17644fe94098cdac3778fa \
    "${hw_root}/dc_handoff/scripts/run_dc_m750_m528_macro_integrated_exact_sha_r1.sh"
expect_sha 73208f7ce9ef2491f81028c153532e152aa8c86dfcbe04c3a5e41ff1afdc5f89 \
    "${hw_root}/contracts/m750_m528_macro_integrated_dc_source_only_contract_r1_20260828.json"
expect_sha 661baa6e7618e41aa83f6d290000cf7c93797fa14a8dd5cb5b9f0dfab507a550 \
    "${hw_root}/contracts/m750_m528_macro_integrated_dc_launch_admission_candidate_r1_20260828.json"

rg -q '^m765_final_review=reviews/m765_m528_macro_integrated_dc_final_launch_release_hammer_r1_20260828/review\.json$' "${runner}"
rg -q 'M765_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256' "${runner}"
rg -q 'm765_expect "\$\{m765_final_review\}" "\$\{M765_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256\}"' "${runner}"
rg -q 'final_release_sha256.*sys\.argv\[2\]' "${runner}"
! rg -q 'fresh_hammers\.review_sha256_by_path' "${runner}"

expect_sha 8b84530b0666b2b52617d25e8cf2e5fd2f0f2fe45b0c5242f45528d38803f991 \
    "${hw_root}/reviews/m757_m533_r12_premkdir_sha_literal_failure_fresh_hammer_r1_20260828/review.json"
expect_sha fa65fb3f81ada9e93bea8ee7d48de2b2a0c8085568d0924b66997d3e000d820e \
    "${hw_root}/reviews/m761_m533_r13_unit_delay_source_candidate_fresh_hammer_r1_20260828/review.json"
expect_sha 2377819d455bac02101f740f7a966717ddf69e34b4173a042c91b870d78b7123 \
    "${hw_root}/dc_handoff/scripts/run_vcs_m758_m533_m528_dead_write_only_1rw_unit_delay_r13_exact_sha.sh"
expect_sha 06c9b5d5346eb9cecafab0bcc5dbad54a1337916ff47a55f19191147742d1d22 \
    "${hw_root}/contracts/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
expect_sha b2ecd28fe2ae1c4bb988c704d8967fdd8bb6c5b8f063d4292aa4d1f967899a12 \
    "${hw_root}/reviews/m763_m533_r13_unit_delay_vcs_final_launch_release_hammer_REQUEST_r1_20260828/request.json"

jq -e '.schema == "m765_m528_macro_integrated_dc_source_only_contract_v3"
       and .authorization.runner_executions == 0
       and .authorization.dc_runs == 0
       and .acyclic_release_chain.runner_embeds_future_final_review_sha == false
       and .acyclic_release_chain.release_embeds_final_review_sha == false
       and .acyclic_release_chain.final_review_binds_release_sha == true
       and .m750_permanent_no_go.old_files_modified == false
       and .claim_boundary.source_only == true' "${contract}" >/dev/null
jq -e '.schema == "m765_m528_macro_integrated_dc_launch_admission_candidate_v3"
       and .launch_now == false
       and .authorization.run_dc == false
       and .pending_gates.m758_vcs_result_path == "results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828"
       and .pending_gates.m766_vcs_result_hammer_path == "reviews/m766_m758_m533_r13_unit_delay_vcs_result_hammer_r1_20260828/review.json"
       and .pending_gates.m758_vcs_pass_present == false
       and .pending_gates.m766_independent_result_hammer_present == false
       and .pending_gates.m765_source_candidate_hammer_present == false
       and .pending_gates.m765_launch_now_true_release_present == false
       and .pending_gates.m765_final_release_hammer_present == false' "${candidate}" >/dev/null

[[ ! -e "${m758_result}" && ! -e "${m766_result_review}" ]]
[[ ! -e "${release}" && ! -e "${final_review}" && ! -e "${result}" && ! -e "${attempt}" ]]
expect_sha dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
    "${hw_root}/docs/359_DATE终局冻结_20260813.md"

printf 'PASS M765 R3 static NO_EDA selftest; runner_executions=0; dc_runs=0; m758_result_gate=future\n'
