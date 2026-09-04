#!/usr/bin/env bash
set -euo pipefail

m528_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m528_hw_root}"

m528_runner="$(realpath "${BASH_SOURCE[0]}")"
m528_analyzer="system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute_r4.py"
m528_execution="contracts/m528_h67_single_port_same_ledger_execution_contract_r4_20260827.json"
m528_governing="reviews/m528_false_kill_audit_r1_20260827/m528_single_port_same_ledger_recompute_contract_r1_20260827.json"
m528_strict_tool="system_simulator/scripts/strict_json_no_duplicate_keys_m528_r4.py"
m528_python="/opt/anaconda3/envs/pytorch310/bin/python"
m528_author_dir="reviews/m528_single_port_same_ledger_recompute_author_handoff_r4_20260827"
m528_static_dir="reviews/m528_r4_recovery_static_hammer_r1_20260827"
m528_static_json="${m528_static_dir}/review.json"
m528_preflight_dir="results/m528_r4_preflight_spawn_schema_r1_20260827"
m528_preflight_json="${m528_preflight_dir}/m528_r4_preflight_receipt_r1.json"
m528_preflight_hammer_dir="reviews/m528_r4_preflight_receipt_hammer_r1_20260827"
m528_preflight_hammer_json="${m528_preflight_hammer_dir}/review.json"
m528_redteam_dir="reviews/m528_r3_wrapper_runner_redteam_r1_20260827"
m528_redteam_json="${m528_redteam_dir}/review.json"
m528_withdrawal_dir="reviews/m528_r3_recovery_static_hammer_withdrawal_r1_20260827"
m528_withdrawal_json="${m528_withdrawal_dir}/withdrawal.json"
m528_r2_review_dir="reviews/m528_r2_consumed_failure_failclosed_hammer_r1_20260827"
m528_r2_review_json="${m528_r2_review_dir}/review.json"
m528_r2_attempt="results/.m528_h67_single_port_same_ledger_recompute_r2_20260827.attempt_consumed"
m528_r2_quarantine="results/m528_h67_single_port_same_ledger_recompute_r2_20260827.failed_or_incomplete.3515398.quarantine"
m528_r2_canonical="results/m528_h67_single_port_same_ledger_recompute_r2_20260827"
m528_canonical="results/m528_h67_single_port_same_ledger_recompute_r4_20260827"
m528_attempt="results/.m528_h67_single_port_same_ledger_recompute_r4_20260827.attempt_consumed"
m528_work="results/.m528_h67_single_port_same_ledger_recompute_r4_work.$$"
m528_pre_quarantine="${m528_canonical}.pre_attempt_failure.$$.quarantine"
m528_post_quarantine="${m528_canonical}.failed_or_incomplete.$$.quarantine"

m528_sha() { sha256sum "$1" | awk '{print $1}'; }
m528_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m528_sha "${path}")" == "${expected}" ]] || {
        echo "M528 r4 production identity mismatch: ${path}" >&2
        exit 3
    }
}
m528_strict() { "${m528_python}" "${m528_strict_tool}" "$1" >/dev/null; }
m528_verify_sealed_dir() {
    local dir=$1 expected_outer=$2
    m528_expect "${dir}/SHA256SUMS.seal.sha256" "${expected_outer}"
    (
        cd "${dir}"
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
}
m528_verify_admission_seals() {
    local path=$1 dir base
    dir="$(dirname "${path}")"
    base="$(basename "${path}")"
    [[ -f "${path}.sha256" && -f "${path}.sha256.seal.sha256" ]] || return 1
    (
        cd "${dir}"
        sha256sum -c "${base}.sha256" >/dev/null
        sha256sum -c "${base}.sha256.seal.sha256" >/dev/null
    )
}

[[ -n "${M528_R4_EXPECTED_PRODUCTION_RUNNER_SHA256:-}" && \
   "$(m528_sha "${m528_runner}")" == "${M528_R4_EXPECTED_PRODUCTION_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M528_R4_EXPECTED_PRODUCTION_ADMISSION_PATH:-}" && \
   -n "${M528_R4_EXPECTED_PRODUCTION_ADMISSION_SHA256:-}" ]] || exit 3
[[ "${M528_R4_EXPECTED_PRODUCTION_ADMISSION_PATH}" != /* && \
   "${M528_R4_EXPECTED_PRODUCTION_ADMISSION_PATH}" != *".."* ]] || exit 3
m528_admission="${M528_R4_EXPECTED_PRODUCTION_ADMISSION_PATH}"

[[ ! -e "${m528_canonical}" && ! -e "${m528_attempt}" && ! -e "${m528_work}" ]] || exit 5
[[ -z "${M528_OUTPUT_OVERRIDE:-}" && -z "${M528_WORKERS_OVERRIDE:-}" ]] || exit 5

m528_expect "${m528_analyzer}" c94b2ca031688158b4bdfa9d0c2aa00931fc7ee880acd5f448eced9a022a3d4f
m528_expect "${m528_execution}" c02faf52a1dfbc6936a59dc65c4d089a13e15578407f204c5ef06585b8d68390
m528_expect "${m528_governing}" d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51
m528_expect "${m528_strict_tool}" b2e95ec8e05434eff246ec300a6c8bf9d069011b3741ba6be20a6986c5055cf6
m528_expect "${m528_python}" 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m528_expect "${m528_admission}" "${M528_R4_EXPECTED_PRODUCTION_ADMISSION_SHA256}"
m528_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
m528_strict "${m528_admission}"
m528_verify_admission_seals "${m528_admission}" || exit 3

jq -e '
  .schema == "m528_r4_production_static_admission_v1" and
  .status == "AUTHORIZED_ONE_M528_R4_CPU_PRODUCTION_AFTER_PREFLIGHT_HAMMER" and
  .authorization.cpu_runs == 1 and
  .authorization.pre_attempt_spawn_schema_repetitions == 1 and
  .authorization.eda_runs == 0 and .authorization.gpu_runs == 0 and .authorization.rtl == false and
  .identity.production_runner_sha256 == env.M528_R4_EXPECTED_PRODUCTION_RUNNER_SHA256 and
  .identity.analyzer_sha256 == "c94b2ca031688158b4bdfa9d0c2aa00931fc7ee880acd5f448eced9a022a3d4f" and
  .identity.preflight_runner_sha256 == "893a89c98ae3ea04fc1c316e71c3768fe5189cc4ce54527e352f0c6f3b3a0944" and
  .identity.execution_contract_sha256 == "c02faf52a1dfbc6936a59dc65c4d089a13e15578407f204c5ef06585b8d68390" and
  .identity.governing_contract_sha256 == "d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51" and
  .identity.strict_json_tool_sha256 == "b2e95ec8e05434eff246ec300a6c8bf9d069011b3741ba6be20a6986c5055cf6" and
  .runtime.workers == 3 and .runtime.chunksize == 2 and
  .runtime.minimum_commit_headroom_kib == 50331648 and
  .runtime.minimum_mem_available_kib == 134217728 and
  .runtime.minimum_swap_free_kib == 33554432 and
  .expected.pass_token == "PASS_M528_R4_PREFLIGHT_SCHEMA_AND_SPAWN_IMPORT_SELF_TEST" and
  .docs359_sha256 == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
' "${m528_admission}" >/dev/null || exit 3

for key in author_handoff_outer_seal_file_sha256 static_review_json_sha256 \
        static_review_outer_seal_file_sha256 preflight_receipt_json_sha256 \
        preflight_receipt_outer_seal_file_sha256 preflight_hammer_json_sha256 \
        preflight_hammer_outer_seal_file_sha256 preflight_admission_sha256; do
    value="$(jq -er ".identity.${key}" "${m528_admission}")"
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done

m528_verify_sealed_dir "${m528_author_dir}" "$(jq -er '.identity.author_handoff_outer_seal_file_sha256' "${m528_admission}")"
m528_verify_sealed_dir "${m528_static_dir}" "$(jq -er '.identity.static_review_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_static_json}" "$(jq -er '.identity.static_review_json_sha256' "${m528_admission}")"
m528_strict "${m528_static_json}"
jq -e --arg production_sha "${M528_R4_EXPECTED_PRODUCTION_RUNNER_SHA256}" \
      --arg author_outer "$(jq -er '.identity.author_handoff_outer_seal_file_sha256' "${m528_admission}")" '
  .schema == "m528_r4_recovery_source_only_static_hammer_v1" and
  .status == "PASS_R4_SOURCE_ONLY_STATIC_HAMMER__AUTHORIZE_ONE_PREFLIGHT_ONLY_SUITE" and
  .verdict == "GO_ONE_R4_PREFLIGHT_ONLY_SPAWN_AND_THREE_CASE_SCHEMA_SUITE__NO_GO_PRODUCTION" and
  .p0_count == 0 and .p1_count == 0 and
  .withdrawn == false and
  .identity.analyzer.sha256 == "c94b2ca031688158b4bdfa9d0c2aa00931fc7ee880acd5f448eced9a022a3d4f" and
  .identity.preflight_runner.sha256 == "893a89c98ae3ea04fc1c316e71c3768fe5189cc4ce54527e352f0c6f3b3a0944" and
  .identity.production_runner.sha256 == $production_sha and
  .identity.execution_contract.sha256 == "c02faf52a1dfbc6936a59dc65c4d089a13e15578407f204c5ef06585b8d68390" and
  .identity.strict_json_tool.sha256 == "b2e95ec8e05434eff246ec300a6c8bf9d069011b3741ba6be20a6986c5055cf6" and
  .identity.author_handoff.outer_seal_file_sha256 == $author_outer and
  .authorization.root_may_create_one_preflight_only_admission == true and
  .authorization.root_may_create_one_new_double_sealed_smoke_only_admission == true and
  .authorization.root_may_create_one_production_admission == false and
  .authorization.cpu_production_runs == 0
' "${m528_static_json}" >/dev/null || exit 3

m528_preflight_admission="$(jq -er '.identity.preflight_admission_path' "${m528_admission}")"
[[ "${m528_preflight_admission}" != /* && "${m528_preflight_admission}" != *".."* ]] || exit 3
m528_expect "${m528_preflight_admission}" "$(jq -er '.identity.preflight_admission_sha256' "${m528_admission}")"
m528_strict "${m528_preflight_admission}"
m528_verify_admission_seals "${m528_preflight_admission}" || exit 3
jq -e --arg production_sha "${M528_R4_EXPECTED_PRODUCTION_RUNNER_SHA256}" '
  .schema == "m528_r4_preflight_static_admission_v1" and
  .status == "AUTHORIZED_ONE_M528_R4_NONPRODUCTION_SPAWN_AND_SCHEMA_SUITE" and
  .authorization.preflight_suites == 1 and .authorization.cpu_production_runs == 0 and
  .identity.preflight_runner_sha256 == "893a89c98ae3ea04fc1c316e71c3768fe5189cc4ce54527e352f0c6f3b3a0944" and
  .identity.production_runner_sha256 == $production_sha and
  .identity.analyzer_sha256 == "c94b2ca031688158b4bdfa9d0c2aa00931fc7ee880acd5f448eced9a022a3d4f" and
  .identity.execution_contract_sha256 == "c02faf52a1dfbc6936a59dc65c4d089a13e15578407f204c5ef06585b8d68390"
' "${m528_preflight_admission}" >/dev/null || exit 3

m528_verify_sealed_dir "${m528_preflight_dir}" "$(jq -er '.identity.preflight_receipt_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_preflight_json}" "$(jq -er '.identity.preflight_receipt_json_sha256' "${m528_admission}")"
m528_strict "${m528_preflight_json}"
jq -e --arg preflight_admission_sha "$(jq -er '.identity.preflight_admission_sha256' "${m528_admission}")" \
      --arg static_review_sha "$(m528_sha "${m528_static_json}")" '
  .schema == "m528_r4_preflight_receipt_v1" and
  .status == "PASS_R4_NONPRODUCTION_SPAWN_AND_SCHEMA_SUITE_PENDING_INDEPENDENT_HAMMER" and
  .identity.preflight_runner_sha256 == "893a89c98ae3ea04fc1c316e71c3768fe5189cc4ce54527e352f0c6f3b3a0944" and
  .identity.analyzer_sha256 == "c94b2ca031688158b4bdfa9d0c2aa00931fc7ee880acd5f448eced9a022a3d4f" and
  .identity.execution_contract_sha256 == "c02faf52a1dfbc6936a59dc65c4d089a13e15578407f204c5ef06585b8d68390" and
  .identity.preflight_admission_sha256 == $preflight_admission_sha and
  .identity.static_review_json_sha256 == $static_review_sha and
  .spawn_import_self_test.pass == true and .spawn_import_self_test.workers == 1 and
  .spawn_import_self_test.spawn_process_pool_created == true and
  .spawn_import_self_test.stable_module_name == "analyze_m528_h67_single_port_same_ledger_recompute" and
  .spawn_import_self_test.initializer == "worker_init" and
  .spawn_import_self_test.worker_init_called == true and
  .spawn_import_self_test.submitted_function == "sha256_file" and
  .spawn_import_self_test.worker_phase_pickle_checked == true and
  .spawn_import_self_test.worker_phase_called == false and
  .cases.positive.exit_code == 0 and .cases.positive.exact_pass_token_count == 1 and
  (.cases.wrong_pointer_argument_control.exit_code | type == "number") and
  .cases.wrong_pointer_argument_control.exit_code != 0 and
  .cases.wrong_pointer_argument_control.pass_token_count == 0 and
  (.cases.wrong_corner_argument_control.exit_code | type == "number") and
  .cases.wrong_corner_argument_control.exit_code != 0 and
  .cases.wrong_corner_argument_control.pass_token_count == 0 and
  .forbidden_activity.production_process_pool == false and
  .forbidden_activity.row_ledger_semantic_read == false and
  .forbidden_activity.worker_phase_called == false and
  .forbidden_activity.row_replay == false and
  .forbidden_activity.production_result_created == false and
  .forbidden_activity.production_attempt_consumed == false and
  .forbidden_activity.cpu_production_runs == 0 and
  .forbidden_activity.eda_runs == 0 and .forbidden_activity.gpu_runs == 0 and
  .forbidden_activity.rtl == false and
  .paper_admitted == false and .system_speedup == false and .date_headline == false
' "${m528_preflight_json}" >/dev/null || exit 3

m528_verify_sealed_dir "${m528_preflight_hammer_dir}" "$(jq -er '.identity.preflight_hammer_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_preflight_hammer_json}" "$(jq -er '.identity.preflight_hammer_json_sha256' "${m528_admission}")"
m528_strict "${m528_preflight_hammer_json}"
jq -e --arg receipt_sha "$(m528_sha "${m528_preflight_json}")" \
      --arg receipt_outer "$(m528_sha "${m528_preflight_dir}/SHA256SUMS.seal.sha256")" '
  .schema == "m528_r4_preflight_receipt_hammer_v1" and
  .status == "PASS_R4_PREFLIGHT_RECEIPT_HAMMER__AUTHORIZE_ROOT_TO_SIGN_ONE_PRODUCTION_ADMISSION" and
  .verdict == "GO_ONE_R4_PRODUCTION_AFTER_NEW_DOUBLE_SEALED_ADMISSION" and
  .p0_count == 0 and .p1_count == 0 and
  .identity.reviewed_receipt_sha256 == $receipt_sha and
  .identity.reviewed_receipt_outer_seal_file_sha256 == $receipt_outer and
  .identity.analyzer_sha256 == "c94b2ca031688158b4bdfa9d0c2aa00931fc7ee880acd5f448eced9a022a3d4f" and
  .identity.preflight_runner_sha256 == "893a89c98ae3ea04fc1c316e71c3768fe5189cc4ce54527e352f0c6f3b3a0944" and
  .identity.execution_contract_sha256 == "c02faf52a1dfbc6936a59dc65c4d089a13e15578407f204c5ef06585b8d68390" and
  .authorization.root_may_create_one_production_admission == true and
  .authorization.cpu_production_runs_directly_authorized == 0
' "${m528_preflight_hammer_json}" >/dev/null || exit 3

# Re-parse the r3 NO-GO and r2 consumed-failure evidence; seals alone are not authority.
m528_verify_sealed_dir "${m528_redteam_dir}" f6af2406b87d1ef596fd472754c85bd69e7256c76c615086d8c9f76162f62d8a
m528_expect "${m528_redteam_json}" 870725a10a843d4742b253d4fbbbead597afc6191609cf63c474bdf69eb821e6
m528_strict "${m528_redteam_json}"
jq -e '.schema == "m528_r3_wrapper_runner_source_redteam_v1" and .status == "FAIL_CURRENT_R3_SOURCE_CHAIN__NO_SMOKE_OR_PRODUCTION_ADMISSION" and .p0_count == 3 and .authorization.current_r3_production_admission == false' "${m528_redteam_json}" >/dev/null || exit 3
m528_verify_sealed_dir "${m528_withdrawal_dir}" 1925fd42a711352750c22aac2f041e15e5db06c27cfc02ea2a2b04ae6c5d1bd0
m528_expect "${m528_withdrawal_json}" a94b4306c1e54ca56d312e9a982b5d2798a1c63633470d1da74321f0f6ef4747
m528_strict "${m528_withdrawal_json}"
jq -e '
  .schema == "m528_r3_recovery_static_hammer_withdrawal_redteam_v1" and
  .status == "WITHDRAW_PRIOR_STATIC_PASS__R3_PERMANENTLY_NO_LAUNCH__AUTHOR_MINIMAL_R4" and
  .verdict == "R3_NO_GO_SMOKE_OR_PRODUCTION__PRESERVE_WITHDRAWN_REVIEW_AS_ERROR_EVIDENCE" and
  .withdrawn_review.may_authorize_smoke == false and
  .withdrawn_review.may_authorize_production == false and
  .r3_identity.permanently_no_launch == true and
  .authorization.r3_smoke_admission == false and
  .authorization.r3_production_admission == false and
  .authorization.r3_cpu_production == false and
  .authorization.author_minimal_r4 == true
' "${m528_withdrawal_json}" >/dev/null || exit 3
m528_verify_sealed_dir "${m528_r2_review_dir}" 8f835eef07233491f2a236e026ec3a9e63448567a361ae0584f5c2b64875e5c9
m528_expect "${m528_r2_review_json}" b6508f216bdfcf9610af7eb23c6f1f7ebe3b1607a8f5eabe4135884a12f0c3a2
m528_strict "${m528_r2_review_json}"
jq -e '.schema == "m528_r2_consumed_failure_failclosed_hammer_v1" and .status == "FAIL_R2_PERMANENTLY_CONSUMED__CONDITIONAL_GO_AUTHOR_MINIMAL_R3" and .sealed_failure_evidence.canonical_absent_at_review == true and .r2_permanent_boundary.rerun_forbidden == true and .authorization.r2_rerun == false' "${m528_r2_review_json}" >/dev/null || exit 3
m528_verify_sealed_dir "${m528_r2_attempt}" acd6ff56930fe943fb1c1e52774daf28952f3ddd241501bc12b5fe93a0f6506a
m528_verify_sealed_dir "${m528_r2_quarantine}" 2483f581635f2c3e662ce167360259e5da786e8b54c445f100a229e1d3a7ae79
[[ ! -e "${m528_r2_canonical}" ]] || exit 3

for process_name in dc_shell dc_shell-t fm_shell pt_shell vcs vcs1 vlogan simv; do
    pgrep -u "$(id -u)" -x "${process_name}" >/dev/null && exit 4
done
m528_resource_snapshot() {
    local log=$1 limit committed available swap headroom failcnt under oomkill
    limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    headroom=$((limit - committed))
    failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    under=$(awk '/^under_oom / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    oomkill=$(awk '/^oom_kill / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'timestamp=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "$(date --iso-8601=seconds)" "${headroom}" "${available}" "${swap}" "${failcnt}" "${under}" "${oomkill}" >>"${log}"
    [[ "${headroom}" -ge 50331648 && "${available}" -ge 134217728 && "${swap}" -ge 33554432 \
       && "${failcnt}" -eq 0 && "${under}" -eq 0 && "${oomkill}" -eq 0 ]]
}

mkdir "${m528_work}"
m528_complete=0
m528_canonical_committed=0
m528_attempt_consumed=0
m528_cleanup() {
    local rc=$? target
    set +e
    if [[ "${m528_complete}" -ne 1 && "${m528_canonical_committed}" -ne 1 && -d "${m528_work}" ]]; then
        if [[ "${m528_attempt_consumed}" -eq 0 ]]; then
            target="${m528_pre_quarantine}"
            printf 'status=PRE_ATTEMPT_FAILURE_DOES_NOT_CONSUME_PRODUCTION_ATTEMPT\nrunner_exit_code=%s\n' "${rc}" >"${m528_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        else
            target="${m528_post_quarantine}"
            printf 'status=FAILED_AFTER_ATTEMPT_DO_NOT_CITE\nrunner_exit_code=%s\n' "${rc}" >"${m528_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        fi
        (cd "${m528_work}" && find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
        mv -T "${m528_work}" "${target}"
    fi
    return "${rc}"
}
trap m528_cleanup EXIT

for sample in 1 2 3; do m528_resource_snapshot "${m528_work}/resource_preflight.log" || exit 40; done
"${m528_python}" "${m528_analyzer}" \
    --execution-contract "${m528_execution}" --schema-smoke-only --spawn-import-self-test \
    --smoke-expected-pointer generated_view_inventory.slow.area_um2 \
    --smoke-expected-corner ssg0p9v125c --out "${m528_work}/FORBIDDEN_SMOKE_OUTPUT" \
    >"${m528_work}/pre_attempt_spawn_schema.stdout" 2>"${m528_work}/pre_attempt_spawn_schema.stderr"
[[ "$(grep -Fxc PASS_M528_R4_PREFLIGHT_SCHEMA_AND_SPAWN_IMPORT_SELF_TEST "${m528_work}/pre_attempt_spawn_schema.stdout")" -eq 1 && \
   ! -s "${m528_work}/pre_attempt_spawn_schema.stderr" && ! -e "${m528_work}/FORBIDDEN_SMOKE_OUTPUT" ]] || exit 41

mkdir "${m528_attempt}"
printf 'status=CONSUMED_AT_FIRST_R4_CPU_PRODUCTION_LAUNCH_AFTER_PREFLIGHT\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m528_canonical}" >"${m528_attempt}/ATTEMPT_CONSUMED.txt"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" "${m528_governing}" "${m528_admission}" >"${m528_attempt}/identity.sha256"
(cd "${m528_attempt}" && sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
m528_attempt_consumed=1

set +e
"${m528_python}" "${m528_analyzer}" --execution-contract "${m528_execution}" \
    --workers 3 --chunksize 2 --out "${m528_work}/result" \
    >"${m528_work}/production_stdout.log" 2>"${m528_work}/production_stderr.log"
m528_rc=$?
set -e
[[ "${m528_rc}" -eq 0 ]] || exit "${m528_rc}"
(cd "${m528_work}/result" && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
mv "${m528_work}/production_stdout.log" "${m528_work}/result/"
mv "${m528_work}/production_stderr.log" "${m528_work}/result/"
mv "${m528_work}/resource_preflight.log" "${m528_work}/result/"
mv "${m528_work}/pre_attempt_spawn_schema.stdout" "${m528_work}/result/"
mv "${m528_work}/pre_attempt_spawn_schema.stderr" "${m528_work}/result/"
cp "${m528_admission}" "${m528_work}/result/production_admission.json"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" "${m528_governing}" "${m528_admission}" >"${m528_work}/result/input_identity.sha256"
printf 'status=PASS_M528_R4_RAW_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER\npaper_admitted=false\nsystem_speedup=false\n' >"${m528_work}/result/RUN_COMPLETE.txt"
(cd "${m528_work}/result" && find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
[[ -z "$(find "${m528_work}" -mindepth 1 -maxdepth 1 ! -name result -print -quit)" ]] || exit 6
mv -T "${m528_work}/result" "${m528_canonical}"
m528_canonical_committed=1
rmdir "${m528_work}"
m528_complete=1
trap - EXIT
echo "PASS M528 r4 raw CPU result sealed; independent result hammer required"
