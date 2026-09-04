#!/usr/bin/env bash
set -euo pipefail

m528_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m528_hw_root}"

m528_runner="$(realpath "${BASH_SOURCE[0]}")"
m528_analyzer="system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute_r3.py"
m528_execution="contracts/m528_h67_single_port_same_ledger_execution_contract_r3_20260827.json"
m528_governing="reviews/m528_false_kill_audit_r1_20260827/m528_single_port_same_ledger_recompute_contract_r1_20260827.json"
m528_author_dir="reviews/m528_single_port_same_ledger_recompute_author_handoff_r3_20260827"
m528_static_dir="reviews/m528_r3_recovery_static_hammer_r1_20260827"
m528_smoke_result_dir="results/m528_r3_schema_smoke_r1_20260827"
m528_smoke_hammer_dir="reviews/m528_r3_schema_smoke_receipt_hammer_r1_20260827"
m528_failure_review_dir="reviews/m528_r2_consumed_failure_failclosed_hammer_r1_20260827"
m528_python="/opt/anaconda3/envs/pytorch310/bin/python"
m528_canonical="results/m528_h67_single_port_same_ledger_recompute_r3_20260827"
m528_attempt="results/.m528_h67_single_port_same_ledger_recompute_r3_20260827.attempt_consumed"
m528_work="results/.m528_h67_single_port_same_ledger_recompute_r3_work.$$"
m528_pre_quarantine="${m528_canonical}.pre_attempt_failure.$$.quarantine"
m528_post_quarantine="${m528_canonical}.failed_or_incomplete.$$.quarantine"

m528_sha() { sha256sum "$1" | awk '{print $1}'; }
m528_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m528_sha "${path}")" == "${expected}" ]] || {
        echo "M528 r3 identity mismatch: ${path}" >&2
        exit 3
    }
}

[[ -n "${M528_R3_EXPECTED_RUNNER_SHA256:-}" && \
   "$(m528_sha "${m528_runner}")" == "${M528_R3_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M528 r3 caller must pin the independently reviewed production-runner SHA" >&2
    exit 3
}
[[ -n "${M528_R3_EXPECTED_PRODUCTION_ADMISSION_PATH:-}" && \
   -n "${M528_R3_EXPECTED_PRODUCTION_ADMISSION_SHA256:-}" ]] || {
    echo "M528 r3 caller must pin production admission path and SHA" >&2
    exit 3
}
[[ "${M528_R3_EXPECTED_PRODUCTION_ADMISSION_PATH}" != /* && \
   "${M528_R3_EXPECTED_PRODUCTION_ADMISSION_PATH}" != *".."* ]] || {
    echo "M528 r3 production admission path must be repository-relative" >&2
    exit 3
}
m528_admission="${M528_R3_EXPECTED_PRODUCTION_ADMISSION_PATH}"

[[ ! -e "${m528_canonical}" && ! -e "${m528_attempt}" && ! -e "${m528_work}" ]] || {
    echo "M528 r3 refuses consumed or colliding production identity" >&2
    exit 5
}
[[ -z "${M528_OUTPUT_OVERRIDE:-}" && -z "${M528_WORKERS_OVERRIDE:-}" ]] || {
    echo "M528 r3 output and worker overrides are forbidden" >&2
    exit 5
}

m528_expect "${m528_analyzer}" a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae
m528_expect "${m528_execution}" 680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65
m528_expect "${m528_governing}" d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51
m528_expect "${m528_python}" 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m528_expect "${m528_admission}" "${M528_R3_EXPECTED_PRODUCTION_ADMISSION_SHA256}"
m528_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
m528_expect "${m528_failure_review_dir}/review.json" b6508f216bdfcf9610af7eb23c6f1f7ebe3b1607a8f5eabe4135884a12f0c3a2
m528_expect "${m528_failure_review_dir}/SHA256SUMS.seal.sha256" 8f835eef07233491f2a236e026ec3a9e63448567a361ae0584f5c2b64875e5c9

[[ -d results/.m528_h67_single_port_same_ledger_recompute_r2_20260827.attempt_consumed ]] || {
    echo "M528 r3 refuses missing preserved r2 consumed-attempt evidence" >&2
    exit 3
}
[[ ! -e results/m528_h67_single_port_same_ledger_recompute_r2_20260827 ]] || {
    echo "M528 r3 refuses an unexpected r2 canonical result" >&2
    exit 3
}

jq -e '
  .schema == "m528_single_port_same_ledger_static_admission_v3" and
  .status == "AUTHORIZED_ONE_M528_R3_CPU_PRODUCTION_RUN_AFTER_SMOKE_HAMMER" and
  .authorization.cpu_runs == 1 and
  .authorization.schema_smoke_repetitions_before_attempt == 1 and
  .authorization.eda_runs == 0 and
  .authorization.gpu_runs == 0 and
  .authorization.rtl == false and
  .identity.runner_sha256 == env.M528_R3_EXPECTED_RUNNER_SHA256 and
  .identity.analyzer_sha256 == "a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae" and
  .identity.execution_contract_sha256 == "680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65" and
  .identity.governing_contract_sha256 == "d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51" and
  .runtime.workers == 3 and
  .runtime.chunksize == 2 and
  .runtime.minimum_commit_headroom_kib == 50331648 and
  .runtime.minimum_mem_available_kib == 134217728 and
  .runtime.minimum_swap_free_kib == 33554432 and
  .expected.area_json_pointer == "generated_view_inventory.slow.area_um2" and
  .expected.corner == "ssg0p9v125c" and
  .expected.pass_token == "PASS_M528_R3_SCHEMA_SMOKE_ONLY" and
  .docs359_sha256 == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
' "${m528_admission}" >/dev/null || exit 3

for key in author_handoff_outer_seal_file_sha256 static_review_outer_seal_file_sha256 \
        smoke_receipt_outer_seal_file_sha256 smoke_hammer_outer_seal_file_sha256; do
    value="$(jq -er ".identity.${key}" "${m528_admission}")"
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done
m528_expect "${m528_author_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.author_handoff_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_static_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.static_review_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_smoke_result_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.smoke_receipt_outer_seal_file_sha256' "${m528_admission}")"
m528_expect "${m528_smoke_hammer_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.smoke_hammer_outer_seal_file_sha256' "${m528_admission}")"
for sealed_dir in "${m528_author_dir}" "${m528_static_dir}" "${m528_smoke_result_dir}" "${m528_smoke_hammer_dir}" "${m528_failure_review_dir}"; do
    (
        cd "${sealed_dir}"
        sha256sum -c SHA256SUMS >/dev/null
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    ) || exit 3
done

for process_name in dc_shell dc_shell-t fm_shell pt_shell vcs vcs1 vlogan simv; do
    if pgrep -u "$(id -u)" -x "${process_name}" >/dev/null; then
        echo "M528 r3 refuses local EDA/simulation collision: ${process_name}" >&2
        exit 4
    fi
done

m528_resource_snapshot() {
    local log=$1
    local limit committed available swap headroom failcnt under oomkill
    limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    headroom=$((limit - committed))
    failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    under=$(awk '/^under_oom / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    oomkill=$(awk '/^oom_kill / {print $2}' /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'timestamp=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "$(date --iso-8601=seconds)" "${headroom}" "${available}" "${swap}" \
        "${failcnt}" "${under}" "${oomkill}" >>"${log}"
    [[ "${headroom}" -ge 50331648 && "${available}" -ge 134217728 \
       && "${swap}" -ge 33554432 && "${failcnt}" -eq 0 \
       && "${under}" -eq 0 && "${oomkill}" -eq 0 ]]
}

mkdir "${m528_work}"
m528_complete=0
m528_canonical_committed=0
m528_attempt_consumed=0
m528_cleanup() {
    local rc=$?
    local target
    set +e
    if [[ "${m528_complete}" -ne 1 && "${m528_canonical_committed}" -ne 1 && -d "${m528_work}" ]]; then
        if [[ "${m528_attempt_consumed}" -eq 0 ]]; then
            target="${m528_pre_quarantine}"
            printf 'status=PRE_ATTEMPT_FAILURE_DOES_NOT_CONSUME_PRODUCTION_ATTEMPT\nrunner_exit_code=%s\n' \
                "${rc}" >"${m528_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        else
            target="${m528_post_quarantine}"
            printf 'status=FAILED_OR_INCOMPLETE_AFTER_ATTEMPT_DO_NOT_CITE\nrunner_exit_code=%s\n' \
                "${rc}" >"${m528_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        fi
        (
            cd "${m528_work}"
            find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
                -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        )
        mv -T "${m528_work}" "${target}"
    fi
    return "${rc}"
}
trap m528_cleanup EXIT

for sample in 1 2 3; do
    m528_resource_snapshot "${m528_work}/resource_preflight.log" || exit 40
done

# Repeat the exact reviewed positive smoke after dynamic gates and before the
# production-attempt sentinel exists.  A failure is sealed as PRE_ATTEMPT.
"${m528_python}" "${m528_analyzer}" \
    --execution-contract "${m528_execution}" \
    --schema-smoke-only \
    --smoke-expected-pointer generated_view_inventory.slow.area_um2 \
    --smoke-expected-corner ssg0p9v125c \
    --out "${m528_work}/FORBIDDEN_SMOKE_PRODUCTION_OUTPUT" \
    >"${m528_work}/pre_attempt_schema_smoke.stdout" \
    2>"${m528_work}/pre_attempt_schema_smoke.stderr"
[[ "$(grep -Fxc PASS_M528_R3_SCHEMA_SMOKE_ONLY "${m528_work}/pre_attempt_schema_smoke.stdout")" -eq 1 && \
   ! -s "${m528_work}/pre_attempt_schema_smoke.stderr" && \
   ! -e "${m528_work}/FORBIDDEN_SMOKE_PRODUCTION_OUTPUT" ]] || exit 41

mkdir "${m528_attempt}"
printf 'status=CONSUMED_AT_FIRST_CPU_PRODUCTION_LAUNCH_AFTER_SCHEMA_SMOKE\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m528_canonical}" >"${m528_attempt}/ATTEMPT_CONSUMED.txt"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" \
    "${m528_governing}" "${m528_admission}" >"${m528_attempt}/identity.sha256"
(
    cd "${m528_attempt}"
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)
m528_attempt_consumed=1

set +e
"${m528_python}" "${m528_analyzer}" \
    --execution-contract "${m528_execution}" \
    --workers 3 --chunksize 2 --out "${m528_work}/result" \
    >"${m528_work}/production_stdout.log" \
    2>"${m528_work}/production_stderr.log"
m528_rc=$?
set -e
[[ "${m528_rc}" -eq 0 ]] || exit "${m528_rc}"

(
    cd "${m528_work}/result"
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
mv "${m528_work}/production_stdout.log" "${m528_work}/result/"
mv "${m528_work}/production_stderr.log" "${m528_work}/result/"
mv "${m528_work}/resource_preflight.log" "${m528_work}/result/"
mv "${m528_work}/pre_attempt_schema_smoke.stdout" "${m528_work}/result/"
mv "${m528_work}/pre_attempt_schema_smoke.stderr" "${m528_work}/result/"
cp "${m528_admission}" "${m528_work}/result/production_admission.json"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" \
    "${m528_governing}" "${m528_admission}" >"${m528_work}/result/input_identity.sha256"
printf 'status=PASS_M528_R3_RAW_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER\npaper_admitted=false\nsystem_speedup=false\n' \
    >"${m528_work}/result/RUN_COMPLETE.txt"
(
    cd "${m528_work}/result"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)
[[ -z "$(find "${m528_work}" -mindepth 1 -maxdepth 1 ! -name result -print -quit)" ]] || {
    echo "M528 r3 refuses unexpected work-root residue before canonical commit" >&2
    exit 6
}
mv -T "${m528_work}/result" "${m528_canonical}"
m528_canonical_committed=1
rmdir "${m528_work}"
m528_complete=1
trap - EXIT
echo "PASS M528 r3 raw CPU result sealed; independent result hammer required"
