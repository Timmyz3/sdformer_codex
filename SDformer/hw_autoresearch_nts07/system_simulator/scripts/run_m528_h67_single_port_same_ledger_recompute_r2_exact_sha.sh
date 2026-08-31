#!/usr/bin/env bash
set -euo pipefail

m528_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m528_hw_root}"

m528_runner="$(realpath "${BASH_SOURCE[0]}")"
m528_analyzer="system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute.py"
m528_execution="contracts/m528_h67_single_port_same_ledger_execution_contract_r2_20260827.json"
m528_governing="reviews/m528_false_kill_audit_r1_20260827/m528_single_port_same_ledger_recompute_contract_r1_20260827.json"
m528_author_dir="reviews/m528_single_port_same_ledger_recompute_author_handoff_r2_20260827"
m528_python="/opt/anaconda3/envs/pytorch310/bin/python"
m528_canonical="results/m528_h67_single_port_same_ledger_recompute_r2_20260827"
m528_attempt="results/.m528_h67_single_port_same_ledger_recompute_r2_20260827.attempt_consumed"
m528_work="results/.m528_h67_single_port_same_ledger_recompute_r2_work.$$"
m528_quarantine="${m528_canonical}.failed_or_incomplete.$$.quarantine"

m528_sha() { sha256sum "$1" | awk '{print $1}'; }
m528_expect() {
    local path=$1 expected=$2
    [[ -f "${path}" && "$(m528_sha "${path}")" == "${expected}" ]] || {
        echo "M528 identity mismatch: ${path}" >&2
        exit 3
    }
}

[[ -n "${M528_EXPECTED_RUNNER_SHA256:-}" && \
   "$(m528_sha "${m528_runner}")" == "${M528_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M528 caller must pin the independently reviewed r2 runner SHA" >&2
    exit 3
}
[[ -n "${M528_EXPECTED_STATIC_ADMISSION_PATH:-}" && \
   -n "${M528_EXPECTED_STATIC_ADMISSION_SHA256:-}" ]] || {
    echo "M528 caller must pin r2 static admission path and SHA" >&2
    exit 3
}
[[ "${M528_EXPECTED_STATIC_ADMISSION_PATH}" != /* && \
   "${M528_EXPECTED_STATIC_ADMISSION_PATH}" != *".."* ]] || {
    echo "M528 admission path must be a repository-relative non-parent path" >&2
    exit 3
}
m528_admission="${M528_EXPECTED_STATIC_ADMISSION_PATH}"

[[ ! -e "${m528_canonical}" && ! -e "${m528_attempt}" && \
   ! -e "${m528_work}" && ! -e "${m528_quarantine}" ]] || {
    echo "M528 r2 refuses consumed or colliding result identity" >&2
    exit 5
}
[[ -z "${M528_OUTPUT_OVERRIDE:-}" && -z "${M528_WORKERS_OVERRIDE:-}" ]] || {
    echo "M528 output and worker overrides are forbidden" >&2
    exit 5
}

m528_expect "${m528_analyzer}" \
    c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a
m528_expect "${m528_execution}" \
    "$(jq -er '.identity.execution_contract_sha256' "${m528_admission}" 2>/dev/null || true)"
m528_expect "${m528_governing}" \
    "$(jq -er '.identity.governing_contract_sha256' "${m528_admission}" 2>/dev/null || true)"
m528_expect "${m528_python}" \
    9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m528_expect "${m528_admission}" "${M528_EXPECTED_STATIC_ADMISSION_SHA256}"
m528_expect docs/359_DATE终局冻结_20260813.md \
    dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

jq -e '
  .schema == "m528_single_port_same_ledger_static_admission_v2" and
  .status == "AUTHORIZED_ONE_M528_R2_CPU_PRODUCTION_RUN" and
  .authorization.cpu_runs == 1 and
  .authorization.eda_runs == 0 and
  .authorization.gpu_runs == 0 and
  .authorization.rtl == false and
  .identity.runner_sha256 == env.M528_EXPECTED_RUNNER_SHA256 and
  .identity.analyzer_sha256 == "c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a" and
  .runtime.workers == 3 and
  .runtime.chunksize == 2 and
  .runtime.minimum_commit_headroom_kib == 50331648 and
  .runtime.minimum_mem_available_kib == 134217728 and
  .runtime.minimum_swap_free_kib == 33554432 and
  .docs359_sha256 == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
' "${m528_admission}" >/dev/null || exit 3

for key in execution_contract_sha256 governing_contract_sha256 \
        author_handoff_outer_seal_file_sha256; do
    value="$(jq -er ".identity.${key}" "${m528_admission}")"
    [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || exit 3
done
m528_expect "${m528_author_dir}/SHA256SUMS.seal.sha256" \
    "$(jq -er '.identity.author_handoff_outer_seal_file_sha256' "${m528_admission}")"
(
    cd "${m528_author_dir}"
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
) || exit 3

# A CPU-only milestone must not silently overlap this user's Synopsys/VCS run.
for process_name in dc_shell dc_shell-t fm_shell pt_shell vcs vcs1 vlogan simv; do
    if pgrep -u "$(id -u)" -x "${process_name}" >/dev/null; then
        echo "M528 refuses local EDA/simulation collision: ${process_name}" >&2
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
    under=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    oomkill=$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'timestamp=%s commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "$(date --iso-8601=seconds)" "${headroom}" "${available}" \
        "${swap}" "${failcnt}" "${under}" "${oomkill}" >>"${log}"
    [[ "${headroom}" -ge 50331648 && "${available}" -ge 134217728 \
       && "${swap}" -ge 33554432 && "${failcnt}" -eq 0 \
       && "${under}" -eq 0 && "${oomkill}" -eq 0 ]]
}

mkdir "${m528_work}"
m528_complete=0
m528_canonical_committed=0
m528_cleanup() {
    local rc=$?
    set +e
    if [[ "${m528_complete}" -ne 1 && "${m528_canonical_committed}" -ne 1 \
          && -d "${m528_work}" ]]; then
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "${rc}" >"${m528_work}/RUN_FAILED_OR_INCOMPLETE.txt"
        (
            cd "${m528_work}"
            find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
                -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
            sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
        )
        mv -T "${m528_work}" "${m528_quarantine}"
    fi
    return "${rc}"
}
trap m528_cleanup EXIT

for sample in 1 2 3; do
    m528_resource_snapshot "${m528_work}/resource_preflight.log" || exit 40
done

mkdir "${m528_attempt}"
printf 'status=CONSUMED_AT_FIRST_CPU_PRODUCTION_LAUNCH\ntimestamp=%s\ncanonical=%s\n' \
    "$(date --iso-8601=seconds)" "${m528_canonical}" \
    >"${m528_attempt}/ATTEMPT_CONSUMED.txt"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" \
    "${m528_governing}" "${m528_admission}" \
    >"${m528_attempt}/identity.sha256"
(
    cd "${m528_attempt}"
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)

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
cp "${m528_admission}" "${m528_work}/result/static_admission.json"
sha256sum "${m528_runner}" "${m528_analyzer}" "${m528_execution}" \
    "${m528_governing}" "${m528_admission}" \
    >"${m528_work}/result/input_identity.sha256"
printf 'status=PASS_M528_RAW_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER\npaper_admitted=false\nsystem_speedup=false\n' \
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
    echo "M528 r2 refuses unexpected work-root residue before canonical commit" >&2
    exit 6
}
mv -T "${m528_work}/result" "${m528_canonical}"
m528_canonical_committed=1
rmdir "${m528_work}"
m528_complete=1
trap - EXIT
echo "PASS M528 r2 raw CPU result sealed at ${m528_canonical}; independent result hammer required"
