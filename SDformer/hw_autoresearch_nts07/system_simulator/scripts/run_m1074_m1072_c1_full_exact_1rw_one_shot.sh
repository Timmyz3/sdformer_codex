#!/usr/bin/env bash
set -euo pipefail

# Source-only until a different-author M1075 hammer supplies the exact three
# seal identities. No arguments select workload, rows, cycles or capacity.
[[ "$#" -eq 0 ]] || { echo "M1074 takes no arguments" >&2; exit 2; }

m1074_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1074_runner="$(realpath "${BASH_SOURCE[0]}")"
m1074_engine="${m1074_hw_root}/system_simulator/scripts/execute_m1074_m1072_c1_full_exact_1rw_one_shot.py"
m1074_contract="${m1074_hw_root}/contracts/m1074_m1073_m1072_c1_full_exact_1rw_one_shot_source_contract_r1_20260830.json"
m1074_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m1074_pgrep=/usr/bin/pgrep
m1074_flock=/usr/bin/flock
m1074_meminfo=/proc/meminfo
m1074_lock="${m1074_hw_root}/results/.c1_full_matched_address_replay_global.lock"
m1074_result="${m1074_hw_root}/results/m1074_m1072_c1_full_exact_1rw_replay_r1_20260830"
m1074_attempt="${m1074_hw_root}/results/.m1074_m1072_c1_full_exact_1rw_replay_attempt_consumed"
m1074_work="${m1074_hw_root}/results/.m1074_m1072_c1_full_exact_1rw_replay_work.$$"
m1074_failure="${m1074_hw_root}/results/m1074_m1072_c1_full_exact_1rw_replay_r1_20260830.failed_or_incomplete.$$.quarantine"
m1074_phase=SOURCE_PREFLIGHT
m1074_complete=0

readonly m1074_expected_engine_sha=90ead8cb4a0196114dbb6c51f4fe9e042fee1bf2816855687327221c8c3274e5
readonly m1074_expected_contract_sha=5d385afe4c0b5875568b19f903d1ed56a224d79790c206a62a28fdeefb967a67
readonly m1074_expected_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
readonly m1074_expected_pgrep_sha=338dbc38325ef890569b352e59317d7decf74554006e9aff51f6247ed0a9c595
readonly m1074_expected_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
readonly m1074_min_commit_headroom_kb=16777216
readonly m1074_min_mem_available_kb=16777216

m1074_sha() { sha256sum "$1" | awk '{print $1}'; }
m1074_fail() { printf 'M1074 gate failure: %s\n' "$*" >&2; exit 3; }
m1074_expect_sha() {
    [[ -f "$1" && ! -L "$1" && "$(m1074_sha "$1")" == "$2" ]] \
        || m1074_fail "identity drift: $1"
}
m1074_python_call() {
    PYTHONNOUSERSITE=1 PYTHONPATH= "${m1074_python}" -I "${m1074_engine}" "$@"
}
m1074_authority_args() {
    printf '%s\n' \
      --runner "${m1074_runner}" \
      --expected-m1075-review-sha "${M1074_EXPECTED_M1075_REVIEW_SHA256:-}" \
      --expected-m1075-manifest-sha "${M1074_EXPECTED_M1075_MANIFEST_SHA256:-}" \
      --expected-m1075-outer-sha "${M1074_EXPECTED_M1075_OUTER_SHA256:-}"
}
m1074_process_gate() {
    local name
    for name in vcs1 vlogan dc_shell dc_shell-t fm_shell pt_shell; do
        ! "${m1074_pgrep}" -x "${name}" >/dev/null \
            || m1074_fail "process collision: ${name}"
    done
}
m1074_resource_gate() {
    local commit_limit committed_as mem_available headroom
    [[ -f "${m1074_meminfo}" && ! -L "${m1074_meminfo}" ]] \
        || m1074_fail "meminfo absent or symlink"
    commit_limit=$(awk '$1=="CommitLimit:"{print $2}' "${m1074_meminfo}")
    committed_as=$(awk '$1=="Committed_AS:"{print $2}' "${m1074_meminfo}")
    mem_available=$(awk '$1=="MemAvailable:"{print $2}' "${m1074_meminfo}")
    [[ "${commit_limit}" =~ ^[0-9]+$ && "${committed_as}" =~ ^[0-9]+$ \
       && "${mem_available}" =~ ^[0-9]+$ ]] \
        || m1074_fail "meminfo fields invalid"
    (( commit_limit >= committed_as )) || m1074_fail "negative commit headroom"
    headroom=$((commit_limit - committed_as))
    (( headroom >= m1074_min_commit_headroom_kb )) \
        || m1074_fail "CommitLimit-Committed_AS below 16GiB"
    (( mem_available >= m1074_min_mem_available_kb )) \
        || m1074_fail "MemAvailable below 16GiB"
}
m1074_cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${m1074_complete}" -eq 1 ]]; then
        exit "${rc}"
    fi
    # Atomic result publication is already complete if RESULT exists and WORK
    # does not; a signal in the two instructions after rename must not create a
    # contradictory failure receipt.
    if [[ -d "${m1074_result}" && ! -e "${m1074_work}" ]]; then
        if m1074_python_call --verify-published >/dev/null 2>&1; then
            exit "${rc}"
        fi
    fi
    if [[ -d "${m1074_attempt}" ]]; then
        [[ "${rc}" -ne 0 ]] || rc=97
        m1074_python_call --quarantine-work \
            --work "${m1074_work}" --quarantine "${m1074_failure}" \
            --return-code "${rc}" --phase "${m1074_phase}" \
            || printf 'M1074 quarantine failed; retained stage/work paths\n' >&2
    fi
    exit "${rc}"
}

# Exact source/tool/hammer identities, process exclusions, lock and resources
# all close before the canonical attempt can exist. These modes never open or
# hash the M410 row file.
m1074_expect_sha "${m1074_engine}" "${m1074_expected_engine_sha}"
m1074_expect_sha "${m1074_contract}" "${m1074_expected_contract_sha}"
m1074_expect_sha "${m1074_python}" "${m1074_expected_python_sha}"
m1074_expect_sha "${m1074_pgrep}" "${m1074_expected_pgrep_sha}"
m1074_expect_sha "${m1074_flock}" "${m1074_expected_flock_sha}"
[[ -n "${M1074_EXPECTED_RUNNER_SHA256:-}" \
   && "$(m1074_sha "${m1074_runner}")" == "${M1074_EXPECTED_RUNNER_SHA256}" ]] \
    || m1074_fail "caller must pin exact runner SHA"
[[ "${M1074_EXPECTED_M1075_REVIEW_SHA256:-}" =~ ^[0-9a-f]{64}$ \
   && "${M1074_EXPECTED_M1075_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ \
   && "${M1074_EXPECTED_M1075_OUTER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
    || m1074_fail "caller must pin exact M1075 review/manifest/outer SHAs"
m1074_python_call --validate-source --runner "${m1074_runner}" >/dev/null
mapfile -t m1074_auth < <(m1074_authority_args)
m1074_python_call --validate-authority "${m1074_auth[@]}" >/dev/null
m1074_process_gate
[[ ! -L "${m1074_lock}" ]] || m1074_fail "global replay lock is symlink"
exec {m1074_lock_fd}>>"${m1074_lock}"
"${m1074_flock}" -n "${m1074_lock_fd}" \
    || m1074_fail "global C1 full-replay lock collision"
m1074_resource_gate
[[ ! -e "${m1074_result}" && ! -e "${m1074_attempt}" \
   && ! -e "${m1074_work}" \
   && ! -e "${m1074_failure}" ]] \
    || m1074_fail "canonical result/attempt/stage/work/quarantine collision"

m1074_phase=ATTEMPT_ATOMIC_CONSUME
trap m1074_cleanup EXIT
trap 'exit 130' INT TERM HUP
m1074_python_call --consume-attempt "${m1074_auth[@]}" >/dev/null
[[ -d "${m1074_attempt}" ]] || m1074_fail "attempt was not atomically consumed"

# This is the first operation that advances M1072 and therefore the first one
# that opens/hashes canonical rows.
m1074_phase=FULL_812160_TASK_51840000_ROW_EXACT_1RW_REPLAY
m1074_python_call --execute-full "${m1074_auth[@]}" \
    --work "${m1074_work}" >/dev/null

m1074_phase=ATOMIC_RESULT_PUBLICATION
m1074_python_call --publish "${m1074_auth[@]}" \
    --work "${m1074_work}" >/dev/null
m1074_complete=1
trap - EXIT INT TERM HUP
printf '%s\n' PASS_M1074_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER
