#!/usr/bin/env bash
set -euo pipefail

# Additive CPU-only successor to blocked M1028.  All authority paths, process
# exclusions, global replay lock and memory floors are fixed before attempt.
hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
contract="${hw_root}/contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
engine="${hw_root}/system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
m1025="${hw_root}/reviews/m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_20260829"
m1036="${hw_root}/reviews/m1036_m1026_m1027_m1028_c1_full_replay_cross_hammer_r1_20260829"
release="${hw_root}/contracts/m1038_m1037_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
release_hammer="${hw_root}/reviews/m1039_m1038_m1036_m1040_c1_full_replay_release_hammer_r1_20260829"
python=/opt/anaconda3/envs/pytorch310/bin/python3.10
pgrep_bin=/usr/bin/pgrep
flock_bin=/usr/bin/flock
meminfo=/proc/meminfo
lockfile="${hw_root}/results/.c1_full_matched_address_replay_global.lock"
result="${hw_root}/results/m1040_m1016_c1_full_matched_address_replay_r1_20260829"
attempt="${hw_root}/results/.m1040_m1016_c1_full_matched_address_replay_attempt_consumed"
work="${hw_root}/results/.m1040_m1016_c1_full_matched_address_replay_work.$$"
failure="${hw_root}/results/m1040_m1016_c1_full_matched_address_replay_r1_20260829.failed_or_incomplete.$$.quarantine"
phase=SOURCE_PREFLIGHT
attempt_consumed=0
complete=0

readonly expected_contract_sha=b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90
readonly expected_engine_sha=d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa
readonly expected_m1025_outer=7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff
readonly expected_m1036_outer=476f0779ad32d40831dbcdaa5d4c223d7f6a50d9aecb196e63107ee4c1c8f5ae
readonly expected_pgrep_sha=338dbc38325ef890569b352e59317d7decf74554006e9aff51f6247ed0a9c595
readonly expected_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
readonly min_commit_headroom_kb=16777216
readonly min_mem_available_kb=16777216

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M1040 M1016 gate failure: %s\n' "$*" >&2; exit 3; }
expect_sha() {
    [[ -f "$1" && ! -L "$1" && "$(sha "$1")" == "$2" ]] || fail "identity drift: $1"
}
verify_seal() {
    local dir=$1 expected_outer=$2
    [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
       && -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "missing sealed directory: ${dir}"
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null \
       && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "seal failure: ${dir}"
    [[ "$(sha "${dir}/SHA256SUMS.seal.sha256")" == "${expected_outer}" ]] \
      || fail "outer seal identity drift: ${dir}"
}
verify_release_sidecars() {
    local sidecar="${release}.sha256" outer="${release}.sha256.seal.sha256"
    [[ -f "${release}" && ! -L "${release}" && -f "${sidecar}" && ! -L "${sidecar}" \
       && -f "${outer}" && ! -L "${outer}" ]] || fail "M1038 release/sidecar absent"
    (cd "$(dirname "${release}")" && sha256sum -c "$(basename "${sidecar}")" >/dev/null \
       && sha256sum -c "$(basename "${outer}")" >/dev/null) || fail "M1038 release sidecar failure"
}
seal_dir() {
    local dir=$1
    (cd "${dir}" && find . -type f ! -name SHA256SUMS \
       ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS \
       && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
}
process_collision_gate() {
    local name
    for name in vcs1 vlogan dc_shell dc_shell-t fm_shell pt_shell; do
        ! "${pgrep_bin}" -x "${name}" >/dev/null || fail "process collision: ${name}"
    done
}
resource_gate() {
    local commit_limit committed_as mem_available commit_headroom
    [[ -f "${meminfo}" && ! -L "${meminfo}" ]] || fail "meminfo absent or symlink"
    commit_limit=$(awk '$1=="CommitLimit:"{print $2}' "${meminfo}")
    committed_as=$(awk '$1=="Committed_AS:"{print $2}' "${meminfo}")
    mem_available=$(awk '$1=="MemAvailable:"{print $2}' "${meminfo}")
    [[ "${commit_limit}" =~ ^[0-9]+$ && "${committed_as}" =~ ^[0-9]+$ \
       && "${mem_available}" =~ ^[0-9]+$ ]] || fail "meminfo fields invalid"
    (( commit_limit >= committed_as )) || fail "negative commit headroom"
    commit_headroom=$((commit_limit - committed_as))
    (( commit_headroom >= min_commit_headroom_kb )) \
      || fail "CommitLimit-Committed_AS below 16GiB floor"
    (( mem_available >= min_mem_available_kb )) \
      || fail "MemAvailable below 16GiB floor"
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${complete}" -ne 1 && "${attempt_consumed}" -eq 1 ]]; then
        [[ "${rc}" -ne 0 ]] || rc=97
        failure_work="${failure}.work"
        mkdir "${failure_work}" || { printf 'M1040 failure quarantine collision\n' >&2; exit "${rc}"; }
        if [[ -d "${work}" ]]; then
            mv -T "${work}" "${failure_work}/partial_result"
        fi
        printf '{"status":"FAILED_OR_INCOMPLETE","phase":"%s","return_code":%s}\n' \
          "${phase}" "${rc}" >"${failure_work}/failure.json"
        seal_dir "${failure_work}" || { printf 'M1040 failure seal failed at %s\n' "${failure_work}" >&2; exit "${rc}"; }
        mv -T "${failure_work}" "${failure}" || { printf 'M1040 failure publish failed at %s\n' "${failure_work}" >&2; exit "${rc}"; }
    fi
    exit "${rc}"
}

expect_sha "${pgrep_bin}" "${expected_pgrep_sha}"
expect_sha "${flock_bin}" "${expected_flock_sha}"
[[ -n "${M1040_EXPECTED_RUNNER_SHA256:-}" \
   && "$(sha "${runner}")" == "${M1040_EXPECTED_RUNNER_SHA256}" ]] \
  || fail "caller must pin exact M1040 runner SHA"
[[ "${M1040_EXPECTED_M1025_OUTER_SHA256:-}" == "${expected_m1025_outer}" \
   && "${M1040_EXPECTED_M1036_OUTER_SHA256:-}" == "${expected_m1036_outer}" \
   && -n "${M1040_EXPECTED_M1039_OUTER_SHA256:-}" ]] \
  || fail "caller must pin exact M1025, M1036 and M1039 outer SHAs"
[[ "$(sha "${contract}")" == "${expected_contract_sha}" \
   && "$(sha "${engine}")" == "${expected_engine_sha}" ]] \
  || fail "hardcoded M1016 source identity drift"
verify_seal "${m1025}" "${expected_m1025_outer}"
verify_seal "${m1036}" "${expected_m1036_outer}"
verify_release_sidecars
verify_seal "${release_hammer}" "${M1040_EXPECTED_M1039_OUTER_SHA256}"
[[ "$(jq -r '.status' "${m1025}/review.json")" \
      == PASS_M1025_M1016_C1_FULL_MATCHED_ADDRESS_REPLAY_SOURCE_HAMMER \
   && "$(jq -r '.status' "${m1036}/review.json")" \
      == FAIL_M1036_M1026_M1027_M1028_C1_FULL_REPLAY_CROSS_HAMMER \
   && "$(jq -r '.status' "${release}")" \
      == PASS_M1038_M1037_M1016_C1_FULL_REPLAY_LAUNCH_RELEASE \
   && "$(jq -r '.launch_now' "${release}")" == true \
   && "$(jq -r '.max_attempts' "${release}")" == 1 \
   && "$(jq -r '.runner_sha256' "${release}")" == "$(sha "${runner}")" \
   && "$(jq -r '.source_contract_sha256' "${release}")" == "${expected_contract_sha}" \
   && "$(jq -r '.engine_sha256' "${release}")" == "${expected_engine_sha}" \
   && "$(jq -r '.m1025.outer_seal_file_sha256' "${release}")" == "${expected_m1025_outer}" \
   && "$(jq -r '.m1036.outer_seal_file_sha256' "${release}")" == "${expected_m1036_outer}" \
   && "$(jq -r '.status' "${release_hammer}/review.json")" \
      == PASS_M1039_M1038_M1036_M1040_C1_FULL_REPLAY_RELEASE_HAMMER \
   && "$(jq -r '.identity.m1038_release_sha256' "${release_hammer}/review.json")" == "$(sha "${release}")" \
   && "$(jq -r '.identity.m1040_runner_sha256' "${release_hammer}/review.json")" == "$(sha "${runner}")" \
   && "$(jq -r '.identity.m1025_outer_seal_file_sha256' "${release_hammer}/review.json")" == "${expected_m1025_outer}" \
   && "$(jq -r '.identity.m1036_outer_seal_file_sha256' "${release_hammer}/review.json")" == "${expected_m1036_outer}" ]] \
  || fail "hardcoded M1040 execution authority content mismatch"

# All three runtime gates are before namespace consumption. The lock is held
# on lock_fd until this process exits; unrelated CPU process names are ignored.
process_collision_gate
[[ ! -L "${lockfile}" ]] || fail "global replay lockfile is symlink"
exec {lock_fd}>>"${lockfile}"
"${flock_bin}" -n "${lock_fd}" || fail "C1 full replay lock collision"
resource_gate
[[ ! -e "${result}" && ! -e "${attempt}" && ! -e "${work}" ]] \
  || fail "M1040 result/attempt/work collision"

phase=ATTEMPT_ATOMIC_CONSUME
mkdir "${attempt}" || fail "M1040 attempt already consumed"
attempt_consumed=1
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf '{"status":"M1040_ATTEMPT_CONSUMED","runner_sha256":"%s","contract_sha256":"%s","release_sha256":"%s","lockfile":"%s","min_commit_headroom_kb":%s,"min_mem_available_kb":%s}\n' \
  "$(sha "${runner}")" "${expected_contract_sha}" "$(sha "${release}")" "${lockfile}" \
  "${min_commit_headroom_kb}" "${min_mem_available_kb}" >"${attempt}/attempt.json"
seal_dir "${attempt}"

phase=FULL_51840000_CPU_REPLAY
"${python}" "${engine}" --contract "${contract}" --out "${work}"
payload="${work}/m1016_c1_full_matched_address_replay_result_r1.json"
[[ -f "${payload}" \
   && "$(jq -r '.status' "${payload}")" \
      == PASS_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER \
   && "$(jq -r '.coverage.raw_full_replay_complete' "${payload}")" == true \
   && "$(jq -r '.claim_boundary.capacity_only_214912B_admitted' "${payload}")" == false \
   && "$(jq -r '.claim_boundary.matched_total_cycles_admitted' "${payload}")" == false \
   && "$(jq -r '.claim_boundary.speedup_admitted' "${payload}")" == false ]] \
  || fail "M1016 engine result did not close raw fail-closed contract"
(cd "${work}" && sha256sum -c SHA256SUMS >/dev/null \
  && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "M1040 result seal failure"
phase=SUCCESS_PUBLISH
[[ ! -e "${result}" ]] || fail "M1040 result appeared before publish"
mv -T "${work}" "${result}"
complete=1
trap - EXIT
printf '%s\n' PASS_M1040_M1016_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER
