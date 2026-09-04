#!/usr/bin/env bash
set -euo pipefail

# Exact additive CPU-only execution successor to M1016.  All authority paths
# are hardcoded.  It never invokes VCS/DC/PT/PTPX/GPU/remote tools.
hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
contract="${hw_root}/contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
engine="${hw_root}/system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
m1025="${hw_root}/reviews/m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_20260829"
release="${hw_root}/contracts/m1026_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
release_hammer="${hw_root}/reviews/m1027_m1026_m1016_c1_full_matched_address_replay_release_hammer_r1_20260829"
python=/opt/anaconda3/envs/pytorch310/bin/python3.10
result="${hw_root}/results/m1028_m1016_c1_full_matched_address_replay_r1_20260829"
attempt="${hw_root}/results/.m1028_m1016_c1_full_matched_address_replay_attempt_consumed"
work="${hw_root}/results/.m1028_m1016_c1_full_matched_address_replay_work.$$"
failure="${hw_root}/results/m1028_m1016_c1_full_matched_address_replay_r1_20260829.failed_or_incomplete.$$.quarantine"
phase=SOURCE_PREFLIGHT
attempt_consumed=0
complete=0

expected_contract_sha=b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90
expected_engine_sha=d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa
expected_m1025_outer=7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M1028 M1016 gate failure: %s\n' "$*" >&2; exit 3; }
verify_seal() {
    local dir=$1 expected_outer=$2
    [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
       && -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "missing sealed directory: ${dir}"
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null \
       && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "seal failure: ${dir}"
    [[ "$(sha "${dir}/SHA256SUMS.seal.sha256")" == "${expected_outer}" ]] \
      || fail "outer seal identity drift: ${dir}"
}
seal_dir() {
    local dir=$1
    (cd "${dir}" && find . -type f ! -name SHA256SUMS \
       ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS \
       && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${complete}" -ne 1 && "${attempt_consumed}" -eq 1 ]]; then
        [[ "${rc}" -ne 0 ]] || rc=97
        failure_work="${failure}.work"
        mkdir "${failure_work}" || { printf 'M1028 failure quarantine collision\n' >&2; exit "${rc}"; }
        if [[ -d "${work}" ]]; then
            mv -T "${work}" "${failure_work}/partial_result"
        fi
        printf '{"status":"FAILED_OR_INCOMPLETE","phase":"%s","return_code":%s}\n' \
          "${phase}" "${rc}" >"${failure_work}/failure.json"
        seal_dir "${failure_work}" || { printf 'M1028 failure seal failed at %s\n' "${failure_work}" >&2; exit "${rc}"; }
        mv -T "${failure_work}" "${failure}" || { printf 'M1028 failure publish failed at %s\n' "${failure_work}" >&2; exit "${rc}"; }
    fi
    exit "${rc}"
}

# Caller pins identities, but cannot select any authority path.
[[ -n "${M1028_EXPECTED_RUNNER_SHA256:-}" \
   && "$(sha "${runner}")" == "${M1028_EXPECTED_RUNNER_SHA256}" ]] \
  || fail "caller must pin exact M1028 runner SHA"
[[ "${M1028_EXPECTED_M1025_OUTER_SHA256:-}" == "${expected_m1025_outer}" ]] \
  || fail "caller must pin exact hardcoded M1025 outer SHA"
[[ -n "${M1028_EXPECTED_M1027_OUTER_SHA256:-}" ]] \
  || fail "caller must pin exact hardcoded M1027 outer SHA"
[[ "$(sha "${contract}")" == "${expected_contract_sha}" \
   && "$(sha "${engine}")" == "${expected_engine_sha}" ]] \
  || fail "hardcoded M1016 source identity drift"
verify_seal "${m1025}" "${expected_m1025_outer}"
[[ -f "${release}" && ! -L "${release}" ]] || fail "hardcoded M1026 release absent"
verify_seal "${release_hammer}" "${M1028_EXPECTED_M1027_OUTER_SHA256}"
[[ "$(jq -r '.status' "${m1025}/review.json")" \
      == PASS_M1025_M1016_C1_FULL_MATCHED_ADDRESS_REPLAY_SOURCE_HAMMER \
   && "$(jq -r '.status' "${release}")" \
      == PASS_M1026_M1016_C1_FULL_REPLAY_LAUNCH_RELEASE \
   && "$(jq -r '.launch_now' "${release}")" == true \
   && "$(jq -r '.max_attempts' "${release}")" == 1 \
   && "$(jq -r '.runner_sha256' "${release}")" == "$(sha "${runner}")" \
   && "$(jq -r '.source_contract_sha256' "${release}")" == "${expected_contract_sha}" \
   && "$(jq -r '.engine_sha256' "${release}")" == "${expected_engine_sha}" \
   && "$(jq -r '.m1025_outer_seal_file_sha256' "${release}")" == "${expected_m1025_outer}" \
   && "$(jq -r '.status' "${release_hammer}/review.json")" \
      == PASS_M1027_M1026_M1016_C1_FULL_REPLAY_RELEASE_HAMMER ]] \
  || fail "hardcoded execution authority content mismatch"
[[ ! -e "${result}" && ! -e "${attempt}" && ! -e "${work}" ]] \
  || fail "M1028 result/attempt/work collision"

phase=ATTEMPT_ATOMIC_CONSUME
mkdir "${attempt}" || fail "M1028 attempt already consumed"
attempt_consumed=1
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf '{"status":"M1028_ATTEMPT_CONSUMED","runner_sha256":"%s","contract_sha256":"%s","release_sha256":"%s"}\n' \
  "$(sha "${runner}")" "${expected_contract_sha}" "$(sha "${release}")" \
  >"${attempt}/attempt.json"
seal_dir "${attempt}"

phase=FULL_51840000_CPU_REPLAY
"${python}" "${engine}" --contract "${contract}" --out "${work}"
payload="${work}/m1016_c1_full_matched_address_replay_result_r1.json"
[[ -f "${payload}" \
   && "$(jq -r '.status' "${payload}")" \
      == PASS_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER \
   && "$(jq -r '.coverage.raw_full_replay_complete' "${payload}")" == true \
   && "$(jq -r '.claim_boundary.capacity_only_214912B_admitted' "${payload}")" == false \
   && "$(jq -r '.claim_boundary.speedup_admitted' "${payload}")" == false ]] \
  || fail "M1016 engine result did not close raw fail-closed contract"
(cd "${work}" && sha256sum -c SHA256SUMS >/dev/null \
  && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "M1028 result seal failure"
phase=SUCCESS_PUBLISH
[[ ! -e "${result}" ]] || fail "M1028 result appeared before publish"
mv -T "${work}" "${result}"
complete=1
trap - EXIT
printf '%s\n' PASS_M1028_M1016_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER
