#!/usr/bin/env bash
set -euo pipefail

# Future CPU-only full replay.  Inert without a separately sealed launch
# release and release hammer.  Never invokes VCS/DC/PT/PTPX/GPU/remote tools.
hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
engine="${hw_root}/system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
contract="${hw_root}/contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
m1010="${hw_root}/reviews/m1010_m1007_c1_matched_common_charge_address_replay_source_hammer_r1_20260829"
python=/opt/anaconda3/envs/pytorch310/bin/python3.10
result="${hw_root}/results/m1016_m1010_c1_full_matched_address_replay_r1_20260829"
attempt="${hw_root}/results/.m1016_m1010_c1_full_matched_address_replay_attempt_consumed"
work="${hw_root}/results/.m1016_m1010_c1_full_matched_address_replay_work.$$"
failure="${hw_root}/results/m1016_m1010_c1_full_matched_address_replay_r1_20260829.failed_or_incomplete.$$.quarantine"
phase=SOURCE_PREFLIGHT
consumed=0
complete=0

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M1016 full replay gate failure: %s\n' "$*" >&2; exit 3; }
verify_seal() {
    local dir=$1 expected_outer=$2
    [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
      && -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "missing seal: ${dir}"
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null \
      && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "seal failure: ${dir}"
    [[ "$(sha "${dir}/SHA256SUMS.seal.sha256")" == "${expected_outer}" ]] \
      || fail "outer identity drift: ${dir}"
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${complete}" -ne 1 && "${consumed}" -eq 1 && -d "${work}" ]]; then
        [[ "${rc}" -ne 0 ]] || rc=97
        printf '{"status":"FAILED_OR_INCOMPLETE","phase":"%s","return_code":%s}\n' \
          "${phase}" "${rc}" >"${work}/failure.json"
        (cd "${work}" && find . -type f -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS \
          && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256) || true
        mv -T "${work}" "${failure}" || true
    fi
    exit "${rc}"
}

[[ -n "${M1016_EXPECTED_RUNNER_SHA256:-}" \
   && "$(sha "${runner}")" == "${M1016_EXPECTED_RUNNER_SHA256}" ]] \
  || fail "caller must pin exact M1016 runner"
[[ -n "${M1016_EXPECTED_CONTRACT_SHA256:-}" \
   && "$(sha "${contract}")" == "${M1016_EXPECTED_CONTRACT_SHA256}" ]] \
  || fail "caller must pin exact M1016 contract"
[[ -n "${M1016_RELEASE_JSON:-}" && -n "${M1016_RELEASE_HAMMER_DIR:-}" \
   && -n "${M1016_EXPECTED_RELEASE_SHA256:-}" \
   && -n "${M1016_EXPECTED_RELEASE_HAMMER_OUTER_SHA256:-}" ]] \
  || fail "future release and independent hammer exact pins required"
release="$(realpath "${M1016_RELEASE_JSON}")"
release_hammer="$(realpath "${M1016_RELEASE_HAMMER_DIR}")"
[[ "$(sha "${release}")" == "${M1016_EXPECTED_RELEASE_SHA256}" ]] \
  || fail "future release SHA drift"
verify_seal "${release_hammer}" "${M1016_EXPECTED_RELEASE_HAMMER_OUTER_SHA256}"
verify_seal "${m1010}" 4885bee6283a09551fa5f95088a01683ce2b561e9305a33365ad807bfeb618f7
[[ "$(jq -r '.status' "${contract}")" == PASS_M1016_SOURCE_ONLY__NO_FULL_REPLAY_NO_EDA \
   && "$(jq -r '.launch_now' "${contract}")" == false \
   && "$(jq -r '.status' "${release}")" == PASS_M1016_FULL_REPLAY_LAUNCH_RELEASE \
   && "$(jq -r '.launch_now' "${release}")" == true \
   && "$(jq -r '.max_attempts' "${release}")" == 1 \
   && "$(jq -r '.runner_sha256' "${release}")" == "$(sha "${runner}")" \
   && "$(jq -r '.source_contract_sha256' "${release}")" == "$(sha "${contract}")" \
   && "$(jq -r '.status' "${release_hammer}/review.json")" == PASS_M1016_FULL_REPLAY_RELEASE_HAMMER ]] \
  || fail "future release chain content mismatch"
[[ "$(sha "${engine}")" == "$(jq -r '.source_identity.engine.sha256' "${contract}")" ]] \
  || fail "engine identity drift"
[[ ! -e "${result}" && ! -e "${attempt}" && ! -e "${work}" ]] \
  || fail "M1016 result/attempt/work collision"

phase=ATTEMPT_ATOMIC_CONSUME
mkdir "${attempt}" || fail "M1016 attempt already consumed"
consumed=1
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf '{"status":"M1016_ATTEMPT_CONSUMED","runner_sha256":"%s","contract_sha256":"%s"}\n' \
  "$(sha "${runner}")" "$(sha "${contract}")" >"${attempt}/attempt.json"
(cd "${attempt}" && sha256sum attempt.json >SHA256SUMS \
  && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)

phase=FULL_CPU_REPLAY
"${python}" "${engine}" --contract "${contract}" --out "${work}"
[[ "$(jq -r '.status' "${work}/m1016_c1_full_matched_address_replay_result_r1.json")" \
   == PASS_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER ]] \
  || fail "full replay did not close raw coverage"
(cd "${work}" && sha256sum -c SHA256SUMS >/dev/null \
  && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "result seal failure"
phase=SUCCESS_PUBLISH
[[ ! -e "${result}" ]] || fail "result appeared before publish"
mv -T "${work}" "${result}"
complete=1
trap - EXIT
printf '%s\n' PASS_M1016_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER
