#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH

[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M961 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m961_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m961_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m961_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m961_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m961_driver="${m961_hw}/system_simulator/scripts/execute_m961_m946_decoder_d2d3_10k_bounded_prefix_r1.py"
m961_contract="${m961_hw}/contracts/m961_m946_decoder_d2d3_10k_bounded_prefix_source_contract_r1_20260829.json"
m961_release="${m961_hw}/contracts/m969_m961_decoder_d2d3_10k_bounded_prefix_release_r1_20260829.json"
m961_release_hammer="${m961_hw}/reviews/m970_m969_m961_decoder_d2d3_10k_bounded_prefix_release_hammer_r1_20260829"
m961_result="${m961_hw}/results/m961_m946_decoder_d2d3_10k_bounded_prefix_r1_20260829"
m961_attempt="${m961_hw}/results/.m961_m946_decoder_d2d3_10k_bounded_prefix_r1_attempt_consumed"
m961_attempt_stage="${m961_attempt}.stage.$$.$RANDOM.$RANDOM"
m961_result_stage="${m961_result}.stage.$$.$RANDOM.$RANDOM"
m961_quarantine="${m961_result}.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m961_timeout=/usr/bin/timeout
m961_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m961_started=0
m961_published=0

: "${M961_EXPECTED_RELEASE_SHA256:?M961 inert until exact M969 release SHA is supplied}"
: "${M961_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256:?M961 inert until exact M970 review SHA is supplied}"
: "${M961_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256:?M961 inert until exact M970 manifest SHA is supplied}"
: "${M961_EXPECTED_RELEASE_HAMMER_OUTER_SHA256:?M961 inert until exact M970 outer SHA is supplied}"

m961_sha() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }

[[ "$(readlink -f "${m961_python}")" == "${m961_python}" &&
   "$(m961_sha "${m961_python}")" == "${m961_python_sha}" &&
   "$(m961_sha "${m961_timeout}")" == "${m961_timeout_sha}" ]] || {
    echo "M961 interpreter/timeout identity drift" >&2
    exit 4
}

m961_driver_env() {
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      PYTHONDONTWRITEBYTECODE=1 "${m961_python}" "${m961_driver}" "$@"
}

m961_authority_args() {
    printf '%s\0' \
      --release "${m961_release}" \
      --runner "${m961_runner}" \
      --release-hammer "${m961_release_hammer}" \
      --expected-release-sha256 "${M961_EXPECTED_RELEASE_SHA256}" \
      --expected-release-hammer-review-sha256 \
        "${M961_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" \
      --expected-release-hammer-manifest-sha256 \
        "${M961_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
      --expected-release-hammer-outer-sha256 \
        "${M961_EXPECTED_RELEASE_HAMMER_OUTER_SHA256}"
}

m961_run_driver() {
    local -a authority=()
    while IFS= read -r -d '' item; do authority+=("${item}"); done \
      < <(m961_authority_args)
    m961_driver_env "$@" "${authority[@]}"
}

m961_cleanup() {
    local rc=$?
    if [[ "${m961_started}" -eq 1 && "${m961_published}" -eq 0 &&
          -d "${m961_result_stage}" ]]; then
        /usr/bin/mv -T "${m961_result_stage}" "${m961_quarantine}" || true
    fi
    if [[ -d "${m961_attempt_stage}" ]]; then
        /usr/bin/mv -T "${m961_attempt_stage}" \
          "${m961_quarantine}.attempt_stage" || true
    fi
    exit "${rc}"
}
trap m961_cleanup EXIT HUP INT TERM

[[ ! -e "${m961_result}" && ! -L "${m961_result}" &&
   ! -e "${m961_attempt}" && ! -L "${m961_attempt}" &&
   ! -e "${m961_result_stage}" && ! -L "${m961_result_stage}" &&
   ! -e "${m961_attempt_stage}" && ! -L "${m961_attempt_stage}" ]] || {
    echo "M961 one-attempt namespace is not fresh" >&2
    exit 5
}

read -r m961_disk_kib < <(/usr/bin/df -Pk "$(dirname "${m961_result}")" |
                           /usr/bin/awk 'NR==2 {print $4}')
read -r m961_mem_kib < <(/usr/bin/awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
read -r m961_limit_kib < <(/usr/bin/awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
read -r m961_committed_kib < <(/usr/bin/awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m961_headroom_kib=$((m961_limit_kib - m961_committed_kib))
[[ "${m961_disk_kib}" -ge 2097152 &&
   "${m961_mem_kib}" -ge 16777216 &&
   "${m961_headroom_kib}" -ge 16777216 ]] || {
    echo "M961 requires 2 GiB disk and 16 GiB memory/commit headroom" >&2
    exit 6
}

m961_driver_env --validate-source-contract --contract "${m961_contract}" \
  --runner "${m961_runner}" >/dev/null
m961_run_driver --validate-release >/dev/null
m961_run_driver --consume-attempt --attempt-stage "${m961_attempt_stage}" >/dev/null
m961_started=1

m961_run_driver --validate-release >/dev/null
set +e
"${m961_timeout}" --foreground --signal=TERM --kill-after=30s 1800s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 "${m961_python}" "${m961_driver}" \
  --run-exact-pair --output-stage "${m961_result_stage}" \
  --release "${m961_release}" --runner "${m961_runner}" \
  --release-hammer "${m961_release_hammer}" \
  --expected-release-sha256 "${M961_EXPECTED_RELEASE_SHA256}" \
  --expected-release-hammer-review-sha256 \
    "${M961_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" \
  --expected-release-hammer-manifest-sha256 \
    "${M961_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
  --expected-release-hammer-outer-sha256 \
    "${M961_EXPECTED_RELEASE_HAMMER_OUTER_SHA256}"
m961_rc=$?
set -e
[[ "${m961_rc}" -eq 0 ]] || {
    echo "M961 bounded pair failed or timed out: rc=${m961_rc}" >&2
    exit "${m961_rc}"
}

m961_run_driver --publish-no-replace --output-stage "${m961_result_stage}" >/dev/null
m961_published=1
trap - EXIT HUP INT TERM
printf 'PASS M961 one D2/D3 10K pair published; fresh result hammer required\n'
