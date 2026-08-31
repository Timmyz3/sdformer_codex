#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH

[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
  echo "M972 refuses startup hooks/exported shell functions" >&2
  exit 3
}

m972_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m972_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m972_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m972_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m972_driver="${m972_hw}/system_simulator/scripts/execute_m972_m971_decoder_d2d3_10k_evidence_safe_r1.py"
m972_contract="${m972_hw}/contracts/m972_m971_decoder_d2d3_10k_evidence_safe_source_contract_r1_20260829.json"
m972_release="${m972_hw}/contracts/m974_m972_decoder_d2d3_10k_evidence_safe_release_r1_20260829.json"
m972_hammer="${m972_hw}/reviews/m975_m974_m972_decoder_d2d3_10k_evidence_safe_release_hammer_r1_20260829"
m972_result="${m972_hw}/results/m972_m946_decoder_d2d3_10k_evidence_safe_r1_20260829"
m972_attempt="${m972_hw}/results/.m972_m946_decoder_d2d3_10k_evidence_safe_r1_attempt_consumed"
m972_attempt_stage="${m972_attempt}.stage.$$.$RANDOM.$RANDOM"
m972_work="${m972_result}.work.$$.$RANDOM.$RANDOM"
m972_quarantine="${m972_result}.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m972_timeout=/usr/bin/timeout
m972_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m972_started=0
m972_published=0

: "${M972_EXPECTED_RELEASE_SHA256:?M972 inert until exact M974 release SHA}"
: "${M972_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256:?M972 inert until exact M975 review SHA}"
: "${M972_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256:?M972 inert until exact M975 manifest SHA}"
: "${M972_EXPECTED_RELEASE_HAMMER_OUTER_SHA256:?M972 inert until exact M975 outer SHA}"

m972_sha() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m972_python}")" == "${m972_python}" &&
   "$(m972_sha "${m972_python}")" == "${m972_python_sha}" &&
   "$(m972_sha "${m972_timeout}")" == "${m972_timeout_sha}" ]] || {
  echo "M972 interpreter/timeout identity drift" >&2
  exit 4
}

m972_driver_env() {
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 "${m972_python}" "${m972_driver}" "$@"
}

m972_authority() {
  m972_driver_env "$@" \
    --release "${m972_release}" --runner "${m972_runner}" \
    --release-hammer "${m972_hammer}" \
    --expected-release-sha256 "${M972_EXPECTED_RELEASE_SHA256}" \
    --expected-release-hammer-review-sha256 \
      "${M972_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" \
    --expected-release-hammer-manifest-sha256 \
      "${M972_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
    --expected-release-hammer-outer-sha256 \
      "${M972_EXPECTED_RELEASE_HAMMER_OUTER_SHA256}"
}

m972_cleanup() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m972_started}" -eq 1 && "${m972_published}" -eq 0 &&
        -d "${m972_work}" ]]; then
    if [[ ! -f "${m972_work}/SHA256SUMS" ]]; then
      m972_driver_env --seal-failure-root --work-root "${m972_work}" \
        --return-code "${rc}" >/dev/null 2>&1 || true
    fi
    /usr/bin/mv -T "${m972_work}" "${m972_quarantine}" || true
  fi
  if [[ -d "${m972_attempt_stage}" ]]; then
    /usr/bin/mv -T "${m972_attempt_stage}" \
      "${m972_quarantine}.attempt_stage" || true
  fi
  exit "${rc}"
}
trap m972_cleanup EXIT HUP INT TERM

[[ ! -e "${m972_result}" && ! -L "${m972_result}" &&
   ! -e "${m972_attempt}" && ! -L "${m972_attempt}" &&
   ! -e "${m972_work}" && ! -L "${m972_work}" ]] || {
  echo "M972 one-attempt namespace is not fresh" >&2
  exit 5
}

read -r m972_disk_kib < <(/usr/bin/df -Pk "$(dirname "${m972_result}")" |
                           /usr/bin/awk 'NR==2 {print $4}')
read -r m972_mem_kib < <(/usr/bin/awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
read -r m972_limit_kib < <(/usr/bin/awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
read -r m972_committed_kib < <(/usr/bin/awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m972_headroom_kib=$((m972_limit_kib - m972_committed_kib))
[[ "${m972_disk_kib}" -ge 2097152 && "${m972_mem_kib}" -ge 16777216 &&
   "${m972_headroom_kib}" -ge 16777216 ]] || {
  echo "M972 requires 2 GiB disk and 16 GiB memory/commit headroom" >&2
  exit 6
}

m972_driver_env --validate-source-contract --contract "${m972_contract}" \
  --runner "${m972_runner}" >/dev/null
m972_authority --validate-release >/dev/null
m972_authority --consume-attempt --attempt-stage "${m972_attempt_stage}" >/dev/null
m972_started=1
/usr/bin/mkdir -m 700 "${m972_work}"
/usr/bin/printf '%s\n' 'M972_WORK_ROOT_CREATED_BEFORE_D2' > \
  "${m972_work}/WORK_STARTED.txt"

# Order is evidence authority: D2 is completed and sealed before D3 exists.
"${m972_timeout}" --foreground --signal=TERM --kill-after=30s 1800s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 "${m972_python}" "${m972_driver}" \
    --run-row D2 --row-stage "${m972_work}/D2" \
    --release "${m972_release}" --runner "${m972_runner}" \
    --release-hammer "${m972_hammer}" \
    --expected-release-sha256 "${M972_EXPECTED_RELEASE_SHA256}" \
    --expected-release-hammer-review-sha256 "${M972_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" \
    --expected-release-hammer-manifest-sha256 "${M972_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
    --expected-release-hammer-outer-sha256 "${M972_EXPECTED_RELEASE_HAMMER_OUTER_SHA256}" \
    >/dev/null

"${m972_timeout}" --foreground --signal=TERM --kill-after=30s 1800s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 "${m972_python}" "${m972_driver}" \
    --run-row D3 --row-stage "${m972_work}/D3" \
    --release "${m972_release}" --runner "${m972_runner}" \
    --release-hammer "${m972_hammer}" \
    --expected-release-sha256 "${M972_EXPECTED_RELEASE_SHA256}" \
    --expected-release-hammer-review-sha256 "${M972_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}" \
    --expected-release-hammer-manifest-sha256 "${M972_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}" \
    --expected-release-hammer-outer-sha256 "${M972_EXPECTED_RELEASE_HAMMER_OUTER_SHA256}" \
    >/dev/null

m972_authority --assemble --work-root "${m972_work}" >/dev/null
m972_authority --publish-no-replace --work-root "${m972_work}" >/dev/null
m972_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' 'PASS M972 one D2-then-D3 10K pair published; fresh result hammer required'
