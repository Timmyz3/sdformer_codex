#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m1085_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m1085_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1085_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m1085_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m1085_driver="${m1085_hw}/system_simulator/scripts/execute_m1083_decoder_side_normalized_validator_repair.py"
m1085_driver_sha=44aa7367ea8a679ff2b067c80d79320989030bde24a6e75edf96c07340e9bbec
m1085_contract="${m1085_hw}/contracts/m1083_decoder_side_normalized_validator_repair_contract_r1_20260830.json"
m1085_release="${m1085_hw}/contracts/m1083_decoder_side_normalized_validator_repair_release_r1_20260830.json"
m1085_timeout=/usr/bin/timeout
m1085_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m1085_flock=/usr/bin/flock
m1085_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
m1085_result="${m1085_hw}/results/m1085_m1083_decoder_side_normalized_pilot_r1_20260830"
m1085_attempt="${m1085_hw}/results/.m1085_m1083_decoder_side_normalized_pilot_attempt_consumed"
m1085_work="${m1085_hw}/results/.m1085_m1083_decoder_side_normalized_pilot_r1_20260830.work.$$.$RANDOM.$RANDOM"
m1085_quarantine="${m1085_hw}/results/m1085_m1083_decoder_side_normalized_pilot_r1_20260830.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m1085_started=0
m1085_published=0

: "${M1085_EXPECTED_CONTRACT_SHA:?inert until exact M1083 contract pin}"
: "${M1085_EXPECTED_M1084_REVIEW_SHA:?independent M1084 review pin required}"
: "${M1085_EXPECTED_M1084_MANIFEST_SHA:?independent M1084 manifest pin required}"
: "${M1085_EXPECTED_M1084_OUTER_SHA:?independent M1084 outer pin required}"

m1085_sha(){ /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m1085_python}")" == "${m1085_python}" &&
   "$(m1085_sha "${m1085_python}")" == "${m1085_python_sha}" &&
   "$(m1085_sha "${m1085_driver}")" == "${m1085_driver_sha}" &&
   "$(m1085_sha "${m1085_timeout}")" == "${m1085_timeout_sha}" &&
   "$(m1085_sha "${m1085_flock}")" == "${m1085_flock_sha}" ]] || exit 4

m1085_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "${m1085_python}" "${m1085_driver}" "$@"
}
m1085_auth(){
  m1085_py "$@" --runner "${m1085_runner}" \
    --expected-contract-sha "${M1085_EXPECTED_CONTRACT_SHA}" \
    --expected-review-sha "${M1085_EXPECTED_M1084_REVIEW_SHA}" \
    --expected-manifest-sha "${M1085_EXPECTED_M1084_MANIFEST_SHA}" \
    --expected-outer-sha "${M1085_EXPECTED_M1084_OUTER_SHA}"
}
m1085_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m1085_started}" -eq 1 && "${m1085_published}" -eq 0 && -d "${m1085_work}" ]]; then
    m1085_auth --quarantine --work "${m1085_work}" \
      --quarantine-path "${m1085_quarantine}" --return-code "${rc}" >/dev/null || true
  fi
  exit "${rc}"
}
trap m1085_cleanup EXIT HUP INT TERM

[[ ! -e "${m1085_result}" && ! -L "${m1085_result}" &&
   ! -e "${m1085_attempt}" && ! -L "${m1085_attempt}" &&
   ! -e "${m1085_work}" && ! -L "${m1085_work}" &&
   ! -e "${m1085_quarantine}" && ! -L "${m1085_quarantine}" ]] || exit 5

# All checks above this point are source/review/root-seal metadata only.
m1085_py --validate-source-only --contract "${m1085_contract}" \
  --release "${m1085_release}" --runner "${m1085_runner}" >/dev/null
m1085_auth --validate-authority >/dev/null

exec 9>"/tmp/m1085_decoder_side_normalized_pilot.lock"
"${m1085_flock}" -n 9 || exit 6
for process in dc_shell vcs simv fm_shell pt_shell; do
  /usr/bin/pgrep -u "$(/usr/bin/id -u)" -x "${process}" >/dev/null && exit 7
done
read -r mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${mem}" -ge 16777216 && $((limit-used)) -ge 16777216 ]] || exit 8

# Permanent attempt is the first mutation; payload verification follows it.
m1085_auth --consume-attempt --attempt "${m1085_attempt}" >/dev/null
m1085_started=1
/usr/bin/mkdir -m 700 "${m1085_work}"
m1085_auth --validate-payload-after-attempt --attempt "${m1085_attempt}" \
  --work "${m1085_work}" >/dev/null
"${m1085_timeout}" --foreground --signal=TERM --kill-after=60s 14400s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  "${m1085_python}" "${m1085_driver}" --run-pilot \
  --attempt "${m1085_attempt}" --work "${m1085_work}" \
  --runner "${m1085_runner}" --expected-contract-sha "${M1085_EXPECTED_CONTRACT_SHA}" \
  --expected-review-sha "${M1085_EXPECTED_M1084_REVIEW_SHA}" \
  --expected-manifest-sha "${M1085_EXPECTED_M1084_MANIFEST_SHA}" \
  --expected-outer-sha "${M1085_EXPECTED_M1084_OUTER_SHA}" >/dev/null
m1085_auth --assemble --attempt "${m1085_attempt}" --work "${m1085_work}" >/dev/null
m1085_auth --publish --attempt "${m1085_attempt}" --work "${m1085_work}" \
  --result "${m1085_result}" >/dev/null
m1085_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' 'PASS M1085 side-normalized diagnostic pilot published; result hammer required'
