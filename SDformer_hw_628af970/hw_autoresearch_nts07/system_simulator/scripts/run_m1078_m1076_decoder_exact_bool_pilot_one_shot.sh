#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m1078_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m1078_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1078_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m1078_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m1078_driver="${m1078_hw}/system_simulator/scripts/execute_m1076_decoder_exact_bool_repair.py"
m1078_driver_sha=d3b98ec71c3123c856d6a7ce8c8cee431e4d8d0da75aebf92eee8e144123ec15
m1078_contract="${m1078_hw}/contracts/m1076_decoder_exact_bool_repair_contract_r1_20260830.json"
m1078_result="${m1078_hw}/results/m1078_m1076_decoder_exact_bool_pilot_r1_20260830"
m1078_attempt="${m1078_hw}/results/.m1078_m1076_decoder_exact_bool_pilot_attempt_consumed"
m1078_work="${m1078_hw}/results/.m1078_m1076_decoder_exact_bool_pilot_r1_20260830.work.$$.$RANDOM.$RANDOM"
m1078_quarantine="${m1078_hw}/results/m1078_m1076_decoder_exact_bool_pilot_r1_20260830.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m1078_timeout=/usr/bin/timeout
m1078_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m1078_flock=/usr/bin/flock
m1078_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
m1078_started=0
m1078_published=0

: "${M1078_EXPECTED_CONTRACT_SHA:?M1078 inert until exact M1076 contract pin}"
: "${M1078_EXPECTED_M1077_REVIEW_SHA:?M1078 requires independent M1077 review pin}"
: "${M1078_EXPECTED_M1077_MANIFEST_SHA:?M1078 requires independent M1077 manifest pin}"
: "${M1078_EXPECTED_M1077_OUTER_SHA:?M1078 requires independent M1077 outer pin}"

m1078_sha(){ /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m1078_python}")" == "${m1078_python}" &&
   "$(m1078_sha "${m1078_python}")" == "${m1078_python_sha}" &&
   "$(m1078_sha "${m1078_driver}")" == "${m1078_driver_sha}" &&
   "$(m1078_sha "${m1078_timeout}")" == "${m1078_timeout_sha}" &&
   "$(m1078_sha "${m1078_flock}")" == "${m1078_flock_sha}" ]] || exit 4

m1078_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "${m1078_python}" "${m1078_driver}" "$@"
}
m1078_auth(){
  m1078_py "$@" \
    --expected-review-sha "${M1078_EXPECTED_M1077_REVIEW_SHA}" \
    --expected-manifest-sha "${M1078_EXPECTED_M1077_MANIFEST_SHA}" \
    --expected-outer-sha "${M1078_EXPECTED_M1077_OUTER_SHA}"
}

m1078_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m1078_started}" -eq 1 && "${m1078_published}" -eq 0 &&
        -d "${m1078_work}" ]]; then
    if ! m1078_py --quarantine --work "${m1078_work}" \
         --quarantine-path "${m1078_quarantine}" \
         --return-code "${rc}" >/dev/null; then
      /usr/bin/printf 'M1078_WORK_RETAINED_NOT_MOVED=%s\n' "${m1078_work}" >&2
    fi
  fi
  if [[ -d "${m1078_attempt}" ]]; then
    /usr/bin/printf 'M1078_CANONICAL_ATTEMPT_PRESERVED=%s\n' "${m1078_attempt}" >&2
  fi
  exit "${rc}"
}
trap m1078_cleanup EXIT HUP INT TERM

[[ ! -e "${m1078_result}" && ! -L "${m1078_result}" &&
   ! -e "${m1078_attempt}" && ! -L "${m1078_attempt}" &&
   ! -e "${m1078_work}" && ! -L "${m1078_work}" &&
   ! -e "${m1078_quarantine}" && ! -L "${m1078_quarantine}" ]] || exit 5

# Source/root-seal metadata only: no manifest.json or calls/* member access.
m1078_py --validate-pre-attempt-source --contract "${m1078_contract}" \
  --runner "${m1078_runner}" \
  --expected-contract-sha "${M1078_EXPECTED_CONTRACT_SHA}" >/dev/null
m1078_auth --validate-authority >/dev/null

exec 9>"/tmp/m1078_decoder_exact_bool_pilot.lock"
"${m1078_flock}" -n 9 || exit 6
for m1078_process in dc_shell vcs simv fm_shell pt_shell; do
  if /usr/bin/pgrep -u "$(/usr/bin/id -u)" -x "${m1078_process}" >/dev/null; then
    exit 7
  fi
done
read -r m1078_mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r m1078_limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r m1078_used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${m1078_mem}" -ge 16777216 &&
   $((m1078_limit-m1078_used)) -ge 16777216 ]] || exit 8

# First mutation: consume the permanent attempt; only then validate payload.
m1078_auth --consume-attempt --attempt "${m1078_attempt}" \
  --runner "${m1078_runner}" \
  --expected-contract-sha "${M1078_EXPECTED_CONTRACT_SHA}" >/dev/null
m1078_started=1
/usr/bin/mkdir -m 700 "${m1078_work}"
m1078_auth --validate-payload-after-attempt --attempt "${m1078_attempt}" \
  --work "${m1078_work}" --runner "${m1078_runner}" \
  --expected-contract-sha "${M1078_EXPECTED_CONTRACT_SHA}" >/dev/null

"${m1078_timeout}" --foreground --signal=TERM --kill-after=60s 14400s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  "${m1078_python}" "${m1078_driver}" --run-pilot \
  --attempt "${m1078_attempt}" --work "${m1078_work}" \
  --runner "${m1078_runner}" \
  --expected-contract-sha "${M1078_EXPECTED_CONTRACT_SHA}" \
  --expected-review-sha "${M1078_EXPECTED_M1077_REVIEW_SHA}" \
  --expected-manifest-sha "${M1078_EXPECTED_M1077_MANIFEST_SHA}" \
  --expected-outer-sha "${M1078_EXPECTED_M1077_OUTER_SHA}" >/dev/null
m1078_auth --assemble --attempt "${m1078_attempt}" --work "${m1078_work}" \
  --runner "${m1078_runner}" \
  --expected-contract-sha "${M1078_EXPECTED_CONTRACT_SHA}" >/dev/null
m1078_auth --publish --attempt "${m1078_attempt}" --work "${m1078_work}" \
  --result "${m1078_result}" --runner "${m1078_runner}" \
  --expected-contract-sha "${M1078_EXPECTED_CONTRACT_SHA}" >/dev/null
m1078_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' \
  'PASS M1078 exact-bool identity-bound diagnostic pilot published; result hammer required'
