#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" &&
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || exit 3

m1050_runner="$(readlink -f "${BASH_SOURCE[0]}")"
m1050_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1050_python=/opt/anaconda3/envs/pytorch310/bin/python3.10
m1050_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
m1050_driver="${m1050_hw}/system_simulator/scripts/execute_m1048_decoder_stratified_block_reset_pilot_release.py"
m1050_driver_sha=3e2fa596e7cb0406feecc4124280643eaa093df80e9dcc7915fa9dcc7074267a
m1050_contract="${m1050_hw}/contracts/m1048_decoder_stratified_block_reset_pilot_release_contract_r1_20260829.json"
m1050_result="${m1050_hw}/results/m1050_m1048_decoder_stratified_block_reset_pilot_r1_20260829"
m1050_attempt="${m1050_hw}/results/.m1050_m1048_decoder_stratified_block_reset_pilot_attempt_consumed"
m1050_work="${m1050_hw}/results/.m1050_m1048_decoder_stratified_block_reset_pilot_r1_20260829.work.$$.$RANDOM.$RANDOM"
m1050_quarantine="${m1050_hw}/results/m1050_m1048_decoder_stratified_block_reset_pilot_r1_20260829.failed_or_incomplete.$$.$RANDOM.$RANDOM"
m1050_timeout=/usr/bin/timeout
m1050_timeout_sha=2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02
m1050_flock=/usr/bin/flock
m1050_flock_sha=54f8c6b3011cff78d3bf90ba77bdf34e3017c652510a26134ac3509d70947435
m1050_started=0
m1050_published=0

: "${M1050_EXPECTED_CONTRACT_SHA:?M1050 inert until exact M1048 contract pin}"
: "${M1050_EXPECTED_M1049_REVIEW_SHA:?M1050 requires independent M1049 review pin}"
: "${M1050_EXPECTED_M1049_MANIFEST_SHA:?M1050 requires independent M1049 manifest pin}"
: "${M1050_EXPECTED_M1049_OUTER_SHA:?M1050 requires independent M1049 outer pin}"

m1050_sha(){ /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
[[ "$(readlink -f "${m1050_python}")" == "${m1050_python}" &&
   "$(m1050_sha "${m1050_python}")" == "${m1050_python_sha}" &&
   "$(m1050_sha "${m1050_driver}")" == "${m1050_driver_sha}" &&
   "$(m1050_sha "${m1050_timeout}")" == "${m1050_timeout_sha}" &&
   "$(m1050_sha "${m1050_flock}")" == "${m1050_flock_sha}" ]] || exit 4

m1050_py(){
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "${m1050_python}" "${m1050_driver}" "$@"
}
m1050_auth(){
  m1050_py "$@" \
    --expected-review-sha "${M1050_EXPECTED_M1049_REVIEW_SHA}" \
    --expected-manifest-sha "${M1050_EXPECTED_M1049_MANIFEST_SHA}" \
    --expected-outer-sha "${M1050_EXPECTED_M1049_OUTER_SHA}"
}

m1050_cleanup(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ "${m1050_started}" -eq 1 && "${m1050_published}" -eq 0 &&
        -d "${m1050_work}" ]]; then
    if ! m1050_py --quarantine --work "${m1050_work}" \
         --quarantine-path "${m1050_quarantine}" \
         --return-code "${rc}" >/dev/null; then
      /usr/bin/printf 'M1050_WORK_RETAINED_NOT_MOVED=%s\n' "${m1050_work}" >&2
    fi
  fi
  if [[ -d "${m1050_attempt}" ]]; then
    /usr/bin/printf 'M1050_CANONICAL_ATTEMPT_PRESERVED=%s\n' "${m1050_attempt}" >&2
  fi
  exit "${rc}"
}
trap m1050_cleanup EXIT HUP INT TERM

[[ ! -e "${m1050_result}" && ! -L "${m1050_result}" &&
   ! -e "${m1050_attempt}" && ! -L "${m1050_attempt}" &&
   ! -e "${m1050_work}" && ! -L "${m1050_work}" &&
   ! -e "${m1050_quarantine}" && ! -L "${m1050_quarantine}" ]] || exit 5

# Validate every frozen input and the independent GO before any attempt is
# consumed or any real payload file is opened.
m1050_py --validate-source --contract "${m1050_contract}" \
  --runner "${m1050_runner}" \
  --expected-contract-sha "${M1050_EXPECTED_CONTRACT_SHA}" >/dev/null
m1050_auth --validate-authority >/dev/null

# Fixed nonblocking runner lock plus exact EDA collision and memory gates.
exec 9>"/tmp/m1050_decoder_stratified_block_reset_pilot.lock"
"${m1050_flock}" -n 9 || exit 6
for m1050_process in dc_shell vcs simv fm_shell pt_shell; do
  if /usr/bin/pgrep -u "$(/usr/bin/id -u)" -x "${m1050_process}" >/dev/null; then
    exit 7
  fi
done
read -r m1050_mem < <(/usr/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
read -r m1050_limit < <(/usr/bin/awk '/^CommitLimit:/{print $2}' /proc/meminfo)
read -r m1050_used < <(/usr/bin/awk '/^Committed_AS:/{print $2}' /proc/meminfo)
[[ "${m1050_mem}" -ge 16777216 &&
   $((m1050_limit-m1050_used)) -ge 16777216 ]] || exit 8

# The canonical attempt is atomically consumed before work creation and before
# the M699 payload context can be opened.  It is permanent on every failure.
m1050_auth --consume-attempt --attempt "${m1050_attempt}" \
  --runner "${m1050_runner}" \
  --expected-contract-sha "${M1050_EXPECTED_CONTRACT_SHA}" >/dev/null
m1050_started=1
/usr/bin/mkdir -m 700 "${m1050_work}"

"${m1050_timeout}" --foreground --signal=TERM --kill-after=60s 14400s \
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  "${m1050_python}" "${m1050_driver}" --run-pilot \
  --work "${m1050_work}" --attempt "${m1050_attempt}" \
  --expected-review-sha "${M1050_EXPECTED_M1049_REVIEW_SHA}" \
  --expected-manifest-sha "${M1050_EXPECTED_M1049_MANIFEST_SHA}" \
  --expected-outer-sha "${M1050_EXPECTED_M1049_OUTER_SHA}" >/dev/null
m1050_auth --assemble --work "${m1050_work}" >/dev/null
m1050_auth --publish --work "${m1050_work}" --result "${m1050_result}" >/dev/null
m1050_published=1
trap - EXIT HUP INT TERM
/usr/bin/printf '%s\n' \
  'PASS M1050 diagnostic stratified pilot published; independent result hammer required'
