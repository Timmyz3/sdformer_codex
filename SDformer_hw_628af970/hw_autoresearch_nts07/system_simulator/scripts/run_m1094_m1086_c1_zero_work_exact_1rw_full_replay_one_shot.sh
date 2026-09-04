#!/usr/bin/env bash
set -euo pipefail

# M1094r2 is intentionally not a production launch wrapper.  It proves that
# the source library is frozen and the runtime namespace is fresh, then fails
# closed.  A different-author M1095 successor must create a new additive
# zero-argument wrapper with exact authority paths and digests hardcoded in its
# own source.  No environment variable can turn this stub into a launcher.
[[ "$#" -eq 0 ]] || { echo "M1094r2 non-launch stub takes no arguments" >&2; exit 2; }

m1094_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m1094_runner="$(realpath "${BASH_SOURCE[0]}")"
m1094_engine="${m1094_hw_root}/system_simulator/scripts/execute_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.py"
m1094_contract="${m1094_hw_root}/contracts/m1094r2_m1087r3_m1086r2_c1_zero_work_full_replay_atomic_library_source_contract_r1_20260830.json"
m1094_python=/opt/anaconda3/envs/pytorch310/bin/python3.10

readonly m1094_expected_engine_sha=c8808c0d4cf37a8f279afa128e089c08af3718606061658db8f2047c198c824a
readonly m1094_expected_contract_sha=5278c5fa03a74cf9e3364325865b1bd52a5f75f372de15d5172b0b38bda64be4
readonly m1094_expected_python_sha=9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115

m1094_sha() { sha256sum "$1" | awk '{print $1}'; }
m1094_require_sha() {
    [[ -f "$1" && ! -L "$1" && "$(m1094_sha "$1")" == "$2" ]] || {
        printf 'M1094r2 identity drift: %s\n' "$1" >&2
        exit 3
    }
}

m1094_require_sha "${m1094_engine}" "${m1094_expected_engine_sha}"
m1094_require_sha "${m1094_contract}" "${m1094_expected_contract_sha}"
m1094_require_sha "${m1094_python}" "${m1094_expected_python_sha}"
PYTHONNOUSERSITE=1 PYTHONPATH= "${m1094_python}" -I "${m1094_engine}" \
    --validate-source --runner "${m1094_runner}" >/dev/null

printf '%s\n' \
  'M1094R2_NO_LAUNCH__DIFFERENT_AUTHOR_M1095_HARDCODED_WRAPPER_REQUIRED' >&2
exit 86
