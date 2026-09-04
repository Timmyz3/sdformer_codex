#!/usr/bin/env bash
set -euo pipefail

m381_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m381_contract="${m381_hw}/contracts/m381_q32_o4_burst_streaming_sensitivity_contract_r1_20260825.json"
m381_analyzer="${m381_hw}/system_simulator/scripts/analyze_m381_q32_o4_burst_streaming_sensitivity.py"
m381_output="${M381_OUTPUT_DIR:-${m381_hw}/results/m381_q32_o4_burst_streaming_sensitivity_replay_20260825}"

m381_expect() {
    local m381_path=$1
    local m381_sha=$2
    [[ -f "${m381_path}" ]]
    [[ "$(sha256sum "${m381_path}" | awk '{print $1}')" == "${m381_sha}" ]]
}

m381_expect "${m381_contract}" 8eb81b44fc8182de43530ff899dacd2bd482f3adf41f8eda45d986e0291bb4ce
m381_expect "${m381_analyzer}" 2612264ff946d71d1af4142971e021060b6db8c4752340a3edde2150134e2a79
[[ ! -e "${m381_output}" ]]
python3 "${m381_analyzer}" --contract "${m381_contract}" \
    --output-dir "${m381_output}"
