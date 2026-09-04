#!/usr/bin/env bash
set -euo pipefail

m370_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m370_contract="${m370_hw}/contracts/m370_bottleneck_activation_magnitude_gate_fastkill_contract_r1_20260825.json"
m370_analyzer="${m370_hw}/system_simulator/scripts/analyze_m370_bottleneck_activation_magnitude_gate_fastkill.py"
m370_output="${M370_OUTPUT_DIR:-${m370_hw}/results/m370_bottleneck_activation_magnitude_gate_fastkill_replay_20260825}"

m370_expect() {
    local m370_path=$1
    local m370_sha=$2
    [[ -f "${m370_path}" ]]
    [[ "$(sha256sum "${m370_path}" | awk '{print $1}')" == "${m370_sha}" ]]
}

m370_expect "${m370_contract}" 02ba792e53fe0233c808b6a3e60f9cf0e23479e33fbcf89e9c0da7eca6dc2c00
m370_expect "${m370_analyzer}" 09e726c47f59f6f5d38c111b5d997b9887afe11d6938903780dd4b8db1b07ed3
[[ ! -e "${m370_output}" ]]
python3 "${m370_analyzer}" --contract "${m370_contract}" \
    --output-dir "${m370_output}"
