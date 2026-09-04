#!/usr/bin/env bash
set -euo pipefail

m377_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m377_contract="${m377_hw}/contracts/m377_q32_o4_active_descriptor_compact_executor_contract_r1_20260825.json"
m377_analyzer="${m377_hw}/system_simulator/scripts/analyze_m377_q32_o4_active_descriptor_compact_executor.py"
m377_output="${M377_OUTPUT_DIR:-${m377_hw}/results/m377_q32_o4_active_descriptor_compact_executor_replay_20260825}"

m377_expect() {
    local m377_path=$1
    local m377_sha=$2
    [[ -f "${m377_path}" ]]
    [[ "$(sha256sum "${m377_path}" | awk '{print $1}')" == "${m377_sha}" ]]
}

m377_expect "${m377_contract}" 93f1356f565056213da0d35d0c508df013cbb69a95f4cc5f6aee24121a277f86
m377_expect "${m377_analyzer}" 43ce235936c749c6f0a2382ae363029b3e7646838c78dbed906d3bbb4f89adf2
[[ ! -e "${m377_output}" ]]
python3 "${m377_analyzer}" --contract "${m377_contract}" \
    --output-dir "${m377_output}"
