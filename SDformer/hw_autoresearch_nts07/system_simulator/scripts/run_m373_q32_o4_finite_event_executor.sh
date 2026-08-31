#!/usr/bin/env bash
set -euo pipefail

m373_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m373_contract="${m373_hw}/contracts/m373_q32_o4_finite_event_executor_contract_r1_20260825.json"
m373_analyzer="${m373_hw}/system_simulator/scripts/analyze_m373_q32_o4_finite_event_executor.py"
m373_output="${M373_OUTPUT_DIR:-${m373_hw}/results/m373_q32_o4_finite_event_executor_replay_20260825}"

m373_expect() {
    local m373_path=$1
    local m373_sha=$2
    [[ -f "${m373_path}" ]]
    [[ "$(sha256sum "${m373_path}" | awk '{print $1}')" == "${m373_sha}" ]]
}

m373_expect "${m373_contract}" fbfcb72bb6e8d1404dcc81cf9dea34c0647b7f9752bc6ce81b74fce13c168b24
m373_expect "${m373_analyzer}" d97384ca159ce37f0a9d2c422c8d2cd62ab1cd862997fd2a7cbd432c43d3b851
[[ ! -e "${m373_output}" ]]
python3 "${m373_analyzer}" --contract "${m373_contract}" \
    --output-dir "${m373_output}"
