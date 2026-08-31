#!/usr/bin/env bash
set -euo pipefail

m361r4_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m361r4_contract="${m361r4_hw}/contracts/m361r4_wide_partition_exact_pattern_dse_contract_r1_20260825.json"
m361r4_analyzer="${m361r4_hw}/system_simulator/scripts/analyze_m361r4_wide_partition_exact_pattern_dse.py"
m361r4_output="${M361R4_OUTPUT_DIR:-${m361r4_hw}/results/m361r4_wide_partition_exact_pattern_dse_replay_20260825}"

m361r4_expect() {
    local m361r4_path=$1
    local m361r4_sha=$2
    [[ -f "${m361r4_path}" ]]
    [[ "$(sha256sum "${m361r4_path}" | awk '{print $1}')" == "${m361r4_sha}" ]]
}

m361r4_expect "${m361r4_contract}" ba5b24670dd83f976e6d6d02de27f7862b0ef84c67e414c62171f2358f2651e7
m361r4_expect "${m361r4_analyzer}" 81064c3253f7e4599b40b1a260e8929d88295a94b483e167d5dbed50bd4c3d72
[[ ! -e "${m361r4_output}" ]]
python3 "${m361r4_analyzer}" --contract "${m361r4_contract}" \
    --output-dir "${m361r4_output}"
