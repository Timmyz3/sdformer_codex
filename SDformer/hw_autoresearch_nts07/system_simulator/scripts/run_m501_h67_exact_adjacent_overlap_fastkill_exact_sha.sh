#!/usr/bin/env bash
set -euo pipefail

m501_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m501_contract="${m501_root}/contracts/m501_h67_exact_adjacent_overlap_fastkill_contract_r1_20260827.json"
m501_analyzer="${m501_root}/system_simulator/scripts/analyze_m501_h67_exact_adjacent_overlap_fastkill.py"
m501_m40="${m501_root}/results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json"
m501_m73="${m501_root}/system_handoff/incoming/m73_h67_ep35_train_calibration_sources_s32_r1_20260823/m73_train_calibration_source_manifest.json"
m501_docs359="${m501_root}/docs/359_DATE终局冻结_20260813.md"
m501_output="${M501_OUTPUT_DIR:-${m501_root}/results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827}"
m501_python="/opt/anaconda3/envs/pytorch310/bin/python"

m501_expect() {
    local m501_path=$1
    local m501_sha=$2
    [[ -f "${m501_path}" ]]
    [[ "$(sha256sum "${m501_path}" | awk '{print $1}')" == "${m501_sha}" ]]
}

m501_expect "${m501_contract}" bbb7bce5015ab3a3a5772b86d594853da353380df8dcd85a295e480d422eb2d6
m501_expect "${m501_analyzer}" 5bdfa6f6fa81510d11751d6867748515763d3d4b31927b8cfe03e03ee597b7e7
m501_expect "${m501_m40}" e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3
m501_expect "${m501_m73}" 3fb3468066fe1f7d61f5e39398cb2f8655643080f03e5b1deb58ef2911db17e2
m501_expect "${m501_docs359}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
[[ ! -e "${m501_output}" ]]
[[ -x "${m501_python}" ]]

"${m501_python}" "${m501_analyzer}" \
    --contract "${m501_contract}" \
    --output-dir "${m501_output}"

(
    cd "${m501_root}"
    sha256sum \
        "${m501_contract#${m501_root}/}" \
        "${m501_analyzer#${m501_root}/}" \
        "${m501_m40#${m501_root}/}" \
        "${m501_m73#${m501_root}/}" \
        "${m501_docs359#${m501_root}/}" \
        "${m501_output#${m501_root}/}/m501_h67_exact_adjacent_overlap_fastkill_result_r1.json" \
        "${m501_output#${m501_root}/}/README.md" \
        "${m501_output#${m501_root}/}/RUN_COMPLETE.txt" \
        > "${m501_output}/SHA256SUMS"
)
sha256sum "${m501_output}/SHA256SUMS" > "${m501_output}/SHA256SUMS.seal.sha256"
