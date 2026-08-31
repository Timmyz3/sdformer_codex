#!/usr/bin/env bash
set -euo pipefail

m425_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
m425_hw="$(cd "${m425_script_dir}/../.." && pwd)"
m425_runner="$(realpath "${BASH_SOURCE[0]}")"
m425_contract="contracts/m425_h67_balanced_selected_slice_saif_subset_contract_r1_20260826.json"
m425_exporter="system_simulator/scripts/export_m425_h67_balanced_selected_slice_saif_subset.py"
m425_out="${M425_SUBSET_DIR:-${m425_hw}/results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826}"

m425_sha() { sha256sum "$1" | awk '{print $1}'; }
m425_expect() {
    local m425_path=$1
    local m425_expected=$2
    [[ -f "${m425_path}" ]] || exit 3
    [[ "$(m425_sha "${m425_path}")" == "${m425_expected}" ]] || exit 3
}

[[ ! -e "${m425_out}" ]] || exit 2
cd "${m425_hw}"
m425_expect "${m425_contract}" a0256ba6093e066ae57d10d2153ae102f35f5a16e3e9c9fddacc6ea2debb7ad5
m425_expect "${m425_exporter}" 407d97a5de33e498b77b0de26f08f6aa6b40cf0bf38c50a8980656673cb6ea7c
m425_expect results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS c0db8a02abe47bd43c8131febb3b6968cb2cc36e911b450c17f5b6bd847056bc
m425_expect results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS.seal.sha256 31abafb9e39e2a9fa39b348b0ab9954805ec94e58f1006a6f2d57e5d24946efc
m425_expect results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS a054409aa63b040b4e620cc2f4a08d07eb2cef0d9d00a09b5822329f9f85bda5
m425_expect results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS.seal.sha256 18a610bd03aa6fee665b4557ff6957f4b864d35be462bea881c1e2d4406cc497
m425_expect results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.seal.sha256 bac41c1f8fe14c3250323659e3c5ef02848c55a3e0c28caadc97774e2529f1b6
m425_expect results/m408_h67_q32_static_codec_full_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 9646e1beebab5203782e1d02fabb60fcd3a21e67279b7d6486d6b193f66a04e6
m425_expect results/m411_m410r2_full_runtime_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 6bbcedb292a0d22ace98eeb969ed903d0cf0bd2f348a815d2a8b1a3bf95a68e2
m425_expect dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79
m425_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS >/dev/null
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS.seal.sha256 >/dev/null
sha256sum -c results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS >/dev/null
sha256sum -c results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS.seal.sha256 >/dev/null
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.sha256 >/dev/null
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.seal.sha256 >/dev/null
sha256sum -c results/m408_h67_q32_static_codec_full_vcs_r1_20260826/RUN_MANIFEST.sha256 >/dev/null
sha256sum -c results/m408_h67_q32_static_codec_full_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 >/dev/null
(cd results/m411_m410r2_full_runtime_vcs_independent_hammer_r1_20260826 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
sha256sum -c dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.sha256 >/dev/null
sha256sum -c dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 >/dev/null

python3 "${m425_exporter}" --contract "${m425_contract}" \
    --output-dir "${m425_out}" >"${m425_hw}/m425_subset_export.tmp.log" 2>&1
mv "${m425_hw}/m425_subset_export.tmp.log" "${m425_out}/export.log"
cp "${m425_contract}" "${m425_out}/contract.json"
sha256sum "${m425_runner}" >"${m425_out}/runner_sha256.txt"
printf '%s\n' PASS_M425_FROZEN_PRE_VCS_ACTIVITY_SUBSET_EXPORT \
    >"${m425_out}/RUN_COMPLETE.txt"
find "${m425_out}" -maxdepth 1 -type f \
    ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${m425_out}/SHA256SUMS"
sha256sum "${m425_out}/SHA256SUMS" >"${m425_out}/SHA256SUMS.seal.sha256"
echo "PASS M425 frozen subset sealed at ${m425_out}"
