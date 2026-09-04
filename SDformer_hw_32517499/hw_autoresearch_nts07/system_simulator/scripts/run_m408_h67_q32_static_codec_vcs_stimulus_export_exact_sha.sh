#!/usr/bin/env bash
set -euo pipefail

task_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M408_EXPORT_DIR:-${task_hw_root}/results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826}"
task_log="$(mktemp)"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >&2; fi; rm -f "${task_log}"' EXIT
cd "${task_hw_root}"

[[ ! -e "${task_run}" ]] || exit 2

declare -A task_expected=(
    ["system_simulator/scripts/export_m408_h67_q32_static_codec_vcs_stimulus.py"]="2143d0f1a91cf87b84d3258b8c60942411b6afdabf8a3b05c3932e62965a60b1"
    ["contracts/m408_h67_q32_static_codec_vcs_stimulus_export_contract_r1_20260826.json"]="7b1638dc15ef1d66472e4ae36966f695cc99b72bf02edc7fbccdb5e11d14d30e"
    ["contracts/m401_h67_q32_elastic_pwp_full_replay_contract_r1_20260826.json"]="7a8a594d40b23a5e399b4f0670ed8bcb87cf7edc4409b49b5818adffe49a11b1"
    ["results/m407_m405r3_integration_independent_hammer_r1_20260826/m407_m405r3_integration_independent_hammer_review_r1.json"]="af279c4d7cc07d8517cbf72fb12ccf4600b66609493af0cda35cb1251b2285e6"
    ["results/m407_m405r3_integration_independent_hammer_r1_20260826/SHA256SUMS"]="0ba0a7d2beeb5964d41d0c95f5b257a6f7b3fb1a905db3127ef5531bfe8a5723"
    ["results/m407_m405r3_integration_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256"]="d2ecf11f0b6fd0a710e961350329b4c728a033816259a32777ef0bb1b0f40fbf"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done

(
    cd results/m407_m405r3_integration_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_log}" 2>&1

python3 system_simulator/scripts/export_m408_h67_q32_static_codec_vcs_stimulus.py \
    --m401-contract contracts/m401_h67_q32_elastic_pwp_full_replay_contract_r1_20260826.json \
    --output-dir "${task_run}" >>"${task_log}" 2>&1
mv "${task_log}" "${task_run}/export.log"
task_log="$(mktemp)"

cp contracts/m408_h67_q32_static_codec_vcs_stimulus_export_contract_r1_20260826.json \
    "${task_run}/contract.json"
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"

python3 - "${task_run}/m408_h67_q32_static_codec_vcs_stimulus_r1.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
pairs_seen = []
def strict_pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise SystemExit("duplicate JSON key: " + key)
        out[key] = value
    return out
data = json.loads(path.read_text(), object_pairs_hook=strict_pairs)
if data["schema"] != "m408_h67_q32_static_codec_vcs_stimulus_v1":
    raise SystemExit("M408 schema drift")
expected = {
    "blocks": 442368,
    "lanes": 42467328,
    "narrow_blocks": 112167,
    "wide_blocks": 330201,
    "expected_accepted_contributions": 772569,
    "global_minimum": -1089,
    "global_maximum": 1059,
    "signed12_violations": 0,
    "wide_reconstruction_mismatches": 0,
    "narrow_reconstruction_mismatches": 0,
    "nonzero_padding_bits": 0,
}
if data["population"] != expected:
    raise SystemExit("M408 population/exactness gate drift")
if data["output"]["bytes"] != 142442496:
    raise SystemExit("M408 memh byte extent drift")
if data["claim_boundary"] != {
        "derived_static_stimulus": True,
        "vcs_executed": False,
        "rtl_measured_speedup": False,
        "system_speedup": False,
        "headline": False}:
    raise SystemExit("M408 claim boundary drift")
PY

printf '%s\n' PASS_M408_H67_Q32_STATIC_CODEC_STIMULUS_EXPORTED \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${task_run}/SHA256SUMS"
sha256sum "${task_run}/SHA256SUMS" >"${task_run}/SHA256SUMS.seal.sha256"
task_complete=1
echo "PASS M408 static stimulus sealed at ${task_run}"
