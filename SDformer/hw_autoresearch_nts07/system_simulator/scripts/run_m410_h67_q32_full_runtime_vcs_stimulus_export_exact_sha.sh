#!/usr/bin/env bash
set -euo pipefail

task_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M410_EXPORT_DIR:-${task_hw_root}/results/m410_h67_q32_full_runtime_vcs_stimulus_r1_20260826}"
task_log="$(mktemp)"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >&2; fi; rm -f "${task_log}"' EXIT
cd "${task_hw_root}"
[[ ! -e "${task_run}" ]] || exit 2

declare -A task_expected=(
    ["system_simulator/scripts/export_m410_h67_q32_full_runtime_vcs_stimulus.py"]="2c8cd0dabd4aff408406d7b25fbcd7231cd8115513ac0991d9f9339407eec6af"
    ["contracts/m410_h67_q32_full_runtime_vcs_stimulus_export_contract_r1_20260826.json"]="b72d1de9c8a700fe2af10f28c90d0d8faccf609327a710a5df2d169caeb4f1ed"
    ["contracts/m401_h67_q32_elastic_pwp_full_replay_contract_r1_20260826.json"]="7a8a594d40b23a5e399b4f0670ed8bcb87cf7edc4409b49b5818adffe49a11b1"
    ["results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json"]="e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3"
    ["results/m338_trainonly_nested_q128_catalog_r1_20260825/m338_trainonly_nested_q128_catalog_r1.json"]="b7c9e19166d3abfbe696df74dcfb99ef65607d209b18ca51b8456f15bcb6c2b1"
    ["system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"]="a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/m408_h67_q32_static_codec_vcs_stimulus_r1.json"]="fbf1454675f6c41162503fe258927fdc6fd5ee36a19c163ed0133068517f4111"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/m408_h67_q32_static_codec_1281.memh"]="a7c0f76187ed57cfedb94bae1ab8bb75513f9959df8fae1fc38eeb95818dd81c"
    ["results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS.seal.sha256"]="18a610bd03aa6fee665b4557ff6957f4b864d35be462bea881c1e2d4406cc497"
    ["results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/m409_m408_static_codec_vcs_independent_hammer_review_r1.json"]="076fdb4e4a2bd7464f01618a389535b4b404acde57cc37bc7aae2b39a0f9adc4"
    ["results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256"]="7fbb0caaa935451edcbf08a965b6bd99fda33ee01d626a53eb4d5a2559b2d8ec"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
(
    cd results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_log}" 2>&1
sha256sum -c \
    results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/SHA256SUMS \
    >>"${task_log}" 2>&1
sha256sum -c \
    results/m409_m408_static_codec_vcs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    >>"${task_log}" 2>&1

python3 system_simulator/scripts/export_m410_h67_q32_full_runtime_vcs_stimulus.py \
    --contract contracts/m410_h67_q32_full_runtime_vcs_stimulus_export_contract_r1_20260826.json \
    --output-dir "${task_run}" >>"${task_log}" 2>&1
mv "${task_log}" "${task_run}/export.log"
task_log="$(mktemp)"

cp contracts/m410_h67_q32_full_runtime_vcs_stimulus_export_contract_r1_20260826.json \
    "${task_run}/contract.json"
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"

python3 - "${task_run}/m410_h67_q32_full_runtime_vcs_stimulus_r1.json" <<'PY'
import json
import sys
from pathlib import Path

def strict_pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise SystemExit("duplicate JSON key: " + key)
        out[key] = value
    return out

data = json.loads(Path(sys.argv[1]).read_text(),
                  object_pairs_hook=strict_pairs)
if data["schema"] != "m410_h67_q32_full_runtime_vcs_stimulus_v1":
    raise SystemExit("M410 schema drift")
expected = {
    "phases": 17280,
    "config_beats": 51840,
    "source_rows": 51840000,
    "zero_rows": 24534432,
    "pop1_rows": 7516420,
    "pass1_tasks": 16037540,
    "early_stops": 3751608,
    "pwp_rows": 16971357,
    "matcher_task_cycles": 67877540,
    "m401_matcher_cycles_with_two_cycle_phase_overhead": 67912100,
}
if data["population"] != expected:
    raise SystemExit("M410 population drift")
if data["output"]["config"]["bytes"] != 3335040:
    raise SystemExit("M410 config byte extent drift")
if data["output"]["rows"]["bytes"] != 466560000:
    raise SystemExit("M410 row byte extent drift")
if data["claim_boundary"] != {
        "full_ordered_runtime_stimulus": True,
        "vcs_executed": False,
        "rtl_realtrace_cycle_match": False,
        "rtl_measured_speedup": False,
        "system_speedup": False,
        "headline": False}:
    raise SystemExit("M410 claim boundary drift")
PY

printf '%s\n' PASS_M410_H67_Q32_FULL_RUNTIME_STIMULUS_EXPORTED \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum >"${task_run}/SHA256SUMS"
sha256sum "${task_run}/SHA256SUMS" >"${task_run}/SHA256SUMS.seal.sha256"
task_complete=1
echo "PASS M410 full runtime stimulus sealed at ${task_run}"
