#!/usr/bin/env bash
set -euo pipefail

task_repo=/root/private_data/work/sdformer_codex/SDformer
task_hw="${task_repo}/hw_autoresearch_nts07"
task_python=/opt/conda/envs/sdformerflow/bin/python
task_tracer="${task_hw}/system_simulator/scripts/trace_m252_control_ep4_running_bn_bottleneck_sources.py"
task_m248="${task_hw}/system_simulator/scripts/trace_m248_paft_ep4_running_bn_bottleneck_sources.py"
task_base="${task_hw}/system_simulator/scripts/trace_m40_bottleneck_packed_sources.py"
task_profile="${task_repo}/neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py"
task_checkpoint="${task_repo}/neuron_experiments/H9_bipolar_self_attention/results/m87_h67_trainonly_paft_paired_20260823/no_paft_control_full5/checkpoint_epoch4.pth"
task_config="${task_hw}/results/m87_h67_trainonly_paft_config_bundle_r2_20260823/no_paft_control_full5.yml"
task_workload="${task_hw}/results/m36_h67_ep35_patch_embed_profile_s10_r1_20260822/sample_workload.csv"
task_docs359="${task_hw}/docs/359_DATE终局冻结_20260813.md"
task_output="${task_hw}/results/m252_control_ep4_running_bn_bottleneck_sources_s10_r1_20260825"
task_stage="${task_output}.partial.$$"
task_receipt="${task_output}.queue_receipt"
task_log="${task_hw}/system_handoff/logs/m252_control_ep4_running_bn_bottleneck_trace_20260825.log"
task_phase=preflight
task_success=0

mkdir -p "$(dirname "${task_log}")"
exec > >(tee -a "${task_log}") 2>&1

task_fail() {
    local task_rc="$1"
    if [[ ${task_success} -ne 1 ]]; then
        printf 'status=FAILED_M252_DO_NOT_CITE\nphase=%s\nexit_code=%s\nretained_partial=%s\n' \
            "${task_phase}" "${task_rc}" "${task_stage}" \
            >"${task_receipt}.FAILED.$(date -u +%Y%m%dT%H%M%SZ).$$"
    fi
}
trap 'task_fail $?' EXIT

task_check_sha() {
    local task_path="$1"
    local task_expected="$2"
    local task_observed
    [[ -f "${task_path}" ]] || return 1
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    [[ "${task_observed}" == "${task_expected}" ]] || {
        echo "M252 SHA drift path=${task_path} expected=${task_expected} observed=${task_observed}" >&2
        return 1
    }
}

task_check_all() {
    task_check_sha "${task_tracer}" 6cd96c2798d4e54de410afca43c0163190ad2ee3300368c24256923e3b0fc75c
    task_check_sha "${task_m248}" d5c18ec3dd358b0ef10e66f7682bd87442be1cf5ddc9de53ea761a42a5451bf6
    task_check_sha "${task_base}" b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19
    task_check_sha "${task_profile}" 04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684
    task_check_sha "${task_checkpoint}" 6964264bde719f36c8ef7537d50ab90a7237973109f610c662d95d071bb0ab4b
    task_check_sha "${task_config}" 12e36e7b88699ebc7ccd138c9e7c16991d6abe029773fb11b037a3463116f739
    task_check_sha "${task_workload}" bb45f8b5406e34835f05e1993692d8cba241c748471037d75fcfa1ec2478cffa
    task_check_sha "${task_docs359}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
}

[[ ! -e "${task_output}" && ! -e "${task_stage}" && ! -e "${task_receipt}" ]] || exit 2
task_check_all

task_phase=wait_for_gpu
task_idle=0
while (( task_idle < 4 )); do
    if ! task_active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
        echo "M252 nvidia-smi failed; refusing launch" >&2
        exit 3
    fi
    task_active="$(printf '%s\n' "${task_active}" | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "${task_active}" == 0 ]]; then
        task_idle=$((task_idle + 1))
    else
        task_idle=0
    fi
    echo "M252_IDLE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) active=${task_active} consecutive=${task_idle}"
    if (( task_idle < 4 )); then sleep 15; fi
done

task_phase=post_wait_sha
task_check_all
task_phase=capture
cd "${task_repo}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${task_python}" -u "${task_tracer}" \
    --config "${task_config}" \
    --checkpoint "${task_checkpoint}" \
    --sample-workload "${task_workload}" \
    --output-dir "${task_stage}"

task_phase=post_capture_sha
task_check_all
task_manifest="${task_stage}/m252_control_ep4_running_bn_bottleneck_source_manifest.json"
test -s "${task_manifest}"
"${task_python}" - "${task_manifest}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], "r"))
assert value["status"] == "PASS_CONTROL_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE"
assert value["identity"]["capture_bn_policy"] == "running"
assert value["identity"]["paired_arm"] == "NO_PAFT_CONTROL"
assert value["identity"]["checkpoint_load_audit"]["missing_count"] == 0
assert value["identity"]["checkpoint_load_audit"]["unexpected_count"] == 0
assert value["cohort"]["records"] == 40
assert value["admission"]["paft_hardware_gain"] is False
PY

task_phase=atomic_publish
mv "${task_stage}" "${task_output}"
task_manifest="${task_output}/m252_control_ep4_running_bn_bottleneck_source_manifest.json"
task_manifest_sha="$(sha256sum "${task_manifest}" | awk '{print $1}')"
{
    echo status=PASS_M252_CONTROL_EP4_RUNNING_BN_BOTTLENECK_TRACE
    echo completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo checkpoint_sha256=6964264bde719f36c8ef7537d50ab90a7237973109f610c662d95d071bb0ab4b
    echo config_sha256=12e36e7b88699ebc7ccd138c9e7c16991d6abe029773fb11b037a3463116f739
    echo tracer_sha256=6cd96c2798d4e54de410afca43c0163190ad2ee3300368c24256923e3b0fc75c
    echo manifest_sha256="${task_manifest_sha}"
    echo paft_hardware_gain=false
    echo system_speedup=false
    echo headline=false
} >"${task_receipt}.tmp.$$"
mv "${task_receipt}.tmp.$$" "${task_receipt}"
task_success=1
cat "${task_receipt}"
