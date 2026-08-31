#!/usr/bin/env bash
set -euo pipefail

task_repo=/root/private_data/work/sdformer_codex/SDformer
task_hw="${task_repo}/hw_autoresearch_nts07"
task_python=/opt/conda/envs/sdformerflow/bin/python
task_contract="${task_hw}/contracts/m460_h67_g8_ffn_token_residual_s10_capture_contract_r1_20260826.json"
task_script="${task_hw}/system_handoff/scripts/capture_m460_h67_g8_ffn_token_residual_s10.py"
task_test="${task_hw}/tests/test_m460_g8_ffn_residual_stream_capture.py"
task_output="${task_hw}/results/m460_h67_g8_ffn_token_residual_s10_r1_20260826"
task_stage="${task_output}.partial.$$"
task_receipt="${task_output}.queue_receipt"
task_log="${task_hw}/system_handoff/logs/m460_h67_g8_ffn_token_residual_s10_20260826.log"
task_contract_sha=f84d959c45a65ca5f3e2cdc7221e9437abbb557def2dafe690e75312b1f321c1
task_script_sha=5f90d3711da5524a883e485c66bc02353ec6ca3dc394286e7ff6d049ba3b6b35
task_test_sha=65e4f246bfc23ad8acc37246e792912e91aecb006f170d977f785933ff3bfa27
task_explicit_launch="${M460_EXPLICIT_REMOTE_LAUNCH:-0}"
task_phase=preflight
task_success=0

mkdir -p "$(dirname "${task_log}")"
exec > >(tee -a "${task_log}") 2>&1

task_fail() {
    local task_rc="$1"
    if [[ ${task_success} -ne 1 && "${task_explicit_launch}" == 1 ]]; then
        printf 'status=FAILED_M460_DO_NOT_CITE\nphase=%s\nexit_code=%s\nretained_partial=%s\n' \
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
        echo "M460 SHA drift path=${task_path} expected=${task_expected} observed=${task_observed}" >&2
        return 1
    }
}

task_check_all() {
    local task_tmp_dry
    local task_tmp_test
    task_tmp_dry="$(mktemp /tmp/m460_dry_run.XXXXXX.json)"
    task_tmp_test="$(mktemp /tmp/m460_cpu_micro.XXXXXX.json)"
    trap 'rm -f "${task_tmp_dry}" "${task_tmp_test}"' RETURN
    task_check_sha "${task_contract}" "${task_contract_sha}"
    task_check_sha "${task_script}" "${task_script_sha}"
    task_check_sha "${task_test}" "${task_test_sha}"
    "${task_python}" "${task_script}" \
        --contract "${task_contract}" --dry-run >"${task_tmp_dry}"
    "${task_python}" "${task_test}" >"${task_tmp_test}"
    grep -q PASS_M460_STATIC_EXACT_SHA_PREINPUT_DRY_RUN "${task_tmp_dry}"
    grep -q PASS_M460_CPU_MICRO_12_FFN_HOOK_AND_REFERENCE "${task_tmp_test}"
    "${task_python}" - "${task_tmp_dry}" "${task_tmp_test}" <<'PY'
import json
import sys
dry = json.load(open(sys.argv[1], "r"))
test = json.load(open(sys.argv[2], "r"))
assert dry["ffn_modules"] == 12 and dry["hook_points"] == 60
assert dry["gpu_touched"] is False and dry["automatic_launch"] is False
assert test["installed_ffn"] == 12 and test["installed_hooks"] == 60
assert test["independent_reference_mismatches"] == 0
assert test["gpu_touched"] is False and test["remote_launched"] is False
PY
    rm -f "${task_tmp_dry}" "${task_tmp_test}"
    trap - RETURN
}

task_ml_processes() {
    ps -eo pid=,args= | awk '
        BEGIN { found=0 }
        {
            line=tolower($0)
            if (line ~ /python/ && line ~ /(train|eval|valid|profile)/ &&
                line !~ /capture_m460_h67_g8_ffn_token_residual_s10/ &&
                line !~ /test_m460_g8_ffn_residual_stream_capture/) {
                print $0
                found=1
            }
        }
        END { exit(found ? 0 : 1) }
    '
}

task_idle_snapshot() {
    local task_active
    local task_busy
    task_active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | wc -l)"
    task_busy=0
    if task_ml_processes; then
        task_busy=1
    fi
    echo "M460_IDLE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_contexts=${task_active} ml_process_busy=${task_busy}"
    [[ "${task_active}" == 0 && "${task_busy}" == 0 ]]
}

task_check_all
task_phase=gpu_safety
task_idle=0
while (( task_idle < 4 )); do
    if task_idle_snapshot; then
        task_idle=$((task_idle + 1))
    else
        echo status=BLOCKED_M460_GPU_NOT_SAFE_NO_LAUNCH
        task_success=1
        exit 3
    fi
    if (( task_idle < 4 )); then sleep 10; fi
done

if [[ "${task_explicit_launch}" != 1 ]]; then
    echo status=PASS_M460_FOUR_CONSECUTIVE_IDLE_PREFLIGHT__NO_LAUNCH
    echo launch_command='M460_EXPLICIT_REMOTE_LAUNCH=1 ./hw_autoresearch_nts07/system_handoff/run_m460_h67_g8_ffn_token_residual_s10_when_gpu_idle_20260826.sh'
    task_success=1
    exit 0
fi

[[ ! -e "${task_output}" && ! -e "${task_stage}" && ! -e "${task_receipt}" ]] || exit 2
task_phase=post_idle_sha
task_check_all
task_phase=capture
cd "${task_repo}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${task_python}" -u "${task_script}" \
    --contract "${task_contract}" \
    --output-dir "${task_stage}"

task_phase=post_capture_sha
task_check_all
task_summary="${task_stage}/m460_h67_g8_ffn_token_residual_s10_capture.json"
task_manifest="${task_stage}/manifest.sha256"
test -s "${task_summary}"
(cd "${task_stage}" && sha256sum -c manifest.sha256)
"${task_python}" - "${task_summary}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], "r"))
assert value["schema"] == "m460_h67_g8_ffn_token_residual_s10_capture_v1"
assert value["identity"]["checkpoint_load_audit"]["missing_count"] == 0
assert value["identity"]["checkpoint_load_audit"]["unexpected_count"] == 0
assert value["identity"]["capture_bn_policy"] == "no_running/current-batch"
assert value["population"]["samples"] == 10
assert value["population"]["ffn_modules"] == 12
assert value["population"]["sample_module_records"] == 120
assert value["population"]["tokens"] == value["population"]["expected_tokens"] == 5580000
assert value["semantics"]["full_tensor_dumped"] is False
assert value["admission"]["training"] is False
assert value["admission"]["cycle_speedup"] is False
assert value["admission"]["system_speedup"] is False
assert value["admission"]["headline"] is False
PY

task_phase=atomic_publish
mv "${task_stage}" "${task_output}"
task_summary="${task_output}/m460_h67_g8_ffn_token_residual_s10_capture.json"
task_summary_sha="$(sha256sum "${task_summary}" | awk '{print $1}')"
{
    echo status=PASS_M460_H67_EP35_NO_RUNNING_S10_STREAM_CAPTURE
    echo completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo contract_sha256="${task_contract_sha}"
    echo capture_script_sha256="${task_script_sha}"
    echo cpu_micro_test_sha256="${task_test_sha}"
    echo capture_summary_sha256="${task_summary_sha}"
    echo training=false
    echo executable_skip=false
    echo system_speedup=false
    echo headline=false
} >"${task_receipt}.tmp.$$"
mv "${task_receipt}.tmp.$$" "${task_receipt}"
task_success=1
cat "${task_receipt}"
