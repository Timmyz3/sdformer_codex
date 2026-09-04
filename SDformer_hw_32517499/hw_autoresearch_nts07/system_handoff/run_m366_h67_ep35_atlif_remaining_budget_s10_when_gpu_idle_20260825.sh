#!/usr/bin/env bash
set -euo pipefail

task_repo=/root/private_data/work/sdformer_codex/SDformer
task_hw="${task_repo}/hw_autoresearch_nts07"
task_python=/opt/conda/envs/sdformerflow/bin/python
task_script="${task_hw}/system_simulator/scripts/trace_m366_h67_ep35_atlif_remaining_budget_s10.py"
task_contract="${task_hw}/contracts/m366_h67_ep35_atlif_remaining_budget_s10_contract_r1_20260825.json"
task_output="${task_hw}/results/m366_h67_ep35_atlif_remaining_budget_s10_r1_20260825"
task_stage="${task_output}.partial.$$"
task_receipt="${task_output}.queue_receipt"
task_log="${task_hw}/system_handoff/logs/m366_h67_ep35_atlif_remaining_budget_s10_20260825.log"
task_contract_sha=95f031569b1695c9c74e7862ac1abd3a95465789bd8c1e4ebe4a658b1bc4cdc2
task_script_sha=c4b2e83b2a1341f9790038d395aa8ed4c25c75bc441e932def4e2e32b1ba4045
task_preflight_only="${M366_PREFLIGHT_ONLY:-0}"
task_phase=preflight
task_success=0

mkdir -p "$(dirname "${task_log}")"
exec > >(tee -a "${task_log}") 2>&1

task_fail() {
    local task_rc="$1"
    if [[ ${task_success} -ne 1 && ${task_preflight_only} -ne 1 ]]; then
        printf 'status=FAILED_M366_DO_NOT_CITE\nphase=%s\nexit_code=%s\nretained_partial=%s\n' \
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
        echo "M366 SHA drift path=${task_path} expected=${task_expected} observed=${task_observed}" >&2
        return 1
    }
}

task_check_all() {
    task_check_sha "${task_contract}" "${task_contract_sha}"
    task_check_sha "${task_script}" "${task_script_sha}"
    "${task_python}" "${task_script}" --contract "${task_contract}" --dry-run >/tmp/m366_dry_run.$$.json
    grep -q PASS_M366_STATIC_EXACT_SHA_AND_PROOF_DRY_RUN /tmp/m366_dry_run.$$.json
    rm -f /tmp/m366_dry_run.$$.json
}

task_ml_processes() {
    ps -eo pid=,args= | awk '
        BEGIN { found=0 }
        {
            line=tolower($0)
            if (line ~ /python/ && line ~ /(train|eval|valid|profile)/ &&
                line !~ /trace_m366_h67_ep35_atlif_remaining_budget_s10/) {
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
    echo "M366_IDLE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_contexts=${task_active} ml_process_busy=${task_busy}"
    [[ "${task_active}" == 0 && "${task_busy}" == 0 ]]
}

task_check_all
task_phase=gpu_safety

if [[ "${task_preflight_only}" == 1 ]]; then
    if task_idle_snapshot; then
        echo status=PASS_M366_PREFLIGHT_GPU_SAFE_NOW
        task_success=1
        exit 0
    fi
    echo status=BLOCKED_M366_GPU_NOT_SAFE_NO_LAUNCH
    task_success=1
    exit 3
fi

[[ ! -e "${task_output}" && ! -e "${task_stage}" && ! -e "${task_receipt}" ]] || exit 2
task_idle=0
while (( task_idle < 4 )); do
    if task_idle_snapshot; then
        task_idle=$((task_idle + 1))
    else
        echo status=BLOCKED_M366_GPU_NOT_SAFE_NO_LAUNCH
        exit 3
    fi
    if (( task_idle < 4 )); then sleep 10; fi
done

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
task_manifest="${task_stage}/m366_h67_ep35_atlif_remaining_budget_s10_capture.json"
test -s "${task_manifest}"
"${task_python}" - "${task_manifest}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], "r"))
assert value["schema"] == "m366_h67_ep35_atlif_remaining_budget_s10_capture_v1"
assert value["identity"]["checkpoint_load_audit"]["missing_count"] == 0
assert value["identity"]["checkpoint_load_audit"]["unexpected_count"] == 0
assert value["population"]["samples"] == 10
assert value["population"]["live_sites"] == 81
assert value["t10_nonattention_main"]["integer_early_mismatches"] == 0
assert value["t10_nonattention_main"]["bound_violations"] == 0
assert value["admission"]["system_speedup"] is False
assert value["admission"]["headline"] is False
PY

task_phase=atomic_publish
mv "${task_stage}" "${task_output}"
task_manifest="${task_output}/m366_h67_ep35_atlif_remaining_budget_s10_capture.json"
task_manifest_sha="$(sha256sum "${task_manifest}" | awk '{print $1}')"
task_decision="$("${task_python}" - "${task_manifest}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["promotion_gates"]["rtl_decision"])
PY
)"
{
    echo status=PASS_M366_H67_EP35_ATLIF_REMAINING_BUDGET_S10
    echo completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo contract_sha256="${task_contract_sha}"
    echo capture_script_sha256="${task_script_sha}"
    echo manifest_sha256="${task_manifest_sha}"
    echo rtl_decision="${task_decision}"
    echo system_speedup=false
    echo headline=false
} >"${task_receipt}.tmp.$$"
mv "${task_receipt}.tmp.$$" "${task_receipt}"
task_success=1
cat "${task_receipt}"
