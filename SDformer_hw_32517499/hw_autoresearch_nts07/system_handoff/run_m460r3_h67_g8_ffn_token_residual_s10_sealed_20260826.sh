#!/usr/bin/env bash
set -euo pipefail

task_mode="${1:---preflight-no-launch}"
task_remote_repo=/root/private_data/work/sdformer_codex/SDformer
task_remote_python=/opt/conda/envs/sdformerflow/bin/python
task_repo="${task_remote_repo}"
task_python="${task_remote_python}"
task_local_static="${M460R3_LOCAL_STATIC_TEST:-0}"

if [[ "${task_local_static}" == 1 ]]; then
    [[ -n "${M460R3_TEST_REPO_OVERRIDE:-}" ]] || {
        echo status=FAILED_M460R3_LOCAL_TEST_REPO_OVERRIDE_REQUIRED >&2
        exit 2
    }
    task_repo="${M460R3_TEST_REPO_OVERRIDE}"
    task_python="${M460R3_TEST_PYTHON_OVERRIDE:-/opt/anaconda3/bin/python}"
    task_mode=--local-static-no-launch
fi

case "${task_mode}" in
    --preflight-no-launch|--capture|--local-static-no-launch) ;;
    *) echo "M460R3 invalid mode: ${task_mode}" >&2; exit 2 ;;
esac

task_hw="${task_repo}/hw_autoresearch_nts07"
task_bundle_rel=hw_autoresearch_nts07/system_handoff/m460r3_launch_bundle_20260826
task_manifest_rel="${task_bundle_rel}/M460R3_LAUNCH_SHA256SUMS"
task_outer_rel="${task_bundle_rel}/M460R3_LAUNCH_SHA256SUMS.outer.seal.sha256"
task_manifest="${task_repo}/${task_manifest_rel}"
task_outer="${task_repo}/${task_outer_rel}"
task_contract="${task_hw}/contracts/m460r3_h67_g8_ffn_token_residual_s10_capture_contract_r1_20260826.json"
task_capture="${task_hw}/system_handoff/scripts/capture_m460r3_h67_g8_ffn_token_residual_s10.py"
task_test="${task_hw}/tests/test_m460r3_g8_strict_capture_and_launch.py"
task_preflight="${task_hw}/system_handoff/scripts/preflight_m460r3_launch_closure.py"
task_runner="${task_hw}/system_handoff/run_m460r3_h67_g8_ffn_token_residual_s10_sealed_20260826.sh"
task_output="${task_hw}/results/m460r3_h67_g8_ffn_token_residual_s10_r1_20260826"
task_stage="${task_output}.partial.$$"
task_receipt="${task_output}.queue_receipt"
task_log="${task_hw}/system_handoff/logs/m460r3_h67_g8_ffn_token_residual_s10_20260826.log"
task_expected_outer="${M460R3_EXPECTED_OUTER_SEAL_SHA256:-}"
task_explicit_capture="${M460R3_EXPLICIT_CAPTURE:-0}"
task_phase=launch_trust_root
task_success=0

[[ "${task_expected_outer}" =~ ^[0-9a-f]{64}$ ]] || {
    echo status=FAILED_M460R3_REVIEW_APPROVED_OUTER_SEAL_SHA_REQUIRED >&2
    exit 2
}
[[ -f "${task_outer}" && -f "${task_manifest}" ]] || {
    echo status=FAILED_M460R3_DETACHED_LAUNCH_SEAL_ABSENT >&2
    exit 2
}
task_observed_outer="$(sha256sum "${task_outer}" | awk '{print $1}')"
[[ "${task_observed_outer}" == "${task_expected_outer}" ]] || {
    echo "M460R3 outer seal trust-root mismatch expected=${task_expected_outer} observed=${task_observed_outer}" >&2
    exit 2
}
(
    cd "${task_repo}"
    sha256sum -c "${task_outer_rel}"
    sha256sum -c "${task_manifest_rel}"
)

task_fail() {
    local task_rc="$1"
    if [[ ${task_success} -ne 1 && "${task_mode}" == --capture ]]; then
        printf 'status=FAILED_M460R3_DO_NOT_CITE\nphase=%s\nexit_code=%s\nretained_partial=%s\n' \
            "${task_phase}" "${task_rc}" "${task_stage}" \
            >"${task_receipt}.FAILED.$(date -u +%Y%m%dT%H%M%SZ).$$"
    fi
}
trap 'task_fail $?' EXIT

task_static_checks() {
    local task_dry
    local task_micro
    local task_preflight_mode=remote
    if [[ "${task_mode}" == --local-static-no-launch ]]; then
        task_preflight_mode=local
    fi
    task_dry="$("${task_python}" "${task_capture}" \
        --contract "${task_contract}" --dry-run)"
    task_micro="$("${task_python}" "${task_test}")"
    grep -q PASS_M460R3_STATIC_EXACT_SHA_AND_STRICT_ORDER_DRY_RUN \
        <<<"${task_dry}"
    grep -q PASS_M460R3_CPU_MICRO_AND_STRICT_ORDER_ATTACKS \
        <<<"${task_micro}"
    "${task_python}" "${task_preflight}" \
        --repo "${task_repo}" \
        --contract "${task_contract}" \
        --launch-manifest "${task_manifest}" \
        --outer-seal "${task_outer}" \
        --mode "${task_preflight_mode}"
    "${task_python}" -c \
        'import json,sys; x=json.loads(sys.argv[1]); assert x["attack_total"] >= 7; assert x["attack_total"] == x["attack_passes"]; assert x["independent_reference_mismatches"] == 0; assert x["sn2_fc2_sn1_rejected"] is True' \
        "${task_micro}"
}

task_static_checks

if [[ "${task_mode}" == --local-static-no-launch ]]; then
    echo status=PASS_M460R3_RUNNER_DEFAULT_NO_LAUNCH_LOCAL_STATIC_TEST
    echo expected_outer_seal_sha256="${task_expected_outer}"
    echo gpu_touched=false
    echo remote_contacted=false
    echo capture_launched=false
    task_success=1
    exit 0
fi

mkdir -p "$(dirname "${task_log}")"
exec > >(tee -a "${task_log}") 2>&1

task_ml_processes() {
    ps -eo pid=,args= | awk '
        BEGIN { found=0 }
        {
            line=tolower($0)
            if (line ~ /python/ && line ~ /(train|eval|valid|profile)/ &&
                line !~ /capture_m460r3_h67_g8_ffn_token_residual_s10/ &&
                line !~ /test_m460r3_g8_strict_capture_and_launch/ &&
                line !~ /preflight_m460r3_launch_closure/) {
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
    echo "M460R3_IDLE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_contexts=${task_active} ml_process_busy=${task_busy}"
    [[ "${task_active}" == 0 && "${task_busy}" == 0 ]]
}

task_phase=gpu_idle_guard
task_idle=0
while (( task_idle < 4 )); do
    if task_idle_snapshot; then
        task_idle=$((task_idle + 1))
    else
        echo status=BLOCKED_M460R3_GPU_NOT_SAFE_NO_LAUNCH
        task_success=1
        exit 3
    fi
    if (( task_idle < 4 )); then sleep 10; fi
done

if [[ "${task_mode}" == --preflight-no-launch ]]; then
    echo status=PASS_M460R3_SEALED_REMOTE_PREFLIGHT_FOUR_IDLE__NO_LAUNCH
    echo expected_outer_seal_sha256="${task_expected_outer}"
    echo capture_launched=false
    task_success=1
    exit 0
fi

[[ "${task_explicit_capture}" == 1 ]] || {
    echo status=FAILED_M460R3_EXPLICIT_CAPTURE_OPT_IN_REQUIRED >&2
    exit 2
}
[[ ! -e "${task_output}" && ! -e "${task_stage}" && ! -e "${task_receipt}" ]] || {
    echo status=FAILED_M460R3_OUTPUT_OR_RECEIPT_ALREADY_EXISTS >&2
    exit 2
}

task_phase=post_idle_trust_and_closure
task_observed_outer="$(sha256sum "${task_outer}" | awk '{print $1}')"
[[ "${task_observed_outer}" == "${task_expected_outer}" ]]
(
    cd "${task_repo}"
    sha256sum -c "${task_outer_rel}"
    sha256sum -c "${task_manifest_rel}"
)
task_static_checks

task_phase=capture
cd "${task_repo}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${task_python}" -u "${task_capture}" \
    --contract "${task_contract}" \
    --output-dir "${task_stage}"

task_phase=post_capture_result_seals
task_summary="${task_stage}/m460_h67_g8_ffn_token_residual_s10_capture.json"
task_inner="${task_stage}/manifest.sha256"
task_result_outer="${task_stage}/manifest.sha256.outer.seal.sha256"
test -s "${task_summary}"
test -s "${task_inner}"
test -s "${task_result_outer}"
(
    cd "${task_stage}"
    sha256sum -c manifest.sha256.outer.seal.sha256
    sha256sum -c manifest.sha256
)
"${task_python}" - "${task_summary}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], "r"))
assert value["schema"] == "m460r3_h67_g8_ffn_token_residual_s10_capture_v1"
assert value["identity"]["checkpoint_load_audit"]["missing_count"] == 0
assert value["identity"]["checkpoint_load_audit"]["unexpected_count"] == 0
assert value["identity"]["capture_bn_policy"] == "no_running/current-batch"
assert value["population"]["samples"] == 10
assert value["population"]["ffn_modules"] == 12
assert value["population"]["sample_module_records"] == 120
assert value["population"]["tokens"] == value["population"]["expected_tokens"] == 5580000
assert value["strict_runtime_state_machine"]["sn2_fc2_sn1_attack_accepted"] is False
assert value["admission"]["double_sealed_payload"] is True
assert value["admission"]["training"] is False
assert value["admission"]["cycle_speedup"] is False
assert value["admission"]["system_speedup"] is False
assert value["admission"]["headline"] is False
PY

task_phase=atomic_publish
mv "${task_stage}" "${task_output}"
task_summary="${task_output}/m460_h67_g8_ffn_token_residual_s10_capture.json"
task_inner="${task_output}/manifest.sha256"
task_result_outer="${task_output}/manifest.sha256.outer.seal.sha256"
task_summary_sha="$(sha256sum "${task_summary}" | awk '{print $1}')"
task_inner_sha="$(sha256sum "${task_inner}" | awk '{print $1}')"
task_result_outer_sha="$(sha256sum "${task_result_outer}" | awk '{print $1}')"
{
    echo status=PASS_M460R3_H67_EP35_NO_RUNNING_S10_STRICT_ORDER_DOUBLE_SEAL
    echo completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo launch_outer_seal_sha256="${task_expected_outer}"
    echo capture_summary_sha256="${task_summary_sha}"
    echo capture_inner_manifest_sha256="${task_inner_sha}"
    echo capture_outer_seal_file_sha256="${task_result_outer_sha}"
    echo g8_readiness_source_m159_sha256=6c67a75d052080cf58e558f960f23bea64d841087967de044fef898ad46c7f89
    echo g8_readiness_ffn_share_fraction=0.331103
    echo g8_readiness_ideal_skip_for_1p15=0.393940
    echo g8_readiness_ideal_skip_for_1p20=0.503368
    echo g8_readiness_ideal_skip_for_1p30=0.696971
    echo g8_readiness_only=true
    echo g8_oracle_measured=false
    echo training=false
    echo executable_skip=false
    echo system_speedup=false
    echo headline=false
} >"${task_receipt}.tmp.$$"
mv "${task_receipt}.tmp.$$" "${task_receipt}"
task_success=1
cat "${task_receipt}"
