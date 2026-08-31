#!/usr/bin/env bash
set -euo pipefail

task_mode="${1:---preflight-no-capture}"
task_git_worktree=/root/private_data/work/sdformer_codex/SDformer_m460r4_c153
task_code_repo=/root/private_data/work/sdformer_codex/SDformer_m460r4_c153/SDformer
task_immutable_data_repo=/root/private_data/work/sdformer_codex/SDformer
task_immutable_data_git_root=/root/private_data/work/sdformer_codex
task_python=/opt/conda/envs/sdformerflow/bin/python
task_local_static="${M460R4_LOCAL_STATIC_TEST:-0}"

if [[ "${task_local_static}" == 1 ]]; then
    [[ -n "${M460R4_TEST_CODE_REPO_OVERRIDE:-}" ]] || {
        echo status=FAILED_M460R4_LOCAL_CODE_REPO_OVERRIDE_REQUIRED >&2
        exit 2
    }
    task_code_repo="${M460R4_TEST_CODE_REPO_OVERRIDE}"
    task_python="${M460R4_TEST_PYTHON_OVERRIDE:-/usr/bin/python3.6}"
    task_mode=--local-static-no-launch
fi

case "${task_mode}" in
    --preflight-no-capture|--local-static-no-launch) ;;
    *) echo "M460R4 invalid or forbidden mode: ${task_mode}" >&2; exit 2 ;;
esac

task_hw="${task_code_repo}/hw_autoresearch_nts07"
task_bundle_rel=hw_autoresearch_nts07/system_handoff/m460r4_launch_bundle_20260826
task_manifest_rel="${task_bundle_rel}/M460R4_LAUNCH_SHA256SUMS"
task_outer_rel="${task_bundle_rel}/M460R4_LAUNCH_SHA256SUMS.outer.seal.sha256"
task_manifest="${task_code_repo}/${task_manifest_rel}"
task_outer="${task_code_repo}/${task_outer_rel}"
task_contract="${task_hw}/contracts/m460r4_h67_g8_environment_preflight_contract_r1_20260826.json"
task_freeze="${task_hw}/system_handoff/m460r4_launch_bundle_20260826/m460r4_remote_environment_freeze.json"
task_inventory_builder="${task_hw}/system_handoff/scripts/build_m460r4_package_inventory.py"
task_preflight="${task_hw}/system_handoff/scripts/preflight_m460r4_code_data_environment.py"
task_capture_advisory="${task_hw}/system_handoff/scripts/capture_m460r4_h67_g8_ffn_token_residual_s10.py"
task_test="${task_hw}/tests/test_m460r4_environment_and_receipt_hardening.py"
task_output="${task_hw}/results/m460r4_remote_preflight_r1_20260826"
task_stage="${task_output}.partial.$$"
task_expected_outer="${M460R4_EXPECTED_OUTER_SEAL_SHA256:-}"
task_post_capture_advisory_receipt_fields=launch_outer_seal_sha256,capture_summary_sha256,capture_inner_manifest_sha256,capture_outer_seal_file_sha256

[[ "${task_expected_outer}" =~ ^[0-9a-f]{64}$ ]] || {
    echo status=FAILED_M460R4_REVIEW_APPROVED_OUTER_SEAL_SHA_REQUIRED >&2
    exit 2
}
[[ -f "${task_outer}" && -f "${task_manifest}" ]] || {
    echo status=FAILED_M460R4_DETACHED_LAUNCH_SEAL_ABSENT >&2
    exit 2
}
task_observed_outer="$(sha256sum "${task_outer}" | awk '{print $1}')"
[[ "${task_observed_outer}" == "${task_expected_outer}" ]] || {
    echo "M460R4 outer seal trust-root mismatch expected=${task_expected_outer} observed=${task_observed_outer}" >&2
    exit 2
}
(
    cd "${task_code_repo}"
    sha256sum -c "${task_outer_rel}"
    sha256sum -c "${task_manifest_rel}"
)

export PYTHONNOUSERSITE=1
unset PYTHONPATH

task_test_json="$(cd /tmp && "${task_python}" -I "${task_test}")"
grep -q PASS_M460R4_CPU_P1_CLOSURE_AND_TAMPER_TESTS <<<"${task_test_json}"
"${task_python}" -I -c \
    'import json,sys; x=json.loads(sys.argv[1]); assert x["attack_total"] == x["attack_passes"] >= 9; assert x["runner_capture_mode_exposed"] is False; assert x["gpu_touched"] is False' \
    "${task_test_json}"

if [[ "${task_mode}" == --local-static-no-launch ]]; then
    task_dry="$(cd /tmp && "${task_python}" -I "${task_capture_advisory}" \
        --contract "${task_contract}" --dry-run)"
    grep -q PASS_M460R4_STATIC_CODE_DATA_AND_RECEIPT_SCHEMA_DRY_RUN \
        <<<"${task_dry}"
    echo status=PASS_M460R4_LOCAL_STATIC_NO_LAUNCH
    echo expected_outer_seal_sha256="${task_expected_outer}"
    echo post_capture_advisory_receipt_fields="${task_post_capture_advisory_receipt_fields}"
    echo remote_contacted=false
    echo gpu_touched=false
    echo capture_launched=false
    echo training=false
    exit 0
fi

[[ "$(readlink -f "${task_code_repo}")" == "${task_code_repo}" ]] || {
    echo status=FAILED_M460R4_CODE_REPO_SYMLINK_OR_PATH_DRIFT >&2
    exit 2
}
[[ "$(readlink -f "${task_immutable_data_repo}")" == "${task_immutable_data_repo}" ]] || {
    echo status=FAILED_M460R4_DATA_REPO_SYMLINK_OR_PATH_DRIFT >&2
    exit 2
}
[[ ! -e "${task_output}" && ! -e "${task_stage}" ]] || {
    echo status=FAILED_M460R4_PREFLIGHT_OUTPUT_EXISTS >&2
    exit 2
}
mkdir -p "${task_stage}"

cd /tmp
"${task_python}" -I "${task_inventory_builder}" \
    --code-repo "${task_code_repo}" \
    --freeze "${task_freeze}" \
    --output "${task_stage}/package_build_inventory.json"

"${task_python}" -I "${task_preflight}" \
    --contract "${task_contract}" \
    --code-repo "${task_code_repo}" \
    --git-worktree-root "${task_git_worktree}" \
    --immutable-data-repo "${task_immutable_data_repo}" \
    --immutable-data-git-root "${task_immutable_data_git_root}" \
    --freeze "${task_freeze}" \
    --inventory "${task_stage}/package_build_inventory.json" \
    --launch-manifest "${task_manifest}" \
    --launch-outer-seal "${task_outer}" \
    --output "${task_stage}/preflight_receipt.json"

task_idle_csv="${task_stage}/idle_receipt.csv"
printf 'snapshot,utc,gpu_contexts,ml_processes,driver,gpu_name\n' >"${task_idle_csv}"
task_idle=0
while (( task_idle < 4 )); do
    task_contexts="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | wc -l)"
    task_ml="$(ps -eo args= | awk '
        BEGIN { count=0 }
        {
            line=tolower($0)
            if (line ~ /python/ && line ~ /(train|eval|valid|profile|capture)/ &&
                line !~ /m460r4/ && line !~ /awk/) count++
        }
        END { print count }
    ')"
    task_gpu_line="$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
    task_driver="${task_gpu_line%%,*}"
    task_name="${task_gpu_line#*, }"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "${task_idle}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${task_contexts}" "${task_ml}" "${task_driver}" "${task_name}" \
        >>"${task_idle_csv}"
    if [[ "${task_contexts}" != 0 || "${task_ml}" != 0 ]]; then
        echo status=BLOCKED_M460R4_REMOTE_NOT_IDLE_NO_CAPTURE
        echo retained_preflight_stage="${task_stage}"
        exit 3
    fi
    task_idle=$((task_idle + 1))
    if (( task_idle < 4 )); then sleep 10; fi
done

cd "${task_stage}"
sha256sum package_build_inventory.json preflight_receipt.json idle_receipt.csv \
    > manifest.sha256
sha256sum manifest.sha256 > manifest.sha256.outer.seal.sha256
sha256sum -c manifest.sha256.outer.seal.sha256
sha256sum -c manifest.sha256

cd "${task_code_repo}"
mv "${task_stage}" "${task_output}"
echo status=PASS_M460R4_SEALED_REMOTE_INVENTORY_PREFLIGHT_IDLE_NO_CAPTURE
echo result="${task_output}"
echo launch_outer_seal_sha256="${task_expected_outer}"
echo post_capture_advisory_receipt_fields="${task_post_capture_advisory_receipt_fields}"
echo gpu_touched=false
echo capture_launched=false
echo training=false
