#!/usr/bin/env bash
set -euo pipefail

task_mode="${1:---invalid-no-default-capture}"
task_git_worktree=/root/private_data/work/sdformer_codex/SDformer_m460r4_c153
task_code_repo=/root/private_data/work/sdformer_codex/SDformer_m460r4_c153/SDformer
task_immutable_data_repo=/root/private_data/work/sdformer_codex/SDformer
task_immutable_data_git_root=/root/private_data/work/sdformer_codex
task_python=/opt/conda/envs/sdformerflow/bin/python
task_local_static="${M460R5_LOCAL_STATIC_TEST:-0}"
maximum_capture_attempts=1

if [[ "${task_local_static}" == 1 ]]; then
    [[ -n "${M460R5_TEST_CODE_REPO_OVERRIDE:-}" ]] || {
        echo status=FAILED_M460R5_LOCAL_CODE_REPO_OVERRIDE_REQUIRED >&2
        exit 2
    }
    task_code_repo="${M460R5_TEST_CODE_REPO_OVERRIDE}"
    task_python="${M460R5_TEST_PYTHON_OVERRIDE:-/usr/bin/python3.6}"
    task_mode=--local-static-no-launch
fi

case "${task_mode}" in
    --capture-once|--local-static-no-launch) ;;
    *) echo "M460R5 invalid mode; capture has no default: ${task_mode}" >&2; exit 2 ;;
esac

task_hw="${task_code_repo}/hw_autoresearch_nts07"
task_bundle_rel=hw_autoresearch_nts07/system_handoff/m460r5_launch_bundle_20260826
task_manifest_rel="${task_bundle_rel}/M460R5_LAUNCH_SHA256SUMS"
task_outer_rel="${task_bundle_rel}/M460R5_LAUNCH_SHA256SUMS.outer.seal.sha256"
task_manifest="${task_code_repo}/${task_manifest_rel}"
task_outer="${task_code_repo}/${task_outer_rel}"
task_contract="${task_hw}/contracts/m460r5_h67_g8_one_shot_capture_contract_r1_20260826.json"
task_freeze="${task_hw}/system_handoff/m460r4_launch_bundle_20260826/m460r4_remote_environment_freeze.json"
task_inventory_builder="${task_hw}/system_handoff/scripts/build_m460r4_package_inventory.py"
task_preflight="${task_hw}/system_handoff/scripts/preflight_m460r5_one_shot_capture.py"
task_capture="${task_hw}/system_handoff/scripts/capture_m460r5_h67_g8_ffn_token_residual_s10_one_shot.py"
task_sealer="${task_hw}/system_handoff/scripts/seal_m460r5_one_shot_result.py"
task_test="${task_hw}/tests/test_m460r5_one_shot_capture_compatibility.py"
task_output="${task_hw}/results/m460r5_h67_g8_one_shot_s10_r1_20260826"
task_stage="${task_output}.partial.$$"
task_consumed="${task_hw}/results/M460R5_G8_S10_ONE_SHOT_CAPTURE.consumed"
task_expected_outer="${M460R5_EXPECTED_OUTER_SEAL_SHA256:-}"

task_r4_launch_dir="${task_hw}/system_handoff/m460r4_launch_bundle_20260826"
task_r4_launch_manifest="${task_r4_launch_dir}/M460R4_LAUNCH_SHA256SUMS"
task_r4_launch_outer="${task_r4_launch_dir}/M460R4_LAUNCH_SHA256SUMS.outer.seal.sha256"
task_r4_result_dir="${task_hw}/results/m460r4_remote_preflight_r1_20260826"
task_r4_result_manifest="${task_r4_result_dir}/manifest.sha256"
task_r4_result_outer="${task_r4_result_dir}/manifest.sha256.outer.seal.sha256"
task_review_dir="${task_hw}/results/m460r4_preflight_independent_hammer_r1_20260826"
task_review_manifest="${task_review_dir}/M460R4_PREFLIGHT_INDEPENDENT_HAMMER_SHA256SUMS"
task_review_outer="${task_review_dir}/M460R4_PREFLIGHT_INDEPENDENT_HAMMER_SHA256SUMS.outer.seal.sha256"

task_expected_r4_launch_outer=4a9d8effe78878774c910284d256537fc258290a015c6c174af1850acd72e604
task_expected_r4_result_outer=317d36436de1d02b94597f4ad1946ff13924da2c03dedc67921d614b0549acd5
task_expected_review_outer=4b788b9f86bcf2de70ffc08dc4f2e7a67fa62e1d8f148d25729db54294ddbf99

fail() {
    echo "status=FAILED_M460R5_$1" >&2
    exit 2
}

verify_file_sha() {
    local expected="$1"
    local path="$2"
    [[ -f "${path}" && ! -L "${path}" ]] || fail "TRUST_ROOT_ABSENT_${path##*/}"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || {
        echo "M460R5 SHA mismatch path=${path} expected=${expected} actual=${actual}" >&2
        exit 2
    }
}

verify_rooted_manifest() {
    local directory="$1"
    local outer_name="$2"
    local manifest_name="$3"
    (
        cd "${directory}"
        sha256sum -c "${outer_name}"
        sha256sum -c "${manifest_name}"
    )
}

# External non-circular launch trust is authenticated before Python or GPU tools.
[[ "${task_expected_outer}" =~ ^[0-9a-f]{64}$ ]] || \
    fail REVIEW_APPROVED_OUTER_SEAL_SHA_REQUIRED
verify_file_sha "${task_expected_outer}" "${task_outer}"
(
    cd "${task_code_repo}"
    sha256sum -c "${task_outer_rel}"
    sha256sum -c "${task_manifest_rel}"
)

# The exact R4 launch, remote preflight and independent authorization roots are
# also authenticated before any executable in the R5 payload is started.
verify_file_sha "${task_expected_r4_launch_outer}" "${task_r4_launch_outer}"
verify_file_sha "${task_expected_r4_result_outer}" "${task_r4_result_outer}"
verify_file_sha "${task_expected_review_outer}" "${task_review_outer}"
verify_rooted_manifest "${task_code_repo}" \
    hw_autoresearch_nts07/system_handoff/m460r4_launch_bundle_20260826/M460R4_LAUNCH_SHA256SUMS.outer.seal.sha256 \
    hw_autoresearch_nts07/system_handoff/m460r4_launch_bundle_20260826/M460R4_LAUNCH_SHA256SUMS
verify_rooted_manifest "${task_r4_result_dir}" \
    manifest.sha256.outer.seal.sha256 manifest.sha256
verify_rooted_manifest "${task_review_dir}" \
    M460R4_PREFLIGHT_INDEPENDENT_HAMMER_SHA256SUMS.outer.seal.sha256 \
    M460R4_PREFLIGHT_INDEPENDENT_HAMMER_SHA256SUMS

export PYTHONNOUSERSITE=1
unset PYTHONPATH

task_test_json="$(cd /tmp && "${task_python}" -I "${task_test}")"
grep -q PASS_M460R5_CPU_FAKE_EXECUTE_AND_ADVERSARIAL_TESTS \
    <<<"${task_test_json}"
"${task_python}" -I -c \
    'import json,sys; x=json.loads(sys.argv[1]); assert x["attack_total"] == x["attack_passes"] >= 10; assert x["gpu_touched"] is False; assert x["capture_launched"] is False' \
    "${task_test_json}"

if [[ "${task_mode}" == --local-static-no-launch ]]; then
    task_dry="$(cd /tmp && "${task_python}" -I "${task_capture}" \
        --contract "${task_contract}" --dry-run)"
    grep -q PASS_M460R5_STATIC_ONE_SHOT_AND_BASE_COMPATIBILITY_DRY_RUN \
        <<<"${task_dry}"
    echo status=PASS_M460R5_LOCAL_STATIC_NO_LAUNCH
    echo expected_outer_seal_sha256="${task_expected_outer}"
    echo maximum_capture_attempts="${maximum_capture_attempts}"
    echo remote_contacted=false
    echo gpu_touched=false
    echo capture_launched=false
    echo training=false
    exit 0
fi

[[ "${M460R5_EXPLICIT_ONE_SHOT_CAPTURE:-0}" == 1 ]] || \
    fail EXPLICIT_ONE_SHOT_CAPTURE_GATE_REQUIRED
[[ "${maximum_capture_attempts}" == 1 ]] || fail MAXIMUM_CAPTURE_ATTEMPTS_DRIFT
[[ "$(readlink -f "${task_code_repo}")" == "${task_code_repo}" ]] || \
    fail CODE_REPO_SYMLINK_OR_PATH_DRIFT
[[ "$(readlink -f "${task_immutable_data_repo}")" == \
    "${task_immutable_data_repo}" ]] || fail DATA_REPO_SYMLINK_OR_PATH_DRIFT
[[ ! -e "${task_output}" && ! -e "${task_stage}" ]] || \
    fail ONE_SHOT_OUTPUT_EXISTS
[[ ! -e "${task_consumed}" ]] || fail ONE_SHOT_ALREADY_CONSUMED

task_attempt_started=0
failure_receipt() {
    local rc=$?
    trap - EXIT
    if [[ -d "${task_stage}" ]]; then
        printf 'status=FAILED_M460R5_ONE_SHOT\nexit_code=%s\nattempt_started=%s\nmaximum_capture_attempts=1\nrerun_forbidden=true\n' \
            "${rc}" "${task_attempt_started}" >"${task_stage}/failure.receipt"
    fi
    exit "${rc}"
}
trap failure_receipt EXIT
mkdir "${task_stage}"
mkdir "${task_stage}/fresh_preflight"

cd /tmp
"${task_python}" -I "${task_inventory_builder}" \
    --code-repo "${task_code_repo}" \
    --freeze "${task_freeze}" \
    --output "${task_stage}/fresh_preflight/package_build_inventory.json"

"${task_python}" -I "${task_preflight}" \
    --contract "${task_contract}" \
    --code-repo "${task_code_repo}" \
    --git-worktree-root "${task_git_worktree}" \
    --immutable-data-repo "${task_immutable_data_repo}" \
    --immutable-data-git-root "${task_immutable_data_git_root}" \
    --freeze "${task_freeze}" \
    --inventory "${task_stage}/fresh_preflight/package_build_inventory.json" \
    --launch-manifest "${task_manifest}" \
    --launch-outer-seal "${task_outer}" \
    --output "${task_stage}/fresh_preflight/preflight_receipt.json"

idle_snapshot() {
    local receipt="$1"
    local snapshot="$2"
    local contexts ml gpu_line driver gpu_name
    contexts="$(nvidia-smi --query-compute-apps=pid \
        --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)"
    ml="$(ps -eo args= | awk '
        BEGIN { count=0 }
        {
            line=tolower($0)
            if (line ~ /python/ && line ~ /(train|eval|valid|profile|capture)/ &&
                line !~ /m460r5/ && line !~ /awk/) count++
        }
        END { print count }
    ')"
    gpu_line="$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
    driver="${gpu_line%%,*}"
    gpu_name="${gpu_line#*, }"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "${snapshot}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${contexts}" "${ml}" "${driver}" "${gpu_name}" >>"${receipt}"
    [[ "${contexts}" == 0 && "${ml}" == 0 ]] || {
        echo "status=BLOCKED_M460R5_REMOTE_NOT_IDLE_NO_CAPTURE snapshot=${snapshot}" >&2
        return 3
    }
}

task_pre_idle="${task_stage}/fresh_preflight/pre_capture_idle_receipt.csv"
printf 'snapshot,utc,gpu_contexts,ml_processes,driver,gpu_name\n' >"${task_pre_idle}"
for task_idle in 0 1 2 3; do
    idle_snapshot "${task_pre_idle}" "${task_idle}"
    [[ "${task_idle}" == 3 ]] || sleep 5
done

# Recheck all launch/authorization roots immediately before irreversibly
# consuming the sole capture attempt.
verify_file_sha "${task_expected_outer}" "${task_outer}"
verify_file_sha "${task_expected_r4_launch_outer}" "${task_r4_launch_outer}"
verify_file_sha "${task_expected_r4_result_outer}" "${task_r4_result_outer}"
verify_file_sha "${task_expected_review_outer}" "${task_review_outer}"

task_contract_sha="$(sha256sum "${task_contract}" | awk '{print $1}')"
(
    set -o noclobber
    printf 'schema=m460r5_one_shot_consumed_marker_v1\nutc=%s\npid=%s\ncontract_sha256=%s\nmaximum_capture_attempts=1\nrerun_forbidden=true\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${task_contract_sha}" \
        >"${task_consumed}"
)
cp --no-preserve=mode,ownership,timestamps "${task_consumed}" \
    "${task_stage}/one_shot_consumed.marker"
task_attempt_started=1

[[ ! -e "${task_stage}/capture_payload" ]] || fail CAPTURE_PAYLOAD_EXISTS
cd /tmp
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${task_python}" -I -u "${task_capture}" \
    --contract "${task_contract}" \
    --output-dir "${task_stage}/capture_payload"

task_post_idle="${task_stage}/post_capture_idle_receipt.csv"
printf 'snapshot,utc,gpu_contexts,ml_processes,driver,gpu_name\n' >"${task_post_idle}"
idle_snapshot "${task_post_idle}" 0

"${task_python}" -I "${task_sealer}" \
    --stage-root "${task_stage}" \
    --contract "${task_contract}" \
    --launch-outer-seal "${task_outer}" \
    --consumed-marker "${task_consumed}"

(
    cd "${task_stage}/capture_payload"
    sha256sum -c manifest.sha256.outer.seal.sha256
    sha256sum -c manifest.sha256
)
(
    cd "${task_stage}"
    sha256sum -c manifest.sha256.outer.seal.sha256
    sha256sum -c manifest.sha256
)

cd "${task_code_repo}"
mv "${task_stage}" "${task_output}"
trap - EXIT
echo status=PASS_M460R5_ONE_SHOT_S10_POSTCOMPUTE_ORACLE_CAPTURE
echo result="${task_output}"
echo maximum_capture_attempts="${maximum_capture_attempts}"
echo one_shot_attempts_consumed=1
echo postcompute_oracle_only=true
echo executable_skip=false
echo system_speedup=false
echo headline=false
echo training=false
