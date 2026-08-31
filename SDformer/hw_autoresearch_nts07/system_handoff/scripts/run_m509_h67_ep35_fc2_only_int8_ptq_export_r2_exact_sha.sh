#!/usr/bin/env bash
set -euo pipefail

m509_runner_abs="$(readlink -f "${BASH_SOURCE[0]}")"
m509_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m509_root}"

m509_python="/opt/anaconda3/envs/pytorch310/bin/python"
m509_exporter="system_handoff/scripts/export_m509_h67_ep35_fc2_only_int8_ptq_checkpoint.py"
m509_verifier="system_handoff/scripts/verify_m509_h67_ep35_fc2_only_int8_ptq_export.py"
m509_contract="contracts/m509_h67_ep35_fc2_only_int8_ptq_export_contract_r2_20260827.json"
m509_review="reviews/m509_export_preflight_hammer_r2_20260827"
m509_output_parent="system_handoff/outgoing"
m509_output="${m509_output_parent}/m509_h67_ep35_fc2_only_int8_ptq_export_r2_20260827"
m509_verify_output="results/m509_h67_ep35_fc2_only_int8_ptq_postexport_verify_r2_20260827"
m509_attempt="results/.m509_h67_ep35_fc2_only_int8_ptq_export_r2_attempt_consumed"

[[ "${m509_runner_abs}" == \
   "${m509_root}/system_handoff/scripts/run_m509_h67_ep35_fc2_only_int8_ptq_export_r2_exact_sha.sh" ]] || {
    echo "M509 runner canonical path drift" >&2
    exit 3
}
[[ -n "${M509_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m509_runner_abs}" | awk '{print $1}')" == \
   "${M509_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M509 caller did not pin the reviewed runner SHA" >&2
    exit 3
}

m509_sha() { sha256sum "$1" | awk '{print $1}'; }
m509_expect() {
    local m509_path=$1 m509_expected=$2
    [[ -f "${m509_path}" ]] || {
        echo "M509 missing ${m509_path}" >&2
        exit 3
    }
    [[ "$(m509_sha "${m509_path}")" == "${m509_expected}" ]] || {
        echo "M509 SHA mismatch ${m509_path}" >&2
        exit 3
    }
}

[[ -x "${m509_python}" ]] || {
    echo "M509 missing pinned Python ${m509_python}" >&2
    exit 3
}
[[ ! -e "${m509_output}" && ! -e "${m509_verify_output}" && \
   ! -e "${m509_attempt}" ]] || {
    echo "M509 one-shot already consumed or output exists" >&2
    exit 5
}
[[ ! -e "${m509_output_parent}" || -d "${m509_output_parent}" ]] || {
    echo "M509 output parent exists but is not a directory" >&2
    exit 5
}
mkdir -p "${m509_output_parent}"
[[ -d "${m509_output_parent}" && -w "${m509_output_parent}" && \
   -d "$(dirname "${m509_verify_output}")" && \
   -w "$(dirname "${m509_verify_output}")" ]] || {
    echo "M509 output parents are not writable directories" >&2
    exit 5
}
m509_export_free_kib=$(df -Pk "${m509_output_parent}" | awk 'NR==2 {print $4}')
m509_verify_free_kib=$(df -Pk "$(dirname "${m509_verify_output}")" | \
    awk 'NR==2 {print $4}')
[[ "${m509_export_free_kib}" -ge 2097152 && \
   "${m509_verify_free_kib}" -ge 2097152 ]] || {
    echo "M509 requires at least 2 GiB free in both output filesystems" >&2
    exit 5
}

m509_identity_paths=(
    "${m509_exporter}"
    "${m509_verifier}"
    "${m509_contract}"
    "${m509_review}/SHA256SUMS.seal.sha256"
    "docs/359_DATE终局冻结_20260813.md"
    "system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
    "system_handoff/incoming/m51_capture_bundle_r2_20260823/manifest.json"
    "results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/per_ffn_bn_atlif_fusion.csv"
    "../neuron_experiments/H9_bipolar_self_attention/entrypoints/extract_m32_h67_threshold_manifest.py"
    "../neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py"
    "../neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/__init__.py"
    "../neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py"
    "../third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py"
    "../third_party/SDformerFlow/utils/utils.py"
)

m509_verify_all_identities() {
    m509_expect "${m509_exporter}" 755bf1111d986de387714356092b3d25c6c3029f83fe738a294edbbaca0739ec
    m509_expect "${m509_verifier}" 660f9a28056350a558e48ea3bdcfd8420c062686047c52b2ca96bf8ba2ffcf7b
    m509_expect "${m509_contract}" 133fad77621e7c3c3feacc6c2ce1dd1e740420f7ef81a706b7645d18426a8c8c
    m509_expect "${m509_review}/SHA256SUMS.seal.sha256" 9562d8c809f4fd0d88966c5f2cb9b1134dc65da43f6f04d9de344aecc9cb3618
    m509_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
    m509_expect "system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth" 4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158
    m509_expect "system_handoff/incoming/m51_capture_bundle_r2_20260823/manifest.json" 2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e
    m509_expect "results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/per_ffn_bn_atlif_fusion.csv" 309a5d802c7e49d432285f09ff43b9d1ec797db815b949cd34798c0a94f4f464
    m509_expect "../neuron_experiments/H9_bipolar_self_attention/entrypoints/extract_m32_h67_threshold_manifest.py" f3e213a814d5b9eb3af725009222624aaa8d1c8f4c5eb9fc2a539226e3d6dd69
    m509_expect "../neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py" 0f77f66dbd331daa77a284199cda33125a1959a005b6f4d592e2e6cda5317187
    m509_expect "../neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/__init__.py" f0e408c6bd136d7ce36b779881ca37a04de6f0cb6220701431b0a05b338f6d6b
    m509_expect "../neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py" d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3
    m509_expect "../third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py" b8d969f9b91c292197dbe47c7b9a11803f10b7c604daaf911ed4bb5d00999b71
    m509_expect "../third_party/SDformerFlow/utils/utils.py" f47d22bad7befa9b8a093f3a693ac92f3ffaac87c050fc8a9392a688b9924ae7
    (cd "${m509_review}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

m509_forbidden_process_gate() {
    local m509_hits=0
    for m509_name in dc_shell dc_shell-t fm_shell pt_shell vcs simv; do
        if pgrep -x "${m509_name}" >/dev/null; then
            echo "M509 resource gate: forbidden ${m509_name} process" >&2
            pgrep -a -x "${m509_name}" >&2 || true
            m509_hits=1
        fi
    done
    if ps -eo pid=,comm=,args= | awk '
        $2 ~ /^python(3([.][0-9]+)?)?$/ &&
        $0 ~ /(train[.]py|eval_.*DSEC|analyze_m[0-9]|simulate_m[0-9]|run_m[0-9]|sweep_m[0-9]|dse_m[0-9])/ {found=1}
        END {exit !found}
    '; then
        echo "M509 resource gate: forbidden project CPU/GPU experiment is active" >&2
        m509_hits=1
    fi
    [[ "${m509_hits}" -eq 0 ]]
}

m509_resource_snapshot() {
    local m509_limit m509_committed m509_available m509_swap m509_headroom
    local m509_failcnt m509_under_oom m509_oom_kill
    m509_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    m509_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    m509_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    m509_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    m509_headroom=$((m509_limit - m509_committed))
    m509_failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    m509_under_oom=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    m509_oom_kill=$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "${m509_headroom}" "${m509_available}" "${m509_swap}" \
        "${m509_failcnt}" "${m509_under_oom}" "${m509_oom_kill}"
    [[ "${m509_headroom}" -ge 8388608 && \
       "${m509_available}" -ge 8388608 && \
       "${m509_swap}" -ge 8388608 && \
       "${m509_failcnt}" -eq 0 && "${m509_under_oom}" -eq 0 ]]
}

m509_verify_all_identities
m509_preflight_log=$(mktemp)
m509_preflight_ok=1
for m509_sample in 1 2 3; do
    printf 'timestamp=%s sample=%s\n' "$(date --iso-8601=seconds)" \
        "${m509_sample}" >>"${m509_preflight_log}"
    m509_resource_snapshot >>"${m509_preflight_log}" || m509_preflight_ok=0
    m509_forbidden_process_gate || m509_preflight_ok=0
    if [[ "${m509_sample}" -ne 3 ]]; then sleep 10; fi
done
if [[ "${m509_preflight_ok}" -ne 1 ]]; then
    cat "${m509_preflight_log}" >&2
    rm -f "${m509_preflight_log}"
    echo "M509 resource gate failed without consuming the one-shot" >&2
    exit 40
fi
m509_forbidden_process_gate || {
    rm -f "${m509_preflight_log}"
    exit 41
}
m509_verify_all_identities

m509_failcnt_start=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
m509_under_oom_start=$(awk '/^under_oom / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
m509_oom_kill_start=$(awk '/^oom_kill / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
[[ "${m509_failcnt_start}" -eq 0 && "${m509_under_oom_start}" -eq 0 ]] || {
    rm -f "${m509_preflight_log}"
    echo "M509 cgroup state changed before attempt consumption" >&2
    exit 42
}
m509_identity_tmp=$(mktemp)
trap 'rm -f "${m509_identity_tmp}" "${m509_preflight_log}"' EXIT
sha256sum "${m509_runner_abs}" "${m509_identity_paths[@]}" \
    >"${m509_identity_tmp}"
sha256sum -c "${m509_identity_tmp}" >/dev/null

mkdir "${m509_attempt}"
mkdir "${m509_attempt}/initial"
{
    echo "status=CONSUMED_IMMEDIATELY_BEFORE_ONE_SHOT_EXPORT"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "export_output=${m509_output}"
    echo "verify_output=${m509_verify_output}"
    echo "cgroup_failcnt_start=${m509_failcnt_start}"
    echo "cgroup_under_oom_start=${m509_under_oom_start}"
    echo "cgroup_oom_kill_start=${m509_oom_kill_start}"
} >"${m509_attempt}/initial/ATTEMPT_CONSUMED.txt"
mv "${m509_preflight_log}" "${m509_attempt}/initial/resource_preflight.log"
mv "${m509_identity_tmp}" "${m509_attempt}/initial/identity.sha256"
trap - EXIT
(cd "${m509_attempt}/initial" && sha256sum ATTEMPT_CONSUMED.txt \
    resource_preflight.log identity.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

"${m509_python}" "${m509_exporter}" \
    --contract "${m509_contract}" \
    --output-dir "${m509_output}"

"${m509_python}" "${m509_verifier}" \
    --contract "${m509_contract}" \
    --export-dir "${m509_output}" \
    --output-dir "${m509_verify_output}"

(cd "${m509_output}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd "${m509_verify_output}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m509_verify_all_identities
sha256sum -c "${m509_attempt}/initial/identity.sha256" >/dev/null
m509_failcnt_end=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
m509_under_oom_end=$(awk '/^under_oom / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
m509_oom_kill_end=$(awk '/^oom_kill / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
[[ "${m509_failcnt_end}" -eq "${m509_failcnt_start}" && \
   "${m509_under_oom_end}" -eq 0 && \
   "${m509_oom_kill_end}" -eq "${m509_oom_kill_start}" ]] || {
    echo "M509 cgroup OOM/oom_kill/failcnt changed during export" >&2
    exit 43
}
{
    echo "status=PASS_EXPORT_AND_INDEPENDENT_POSTEXPORT_REHASH"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "export_seal_sha256=$(m509_sha "${m509_output}/SHA256SUMS.seal.sha256")"
    echo "verify_seal_sha256=$(m509_sha "${m509_verify_output}/SHA256SUMS.seal.sha256")"
    echo "cgroup_failcnt_end=${m509_failcnt_end}"
    echo "cgroup_under_oom_end=${m509_under_oom_end}"
    echo "cgroup_oom_kill_end=${m509_oom_kill_end}"
} >"${m509_attempt}/POSTEXPORT_PASS.txt"
(cd "${m509_attempt}" && sha256sum initial/SHA256SUMS.seal.sha256 \
    POSTEXPORT_PASS.txt >SHA256SUMS && sha256sum SHA256SUMS \
    >SHA256SUMS.seal.sha256 && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
