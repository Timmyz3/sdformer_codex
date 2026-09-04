#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M511 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m511_runner_abs="$(readlink -f "${BASH_SOURCE[0]}")"
m511_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m511_repo_root="$(cd "${m511_hw_root}/.." && pwd)"
cd "${m511_hw_root}"

m511_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m511_launch_wrapper="${m511_hw_root}/system_handoff/scripts/run_m632_m511_local_rtx3090_capture_exact_sha.sh"
m511_hostname_tool="/usr/bin/hostname"
m511_nvidia_smi_tool="/usr/bin/nvidia-smi"
m511_expected_hostname="ic.ismd-nemo"
m511_expected_gpu_name="NVIDIA GeForce RTX 3090"
m511_expected_gpu_uuid="GPU-2b9bf62c-21f9-6c5e-8ace-ee867d88a037"
m511_expected_gpu_driver="575.64"
m511_expected_gpu_memory_total_mib="24576"
m511_producer="${m511_repo_root}/neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m511_h67_convtranspose_binary_inputs.py"
m511_contract="${m511_hw_root}/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
m511_review="${m511_hw_root}/reviews/m511_capture_static_hammer_r4_20260827"
m511_config="${m511_repo_root}/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
m511_checkpoint="${m511_hw_root}/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
m511_output_parent="${m511_hw_root}/system_handoff/outgoing"
m511_output="${m511_output_parent}/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827"
m511_attempt="${m511_hw_root}/results/.m511_h67_ep35_convtranspose_binary_input_capture_r1_attempt_consumed"
m511_quarantine="${m511_output}.runner_quarantine.$$.${RANDOM}.${RANDOM}"
m511_capture_started=0
m511_runner_success=0
m511_preflight=""
m511_identity_tmp=""

m511_fail_closed_exit() {
    local m511_rc=$?
    if [[ "${m511_capture_started}" -eq 1 && \
          "${m511_runner_success}" -ne 1 ]]; then
        if [[ -e "${m511_output}" || -L "${m511_output}" ]]; then
            if ! mv -- "${m511_output}" "${m511_quarantine}"; then
                echo "M511 FATAL: failed to quarantine canonical output" >&2
                exit 99
            fi
            if [[ -e "${m511_output}" || ! -d "${m511_quarantine}" ]]; then
                echo "M511 FATAL: canonical rollback postcondition failed" >&2
                exit 99
            fi
        fi
    fi
    [[ -z "${m511_preflight}" || ! -f "${m511_preflight}" ]] || \
        rm -f -- "${m511_preflight}"
    [[ -z "${m511_identity_tmp}" || ! -f "${m511_identity_tmp}" ]] || \
        rm -f -- "${m511_identity_tmp}"
    exit "${m511_rc}"
}
trap m511_fail_closed_exit EXIT

[[ "${m511_runner_abs}" == \
   "${m511_hw_root}/system_handoff/scripts/run_m511_h67_ep35_convtranspose_binary_input_capture_r1_exact_sha.sh" ]] || {
    echo "M511 runner canonical path drift" >&2
    exit 3
}
[[ -n "${M511_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m511_runner_abs}" | awk '{print $1}')" == \
   "${M511_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M511 caller did not supply the literal reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M511_EXPECTED_REPO_ROOT:-}" && \
   "${m511_repo_root}" == "${M511_EXPECTED_REPO_ROOT}" ]] || {
    echo "M511 caller did not pin the isolated repo root" >&2
    exit 3
}
[[ -n "${M632_LAUNCH_WRAPPER_PATH:-}" && \
   "$(readlink -f "${M632_LAUNCH_WRAPPER_PATH}")" == \
   "${m511_launch_wrapper}" && \
   -n "${M632_EXPECTED_WRAPPER_SHA256:-}" && \
   "$(sha256sum "${m511_launch_wrapper}" | awk '{print $1}')" == \
   "${M632_EXPECTED_WRAPPER_SHA256}" ]] || {
    echo "M511 missing independently frozen M632 launch-wrapper identity" >&2
    exit 3
}
[[ -x "${m511_python}" && \
   ! -L "${m511_python}" && \
   "$(sha256sum "${m511_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M511 missing pinned Python environment" >&2
    exit 3
}
[[ ! -e "${m511_output}" && ! -L "${m511_output}" && \
   ! -e "${m511_attempt}" && ! -L "${m511_attempt}" && \
   ! -e "${m511_quarantine}" && ! -L "${m511_quarantine}" ]] || {
    echo "M511 one-shot/output/quarantine already exists" >&2
    exit 5
}
[[ ! -e "${m511_output_parent}" || -d "${m511_output_parent}" ]] || {
    echo "M511 output parent exists but is not a directory" >&2
    exit 5
}
mkdir -p "${m511_output_parent}"
[[ -w "${m511_output_parent}" && -w "$(dirname "${m511_attempt}")" ]] || {
    echo "M511 output/attempt parent is not writable" >&2
    exit 5
}
m511_free_kib=$(df -Pk "${m511_output_parent}" | awk 'NR==2 {print $4}')
[[ "${m511_free_kib}" -ge 2097152 ]] || {
    echo "M511 requires at least 2 GiB free before capture" >&2
    exit 5
}

m511_sha() { sha256sum "$1" | awk '{print $1}'; }
m511_expect() {
    local m511_path=$1 m511_expected=$2
    [[ -f "${m511_path}" && "$(m511_sha "${m511_path}")" == \
       "${m511_expected}" ]] || {
        echo "M511 identity mismatch ${m511_path}" >&2
        exit 6
    }
}

m511_verify_host_gpu_identity() {
    m511_observed_hostname=$("${m511_hostname_tool}")
    m511_observed_gpu_name=$("${m511_nvidia_smi_tool}" --query-gpu=name \
        --format=csv,noheader,nounits 2>/dev/null | /usr/bin/sed -n '1p')
    m511_observed_gpu_uuid=$("${m511_nvidia_smi_tool}" --query-gpu=uuid \
        --format=csv,noheader,nounits 2>/dev/null | /usr/bin/sed -n '1p')
    m511_observed_gpu_driver=$("${m511_nvidia_smi_tool}" --query-gpu=driver_version \
        --format=csv,noheader,nounits 2>/dev/null | /usr/bin/sed -n '1p')
    m511_observed_gpu_memory_total_mib=$("${m511_nvidia_smi_tool}" --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null | /usr/bin/sed -n '1p')
    [[ "${m511_observed_hostname}" == "${m511_expected_hostname}" && \
       "${m511_observed_gpu_name}" == "${m511_expected_gpu_name}" && \
       "${m511_observed_gpu_uuid}" == "${m511_expected_gpu_uuid}" && \
       "${m511_observed_gpu_driver}" == "${m511_expected_gpu_driver}" && \
       "${m511_observed_gpu_memory_total_mib}" == \
       "${m511_expected_gpu_memory_total_mib}" ]] || {
        echo "M511 literal host/GPU admission identity drift" >&2
        return 1
    }
}

m511_verify_identity() {
    m511_verify_host_gpu_identity
    m511_expect "${m511_hostname_tool}" c1f8c2c26baa42a5896989353aa7330cd41693435b5fe08386a8b7aa998629dc
    m511_expect "${m511_nvidia_smi_tool}" 6b8be04c92bf327401faa99d6c7aa7da351b0d4aca8531b422efe2e58b456886
    m511_expect "${m511_launch_wrapper}" "${M632_EXPECTED_WRAPPER_SHA256}"
    m511_expect "${m511_python}" 9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115
    m511_expect "${m511_producer}" e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a
    m511_expect "${m511_contract}" e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e
    m511_expect "${m511_review}/SHA256SUMS.seal.sha256" 1d2334c7a73bfc84c3067089d953c43f775ae5e6d98c2788172fa8de244aa748
    m511_expect "${m511_hw_root}/docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
    (cd "${m511_review}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
    "${m511_python}" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "numpy": "2.1.2",
    "spikingjelly": "0.0.0.0.14",
    "timm": "0.6.13",
    "einops": "0.8.2",
    "PyYAML": "6.0.3",
    "opencv-python-headless": "4.11.0.86",
    "h5py": "3.16.0",
}
for name, wanted in expected.items():
    observed = version(name)
    if observed != wanted:
        raise SystemExit(
            "M511 package identity mismatch {} {} != {}".format(
                name, observed, wanted))
PY
    "${m511_python}" - "${m511_contract}" "${m511_repo_root}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

contract_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
assert len(contract["inputs"]) == 21
for name, entry in contract["inputs"].items():
    path = repo_root / entry["path"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise SystemExit("M511 contract input mismatch: " + name)
PY
}

m511_resource_snapshot() {
    local m511_limit m511_committed m511_headroom m511_available m511_swap
    local m511_gpu_free m511_failcnt m511_under_oom m511_oom_kill
    m511_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    m511_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    m511_headroom=$((m511_limit - m511_committed))
    m511_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    m511_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    m511_gpu_free=$("${m511_nvidia_smi_tool}" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $1}')
    m511_failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    m511_under_oom=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    m511_oom_kill=$(awk '/^oom_kill / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s gpu_free_mib=%s cgroup_failcnt=%s cgroup_under_oom=%s cgroup_oom_kill=%s\n' \
        "${m511_headroom}" "${m511_available}" "${m511_swap}" \
        "${m511_gpu_free}" "${m511_failcnt}" "${m511_under_oom}" \
        "${m511_oom_kill}"
    [[ "${m511_headroom}" -ge 8388608 && \
       "${m511_available}" -ge 8388608 && \
       "${m511_swap}" -ge 8388608 && \
       "${m511_gpu_free}" -ge 20480 && \
       "${m511_failcnt}" -eq 0 && "${m511_under_oom}" -eq 0 ]]
}

m511_idle_gate() {
    if pgrep -af '(train[.]py|eval_.*DSEC|capture_m511.*[.]py)' >/dev/null; then
        echo "M511 local workload gate: training/evaluation/capture is active" >&2
        pgrep -af '(train[.]py|eval_.*DSEC|capture_m511.*[.]py)' >&2 || true
        return 1
    fi
    local m511_gpu_rows
    m511_gpu_rows=$("${m511_nvidia_smi_tool}" --query-compute-apps=pid,process_name \
        --format=csv,noheader 2>/dev/null || return 1)
    [[ -z "${m511_gpu_rows}" ]] || {
        echo "M511 GPU is not idle" >&2
        printf '%s\n' "${m511_gpu_rows}" >&2
        return 1
    }
}

m511_verify_identity
m511_preflight=$(mktemp)
m511_preflight_ok=1
for m511_sample in 1 2 3; do
    printf 'timestamp=%s sample=%s\n' "$(date --iso-8601=seconds)" \
        "${m511_sample}" >>"${m511_preflight}"
    m511_resource_snapshot >>"${m511_preflight}" || m511_preflight_ok=0
    m511_idle_gate || m511_preflight_ok=0
    if [[ "${m511_sample}" -ne 3 ]]; then sleep 10; fi
done
if [[ "${m511_preflight_ok}" -ne 1 ]]; then
    cat "${m511_preflight}" >&2
    rm -f "${m511_preflight}"
    echo "M511 resource gate failed without consuming one-shot" >&2
    exit 40
fi
m511_verify_identity
m511_idle_gate

m511_failcnt_start=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
m511_under_oom_start=$(awk '/^under_oom / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
m511_oom_kill_start=$(awk '/^oom_kill / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
[[ "${m511_failcnt_start}" -eq 0 && "${m511_under_oom_start}" -eq 0 ]] || {
    rm -f "${m511_preflight}"
    echo "M511 cgroup state changed before attempt" >&2
    exit 41
}

m511_identity_tmp=$(mktemp)
sha256sum "${m511_runner_abs}" "${m511_launch_wrapper}" \
    "${m511_hostname_tool}" "${m511_nvidia_smi_tool}" \
    "${m511_python}" "${m511_producer}" "${m511_contract}" \
    "${m511_review}/SHA256SUMS.seal.sha256" \
    "${m511_hw_root}/docs/359_DATE终局冻结_20260813.md" \
    >"${m511_identity_tmp}"
sha256sum -c "${m511_identity_tmp}" >/dev/null

mkdir "${m511_attempt}"
mkdir "${m511_attempt}/initial"
{
    echo "status=CONSUMED_IMMEDIATELY_BEFORE_M511_ONE_SHOT"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "superseded_static_reviews=r1,r2,r3"
    echo "authorized_static_review=r4"
    echo "repo_root=${m511_repo_root}"
    echo "output=${m511_output}"
    echo "quarantine=${m511_quarantine}"
    echo "hostname=${m511_observed_hostname}"
    echo "gpu_name=${m511_observed_gpu_name}"
    echo "gpu_uuid=${m511_observed_gpu_uuid}"
    echo "gpu_driver=${m511_observed_gpu_driver}"
    echo "gpu_memory_total_mib=${m511_observed_gpu_memory_total_mib}"
    echo "cgroup_failcnt_start=${m511_failcnt_start}"
    echo "cgroup_under_oom_start=${m511_under_oom_start}"
    echo "cgroup_oom_kill_start=${m511_oom_kill_start}"
} >"${m511_attempt}/initial/ATTEMPT_CONSUMED.txt"
mv "${m511_preflight}" "${m511_attempt}/initial/resource_preflight.log"
mv "${m511_identity_tmp}" "${m511_attempt}/initial/identity.sha256"
(cd "${m511_attempt}/initial" && sha256sum ATTEMPT_CONSUMED.txt \
    resource_preflight.log identity.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m511_capture_started=1
"${m511_python}" "${m511_producer}" \
    --contract "${m511_contract}" \
    --config "${m511_config}" \
    --checkpoint "${m511_checkpoint}" \
    --output-dir "${m511_output}" \
    --samples 10 --num-workers 0 --chunk-elements 8388608

(cd "${m511_output}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m511_verify_identity
sha256sum -c "${m511_attempt}/initial/identity.sha256" >/dev/null
m511_failcnt_end=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
m511_under_oom_end=$(awk '/^under_oom / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
m511_oom_kill_end=$(awk '/^oom_kill / {print $2}' \
    /sys/fs/cgroup/memory/user.slice/memory.oom_control)
[[ "${m511_failcnt_end}" -eq "${m511_failcnt_start}" && \
   "${m511_under_oom_end}" -eq 0 && \
   "${m511_oom_kill_end}" -eq "${m511_oom_kill_start}" ]] || {
    echo "M511 cgroup state changed during capture" >&2
    exit 42
}
{
    echo "status=PASS_EXACT_CAPTURE_AND_RUNNER_REHASH"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "capture_manifest_sha256=$(m511_sha "${m511_output}/manifest.json")"
    echo "capture_seal_file_sha256=$(m511_sha "${m511_output}/SHA256SUMS.seal.sha256")"
    echo "hostname=${m511_observed_hostname}"
    echo "gpu_name=${m511_observed_gpu_name}"
    echo "gpu_uuid=${m511_observed_gpu_uuid}"
    echo "gpu_driver=${m511_observed_gpu_driver}"
    echo "gpu_memory_total_mib=${m511_observed_gpu_memory_total_mib}"
    echo "cgroup_failcnt_end=${m511_failcnt_end}"
    echo "cgroup_under_oom_end=${m511_under_oom_end}"
    echo "cgroup_oom_kill_end=${m511_oom_kill_end}"
    echo "claim_boundary=CAPTURE_ONLY_NO_CYCLES_SPEEDUP_RTL_ENERGY_PPA_OR_HEADLINE"
} >"${m511_attempt}/POSTCAPTURE_PASS.txt"
(cd "${m511_attempt}/initial" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd "${m511_attempt}" && sha256sum initial/SHA256SUMS.seal.sha256 \
    POSTCAPTURE_PASS.txt >SHA256SUMS && sha256sum SHA256SUMS \
    >SHA256SUMS.seal.sha256 && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m511_runner_success=1
trap - EXIT
echo "PASS M511 one-shot capture; independent payload verification required"
