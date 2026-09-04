#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" && \
   -z "$(/usr/bin/env | /usr/bin/grep '^BASH_FUNC_' || true)" ]] || {
    echo "M660 refuses startup hooks or exported shell functions" >&2
    exit 3
}

m660_runner_abs="$(readlink -f "${BASH_SOURCE[0]}")"
m660_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m660_repo_root="$(cd "${m660_hw_root}/.." && pwd)"
m660_python="/opt/anaconda3/envs/pytorch310/bin/python3.10"
m660_producer="${m660_repo_root}/neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m660_h67_layer_static_decoder_payload.py"
m660_contract="${m660_hw_root}/contracts/m660_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json"
m660_m511_contract="${m660_hw_root}/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
m660_config="${m660_repo_root}/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
m660_checkpoint="${m660_hw_root}/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
m660_output_parent="${m660_hw_root}/system_handoff/outgoing"
m660_output="${m660_output_parent}/m660_h67_ep35_layer_static_decoder_payload_s10_r1_20260828"
m660_attempt="${m660_hw_root}/results/.m660_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed"
m660_quarantine="${m660_output}.runner_quarantine.$$.${RANDOM}.${RANDOM}"
m660_capture_started=0
m660_success=0

m660_fail_closed() {
    local m660_rc=$?
    if [[ "${m660_capture_started}" -eq 1 && "${m660_success}" -ne 1 && \
          ( -e "${m660_output}" || -L "${m660_output}" ) ]]; then
        mv -- "${m660_output}" "${m660_quarantine}" || exit 99
    fi
    exit "${m660_rc}"
}
trap m660_fail_closed EXIT

[[ "${m660_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "${m660_runner_abs}" == \
   "${m660_hw_root}/system_handoff/scripts/run_m660_h67_layer_static_decoder_payload_one_shot.sh" ]] || {
    echo "M660 canonical path drift" >&2
    exit 3
}
[[ -n "${M660_EXPECTED_RUNNER_SHA256:-}" && \
   "$(sha256sum "${m660_runner_abs}" | awk '{print $1}')" == \
   "${M660_EXPECTED_RUNNER_SHA256}" ]] || {
    echo "M660 caller must supply the independently reviewed runner SHA" >&2
    exit 3
}
[[ -n "${M660_EXPECTED_CONTRACT_SHA256:-}" && \
   -f "${m660_contract}" && ! -L "${m660_contract}" && \
   "$(sha256sum "${m660_contract}" | awk '{print $1}')" == \
   "${M660_EXPECTED_CONTRACT_SHA256}" ]] || {
    echo "M660 caller must supply the independently reviewed contract SHA" >&2
    exit 3
}
[[ -x "${m660_python}" && ! -L "${m660_python}" && \
   "$(sha256sum "${m660_python}" | awk '{print $1}')" == \
   "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
    echo "M660 Python identity drift" >&2
    exit 4
}
[[ "$(/usr/bin/hostname)" == "ic.ismd-nemo" && \
   "$(sha256sum /usr/bin/hostname | awk '{print $1}')" == \
   "c1f8c2c26baa42a5896989353aa7330cd41693435b5fe08386a8b7aa998629dc" && \
   "$(sha256sum /usr/bin/nvidia-smi | awk '{print $1}')" == \
   "6b8be04c92bf327401faa99d6c7aa7da351b0d4aca8531b422efe2e58b456886" ]] || {
    echo "M660 host identity/tool drift" >&2
    exit 4
}
m660_gpu=$(/usr/bin/nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader,nounits 2>/dev/null)
[[ "${m660_gpu}" == \
   "0, NVIDIA GeForce RTX 3090, GPU-2b9bf62c-21f9-6c5e-8ace-ee867d88a037, 575.64, 24576" ]] || {
    echo "M660 requires the frozen single local RTX3090 identity" >&2
    exit 4
}
[[ ! -e "${m660_output}" && ! -L "${m660_output}" && \
   ! -e "${m660_attempt}" && ! -L "${m660_attempt}" && \
   ! -e "${m660_quarantine}" && ! -L "${m660_quarantine}" ]] || {
    echo "M660 one-shot/output/quarantine already exists" >&2
    exit 5
}
mkdir -p "${m660_output_parent}"
[[ -w "${m660_output_parent}" && -w "$(dirname "${m660_attempt}")" ]] || {
    echo "M660 output/attempt parent is not writable" >&2
    exit 5
}
m660_free_kib=$(df -Pk "${m660_output_parent}" | awk 'NR==2 {print $4}')
[[ "${m660_free_kib}" -ge 1048576 ]] || {
    echo "M660 requires at least 1 GiB free" >&2
    exit 5
}
m660_gpu_free=$(/usr/bin/nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $1}')
m660_mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
m660_commit_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
m660_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
m660_commit_headroom=$((m660_commit_limit - m660_committed))
[[ "${m660_gpu_free}" -ge 20480 && "${m660_mem_available}" -ge 8388608 && \
   "${m660_commit_headroom}" -ge 8388608 ]] || {
    echo "M660 resource gate failed without consuming one-shot" >&2
    exit 40
}
[[ -z "$(/usr/bin/nvidia-smi --query-compute-apps=pid,process_name \
    --format=csv,noheader 2>/dev/null)" ]] || {
    echo "M660 GPU is not idle; one-shot remains unconsumed" >&2
    exit 40
}

"${m660_python}" - "${m660_contract}" "${m660_repo_root}" \
    "${m660_runner_abs}" "${M660_EXPECTED_RUNNER_SHA256}" \
    "${M660_EXPECTED_CONTRACT_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

contract_path = Path(sys.argv[1])
root = Path(sys.argv[2])
runner = Path(sys.argv[3])
runner_sha = sys.argv[4]
contract_sha = sys.argv[5]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

contract = json.loads(contract_path.read_text(encoding="utf-8"))
if digest(contract_path) != contract_sha:
    raise SystemExit("M660 reviewed contract changed during preflight")
if contract["status"] != "STATIC_AUTHOR_HANDOFF_ONLY__FRESH_HAMMER_REQUIRED_BEFORE_GPU":
    raise SystemExit("M660 contract status drift")
if digest(runner) != runner_sha:
    raise SystemExit("M660 runner changed during preflight")
for name, entry in contract["inputs"].items():
    path = root / entry["path"]
    if not path.is_file() or path.is_symlink() or digest(path) != entry["sha256"]:
        raise SystemExit("M660 contract input drift: " + name)
if contract["inputs"]["runner"]["sha256"] != runner_sha:
    raise SystemExit("M660 contract does not bind reviewed runner")
PY

"${m660_python}" - <<'PY'
from importlib.metadata import version
expected = {
    "torch": "2.7.1+cu128",
    "numpy": "2.1.2",
    "spikingjelly": "0.0.0.0.14",
}
for name, wanted in expected.items():
    if version(name) != wanted:
        raise SystemExit("M660 package drift: " + name)
PY

mkdir "${m660_attempt}"
mkdir "${m660_attempt}/initial"
{
    echo "status=CONSUMED_IMMEDIATELY_BEFORE_M660_ONE_SHOT"
    echo "timestamp=$(/usr/bin/date --iso-8601=seconds)"
    echo "repo_root=${m660_repo_root}"
    echo "runner=${m660_runner_abs}"
    echo "runner_sha256=${M660_EXPECTED_RUNNER_SHA256}"
    echo "contract_sha256=${M660_EXPECTED_CONTRACT_SHA256}"
    echo "output=${m660_output}"
    echo "hostname=$(/usr/bin/hostname)"
    echo "gpu=${m660_gpu}"
    echo "gpu_free_mib=${m660_gpu_free}"
    echo "mem_available_kib=${m660_mem_available}"
    echo "commit_headroom_kib=${m660_commit_headroom}"
    echo "claim_boundary=CAPTURE_ONLY_NO_CYCLES_SPEEDUP_RTL_EDA_ENERGY_PPA_OR_HEADLINE"
} >"${m660_attempt}/initial/ATTEMPT_CONSUMED.txt"
sha256sum "${m660_runner_abs}" "${m660_python}" "${m660_producer}" \
    "${m660_contract}" /usr/bin/hostname /usr/bin/nvidia-smi \
    "${m660_hw_root}/docs/359_DATE终局冻结_20260813.md" \
    >"${m660_attempt}/initial/identity.sha256"
(cd "${m660_attempt}/initial" && \
    sha256sum ATTEMPT_CONSUMED.txt identity.sha256 >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

m660_capture_started=1
/usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SDFORMER_USE_MLFLOW=0 \
    M660_EXPECTED_RUNNER_SHA256="${M660_EXPECTED_RUNNER_SHA256}" \
    M660_EXPECTED_CONTRACT_SHA256="${M660_EXPECTED_CONTRACT_SHA256}" \
    M660_RUNNER_PATH="${m660_runner_abs}" \
    M660_ATTEMPT_DIRECTORY="${m660_attempt}" \
    "${m660_python}" "${m660_producer}" \
    --contract "${m660_contract}" \
    --m511-contract "${m660_m511_contract}" \
    --config "${m660_config}" \
    --checkpoint "${m660_checkpoint}" \
    --output-dir "${m660_output}" \
    --samples 10 --num-workers 0 --chunk-elements 8388608

(cd "${m660_output}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
sha256sum -c "${m660_attempt}/initial/identity.sha256" >/dev/null
{
    echo "status=PASS_CAPTURE_AND_RUNNER_REHASH"
    echo "timestamp=$(/usr/bin/date --iso-8601=seconds)"
    echo "manifest_sha256=$(sha256sum "${m660_output}/manifest.json" | awk '{print $1}')"
    echo "outer_seal_file_sha256=$(sha256sum "${m660_output}/SHA256SUMS.seal.sha256" | awk '{print $1}')"
    echo "claim_boundary=CAPTURE_ONLY_NO_CYCLES_SPEEDUP_RTL_EDA_ENERGY_PPA_OR_HEADLINE"
} >"${m660_attempt}/POSTCAPTURE_PASS.txt"
(cd "${m660_attempt}" && \
    sha256sum initial/SHA256SUMS.seal.sha256 POSTCAPTURE_PASS.txt \
        >SHA256SUMS && \
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m660_success=1
trap - EXIT
echo "PASS M660 one-shot capture; independent result hammer still required"
