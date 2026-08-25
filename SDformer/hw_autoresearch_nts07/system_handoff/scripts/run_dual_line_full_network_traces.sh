#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "${HW_ROOT}/.." && pwd)"
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PYTHON="${PYTHON:-/opt/conda/envs/sdformerflow/bin/python}"
MODE="${MODE:-all}"
SAMPLES="${SAMPLES:-1}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-1024}"

if ! [[ "${SAMPLES}" =~ ^[0-9]+$ ]] || (( SAMPLES < 1 || SAMPLES > 100 )); then
  echo "FAIL: SAMPLES must be in [1,100]" >&2
  exit 2
fi
if [[ "${MODE}" != "all" && "${MODE}" != "motion" && "${MODE}" != "local" ]]; then
  echo "FAIL: MODE must be all, motion, or local" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "BLOCKED: nvidia-smi is unavailable" >&2
  exit 75
fi

gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if ! [[ "${gpu_used_mib}" =~ ^[0-9]+$ ]] || (( gpu_used_mib > MAX_GPU_USED_MIB )); then
  echo "BLOCKED: GPU usage ${gpu_used_mib:-unknown} MiB exceeds ${MAX_GPU_USED_MIB} MiB" >&2
  exit 75
fi

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_profile() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local output="$4"
  for input in "${config}" "${checkpoint}"; do
    if [[ ! -f "${input}" ]]; then
      echo "BLOCKED: missing ${name} input ${input}" >&2
      return 75
    fi
  done
  if [[ -e "${output}" ]]; then
    echo "FAIL: refusing to overwrite ${output}" >&2
    return 2
  fi
  "${PYTHON}" -u "${EXP}/entrypoints/profile_nts11_hardware_p0.py" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output}" \
    --samples "${SAMPLES}" \
    --num-workers 0 \
    --ordered-trace \
    --dual-line-trace
  test -s "${output}/execution_trace.csv"
  test -s "${output}/dual_line_operator_trace.csv"
  sha256sum \
    "${output}/execution_trace.csv" \
    "${output}/dual_line_operator_trace.csv" \
    "${output}/nts11_hardware_p0_profile.json" \
    >"${output}/dual_line_trace.sha256"
  echo "PASS ${name} full-network dual-line trace samples=${SAMPLES}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "motion" ]]; then
  run_profile \
    motion_h67_ep35 \
    "${EXP}/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml" \
    "${EXP}/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth" \
    "${HW_ROOT}/results/h67_ep35_full_network_ordered_trace_s${SAMPLES}_20260821"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "local" ]]; then
  run_profile \
    local_ep44 \
    "${EXP}/configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml" \
    "${EXP}/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth" \
    "${HW_ROOT}/results/local_ep44_full_network_ordered_trace_s${SAMPLES}_20260821"
fi
