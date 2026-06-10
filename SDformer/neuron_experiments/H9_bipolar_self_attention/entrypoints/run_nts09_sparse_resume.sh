#!/usr/bin/env bash
# Resume NTS09 sparse short sweep: skip 09e if checkpoint+valid10 already exist
# in a prior nts09_sparse_* directory; run remaining candidates to completion.
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
GEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts09_threshold_freeze_hw_configs.py"
SCREEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py"
PROFILE="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py"
CKPT="${ROOT}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
CFG_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/configs/generated"
PREV_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/results/nts09_sparse_20260609_185456"
LOG="${ROOT}/neuron_experiments/H9_bipolar_self_attention/results/nts09_sparse_resume_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "[nts09_sparse_resume] log=${LOG}"
echo "[nts09_sparse_resume] started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow

"${PY}" "${GEN}"

# 1) Remaining rapid_screen candidates (09b was interrupted; 09f-i never started)
"${PY}" "${SCREEN}" \
  --config "${CFG_DIR}/nts09b_hw_h60_freeze918_s1224.yml" \
  --config "${CFG_DIR}/nts09f_hw_h60_freeze6120_s1224.yml" \
  --config "${CFG_DIR}/nts09g_hw_h60_freeze12240_s1224.yml" \
  --config "${CFG_DIR}/nts09h_hw_h60_cap115_freeze12240_s1224.yml" \
  --config "${CFG_DIR}/nts09i_hw_h60_eta0013_freeze6120_s1224.yml" \
  --steps 1224 \
  --prev-runid "${CKPT}" \
  --batch-size 8 \
  --valid-samples 10 \
  --confirm-steps 1224 \
  --promote-samples 40 \
  --promote-aee 1.75 \
  --promote-aae 16.0 \
  --promote-sops-g 6.0 \
  --workers 8 \
  --prefetch-factor 4 \
  --pin-memory \
  --tag nts09_sparse_resume

RESUME_DIR="$(ls -td ${ROOT}/neuron_experiments/H9_bipolar_self_attention/results/nts09_sparse_resume_* 2>/dev/null | head -1)"
echo "[nts09_sparse_resume] resume dir=${RESUME_DIR}"

# 2) Carry forward 09e from interrupted run +补 valid40
if [[ -d "${PREV_DIR}/runs/nts09e_hw_h60_freeze1224_s1224_steps1224" ]]; then
  mkdir -p "${RESUME_DIR}/runs"
  cp -a "${PREV_DIR}/runs/nts09e_hw_h60_freeze1224_s1224_steps1224" \
    "${RESUME_DIR}/runs/"
  CFG_E="${RESUME_DIR}/configs/nts09e_hw_h60_freeze1224_s1224_steps1224.yml"
  if [[ ! -f "${CFG_E}" ]]; then
    mkdir -p "${RESUME_DIR}/configs"
    cp "${PREV_DIR}/configs/nts09e_hw_h60_freeze1224_s1224_steps1224.yml" "${CFG_E}" 2>/dev/null || \
      cp "${CFG_DIR}/nts09e_hw_h60_freeze1224_s1224.yml" "${CFG_E}"
  fi
  CKPT_E="${RESUME_DIR}/runs/nts09e_hw_h60_freeze1224_s1224_steps1224/checkpoint_epoch0.pth"
  for NSAMPLES in 10 40; do
    OUT="${RESUME_DIR}/profiles/nts09e_hw_h60_freeze1224_s1224_steps1224_valid${NSAMPLES}"
    if [[ ! -f "${OUT}/sops_summary.json" ]]; then
      "${PY}" -u "${PROFILE}" \
        --config "${CFG_E}" \
        --checkpoint "${CKPT_E}" \
        --output-dir "${OUT}" \
        --split valid --num-samples "${NSAMPLES}" \
        --batch-size 1 --num-workers 4 --metric AEE --metric AAE
    fi
  done
  echo "[nts09_sparse_resume] merged 09e from ${PREV_DIR}"
fi

# 3) Rebuild combined ranking (resume dir + 09e rows)
COMBINE="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/merge_rapid_screen_summaries.py"
if [[ -f "${COMBINE}" ]]; then
  "${PY}" "${COMBINE}" \
    --out "${RESUME_DIR}" \
    --input "${RESUME_DIR}" \
    --input "${PREV_DIR}" \
    --tag nts09_sparse_combined || true
fi

echo "[nts09_sparse_resume] finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"