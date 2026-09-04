#!/usr/bin/env bash
# Run on the SECOND server only. Main server is running phase-4 scope short sweep.
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
EXP="${ROOT}/neuron_experiments/H9_bipolar_self_attention"
GEN="${EXP}/entrypoints/make_nts11_secondary_ac_config.py"
VERIFY="${EXP}/entrypoints/verify_nts11_chain.py"
CFG="${EXP}/configs/generated/nts11ac_hw_h60_s23_sn2qbin_fastlr_freeze816_warm720_full30.yml"
CKPT="${ROOT}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${EXP}/results/nts11ac_secondary_full30_bs8_${STAMP}_setsid"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PY}" "${GEN}"
"${PY}" "${VERIFY}" "${CFG}"

mkdir -p "${RUN_DIR}"
echo "run_dir=${RUN_DIR}" | tee "${RUN_DIR}/launch.txt"

"${PY}" -u "${EXP}/entrypoints/train.py" \
  --config "${CFG}" \
  --prev_runid "${CKPT}" \
  --save_path "${RUN_DIR}/checkpoint_epoch{}.pth" \
  2>&1 | tee "${RUN_DIR}/train.log"

for EPOCH in 9 14 19 24 28 29; do
  CK="${RUN_DIR}/checkpoint_epoch${EPOCH}.pth"
  [[ -f "${CK}" ]] || continue
  OUT="${RUN_DIR}/standard_valid825/epoch${EPOCH}"
  mkdir -p "${OUT}"
  "${PY}" -u "${ROOT}/third_party/SDformerFlow/eval_DSEC_flow_SNN.py" \
    --config "${CFG}" \
    --checkpoint "${CK}" \
    --path_results "${OUT}" \
    --mode valid \
    2>&1 | tee "${OUT}/eval.log"
done

echo "DONE. Results: ${RUN_DIR}"