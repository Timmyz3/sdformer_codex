#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "${REPO}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/sdformerflow/bin/python}"
CONFIG="${CONFIG:-configs/generated/train_mdr_ttx_mvsec_ep20_calib_lr025_ep26.yml}"
CKPT="${CKPT:-neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956/local_ckpts/checkpoint_epoch20.pth}"
MLFLOW_URI="${MLFLOW_URI:-file:///root/private_data/work/sdformer_codex/SDformer/mlruns_local}"
LABEL="${LABEL:-ttx_mdr_ep20_calib_lr025_ep21_25_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="neuron_experiments/H9_bipolar_self_attention/results/${LABEL}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-${RESULT_ROOT}/local_ckpts}"

mkdir -p "${RESULT_ROOT}" "${LOCAL_CKPT_DIR}"

echo "[ttx-ep20-calib] start $(date -Iseconds)"
echo "[ttx-ep20-calib] config=${CONFIG}"
echo "[ttx-ep20-calib] checkpoint=${CKPT}"
echo "[ttx-ep20-calib] result_root=${RESULT_ROOT}"

(
  cd third_party/SDformerFlow
  SDFORMER_SNN_BACKEND="${SDFORMER_SNN_BACKEND:-cupy}" \
  SDFORMER_USE_MLFLOW=0 \
  SDFORMER_MLFLOW_MODEL_LOGGING=0 \
  SDFORMER_MDR_DETECT_ANOMALY=0 \
  SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1 \
  SDFORMER_MDR_VOXEL_GPU=0 \
  SDFORMER_MDR_RESET_LR_FROM_CONFIG=1 \
  SDFORMER_MDR_LOCAL_CHECKPOINT_DIR="${REPO}/${LOCAL_CKPT_DIR}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  KMP_DUPLICATE_LIB_OK=TRUE \
  "${PYTHON_BIN}" -u train_mdr_supervised_SNN.py \
    --config "../../${CONFIG}" \
    --prev_runid "../../${CKPT}" \
    --path_mlflow "${MLFLOW_URI}" \
    --resume 1
) 2>&1 | tee "${RESULT_ROOT}/train.log"

echo "[ttx-ep20-calib] complete $(date -Iseconds)"
