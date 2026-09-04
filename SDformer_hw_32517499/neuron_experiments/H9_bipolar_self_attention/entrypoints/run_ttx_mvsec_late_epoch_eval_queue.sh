#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/private_data/work/sdformer_codex/SDformer"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/sdformerflow/bin/python}"
CONFIG="${CONFIG:-configs/generated/eval_mvsec_dt1_ttx_mdr_epoch20_route.yml}"
RUN_ROOT="${RUN_ROOT:-neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956}"
CKPT_ROOT="${CKPT_ROOT:-${RUN_ROOT}/local_ckpts}"
CURRENT_OUT="${CURRENT_OUT:-results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EPOCHS=("$@")

if [ "${#EPOCHS[@]}" -eq 0 ]; then
  EPOCHS=(40 43)
fi

cd "${REPO_ROOT}"
export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_MDR_VOXEL_GPU=0
export SDFORMER_SNN_BACKEND="${SDFORMER_SNN_BACKEND:-cupy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[late-eval-queue] repo=${REPO_ROOT}"
echo "[late-eval-queue] config=${CONFIG}"
echo "[late-eval-queue] ckpt_root=${CKPT_ROOT}"
echo "[late-eval-queue] waiting for current_out=${CURRENT_OUT}"
echo "[late-eval-queue] epochs=${EPOCHS[*]}"

while pgrep -af "run_h9_standard_mvsec_eval.py.*${CURRENT_OUT}" >/dev/null; do
  date '+[late-eval-queue] %F %T waiting for epoch20 eval to finish'
  sleep 120
done

if [ ! -f "${CURRENT_OUT}/mvsec_ranking.md" ]; then
  echo "[late-eval-queue] warning: ${CURRENT_OUT}/mvsec_ranking.md not found; continuing with late checkpoints"
fi

for epoch in "${EPOCHS[@]}"; do
  ckpt="${CKPT_ROOT}/checkpoint_epoch${epoch}.pth"
  out_dir="results_inference/mvsec_ttx_mdr_epoch${epoch}_dt1_full4_${STAMP}"
  if [ ! -f "${ckpt}" ]; then
    echo "[late-eval-queue] missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
  if [ -f "${out_dir}/mvsec_ranking.md" ]; then
    echo "[late-eval-queue] skip epoch${epoch}; ranking exists: ${out_dir}/mvsec_ranking.md"
    continue
  fi
  date "+[late-eval-queue] %F %T starting epoch${epoch}"
  "${PYTHON_BIN}" neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_mvsec_eval.py \
    --config "${CONFIG}" \
    --checkpoint "${ckpt}" \
    --out-dir "${out_dir}"
  date "+[late-eval-queue] %F %T finished epoch${epoch}"
done

date '+[late-eval-queue] %F %T done'
