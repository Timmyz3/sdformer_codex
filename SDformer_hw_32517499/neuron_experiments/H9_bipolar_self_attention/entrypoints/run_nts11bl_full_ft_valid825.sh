#!/usr/bin/env bash
# 11bd-v2 finetune full: nts11bl (fastlr + warm360, 5ep) from ep19 → valid825
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/bin/python3
CONFIG="${EXP}/configs/generated/nts11bl_u12_ds_w720_fastlr_w360_ftbd19_ft5.yml"
RESUME="${EXP}/results/nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid/checkpoint_epoch19.pth"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${EXP}/results/nts11bl_u12_ds_w720_fastlr_w360_ftbd19_ft5_bs8_${STAMP}_setsid"
LOG="${RUN_DIR}/pipeline.log"

mkdir -p "${RUN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== 11bl full finetune+valid825 start $(date -Is) ==="
echo "config=${CONFIG}"
echo "resume=${RESUME}"
echo "run_dir=${RUN_DIR}"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
"${PY}" -u "${EXP}/entrypoints/train.py" \
  --config "${CONFIG}" \
  --prev_runid "${RESUME}" \
  --save_path "${RUN_DIR}/checkpoint_epoch{}.pth"

echo "=== train done $(date -Is) ==="
ls -la "${RUN_DIR}"/checkpoint_epoch*.pth || true

"${PY}" -u "${EXP}/entrypoints/run_h9_standard_valid825_eval.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --epoch 0 --epoch 1 --epoch 2 --epoch 3 --epoch 4

echo "=== valid825 done $(date -Is) ==="
echo "=== 11bl pipeline complete ==="