#!/usr/bin/env bash
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/envs/sdformerflow/bin/python
CONFIG="${EXP}/configs/generated/nts11aqa_hw_h60_s23_ds_w720_fastlr_ftaq2_ft5.yml"
RESUME="${EXP}/results/nts11aq_hw_h60_s23_ds_w720_fastlr_ftaa19_full_20260613_070741_bs8_20260613_070741_setsid/checkpoint_epoch2.pth"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${EXP}/results/nts11aqa_hw_h60_s23_ds_w720_fastlr_ftaq2_ft5_bs8_${STAMP}_setsid"
LOG="${RUN_DIR}/pipeline.log"

mkdir -p "${RUN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== 11aqa continue+valid825 start $(date -Is) ==="
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
  --epoch 3 --epoch 4 --epoch 5 --epoch 6 --epoch 7

echo "=== valid825 done $(date -Is) ==="
echo "=== 11aqa pipeline complete ==="