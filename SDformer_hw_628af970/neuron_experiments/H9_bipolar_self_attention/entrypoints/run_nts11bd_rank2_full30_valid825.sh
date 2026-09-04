#!/usr/bin/env bash
# Rank2: nts11bd_u12_ds_w720_fastlr — full30 + standard valid825
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/bin/python3
STAMP=$(date +%Y%m%d_%H%M%S)
META_NAME=nts11bd_u12_ds_w720_fastlr
SHORT_CFG="${EXP}/configs/generated/${META_NAME}_s1224.yml"
RESUME="${REPO}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
FULL_CFG="${EXP}/configs/${META_NAME}_full30_${STAMP}.yml"
RUN_DIR="${EXP}/results/${META_NAME}_full30_${STAMP}_bs8_${STAMP}_setsid"
LOG="${RUN_DIR}/pipeline.log"

mkdir -p "${RUN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== 11bd rank2 (${META_NAME}) full30+valid825 start $(date -Is) ==="

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

"${PY}" - <<'PY' "${SHORT_CFG}" "${FULL_CFG}" "${META_NAME}"
import sys
from copy import deepcopy
from pathlib import Path
import yaml

short_cfg = Path(sys.argv[1])
full_cfg = Path(sys.argv[2])
meta_name = sys.argv[3]
cfg = yaml.safe_load(short_cfg.read_text(encoding="utf-8"))
cfg["experiment"] = meta_name + "_full30"
loader = cfg.setdefault("loader", {})
loader["n_epochs"] = 30
loader["batch_size"] = 8
loader["n_workers"] = 8
loader["persistent_workers"] = True
loader["prefetch_factor"] = 4
runtime = cfg.setdefault("runtime", {})
runtime["max_train_steps"] = 0
runtime["skip_state_save"] = False
runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
optimizer = cfg.setdefault("optimizer", {})
optimizer["use_amp"] = True
optimizer["milestones"] = [20, 25]
cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
cfg.setdefault("test", {})["sample"] = 10
cfg["note"] = (
    str(cfg.get("note", ""))
    + "\n11bd rank2 manual launch: h60 all12, downsample_ternary + w720_fastlr."
)
full_cfg.parent.mkdir(parents=True, exist_ok=True)
full_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"full_config={full_cfg}")
PY

echo "config=${FULL_CFG}"
echo "resume=${RESUME}"
echo "run_dir=${RUN_DIR}"

echo "=== verify chain ==="
"${PY}" -u "${EXP}/entrypoints/verify_nts11_chain.py" "${FULL_CFG}"

echo "=== train full30 ==="
"${PY}" -u "${EXP}/entrypoints/train.py" \
  --config "${FULL_CFG}" \
  --prev_runid "${RESUME}" \
  --save_path "${RUN_DIR}/checkpoint_epoch{}.pth"

echo "=== train done $(date -Is) ==="
ls -la "${RUN_DIR}"/checkpoint_epoch*.pth || true

echo "=== standard valid825 ==="
"${PY}" -u "${EXP}/entrypoints/run_h9_standard_valid825_eval.py" \
  --config "${FULL_CFG}" \
  --run-dir "${RUN_DIR}" \
  --epoch 9 --epoch 14 --epoch 19 --epoch 24 --epoch 28 --epoch 29

echo "=== 11bd rank2 pipeline complete $(date -Is) ==="