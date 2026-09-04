#!/usr/bin/env bash
# 11bd-v2 finetune recipe short screen: 8 configs × 1224 step + valid10
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/envs/sdformerflow/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
DRIVER="${EXP}/results/nts11bd_v2_short_${STAMP}"
MANIFEST="${EXP}/configs/generated/nts11bd_v2_tune_manifest.json"
STATUS="${DRIVER}/status.log"
LOG="${DRIVER}/rapid_screen.log"

mkdir -p "${DRIVER}"
exec >>"${LOG}" 2>&1
echo "=== 11bd-v2 finetune short screen start $(date -Is) ==="
echo "driver=${DRIVER}"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
"${PY}" "${EXP}/entrypoints/make_nts11bd_v2_tune_configs.py"

RESUME=$("${PY}" -c "import json; print(json.load(open('${MANIFEST}'))[0]['resume'])")
echo "[$(date -Is)] resume=${RESUME}" | tee -a "${STATUS}"

CMD=(
  "${PY}" -u "${EXP}/entrypoints/rapid_screen.py"
  --steps 1224
  --prev-runid "${RESUME}"
  --batch-size 8
  --workers 8
  --prefetch-factor 4
  --valid-samples 10
  --confirm-steps 1224
  --no-promote-valid40
  --tag nts11bd_v2_short
)

while IFS= read -r cfg; do
  CMD+=(--config "${cfg}")
done < <("${PY}" -c "import json; [print(x['config']) for x in json.load(open('${MANIFEST}'))]")

echo "[$(date -Is)] n_configs=$("${PY}" -c "import json; print(len(json.load(open('${MANIFEST}'))))")" | tee -a "${STATUS}"
"${CMD[@]}"
SHORT_DIR=$(ls -dt "${EXP}/results/nts11bd_v2_short_"* 2>/dev/null | head -1)
cp -f "${SHORT_DIR}/summary.csv" "${DRIVER}/summary.csv" 2>/dev/null || true
cp -f "${SHORT_DIR}/summary.md" "${DRIVER}/summary.md" 2>/dev/null || true
echo "[$(date -Is)] done short_dir=${SHORT_DIR}" | tee -a "${STATUS}"
echo "=== 11bd-v2 short screen complete $(date -Is) ==="