#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/work/sdformer_codex/SDformer
EXP=neuron_experiments/H9_bipolar_self_attention
PY=${PYTHON_BIN:-/opt/conda/envs/sdformerflow/bin/python}
CANDIDATE=${1:-h66c_allbinary_all12_tp_ttx_s120}
STEPS=${2:-120}
STAMP=$(date +%Y%m%d_%H%M%S)
TTX_CKPT="$ROOT/$EXP/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth"
LOG="$ROOT/$EXP/results/${CANDIDATE}_launcher_${STAMP}.log"

cd "$ROOT"
export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" "$EXP/entrypoints/make_h66_accuracy_first_unified_configs.py"
exec "$PY" -u "$EXP/entrypoints/rapid_screen.py" \
  --config "generated/${CANDIDATE}.yml" \
  --steps "$STEPS" \
  --prev-runid "$TTX_CKPT" \
  --valid-samples 10 \
  --promote-samples 40 \
  --promote-aee 1.65 \
  --promote-aae 20 \
  --promote-sops-g 3.35 \
  --confirm-steps 360 \
  --tag "${CANDIDATE}" 2>&1 | tee "$LOG"
