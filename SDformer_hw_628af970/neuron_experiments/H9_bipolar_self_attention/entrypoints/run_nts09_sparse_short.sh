#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow
PY="/opt/conda/envs/sdformerflow/bin/python"
GEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts09_threshold_freeze_hw_configs.py"
SCREEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py"
CKPT="${ROOT}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
CFG_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/configs/generated"

"${PY}" "${GEN}"

# Sparse-biased NTS09 sweep: later freeze steps push qk threshold higher before
# locking, targeting >=15% total_spikes drop vs NB0 at full30 valid825.
# Priority: mild late-freeze first, then multi-epoch freeze, then cap combo.
"${PY}" "${SCREEN}" \
  --config "${CFG_DIR}/nts09e_hw_h60_freeze1224_s1224.yml" \
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
  --tag nts09_sparse