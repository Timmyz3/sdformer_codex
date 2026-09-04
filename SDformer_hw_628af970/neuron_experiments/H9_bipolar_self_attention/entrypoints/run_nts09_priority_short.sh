#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
GEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts09_threshold_freeze_hw_configs.py"
SCREEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py"
CKPT="${ROOT}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
CFG_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/configs/generated"

"${PY}" "${GEN}"

# Priority-ordered NTS09 short screen for the current DATE-facing mainline:
# 1) freeze816
# 2) cap115 + freeze816
# 3) freeze918
# 4) eta0325 + freeze816
#
# Compared with the broader nts09_thresh_freeze sweep, this launcher uses a
# tighter promotion gate so only runs that stay close to the current NTS07b/NTS08
# quality band are treated as full-training candidates.
"${PY}" "${SCREEN}" \
  --config "${CFG_DIR}/nts09a_hw_h60_freeze816_s1224.yml" \
  --config "${CFG_DIR}/nts09d_hw_h60_cap115_freeze816_s1224.yml" \
  --config "${CFG_DIR}/nts09b_hw_h60_freeze918_s1224.yml" \
  --config "${CFG_DIR}/nts09c_hw_h60_eta0325_freeze816_s1224.yml" \
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
  --tag nts09_priority
