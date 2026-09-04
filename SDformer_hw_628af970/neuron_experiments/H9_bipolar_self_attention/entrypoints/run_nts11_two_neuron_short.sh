#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
GEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts11_two_neuron_only_configs.py"
VERIFY="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_nts11_chain.py"
SCREEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py"
CKPT="${ROOT}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
CFG_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/configs/generated"

"${PY}" "${GEN}"
"${PY}" -m py_compile "${GEN}"
"${PY}" -m py_compile "${VERIFY}"

"${PY}" "${VERIFY}" "${CFG_DIR}/nts11b_hw_h60_s23_two_neuron_freeze1224_s1224.yml"

# Priority short screen on nts11b mainline + LR/threshold variants.
"${PY}" -u "${SCREEN}" \
  --config "${CFG_DIR}/nts11b_hw_h60_s23_two_neuron_freeze1224_s1224.yml" \
  --config "${CFG_DIR}/nts11c_hw_h60_s23_two_neuron_fastlr_s1224.yml" \
  --config "${CFG_DIR}/nts11d_hw_h60_s23_two_neuron_slowlr_s1224.yml" \
  --config "${CFG_DIR}/nts11e_hw_h60_s23_two_neuron_qkscale25k_s1224.yml" \
  --config "${CFG_DIR}/nts11f_hw_h60_s23_two_neuron_freeze816_s1224.yml" \
  --config "${CFG_DIR}/nts11g_hw_h60_s23_two_neuron_eta0325_s1224.yml" \
  --config "${CFG_DIR}/nts11a_hw_h60_s2_two_neuron_freeze1224_s1224.yml" \
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
  --tag nts11_two_neuron