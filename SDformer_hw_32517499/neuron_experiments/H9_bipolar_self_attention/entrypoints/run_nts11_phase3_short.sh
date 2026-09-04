#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
GEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts11_phase3_configs.py"
SCREEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py"
CKPT="${ROOT}/experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
CFG_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/configs/generated"

"${PY}" "${GEN}"
"${PY}" -m py_compile "${GEN}"

cd "${ROOT}"
"${PY}" -u "${SCREEN}" \
  --config "${CFG_DIR}/nts11n_hw_h60_s23_vdec_fastlr_s1224.yml" \
  --config "${CFG_DIR}/nts11o_hw_h60_s23_vdec_fastlr_warm720_s1224.yml" \
  --config "${CFG_DIR}/nts11p_hw_h60_s23_vdec_fastlr_warm720_freeze816_s1224.yml" \
  --config "${CFG_DIR}/nts11j_hw_h60_s23_two_neuron_vanilla_decoder_s1224.yml" \
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
  --tag nts11_phase3