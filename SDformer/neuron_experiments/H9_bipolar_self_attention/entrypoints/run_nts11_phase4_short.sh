#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
PY="/opt/conda/envs/sdformerflow/bin/python"
GEN="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts11_phase4_scope_configs.py"
VERIFY="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_nts11_chain.py"
AUTOPILOT="${ROOT}/neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nts11_scope_autopilot.py"
CFG_DIR="${ROOT}/neuron_experiments/H9_bipolar_self_attention/configs/generated"

"${PY}" "${GEN}"
"${PY}" -m py_compile "${GEN}"
"${PY}" -m py_compile "${AUTOPILOT}"
"${PY}" "${VERIFY}" "${CFG_DIR}/nts11r_hw_h60_s23_scope_sn2q_binary_s1224.yml"

nohup "${PY}" -u "${AUTOPILOT}" \
  > "${ROOT}/neuron_experiments/H9_bipolar_self_attention/results/nts11_scope_autopilot_launcher_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "NTS-11 scope autopilot launched in background."