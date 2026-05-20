#!/usr/bin/env bash
set -u

cd /root/private_data/work/sdformer_codex/SDformer

PY=/opt/conda/envs/sdformerflow/bin/python
WAIT_PID=${WAIT_PID:-619399}

echo "[$(date -Is)] 等待 H23/H24/H25 主短测队列 pid=${WAIT_PID}"
while ps -p "${WAIT_PID}" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date -Is)] H23/H24/H25 已结束，启动 H26 降级注意力回收短测"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py \
  --tag rapid_screen_h26_attention_revisit \
  --config h26a_axnor_l1_sparse040_guard120.yml \
  --config h26b_a2os2a_sparse040_guard120.yml \
  --config h26c_hamming_ternary_sparse040_guard120.yml \
  --config h26d_hamming_binary_sparse040_guard120.yml \
  --config h26e_axnor_shiftmax_signv_sparse040_guard120.yml \
  --config h26f_axnor_l1_ffn_ternary_guard120.yml \
  --config h26g_a2os2a_ffn_sn1_ternary_guard120.yml \
  --config h26h_hamming_ternary_sparse035_guard120.yml \
  --config h26i_axnor_l1_flowreg0003_guard120.yml \
  --steps 120 \
  --valid-samples 10 --promote-samples 40 \
  --promote-aee 1.70 --promote-aae 8.5 --promote-sops-g 3.90 \
  --batch-size 8 --workers 8 --amp
H26_EXIT=$?
echo "[$(date -Is)] H26 rapid_screen exit_code=${H26_EXIT}"

echo "[$(date -Is)] 无论 H26 是否有坏候选，都进入自动全量选择"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py \
  --tag rapid_screen_h23_low_lr_sparse_combo \
  --tag rapid_screen_h24_h9ascope_axnor_hparam \
  --tag rapid_screen_h25_module_combinations \
  --tag rapid_screen_h26_attention_revisit \
  --batch-size 8 --workers 8 --epochs 30 --profile-samples 40
PROMOTE_EXIT=$?
echo "[$(date -Is)] promote/full exit_code=${PROMOTE_EXIT}"
exit "${PROMOTE_EXIT}"
