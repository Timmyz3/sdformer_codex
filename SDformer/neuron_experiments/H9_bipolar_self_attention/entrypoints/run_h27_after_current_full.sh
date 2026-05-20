#!/usr/bin/env bash
set -u

cd /root/private_data/work/sdformer_codex/SDformer

PY=/opt/conda/envs/sdformerflow/bin/python
WAIT_PID=${WAIT_PID:-654045}

echo "[$(date -Is)] 等待当前 H24/H25/H26/auto-full 队列 pid=${WAIT_PID}"
while ps -p "${WAIT_PID}" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date -Is)] 启动 H27 标准 BSA 范式短测"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py \
  --tag rapid_screen_h27_strict_bsa_standard \
  --config h27a_strict_bsa_signv_sqrt_sparse040_guard120.yml \
  --config h27b_strict_bsa_thetav_sqrt_sparse040_guard120.yml \
  --config h27c_strict_bsa_signv_head_sparse040_guard120.yml \
  --config h27d_strict_bsa_thetav_head_sparse040_guard120.yml \
  --config h27e_strict_bsa_signv_active_sparse040_guard120.yml \
  --config h27f_strict_bsa_signv_sqrt_sparse035_guard120.yml \
  --steps 120 \
  --valid-samples 10 --promote-samples 40 \
  --promote-aee 1.70 --promote-aae 8.5 --promote-sops-g 3.90 \
  --batch-size 8 --workers 8 --amp
H27_EXIT=$?
echo "[$(date -Is)] H27 exit_code=${H27_EXIT}"

echo "[$(date -Is)] H27 后再次自动选择候选做全量"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py \
  --tag rapid_screen_h23_low_lr_sparse_combo \
  --tag rapid_screen_h24_h9ascope_axnor_hparam \
  --tag rapid_screen_h24_remaining_after_cleanup \
  --tag rapid_screen_h25_module_combinations \
  --tag rapid_screen_h26_attention_revisit \
  --tag rapid_screen_h27_strict_bsa_standard \
  --batch-size 8 --workers 8 --epochs 30 --profile-samples 40
PROMOTE_EXIT=$?
echo "[$(date -Is)] H27 promote/full exit_code=${PROMOTE_EXIT}"
exit "${PROMOTE_EXIT}"
