#!/usr/bin/env bash
set -u

cd /root/private_data/work/sdformer_codex/SDformer

PY=/opt/conda/envs/sdformerflow/bin/python

echo "[$(date -Is)] 满盘清理后断点续跑：H24 剩余 + H25 + H26 + 自动全量"

echo "[$(date -Is)] 继续 H24d-H24g"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py \
  --tag rapid_screen_h24_remaining_after_cleanup \
  --config h24d_h9ascope_axnor_sparse035_guard120.yml \
  --config h24e_h9ascope_axnor_ang002_guard120.yml \
  --config h24f_h9ascope_axnor_ang005_guard120.yml \
  --config h24g_h9ascope_axnor_flowreg0003_guard120.yml \
  --steps 120 \
  --valid-samples 10 --promote-samples 40 \
  --promote-aee 1.70 --promote-aae 8.5 --promote-sops-g 3.90 \
  --batch-size 8 --workers 8 --amp
H24_REMAIN_EXIT=$?
echo "[$(date -Is)] H24 remaining exit_code=${H24_REMAIN_EXIT}"

echo "[$(date -Is)] 启动 H25 模块组合"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py \
  --tag rapid_screen_h25_module_combinations \
  --config h25a_ffn_sn1_only_binary_guard120.yml \
  --config h25b_ffn_sn2_only_binary_guard120.yml \
  --config h25c_ffn_sn1_ternary_sn2_binary_guard120.yml \
  --config h25d_ffn_sn1_binary_sn2_ternary_guard120.yml \
  --config h25e_no_ffn_downsample_only_guard120.yml \
  --config h25f_ffn_all_ternary_guard120.yml \
  --config h25g_ffn_all_binary_no_downsample_guard120.yml \
  --steps 120 \
  --valid-samples 10 --promote-samples 40 \
  --promote-aee 1.70 --promote-aae 8.5 --promote-sops-g 3.90 \
  --batch-size 8 --workers 8 --amp
H25_EXIT=$?
echo "[$(date -Is)] H25 exit_code=${H25_EXIT}"

echo "[$(date -Is)] 启动 H26 降级注意力回收"
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
echo "[$(date -Is)] H26 exit_code=${H26_EXIT}"

echo "[$(date -Is)] 开始自动选择一个候选做全量训练"
"${PY}" -u neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py \
  --tag rapid_screen_h23_low_lr_sparse_combo \
  --tag rapid_screen_h24_h9ascope_axnor_hparam \
  --tag rapid_screen_h24_remaining_after_cleanup \
  --tag rapid_screen_h25_module_combinations \
  --tag rapid_screen_h26_attention_revisit \
  --batch-size 8 --workers 8 --epochs 30 --profile-samples 40
PROMOTE_EXIT=$?
echo "[$(date -Is)] promote/full exit_code=${PROMOTE_EXIT}"
exit "${PROMOTE_EXIT}"
