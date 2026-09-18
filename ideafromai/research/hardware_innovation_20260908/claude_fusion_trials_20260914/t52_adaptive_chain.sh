#!/bin/bash
# T52 自适应阈值标定：等 T51 评测链跑完 → 跑一个 60 步探针，把 θ 每步实际位移量测出来。
#
# 为什么需要标定：官方 ATLIF 的质量型自适应是 Δθ = threshold_eta * mass * threshold_base_lr
# * threshold_lr_scale，其中 mass = E[Σ_batch zif·active]（official 模式还要 ÷T）。
# 源配置 threshold_eta=0 ⇒ update_value 恒为 0（探针实测 update_mean: 0.0），所以
# mass 从来没有被测出来过，η 没有量纲可依 —— 只能实测，不能拍脑袋。
#
# 用 binary + official_atlif 跑：这样对照就是**已发布的 ep34 锚点**
# （AEE 1.19951 / FR 5.6709%），是单变量消融（只动 threshold_eta 一个键）。
# 用 threshold_eta=0.1 而不是 1.0：让 60 步内 θ 的漂移可忽略，
# 测到的 mass 才代表锚点工作点上的 mass。mass_unit = raw_update_mean / 0.1。
PY=/opt/conda/envs/sdformerflow/bin/python
LOG=/root/t51_lr_chain.log

while ! grep -q 'CHAIN COMPLETE' "$LOG" 2>/dev/null; do sleep 30; done
echo "=== CHAIN DONE, start adaptive calibration $(date -Is) ==="

"${PY}" -u /root/t49_ternary_retrain.py \
    --name t52_eta_calib --epochs 1 --max-steps 60 \
    --binary-official --threshold-eta 0.1
echo "=== CALIB rc=$? $(date -Is) ==="

R=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/t52_eta_calib
tr '\r' '\n' < "$R/train.log" | grep -E '\[H9\] step .* update:' | tail -3
echo "--- threshold/rate summary ---"
tr '\r' '\n' < "$R/train.log" | grep -E 'num_modules|ternary_zero|thresh|update' | tail -8
echo "=== CALIB DONE $(date -Is) ==="
