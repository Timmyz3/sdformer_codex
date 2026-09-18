#!/bin/bash
# T55a: η 预算扫描（固定 1 epoch，只扫 η）—— 描出「SOPS 预算 → 精度」帕累托曲线。
#
# 为什么固定 1 epoch 只扫 η：
#   Δθ/步 = η · mass · 0.15，而 T52 实测 mass 在工作点附近近似常数
#   （effective_update_mean 全 epoch 只从 2.765e-5 掉到 2.687e-5，−2.8%）
#   ⇒ θ_final ≈ 1 + η · mass · 0.15 · steps 是 η 与 steps 乘积的线性函数
#   ⇒ 「扫 η」与「扫 epoch 预算」描的是同一条曲线，但 1 epoch 便宜 2–3 倍。
#
# 对照口径为零成本：θ 冻结臂 t52_eta_off（1 epoch，θ≡1，AEE 1.19565 / FR 5.6986%）
# 已经跑过；因为 η=0 时 θ 不动，该控制点与 η 无关 ⇒ 本扫描只需补 η>0 的臂。
#
# 已有：θ=1.0 无训练（锚点 1.19951 / 5.6709%）、θ=1.0 1 epoch（eta_off）、
#       θ≈1.095（t52_eta_on, η=0.008）。本脚本补 θ≈1.05 与 θ≈1.19 两点。
#
# 用法（远端 /root 下）：bash /root/t55_eta_budget.sh
set -u
PY=/opt/conda/envs/sdformerflow/bin/python
LR=2.5e-5
echo "=== T55a eta sweep start $(date -Is) ==="
for spec in "004:0.004" "016:0.016"; do
    tag=${spec%%:*}; ETA=${spec##*:}
    NAME="t55_eta${tag}"
    echo "--- ${NAME} threshold_eta=${ETA} $(date -Is) ---"
    "${PY}" -u /root/t49_ternary_retrain.py \
        --name "${NAME}" --epochs 1 --binary-official \
        --backbone-lr "${LR}" --threshold-eta "${ETA}"
    echo "--- ${NAME} rc=$? ---"
    R=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/${NAME}
    tr '\r' '\n' < "$R/train.log" 2>/dev/null | grep -E '\[H9\] step .* update:' | tail -1
done
echo "=== train done, evaluating $(date -Is) ==="
"${PY}" -u /root/t52_eval_runs.py t55_eta004 t55_eta016 --epoch 35
echo "=== T55a COMPLETE $(date -Is) ==="
