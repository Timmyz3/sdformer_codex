#!/bin/bash
# T52 自适应阈值正式长跑 + 评测（等标定定出 eta* 与 LR 扫描定出 lr* 之后再跑）。
#
# 为什么要跑 **两个** arm（而不是只跑 eta-on）：
#   只跑 eta-on 再和已发布的 ep34 锚点比，会把「多训 5 个 epoch」和
#   「自适应阈值」两件事混在一起，精度差无法归因。加上同预算的 eta-off 之后：
#     锚点 vs eta_off  = 多训 5 epoch 本身的代价（与自适应无关）
#     eta_off vs eta_on = 自适应阈值的净代价/收益（同预算、同起点、只差一个键）
#   这才是能支撑「自适应阈值省 SOPS」的对照。
#
# 用法（远端 /root 下）：
#   bash /root/t52_adaptive_run.sh <eta> <lr> [epochs]
# 例：bash /root/t52_adaptive_run.sh 0.05 2.5e-5 5
# 注意：预算必须在标定之后定 —— θ 每步位移 = eta * mass * 0.15，
#      预算要长到 θ 能真正走完想走的距离，否则自适应"来不及"表现。
set -u
PY=/opt/conda/envs/sdformerflow/bin/python
ETA="${1:?eta required}"
LR="${2:?lr required}"
EPOCHS="${3:-5}"
EVAL_EPOCH=$((34 + EPOCHS))          # EPOCH_OFFSET=35 ⇒ raw epoch0 落盘为 35

echo "=== T52 adaptive run: eta=${ETA} lr=${LR} epochs=${EPOCHS} $(date -Is) ==="

for spec in "off:0.0" "on:${ETA}"; do
    tag=${spec%%:*}; TETA=${spec##*:}
    echo "--- arm ${tag} threshold_eta=${TETA} $(date -Is) ---"
    "${PY}" -u /root/t49_ternary_retrain.py \
        --name "t52_eta_${tag}" --epochs "${EPOCHS}" --binary-official \
        --backbone-lr "${LR}" --threshold-eta "${TETA}"
    rc=$?
    echo "--- arm ${tag} rc=${rc} ---"
    # θ 到底动没动：last 两条 update 行里的 threshold_mean / effective_update_mean
    R=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/t52_eta_${tag}
    tr '\r' '\n' < "$R/train.log" 2>/dev/null | grep -E '\[H9\] step .* update:' | tail -2
done

echo "=== train done, evaluating $(date -Is) ==="
"${PY}" -u /root/t52_eval_runs.py t52_eta_off t52_eta_on --epoch "${EVAL_EPOCH}"
echo "=== T52 ADAPTIVE COMPLETE $(date -Is) ==="
