#!/bin/bash
# T51 接力：等 LR 扫描结束 → 跑**同预算** 2.5e-5 对照探针 → 四个 arm 全做 valid825 评测。
#
# 为什么要这个对照：主扫描三档（1e-4/3e-4/1e-3）互为同预算可比，但报告里原先拿
# 3672 步的 val 8.4124 当基线是**跨预算**比较，无效。补一个 600 步 @2.5e-5 才能直接
# 证明"2.5e-5 在这个预算下下降更慢"。
#
# 为什么要自动接力：单卡，评测必须等扫描释放 GPU；用远端轮询避免本地长时间挂起。
PY=/opt/conda/envs/sdformerflow/bin/python
LOG=/root/t51_lr_sweep.log

while ! grep -q 'SWEEP COMPLETE' "$LOG" 2>/dev/null; do sleep 30; done
echo "=== SWEEP DONE, start 2.5e-5 control probe $(date -Is) ==="

"${PY}" -u /root/t49_ternary_retrain.py \
    --name t51_lrprobe_2p5e5 --epochs 1 --max-steps 600 --backbone-lr 2.5e-5
echo "=== CONTROL DONE rc=$? $(date -Is) ==="

"${PY}" -u /root/t51_eval_lrprobe.py 1e4 3e4 1e3 2p5e5
echo "=== CHAIN COMPLETE $(date -Is) ==="
