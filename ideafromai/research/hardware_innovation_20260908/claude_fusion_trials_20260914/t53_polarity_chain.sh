#!/bin/bash
# T53 接力：等 T52 自适应长跑 + 评测全部结束 → 跑 h60 极性零样本矩阵。
#
# 为什么要接力而不是马上起：GPU 是串行的，T52 的两个 arm 各 1 epoch
# （3672 步 × ~1.4 s/步 ≈ 86 min）外加两次 valid825 评测，还要 ~3 h。
# 接力避免 GPU 空转。
#
# 用法（远端 /root 下）：
#   setsid nohup bash /root/t53_polarity_chain.sh > /root/t53_chain.log 2>&1 < /dev/null &
set -u
PY=/opt/conda/envs/sdformerflow/bin/python
ADAPT_LOG=/root/t52_adaptive_run.log

echo "=== T53 chain armed $(date -Is) ==="
while ! grep -q 'T52 ADAPTIVE COMPLETE' "${ADAPT_LOG}" 2>/dev/null; do
    sleep 60
done
echo "=== T52 complete, GPU 交棒给 T53 $(date -Is) ==="

"${PY}" -u /root/t53_polarity_zeroshot.py
echo "=== T53 rc=$? $(date -Is) ==="
echo "=== T53 COMPLETE $(date -Is) ==="
