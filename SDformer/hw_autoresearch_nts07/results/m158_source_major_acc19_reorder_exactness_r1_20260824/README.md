# M158 source-major Acc19 重排精确性证明

## 结论

M157 的 source-major/row-interleave 不改变冻结 H67 ep35 的整数 Conv 结果，并且在 exact-SHA 载荷域内不需要 M155 的逐 lane runtime overflow detector。

## 完整 signed tuple 证据

- 20 条 heldout、4 个 Conv3x3 operator。
- 完整 source key：414,720,000。
- active source key：23,522,595，与 M157 一致。
- signed event：188,148,490，其中正 173,840,364，负 14,308,126。
- negative-not-event、event half reconstruction、negative half reconstruction 全部 0 mismatch。

这不只是事件总数相等：对每个 `(record, partition, raw row, source)` 的 8-bit destination mask 与符号都做了互斥 low/high half 分解和重构。

## Acc19 全前缀界

四个冻结 INT8 Conv 权重文件按 `I_KY_KX_O_C_ORDER` 独立重算，每输出通道 6,912 个权重的 `sum(abs(weight))` 最大值分别为：

- 218,338
- 204,866
- 207,239
- 190,753

signed19 范围为 `[-262144, 262143]`，最坏正向余量为 43,805。任何事件子集与符号重排的任意前缀，其绝对值都不可能超过对应通道的 `sum(abs(weight))`。所以不会发生 Acc19 溢出，整数加法可安全交换/结合。

## 硬件含义

- 冻结负载下，可以从最终 fused accumulator 删除 384 lane 的溢出比较和全局 OR tree，解决 M155 的 50-level 时序热点之一。
- 这不是通用安全性豁免。非 exact-SHA checkpoint、非 INT8 权重、非 Acc19 或不可信外部输入必须 fail closed，或恢复 dynamic guard。
- 尚缺 fused cache/accumulator RTL 和真实小样逐拍 miter，因此没有接纳 PPA 或 speedup。

当前等级：`PASS_FROZEN_SIGNED_TUPLE_AND_ACC19_REORDER_PROOF`。`physical_speedup=false`，`system_speedup=false`，`headline=false`。
