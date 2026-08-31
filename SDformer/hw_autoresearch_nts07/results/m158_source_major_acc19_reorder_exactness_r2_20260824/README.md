# M158 r2 source-major Acc19 重排精确性证明

## r1 纠正

r1 误把 `m132.build_record_rows` 的 PWP row ledger 当成 negative mask，并错误广播 destination sign。r1 的 14,308,126 个负事件以及由它导出的 tuple 证明全部撤销；纠正 overlay 已 fail closed 封存，未覆盖旧证据。

## r2 完整 signed tuple 证据

- 直接复用并固定 SHA 的 M150 独立审计 `select_events` 作为 signed-event oracle。
- H67/Motion ep35 heldout 20 条记录、4 个 Conv3x3 operator。
- 完整 source key：414,720,000；active source key：23,522,595。
- signed event：188,148,490，其中正 170,591,133，负 17,557,357。
- negative-not-event、event low/high half reconstruction、negative low/high half reconstruction 全部 0 mismatch。

每个 `(record, partition, raw row, source, destination)` 的 event/sign 位均独立重构，没有跨 destination 广播。

## Acc19 前缀界

四个冻结 INT8 Conv 权重文件按 `I_KY_KX_O_C_ORDER` 独立重算，每输出通道 6,912 个权重的 `sum(abs(weight))` 最大值分别为 218,338、204,866、207,239 和 190,753。

signed19 范围为 `[-262144, 262143]`，最坏正向余量 43,805。只在 exact-SHA checkpoint、INT8 权重、每个 feature 每输出至多贡献一次的冻结 Conv 数值域内，任何事件子集/符号重排的前缀绝对值不超过 `sum(abs(weight))`，因此不会溢出。

## 接纳边界

- 允许继续设计不带 384-lane runtime overflow OR tree 的冻结域 fused accumulator。
- 不可信输入、不同 checkpoint/量化或不同累加语义仍须 fail closed 或恢复动态保护。
- 仍缺 RTL trace miter、cache/accumulator 集成、macro PPA 和实测周期，故 `physical_speedup=false`、`system_speedup=false`、`headline=false`。
- r2 需独立打铁评审后才可进入下一里程碑。
