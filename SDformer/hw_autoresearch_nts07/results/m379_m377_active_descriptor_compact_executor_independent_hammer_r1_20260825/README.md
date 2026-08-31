# M379：M377 active-descriptor compact executor 独立打铁

结论：**92/100，P0/P1/P2 = 0/0/5；允许进入 bounded VCS active-descriptor controller，不准入完整 q32 accelerator、系统速度或论文 PPA。**

M379 fresh exact-SHA replay 与 M377 原 result、candidate CSV、baseline CSV 逐字节一致，原双层 seal 通过。独立按事件端点重算 baseline 543,784,143、candidate 503,016,392 cycles，模块速度比 1.081046566×，节省 40,767,751 cycles（相对 baseline 减 7.497%），超过冻结 1.05× controller gate。

raw trace 重新解包得到严格恒等式 `51,840,000 = 30,368,111 zero + 21,471,889 active`。compact write、tile0 replay、tile1 replay 都逐 phase 守恒为 21,471,889 行；descriptor 每行 6 B，最大 partition 2,400 active，即 14,400 B，低于 18,000 B bank。每 phase active 最少 89，因此冻结 S10 没有 empty partition。

精确删除边界只允许 `original16==0`：此时 `W×0=0`，M339 的严格 PWP 条件也不会选择 PWP。M379 额外统计到 6,762,595 个 popcount=1 行；由于 `1+distance<1` 永不成立，它们全部是非零 bit-sparse fallback，并全部保留在 active stream，误删为 0。

matcher 仍为全部 3,000 行收费：17,280 个 matcher event 最短 3,044、最长 4,991 拍，总计 66,583,854 cycles。每个 phase 的 active write 数都不超过 matcher event 时长。事件守恒为 120,970 candidate / 69,130 baseline；单 DMA overlap、顺序、duration、32 B DMA alignment、slot、descriptor capacity 和 `replay_duration >= 1+4×active` 违规均为 0。34,560 个 tile startup 已显式计入。

真实数据没有 empty partition，因此 M379 定向调用该分支：保留 pattern DMA、3,002-cycle 全行 matcher、tail 与 common commit，descriptor bytes 为 0，并抑制 tile DMA 和 descriptor replay；语义符合合同，但仍需 VCS 覆盖 stall/reset/bank reuse。

五个 P2 都是 controller 验证义务：rowID/48-bit payload 尚未落成逐条 trace；one-write/cycle 没有 row-level retirement timestamp；FIFO occupancy 仍由服务率证明而非逐拍状态；empty branch 没有真实样本；1.081047× 只属于四个 bottleneck Conv 的 S10 模块 schedule。下一里程碑应做 controller RTL + VCS/SVA cycle miter，不能直接跳到完整 matcher DC、system speedup 或 DATE headline。
