# M372：M361r4 宽分区独立打铁

结论：**94/100，P0/P1/P2 = 0/0/4。关闭 k32/k64 宽分区 active line，继续 k16 q32/O4 executable scheduler，判断合理。** “关闭”是当前资源优先级，不是宽分区在 cycle/area/energy 上被永久形式支配。

M372 独立汇总 M361r4 的全部 1,296 个 operator/partition catalog record，逐 partition 检查 q16/32/64/128 nested prefix、center 编码和 train/disjoint candidate work 单调性，再用独立 baseline 重新除法：

| k | q | train work | disjoint S10 work | signed PWP | full PWP capacity |
|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 1.371811x | 1.374535x | 13 bit | 17,252,352 B |
| 32 | 32 | 1.469137x | 1.476186x | 13 bit | 34,504,704 B |
| 32 | 64 | 1.579207x | 1.592538x | 13 bit | 69,009,408 B |
| 32 | 128 | 1.696467x | 1.714495x | 13 bit | 138,018,816 B |
| 64 | 16 | 1.225253x | 1.226243x | 14 bit | 9,289,728 B |
| 64 | 32 | 1.284563x | 1.287258x | 14 bit | 18,579,456 B |
| 64 | 64 | 1.347339x | 1.350725x | 14 bit | 37,158,912 B |
| 64 | 128 | 1.406052x | 1.406600x | 14 bit | 74,317,824 B |

M339 被 M372 单独冻结并重除：k16/q128 为 `67,844,260 / 33,192,878 = 2.043940269x`。k32/q128 的 candidate work 比它多 19.22%，k64/q128 多 45.31%。在每个 matched q，k32 与 k64 的 work 也都差于 k16。

signed PWP 位宽独立按 INT8 累加范围 `[-128k,127k]` 推导：k16/k32/k64 分别需要 12/13/14 bit；96-output vector 分别为 144/156/168 B。pattern/full-PWP capacity 公式也与八行冻结结果逐项一致。

k16 q32/O4 是合理的下一实现靶点：disjoint exact work 为 1.692877x；每 context 24,640 B，双 context 49,280 B，每个 32 KiB context 尚余 8,128 B。它比继续开宽分区更成熟，但 finite-queue executable scheduler 仍未完成，不能把 capacity fit 或 work ratio 写成 cycle speedup。

四个 P2：原 contract 未冻结 M339 比较源；本次从冻结 catalog records 汇总而未重新拟合 raw-trace centers；关闭宽分区不是物理支配证明；q32/O4 scheduler 仍只是下一实现目标。全部数字只属 exact vector work/capacity，无 cycle、energy、system 或 headline 准入。
