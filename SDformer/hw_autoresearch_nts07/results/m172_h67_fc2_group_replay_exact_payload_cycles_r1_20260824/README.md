# M172 H67 FC2 group-held replay exact-payload cycles r1

M172 将 M171 的 64-bit bitmap scanner、one-beat raw prefetch、bank-unique
K1/K4 分组、`Cout/96` group replay 和 token-done 控制延迟逐 token 映射到冻结
M51 的 120 个 FC2 payload。全部 437,760,000 payload bytes 的 SHA、大小和
popcount 均重新检查。

## 结果

| 项 | K1 | K4 |
|---|---:|---:|
| group replay cycles | 412,900,394 | 144,999,276 |
| control + unhidden scan | 33,628,230 | 34,058,679 |
| frontend wall cycles | 446,528,624 | 179,057,955 |

K1/K4 group-replay 比为 **2.847603x**，完整 standalone frontend wall-cycle
比为 **2.493766x**。它们都不是 RTL measured、physical、FC2、FFN 或 system
speedup。

分 stage wall-cycle ratio 为：stage0 **1.761925x**、stage1 **2.425725x**、
stage2 **2.707184x**、stage3 **2.853771x**。stage0 的单 output block 无法隐藏
64-bit scan，是最严瓶颈。

## 硬件结论

M168 允许跨整个 token 的独立 bank queues，K4 replay 下界是 106,536,803
cycles；M171 只在单个 64-bit beat 内分组，变成 144,999,276 cycles，即
**1.361033x fragmentation**。因此 M171 证明了 group hold/replay 和零 token
协议，但 64-bit one-beat scanner 不应作为最终性能结构。

下一版应把 bank grouping 前移到 ATLIF 的原生 96-lane event producer，或加入
跨 beat bank reservoir；这样避免 FC2 输入落 SRAM 后再扫一遍，并恢复跨 beat
bank pairing。

## 严格边界

周期模型假设 group 与 token-done consumer always-ready；没有 weight SRAM
request/response、M169 arithmetic、2304-bit accumulator context、BN2 或 residual。
M171 r1 DC 因 103 logic levels 超过预锁定 80-level 上限而 fail-closed，不能作为
通过的物理证据。

分析器内部将向量 beat-event recurrence 与独立 edge simulator 比较 9,600 个
随机/结构 case，0 mismatch。`docs/359` 未修改。
