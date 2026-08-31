# M179 dual-window reservoir DSE 独立打铁评审 r1

结论：**85/100，`PASS_EXACT_PAYLOAD_DSE_REVISE_BASELINE_AND_IMPLEMENTATION_BEFORE_ADMISSION`，P0/P1/P2 = 4/5/3。**

M179 的冻结 payload 计数、双 ping-pong window recurrence、D1 对 M176 的复现和
全部 D/K 总账都可信。它把 M176/M177 的单 descriptor group fragmentation 从
`144,146,504` K4 wall cycles 压到 stage 自适应点的 `127,581,198`，即额外
`1.129841280x` analytic frontend opportunity。当前还没有 producer、directory、window
storage、cross-entry selector、weight response、arithmetic 或 accumulator context，因此
它不是 RTL、物理、完整 FC2 或系统加速。

## 独立 exact-payload 复算

评审脚本没有导入 M172/M176/M179 analyzer，直接以
`numpy.unpackbits(bitorder="little")` 重解 120 个 FC2 payload。全部
**437,760,000 bytes** 的逐文件 SHA、大小和 popcount 均重新检查：

| 项 | 独立结果 |
|---|---:|
| FC2 records / tokens | 120 / 5,580,000 |
| events | 143,894,510 |
| raw96 / nonzero96 descriptors | 36,480,000 / 18,869,376 |
| zero tokens | 1,863,944 |
| SHA/size/popcount mismatch | 0 |
| scalar/vector recurrence cases | 31,824 |
| D1/explicit-EOT random cases | 5,304 |
| recurrence / aggregate integer mismatch | 0 / 0 |

此外，真实 payload 的每个 token 都在 K1、K4 两档逐项比较了 D1 双窗与独立显式 EOT
recurrence，0 mismatch。D1 严格复现 M176：K1/K4 分别为
`424,060,394 / 144,146,504`。

## D={1,2,4,8,16,32} 全量结果

| D | K1 wall | K4 wall |
|---:|---:|---:|
| 1 | 424,060,394 | 144,146,504 |
| 2 | 427,317,449 | 132,291,199 |
| 4 | 432,614,027 | 130,275,370 |
| 8 | 435,912,546 | 130,711,219 |
| 16 | 438,694,207 | 132,461,776 |
| 32 | 439,213,714 | 132,850,123 |

recurrence 的 fill/drain/reuse 语义成立：source 每拍按序填一个 descriptor；window 填满或
到 token 尾才关闭；drain 每个 output block 一拍产生一个 group result；另一 buffer 可在
drain 时填充；同一物理 buffer 必须等自己的前次 drain 结束才能复用。zero token 计 2 拍。

这仍然隐含一个免费、提前可用的 descriptor-count directory。D1 数字虽然与显式 EOT
相同，却没有实现或计入 directory 的生成、存储与读取，不能把数值等价解释成硬件已经
闭环。

## Stage 选择：数值正确，但不能称无选择偏置

`D={2,4,8,8}` 确实分别是同一批 120 payload 上四个 stage 的唯一 K4 最小点；聚合
K1/K4 为 `430,917,270 / 127,581,198 = 3.377592284x`，D1 K4/selected K4 为
`144,146,504 / 127,581,198 = 1.129841280x`。

但这里有两项必须修正口径：

1. 深度是在同一评测 population 上选出并在其上报告，因此是 in-sample oracle，不是
   holdout-validated point。四个 stage 对 runner-up 的优势只有
   `0.3543% / 0.5869% / 0.2443% / 0.0520%`，缺失的 directory/control/physical cost
   足以改变小差距。
2. `3.377592x` 把 K1 也强制走 K4 选出的 `{2,4,8,8}` window，令 K1 比自身最优 D1
   慢 `1.616957%`。它可保留为 **same-window ablation**，但 independently optimized
   analytic baseline 应用 `424,060,394 / 127,581,198 = 3.323847092x`。

因此安全数字是：cross-entry pooling 相对 M176 D1 K4 的机会为 `1.129841x`；K1/K4
优化基线 analytic ratio 为至多 `3.323847x`。两者都尚非物理 speedup。

## 硬件 P0 边界

P0：

1. 实现或 materialize ATLIF-native/preindexed producer 与 token descriptor-count
   directory；逐 token miter 18,869,376 个 descriptor，并计 directory 生成/lookup 和
   zero-token release。
2. 实现 D8-capable 双 window 与 constructive cross-entry four-source selector，跑
   exact-SHA VCS/SVA 和 3 ns Synopsys DC，覆盖 backpressure、buffer reuse 及全部守恒。
3. 接 M169 arithmetic/accumulator，针对跨 descriptor 重排做冻结 payload 数值 miter，
   证明无 overflow、saturation 或最终输出 mismatch。
4. 接四 bank weight response、accumulator context 与 commit/backpressure；完成 matched
   module cycle/PPA 后才允许称 complete-FC2 或 physical speedup。

P1 是修正 K1 baseline 命名、用独立 tuning/heldout 做 stage 选择、规定 directory/window
metadata 与物理组织、做 D4/D8 Synopsys 面积时序能耗比较、最终 PAFT 后重放。P2 是补
per-token tail、materialized selector schedule 和 stage-mode power gating。

机器可读裁决见 `m179_independent_hammer_review_r1.json`，逐文件身份和全量复算见
`independent_recompute_result.json`。本评审没有修改 M179 主线文件或 `docs/359`。
