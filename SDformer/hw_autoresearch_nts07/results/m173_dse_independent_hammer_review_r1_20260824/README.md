# M173 scan-width exact-payload DSE 独立打铁评审 r1

结论：**82/100，`PASS_EXACT_ISOLATED_LATENCY_DSE_FIX_WALL_LABEL_AND_PHYSICAL_SELECTION_PENDING`，P0/P1/P2 = 4/6/3。**

M173 的 payload 解码、width reshape、modulo-8 bank grouping 和五个宽度点的整数
结果均可独立复现；128-bit 也确实是**已评估的 power-of-two 点**中，第一个在四个
stage 都超过 2x 的点。但生产结果继续继承了 M172 的口径问题：表中所谓 `wall`
是逐 token 从空闲状态开始、到 `token_done_valid` 出现为止的**隔离延迟之和**，不是
当前 M171 RTL 连续接收相邻 token 的串行总周期。

因此，128-bit 的 `432,951,702 / 146,423,753 = 2.956840630x` 数值正确，但必须
改名为 isolated-token latency-sum ratio。按当前 M171 `token_done` 后需要一个 re-arm
edge 的协议，连续串行流是 `438,531,701 / 152,003,752 = 2.885005766x`。选择规则
仍勉强成立，不过 stage0 只剩 `2.007154857x`，距离 2x 仅 **0.3577%**。

## 1. 独立 exact-payload 复算

本评审没有导入 M172/M173 analyzer。新脚本直接使用
`numpy.unpackbits(bitorder="little")` 解 120 个 FC2 bitpack，再按 channel-last C-order
重构 token；bank 由 `channel_index mod 8` 得到。全部 120 个 payload、
**437,760,000 bytes** 的 SHA、大小和 popcount 均重查。

| 项 | 独立结果 |
|---|---:|
| samples / FC2 records | 10 / 120 |
| tokens | 5,580,000 |
| input elements / events | 3,502,080,000 / 143,894,510 |
| recurrence oracle cases | 7,120 |
| recurrence mismatch | 0 |
| aggregate + stage integer mismatch | 0 |

五档 production 数值精确复现：

| scan width | K1 isolated sum | K4 isolated sum | K1/K4 |
|---:|---:|---:|---:|
| 64 | 446,528,624 | 179,057,955 | 2.493765909x |
| 96 | 437,234,151 | 157,504,597 | 2.776008823x |
| **128** | **432,951,702** | **146,423,753** | **2.956840630x** |
| 192 | 428,961,896 | 135,135,765 | 3.174303235x |
| 384 | 425,370,073 | 123,793,034 | 3.436139008x |

64-bit 点也精确复现 M172 的 `446,528,624 / 179,057,955` 身份。分 stage 的
128-bit 比例为 `2.151304070 / 2.898766685 / 3.160067513 / 3.295854900`，因此原
结果给出的四 stage 最小值 `2.151304070` 正确。

## 2. reshape、padding 和 bank grouping

冻结四档 `Cin` 为 384/768/1536/3072，换成每 token byte 数为
48/96/192/384。五档 beat byte 数 8/12/16/24/48 都整除这四档 byte 数，所以本次
真实数据在所有 width 下的 padding 都恰好为 0。M173 的 generic padding 分支写法合理，
但没有被本次冻结几何实际覆盖，后续应补一个非整除 synthetic case。

对一个 beat，令八个 bank 的 event 数为 `n_b`。K4 的最小 group 数

```text
max(max_b(n_b), ceil(sum_b(n_b)/4))
```

既是必要下界，也是充分的调度长度：每 bank 每 group 最多放一项，总 group 容量最多
四项。独立脚本从 bit 级重构 `n_b`，没有复用 M173 的 byte lookup table；全部 aggregate
和 per-stage group/replay/wall 字段 0 mismatch。

## 3. `wall` 口径仍需修正

M171 的 `scan_ready` 被 `done_valid_q` 和 `token_last_seen_q` 同时 gate。即使
`token_done_ready=1`，消费 done 的那个 edge 上也不能同时接收下一 token；清除状态后，
下一拍才重新 ready。M173 的 recurrence 每个 token 都从空闲开始，并在 done 出现时返回，
把 5,580,000 项隔离延迟直接相加，遗漏了相邻 token 间的 5,579,999 个 re-arm cycles。

| width | K1 continuous | K4 continuous | K1/K4 |
|---:|---:|---:|---:|
| 64 | 452,108,623 | 184,637,954 | 2.448622362x |
| 96 | 442,814,150 | 163,084,596 | 2.715242033x |
| **128** | **438,531,701** | **152,003,752** | **2.885005766x** |
| 192 | 434,541,895 | 140,715,764 | 3.088082548x |
| 384 | 430,950,072 | 129,373,033 | 3.331065694x |

128-bit 连续串行口径的四 stage ratio 是
`2.007154857 / 2.826344085 / 3.137640482 / 3.290718462`。所以“每 stage >2x”
在当前 RTL 协议下仍成立，但 stage0 的余量已非常小。另一个可行修复是在下一 RTL 中支持
done retire 与下一 token 首 beat 同拍接受，并由 VCS 锁定；在此之前不能把
`2.956840630x` 称为连续 wall-cycle ratio。

## 4. 128-bit 选择：数值规则成立，物理选择尚不成立

若只按“已评估 power-of-two 且四 stage 全部超过 2x”的预设规则，64-bit 的 stage0
只有 `1.761925091x`，128-bit 为 `2.151304070x`，所以选择 128 的逻辑成立。更严谨
的写法应是“smallest **evaluated** power-of-two point”，因为 32-bit 未实际评估。

但这个规则不是硬件 Pareto 规则。96→128 将 bitmap delivery 从 96 bit/cycle 增加
33.333%，K4 隔离延迟只从 157,504,597 降到 146,423,753，即 **7.035% 周期下降、
1.075677x 吞吐提升**。96-bit 还天然对齐现有 96-lane pool。故 128 可以作为下一 RTL
候选，不能在 matched 96/128 DC、macro 和功耗之前锁成物理最优点。

## 5. 物理与组成缺口

M171 的 64-bit 双套 nested selector 已因 103 logic levels fail-closed。M173 提出的
“one shared hierarchical per-bank selector”目前只是方案：128-bit beat 要做八个 16-row
bank first-event encoder，再选至多四个 bank，同时更新 residual/prefetch。是否能在 3 ns
闭合、是否需要 pipeline、pipeline 后 recurrence 是否改变，都没有 RTL/VCS/DC 证据。

此外每拍 128-bit bitmap delivery 尚无 macro/port；每个 K4 group 还要求四个 distinct-bank
weight response，每 bank 返回 96 个 INT8，即合计 **3072 bit/cycle**。weight latency、
bank conflict、response tag、M169 arithmetic、2304-bit accumulator context、BN2、residual
和 commit 均未组成。always-ready consumer 也会掩盖这些模块造成的 backpressure。

所以安全口径只有：

> M173 是冻结 H67 十样本上的 exact isolated-token frontend latency-sum width DSE；
> 128-bit 是按预设阈值选出的 provisional RTL candidate。

不能称 RTL-measured、physical、complete-FC2、FFN、network 或 system speedup，也不能写成
能效、FPS、P&R PPA 或 DATE headline。

## 6. P0/P1/P2

P0：

1. 修正 `wall` 命名，或实现 next-token same-cycle re-arm；当前连续 128-bit-equivalent
   口径应为 **2.885005766x**，不是 2.956840630x。
2. 用同一 parameterized RTL 对 96/128 两点做 hierarchical selector VCS/DC；64-bit 前代
   已有 103-level 失败，解析宽度不能替代物理证据。
3. 落地 128-bit bitmap 供给和四 bank、3072-bit/cycle weight response，包含 latency、
   tag、冲突和 routing。
4. 把 selector/group hold、weight response、M169、accumulator context 和 final commit 放进
   一个 exact-payload ready/valid shell，再谈完整 FC2 倍率。

P1：保留 96-bit 物理 A/B；用 area/frequency/energy Pareto 取代人为 2x 门槛；测试 finite
buffer/backpressure；PAFT 后重放；扩展 sequence 与 tail 分布；比较 SRAM bitmap reread 和
ATLIF-native producer tap。P2：把“smallest”改成“smallest evaluated”；补 marginal
bandwidth/cycle 表；补非整除 channel 的 padding 测试。

建议下一步：**不要先锁死 128。做一份 parameterized 96/128 shared hierarchical selector，
同时修复 token re-arm，再用 matched VCS/DC 决定宽度；之后接显式 bitmap/weight memory 和
M169。**

机器可读裁决见 `m173_dse_independent_hammer_review.json`，fresh 复算见
`fresh_independent_checks.json`；本评审未修改 M173 source/contract/result 或 `docs/359`。
