# M93 dual descriptor packet issue 独立打铁评审

## 结论

M93 的正确结论是 **NO-GO，且不得放宽 0.5% 门槛或只报 source gain**。

- 综合评分：**82/100**；`P0=0 / P1=5 / P2=4`。
- W1 对 M89 K6 的 aggregate、p95 和十个 sample 的 source/integrated 全部 exact reproduction。
- W2 integrated 仅改善 27,624 cycles，即 **0.0360263% / 1.00036039x**，比冻结
  0.5% 门槛差 355,763 cycles，并使 sample 5、9 分别退化 120、4,208 cycles。
- W4 比 W2 **慢 36,744 cycles**，必须 KILL。
- 当前实现只是 command-calendar width sensitivity，不是真正验证过的 128B/256B packet。

准入范围只有冻结 transaction model 的敏感度和 NO-GO 结论；RTL、等面积、系统倍速和
DATE headline 全部不准入。

## 身份与执行

独立脚本 exact-SHA 核验了 contract、probe、raw result、r1 failure log、r2 complete log、
receipt，以及 M45/M53/M43/M89 输入。主要 SHA 为：

| artifact | SHA256 |
|---|---|
| contract | `28ffb056...a87a4e8` |
| probe | `832042a4...5ed883` |
| raw result | `7345e006...ac7e9` |
| r1 failure log | `846a8303...a9641` |
| r2 complete log | `e1ec609c...c5e83` |
| receipt | `b07fa687...536a9` |

r1 在缺失 M53 result 时 fail closed，没有 final marker。r2 对 W1/W2/W4 各含严格的
`1..40` record marker、十个 sample、四个 operator，以及一个与 raw result exact match 的
compact final marker。

## 独立重算结果

脚本不 import 或执行 M93 producer，直接从三档各 40 条 raw record 重新聚合 sample、
distribution、wait、packet 和 gate。

| width | source | integrated | p95 | command wait | response wait | parent wait | lane util. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 / 64B | 69,964,176 | 76,677,320 | 7,843,680 | 2,624,272 | 2,011,048 | 1,947,448 | 100% |
| W2 / 128B | 69,855,096 | 76,649,696 | 7,841,024 | 2,561,392 | 2,128,032 | 1,974,608 | 58.3653% |
| W4 / 256B | 69,832,224 | 76,686,440 | 7,847,448 | 2,552,856 | 2,175,576 | 1,995,240 | 29.4999% |

每档 descriptor 均为 25,920,000，来源可独立推导为：

`40 records × 10 timesteps × 27 tiles × 300 rows × 8 blocks = 25,920,000`。

calendar instances 为 `40×10×27=10,800`。W2 的 22,204,984 packet cycles 中只有
3,715,016（16.73%）填满两 lane；W4 的 21,966,160 cycles 中只有 247,280
（1.13%）填满四 lane。

### W1 逐样本 exact reproduction

| sample | source | integrated |
|---:|---:|---:|
| 0 | 7,148,648 | 7,802,976 |
| 1 | 7,190,120 | 7,843,680 |
| 2 | 7,075,968 | 7,723,128 |
| 3 | 6,970,992 | 7,667,384 |
| 4 | 6,847,384 | 7,547,360 |
| 5 | 6,897,792 | 7,563,232 |
| 6 | 6,999,096 | 7,670,736 |
| 7 | 6,954,744 | 7,625,264 |
| 8 | 6,938,440 | 7,609,632 |
| 9 | 6,940,992 | 7,623,928 |

十行均与 M89 K6 exact match。

### W2 逐样本 delta（candidate - W1）

| sample | source delta | integrated delta |
|---:|---:|---:|
| 0 | -12,488 | -5,064 |
| 1 | -7,240 | -2,656 |
| 2 | -8,384 | -888 |
| 3 | -12,248 | -3,976 |
| 4 | -13,024 | -4,016 |
| 5 | -8,544 | **+120** |
| 6 | -12,072 | -3,816 |
| 7 | -12,336 | -4,808 |
| 8 | -17,808 | -6,728 |
| 9 | -4,936 | **+4,208** |

所有 sample 的 source 都下降，但两个 sample 的 integrated 退化，证明只报 source 会给出
错误方向。

## 为什么 command 降了，integrated 几乎不降

W2 相对 W1 的精确分解是：

| 项 | delta |
|---|---:|
| source cycles | -109,080 |
| command wait | -62,880 |
| response wait | +116,984 |
| parent wait | +27,160 |
| 其余 end/tail residual | +192 |
| non-source overhead 合计 | +81,456 |
| integrated | **-27,624** |

闭合式为：`-109,080 source + 81,456 overhead = -27,624 integrated`。

宽 admission 改变了哪些 ready descriptor 同时 resident/prepared，因而改变 K6 grouping：W2
少 204,296 个 fusion groups、少 92,544 个 zero groups，source work 下降；但更早、更突发的
residency 把压力移到 context/complete/parent 路径，response 和 parent wait 吃掉绝大部分收益。

W4 延续同一趋势：相对 W2 source 再降 22,872、command wait 再降 8,536，但 response
增加 47,544、parent 增加 20,632，最终 integrated **增加 36,744**。

## 冻结 gate

W1 四项 reproduction gate 全过。W2 的 replay、signed conservation、source、p95、reported
occupancy、command-wait 等 gate 数值上通过，但两个关键 gate 失败：

1. integrated `76,649,696 > 76,293,933`，未达到冻结 0.5%；
2. sample 5、9 退化，违反每 sample 不退化。

因此 `all_width2_gates_pass=false`。W4 的增量 0.5% gate 也失败，而且比 W2 更慢。

不能因只差 0.464 个百分点而事后放宽阈值；该门槛是在结果前冻结，正是防止把噪声级改善
包装成硬件创新。

## 它不是真正的 128B/256B packet

静态重建 M53 transforms 后，M93 对 scheduler source 的唯一 diff 是：

```diff
-    command_port = PortCalendar()
+    command_port = CommandPacketCalendar()
```

新 calendar 确保每 calendar cycle 最多 W 个 issue，并保证 `issue_cycle >= ready_since`；
descriptor 计数和 8-block 缩放也正确。但这不等价于物理 packet：

- 没有 descriptor payload/address、128/256B alignment 或真实 byte transfer；
- 没有 lane-valid、partial-packet 格式、packet membership trace；
- 没有 pack/unpack/dispatch/backpressure；
- 没有宽 wire、寄存器、仲裁、Fmax、area、energy 费用；
- calendar 可以把 issue 排到早于当前 scheduler `now` 的历史空槽。

另有一个重要静态缺口：metadata occupancy 使用 `min(16,len(ready))` 后再检查 `<=16`，
因此检查恒真，底层 ready heap 并没有被 16-entry 容量约束。这使物理 assembly/FIFO 可行性
更不能成立。

## P1 / P2

P1：calendar 不是物理 packet；metadata gate 恒真且 calendar 可 backdate；宽路径成本完全
未收费；没有 per-cycle descriptor/packet identity trace，三档 transformed scheduler SHA 也相同；
范围仅 valid825-internal 四层十窗、无 RTL/准确率/全网。

P2：wait 分解余 192 cycles 未命名；W4 无 1/2/3-lane occupancy histogram；十 sample 的
p95 实际等于最大值；r2 log 不自带 full-output SHA/命令/环境。

## 下一最小方向

下一步建议改做 `METADATA_ONLY_BANK_CYCLE_MONOTONIC_K6_GROUPING`：

- 保持 W1 64B command、K6-C16，不增加 vector payload storage，不等待未来 descriptor；
- 只在当前 prepared descriptors 中，用现有 exact delta/bank-signature metadata 选 group；
- alternate group 的 union bank-issue cycles 必须不高于同一决策点 legacy K6 group，否则
  deterministic fallback 到 legacy；
- 冻结 selector-off W1 exact reproduction、逐决策 bank-cycle monotonic、descriptor/DAG/parent/
  signed conservation、三类 wait 不增加、逐 record/sample source 和 integrated 不退化、aggregate
  integrated 至少改善 0.5%。

M93 已说明瓶颈不是 command issue width，而是 K6 grouping 与下游 context/response 的耦合。
metadata-only monotonic selector 才是当前最小、最可证伪的方向。

机器可读结果见 `m93_dual_descriptor_packet_issue_independent_hammer_review.json` 和
`m93_independent_audit.json`；复跑入口为 `audit_m93_independent.py`。未修改 contract、probe、
raw result、logs 或 receipt。
