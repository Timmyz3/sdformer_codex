# Motion 三类 Zero-K 直注入与 Active 流解耦准入

## 1. 结论

本轮没有直接扩 RTL，而是先用 H67 fullres profile100、sample0/window0 全 12 block
真实 row vector 和现有 RQTB2S 强基线筛选新架构。

结论分为一项负结果和一项准入候选：

1. `[模型]` 通用 TTB8 局部 score-class quotient 虽把 class command 相对 RQTB
   减少 `61.80%`，但 metadata、active score、class commit 完全串行时只得到
   `1.067x`，周期减少 `6.30%`，未过 10% 门槛，因此否决该数据流；
2. `[prof]` both-K-zero temporal pair 占 `75.99%`，且此时 H67 两个 score 均
   严格落在 `{0,1,2}`；
3. `[模型]` 将这三类 multiplicity 直接写入三个专用 counter，同时让非 K-zero
   active bundle 通过 pointer/mask FIFO 进入原 H67 score + RQTB 路径，周期模型
   为 `1.192x`，周期减少 `16.08%`，通过最小 RTL 准入；
4. 该 PASS 只表示值得实现最小 RTL，不表示已有性能、能耗、PPA 或 DATE 新颖性
   结论。

完整机器可读结果：

```text
results/h67_exact_metadata_cascade_profile_20260809/report.json
results/h67_exact_metadata_cascade_profile_20260809/report.md
```

一键 CPU 入口：

```text
sim_h67/run_h67_exact_metadata_cascade_profile.sh
```

当前 `status.tsv` 为 `3/3 PASS`：Python 单元测试、架构准入重算、三份源码
SHA-256 自校验全部通过。

## 2. 为什么不能把 empty/K-zero 直接跳过

H67 score 为：

```text
score = 4 * popcount(Q & Kcur)
      + popcount(Kcur xor Kpeer)
      + round_even(popcount(~Q & ~Kcur) / 16)
```

当 `Kcur=Kpeer=0` 时，前两项为 0，但第三项仍参与 Shiftmax 分母：

```text
score = round_even((32 - popcount(Q)) / 16) in {0,1,2}
```

因此，empty/K-zero 不能被当成“无工作”删除。可做的 exact 变换是：

- 不读 K payload；
- 由 q-count 或预分类 2-bit code 得到 0/1/2；
- 把两个 temporal score 的 multiplicity 累加到三个专用 counter；
- SCS denominator 扫描时与 active score histogram 精确相加；
- gated-K 输出端仍只发射 `Kcur != 0` 的 token。

脚本穷举全部 33 个 q-count，而不是只从 trace 观察：

| q-count | exact score class |
|---:|---:|
| 0..8 | 2 |
| 9..23 | 1 |
| 24..32 | 0 |

这与近似 token pruning 的语义不同：计算表示和执行路径改变，最终整数 gate 与
gated-K 不变。

## 3. Profile100 驱动的四级路径

| 路径 | 占比 | 候选硬件行为 |
|---|---:|---|
| L0 all-four empty | 66.7804% | 固定 zero-K 类计数，不读 Q/K payload |
| L1 both-K-zero nonempty | 9.2077% | 读取/生成 q-count 类码，不读 K payload |
| L2 K-motion-zero non-K-zero | 0.1048% | 关闭 Motion-XOR，保留 overlap/same-zero |
| L3 full | 23.9071% | 完整 Q0/Q1/K0/K1 score |

由此得到两个工作量模型：

| 指标 | 结果 | 证据 |
|---|---:|---|
| metadata-assisted score boolean lane work 减少 | 76.02% | `[模型]` |
| 将 q-count popcount 计回 score 核后的减少 | 72.95% | `[模型]` |
| TTB8、32-bit header、2-bit 预分类 payload 减少 | 71.04% | `[模型]` |
| per-token gated-K emit 减少 | 83.96% | `[prof]` |

前两项是布尔 lane 操作代理，不是功耗。payload 模型假设 TTB FIFO 只存
pointer/mask/class metadata，Q/K bitmap 继续驻留现有 row SRAM；`71.04%` 只指
score 阶段 row-SRAM 读取流量，不包含两种方案共同承担的 Q/K 首次写入。

## 4. 数据流

```text
Q/K event coder + row-resident Q/K SRAM
       |
       +-- TTB8 metadata {both_kzero, zk_class0/1,
       |                  active_mask, row_base}
       |
       +-- zero-K direct path -------------------------+
       |     每bundle统计class0/1/2 multiplicity      |
       |     三个专用counter，无通用class FIFO写入    |
       |                                               |
       +-- active bundle FIFO ---------------------+   |
             只存row pointer + 8-bit active mask  |   |
             depth候选=32                         |   |
                  |                               |   |
                  +-> one active pair/cycle       |   |
                      read Q0/Q1/K0/K1 from SRAM   |   |
                      H67 score + temporal RQTB    |   |
                      active class histogram ------+---+
                                                   |
                              weighted SCS merge <-+
                                      |
                                  gated-K emit
```

两个关键点：

1. active backlog 最大为 197 个 pair，不表示 FIFO 需要存 197 份 128-bit payload；
   FIFO 只存最多 26 个未完成 TTB8 descriptor，payload 由 row SRAM 按 pointer 读取；
2. zero-K 三 counter 与 active score 不能分别做两次 Shiftmax，必须在同一 row 的
   max/denominator 扫描中合并，否则不 bit-exact。

此外，最小 RTL 的强基线也必须使用同一份 row-resident Q/K SRAM。新方案不能把
“上游已经构造好的 TTB metadata”与“基线仍逐 pair 接收原始 Q/K”直接比较；两者
都应先完成相同的 Q/K 驻留，再分别执行 RQTB2S scan 和 TTB8 zero-K/active scan。
共同的填充周期可以同时纳入或同时排除，但不能只对一侧免费。

## 5. Sample0 精确账本

真实向量覆盖 138 个 head-row、62100 个 temporal score。复算结果为：

| 项目 | 数值 | 说明 |
|---|---:|---|
| RQTB score slot | 34052 | 与现有 RQTB2S 报告精确一致 |
| non-K-zero active pair | 14554 | 进入完整 score 路径 |
| active RQTB command | 17170 | 相对原 RQTB 减少 49.58% |
| both-K-zero score class | `[0,1,2]` | 独立复算闭合 |
| active pair temporal score equal | 82.03% | 仍可使用 RQTB |
| TTB8 descriptor group | 4002 | 138×ceil(225/8) |
| descriptor FIFO depth p95/max | 26/26 | 无界队列模型；候选 depth32 |
| active-pair backlog p95/max | 197/197 | payload 不复制进 FIFO |

作为负结果保留的通用类商 DSE：

| TTB spatial | class command | 相对 RQTB 减少 |
|---:|---:|---:|
| 1 | 34052 | 0.00% |
| 2 | 25098 | 26.30% |
| 4 | 18422 | 45.90% |
| 8 | 13007 | 61.80% |
| 16 | 8689 | 74.48% |
| 32 | 5580 | 83.61% |

这些数字只证明代数压缩，不能单独证明周期收益。TTB8 通用类商串行 commit 正是
“压缩率高、周期收益不足”的反例。

## 6. 与既有工作的边界

| 来源 | 借鉴 | 本工作本土化差异 | 不声称 |
|---|---|---|---|
| Bishop | TTB metadata-first 和密度感知执行 | 不设 dense/sparse 双核；不做 ECP；zero-K 仍精确进入分母 | 不声称发明 TTB |
| Prosperity | exact reuse/分类优先于 payload 计算 | 复用对象是 H67 both-K-zero 的三类商与 multiplicity | 不借用其性能/PPA |
| SpAtten | cascade issue 的层次化思想 | 不删 token/head；四级路径保持整数 gate | 不声称无损版 SpAtten 即新架构 |
| RQTB | temporal pair 的 Q7 等价取商 | RQTB 只处理 non-K-zero active 流，zero-K 走三类直注入 | 不把 RQTB 重复列为新贡献 |
| SCS | weighted class denominator 与 gated-K 分离 | 三类 zero-K counter 在 SCS 扫描时合并 | 不声称发明 Shiftmax |

当前最可辩护的架构命题是：

> 面向 all-binary H67 的 denominator-preserving zero-K quotient injection：利用
> both-K-zero score 的三类闭包，把“不可删除的 silent denominator 工作”变成
> 三计数器直注入，并与 pointer-only active bundle 执行解耦。

这比“empty 跳过”更准确，也比通用双核或单纯 TTB 打包更贴网络语义。能否成为
DATE 主贡献仍取决于 RTL 后相对强 RQTB2S 的周期、活动、面积和存储结果。

## 7. 准入与停止条件

本轮六项准入检查均通过：

1. profile100 both-K-zero `75.99% >= 60%`；
2. 含 32-bit header payload 减少 `71.04% >= 60%`；
3. 保守 score lane 工作减少 `72.95% >= 60%`；
4. active command 减少 `49.58% >= 30%`；
5. zero-K score class 精确为 `{0,1,2}`；
6. 解耦周期模型减少 `16.08% >= 10%`。

进入最小 RTL 后采用以下停止条件：

- 与 RQTB2S 等 score lane、等 FIFO 总 bit、共享 SCS backend；
- 同一 138-row real trace 下 gated-K 与 synthetic Acc32 零失配；
- 随机和定向反压下不丢 descriptor、不重复 multiplicity；
- 实测 RTL 周期减少低于 10%，停止继续扩展；
- 若 descriptor SRAM、三 counter merge 和 event preclassifier 的开放映射面积使
  面积归一吞吐不增，降级为负结果；
- 没有 DC/STA/SAIF 前，不声称 ASIC PPA 或能效。

## 8. 当前证据边界

本包是 `[prof]+[prof-sample0]+[模型]` 的架构准入，不是 `[rtl]` 新机制证据。
当前不允许写：

- “three-class zero-K injection 已实现”；
- “Motion 已额外加速 1.192x”；
- “payload 或 score work 减少等于功耗降低”；
- “优于 Bishop/Prosperity”；
- “已满足 DATE 架构创新和工作量要求”。

下一轮只推进这个候选的等资源最小 RTL；Local5 checkpoint watcher 释放后仍按既定
优先级恢复 checkpoint-bound 数值合同与 12-block 调度，Motion 不因此停止。

## 9. 独立评审

独立 DATE 架构审稿人只按本包证据给出：

- Recommendation：`条件准入 RTL`；
- 包级评分：`3.0/5`；
- 创新潜力：`中高`；
- 准入含义：只允许做成本闭环与公平性验证，不得宣称性能收益。

评审没有否定 zero-K 三类闭包，但给出四项 RTL 硬门槛：

1. `1.192x` 未计 metadata SRAM、assembler、event preclassifier、SCS merge、
   真实 SRAM latency 与反压，必须继续标为 `[模型]`；
2. 候选与 RQTB2S 必须使用同一 row-resident Q/K SRAM、同 score lane，preload
   共同计入或共同排除；还要证明 active backlog 存活期间 row 不被覆盖；
3. preclassifier、header decode、bundle formation、metadata 端口和仲裁必须进入
   RTL/开放映射，不能作为上游免费结果；
4. depth26/backlog197 需在满载、突发、跨 row、随机/定向反压下验证 overflow、
   丢序和 deadlock；three-counter merge 需覆盖位宽、清零、累积、重复/遗漏和
   weighted denominator 一致性。

因此本包状态冻结为“条件准入”。下一包的最小 RTL 若未满足上述任一项，不能用
其周期结果支撑新贡献；若等资源 RTL 周期减少低于 10%，或开放映射面积归一吞吐
不增，则将本机制降为负结果。
