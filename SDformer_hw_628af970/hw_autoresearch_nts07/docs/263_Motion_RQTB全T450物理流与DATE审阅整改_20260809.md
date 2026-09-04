# Motion RQTB 全 T450 物理流与 DATE 审阅整改

> **已被双 slot 公平强基线取代：** 本文的 `Fixed-TTB32 -> RQTB16/32`
> `1.471x` 结果含单 slot/cycle 弱基线瓶颈，不再用于论文主结果。
> 当前主证据为 `Fixed2S -> RQTB2S` 的 `1.185x` 公平对照，详见
> `docs/264_Motion_RQTB双slot公平强基线与协议签核_20260809.md`。本文仅保留为
> 早期物理流和审阅整改历史。

## 1. 本轮结论

本轮完成 Motion/H67 的 `Fixed-TTB32` 与 `RQTB16/32` 同约束 row-slice RTL、真实 T450 回放和开放布局布线代理。结论分为三层：

1. `[rtl]` RQTB 的 gated-K 输出与 synthetic Acc32 checksum 在当前真实 trace 上保持精确一致；
2. `[rtl]` 相对单 slot/cycle Fixed-TTB，RQTB 的 row-slice 周期、slot 和 exp 事务均有明确收益；
3. `[DATE审阅]` 当前结果仍是 `2/5 Weak Reject`，主要原因是缺少 2-slot/cycle Fixed 强基线、多样本 RTL trace、真实 SRAM/功耗和 full encoder 系统闭环。

因此，RQTB 当前状态为：

> **通过组件级 RTL 与开放物理筛选，但尚未晋级 DATE 独立主贡献。**

Local5 仍是当前优先补系统完整度的主线；Motion 不冻结，继续维护现有回归，并允许对通过真实 workload 门槛的新机制继续迭代。

## 2. 机制与数据流

### 2.1 核心数据流

```text
Q0/Q1 + K0/K1
      |
      v
双路 Motion-XOR Q7 score
      |
      +---- Fixed-TTB32：每 pair 固定两个 16-bit slot
      |
      +---- RQTB16/32：score0==score1 时形成一个 temporal-mask=11 slot
      |
      v
16-bit x 32 slot FIFO
      |
      v
weighted-SCS class histogram
  multiplicity = popcount(temporal_mask)
      |
      v
Shiftmax exp/denominator + active descriptor
      |
      v
同步 K0/K1 双 bank 按 active-mask 读取
      |
      v
按原时间顺序恢复 gated-K event
```

### 2.2 可逆性合同

RQTB 不是把一个时间 token 删除，而是对量化后 score 相等关系取商：

- `score0 == score1`：只存一次 score，同时保存 `temporal_mask=11`；
- Shiftmax 分母通过 `popcount(mask)=2` 保留两个 token 的 multiplicity；
- `active_mask` 独立记录 K0/K1 是否需要输出；
- gated-K 边界读取对应 K bank，并按 time0、time1 顺序展开；
- `score0 != score1` 时退化为两个独立 slot，无近似 fallback。

因此当前允许使用的架构差分是：

> 面向 T=2 Motion attention 的无损 post-quantization temporal score quotient，在归一化域保留 multiplicity，并将 token 展开推迟到 gated-K 边界。

不能把 TESC 和 RQTB 拆成两项贡献：TESC 是数学合同，RQTB 是该合同的物理流接口。

## 3. 同约束 RTL 结果

### 3.1 覆盖范围

证据绑定 H67 fullres epoch30、`sample0/window0`、全部 12 个 attention block：

| 项目 | 数值 |
|---|---:|
| head-row | `138` |
| token | `62,100` |
| temporal pair | `31,050` |
| gated-K 输出检查 | `20,841` |
| synthetic Acc32 checksum | `4,416` |
| checksum mismatch | `0` |
| Icarus | 首个真实 T450 row PASS |
| Verilator | `138/138` row PASS |
| Verilator + SVA | `138/138` row PASS |

这里的 Acc32 使用确定性的人工 lane weight，只用于检查 Fixed/RQTB 最终整数累加一致，不是真实 projection 权重回放。真实权重 projection 已在既有独立组件流中验证，不能与本轮 checksum 混称一个端到端顶层。

### 3.2 公平边界

两种候选共享：

- 两路 Motion-XOR score 前端；
- `16 bit x 32` FIFO bit 容量；
- weighted-SCS、Shiftmax 和 gated-K backend；
- 同步一拍 K0/K1 双 bank；
- active-mask K bank gating；
- 同一确定性周期反压模式；
- 同一 Q/K、gate 和 Acc32 检查口径。

当前唯一功能差异为 Fixed 每 pair 固定两个 slot，RQTB 在 score 相等时只发一个 slot。

### 3.3 周期与工作量

| 指标 | Fixed-TTB32 | RQTB16/32 | 变化 |
|---|---:|---:|---:|
| 总周期 | `146,948` | `99,917` | `1.471x`，周期 `-32.01%` |
| slot | `62,100` | `34,052` | `-45.17%` |
| exp 事务 | `22,133` | `17,255` | `-22.04%` |
| K read transaction | `20,841` | `20,841` | 相同 |
| K read bits | `666,912` | `666,912` | 相同 |

逐 row 周期：

| 候选 | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|
| Fixed | `1064.84` | `2021.05` | `2103.16` | `2132` |
| RQTB | `724.04` | `1437.20` | `1660.56` | `1720` |

RQTB 在 `138/138` 行均更快，但当前 Fixed 后端每拍只消费一个 slot。该结果只能描述这一具体接口边界，不能越过 2-slot/cycle 强基线直接称为架构加速。

## 4. 活动量证据修正

初版 VCD 解析器以 identifier 为单一字典键，跨层 alias 可能被最后一次声明覆盖。整改后：

- 同时别名到 Fixed/RQTB 的 identifier 被当作共享网络排除；
- 共排除 `4` 个跨设计共享 alias code；
- 只统计各候选独有层次信号的位翻转；
- 单个真实 row 的位翻转为 `75,341 -> 54,552`，下降 `27.59%`。

该数字仍不是 SAIF 功耗、动态能量或 ASIC 功耗。它只能用于决定是否值得进入后续 SAIF/PTPX，不进入论文能效主表。

## 5. OpenROAD 物理代理

### 5.1 流程口径

- 工艺：Nangate45 开放库；
- 时钟：`5 ns`；输入/输出各留 `0.5 ns`；
- T450 K store、slot FIFO 和目录全部映射为 flop；
- 两种候选均 `macro_count=0`；
- 完成详细布线和 post-route RC/STA 报告；
- 本机缺 KLayout，未完成 GDS finish；
- 不是 DC、PrimeTime、PTPX 或目标 SRAM 宏签核。

### 5.2 结果

| 候选 | 标准单元面积 | 单元数 | WNS | setup/hold | max-cap | DRC | 线长 | via |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-TTB32 | `242,969 um2` | `95,324` | `-0.0022 ns` | `1/0` | `63` | `0` | `1,912,760 um` | `691,475` |
| RQTB16/32 | `244,164 um2` | `94,587` | `+0.0951 ns` | `0/0` | `41` | `0` | `1,902,874 um` | `687,175` |

开放代理下 RQTB 面积 `+0.49%`、单元数 `-0.77%`、线长 `-0.52%`、via `-0.62%`。把 RTL 周期与面积简单组合得到的筛选指标为 `1.464x`，但 Fixed 尚有 2.2 ps setup 违例，且两者均有 max-cap 违例，因此该指标不能称闭合 PPA 或正式面积归一吞吐。

未约束端点均为综合后常量化的性能计数器低位，解析器对名称集合做精确 fail-closed 检查；主数据输出没有未约束端点。

## 6. 新机制筛选负结果

本轮额外评估了“RQTB 时间对偶奇多播”：score 相等时 gate 相同，且空间 token 数 `225` 为奇数，同一空间位置的 time0/time1 原始 token ID 奇偶相反，理论上可把一个 gate-product 同时写入偶/奇 Acc 端口。

真实 all12 trace 结果：

| 指标 | 数值 |
|---|---:|
| 标量 lane 命令 | `77,792` |
| 时间对多播命令 | `75,402` |
| 节省命令 | `2,390` |
| 命令降低 | `3.07%` |
| 理想命令加速 | `1.032x` |

虽然 score 相等率高，但两时刻同一 lane 的 K 事件重合不足。该候选未过 `15%` 命令降低准入线，**不扩 RTL、不列为 DATE 贡献**。它与既有跨完整 gate-term 的 PPDI 不是同一统计口径，两者收益不得相加。

## 7. DATE 独立审阅

### 7.1 评分

- 总分：`2/5`；
- 结论：`Weak Reject，Major Revision`；
- 机制新颖性：约 `3/5`；
- 当前投稿证据：约 `2/5`。

### 7.2 审稿人认可项

1. 相对固定 TTB，按 Q7 score 等价取商、保留 multiplicity 并在 gated-K 边界无损展开的差分成立；
2. 与 Bishop 类 TTB 的差分应限定为“exact quotient + normalization multiplicity + late expansion”，不能声称发明 TTB；
3. 当前真实 trace 上的 bit-exact 和 row-slice 工作量收益可信；
4. 负结果和开放物理边界总体诚实。

### 7.3 阻塞接收项

1. 缺 2-slot/cycle Fixed-TTB 或双 histogram update 强基线；
2. RTL trace 只有 sample0/window0，不能外推 100 样本或完整 frame；
3. 缺 post-route SAIF、动态/漏电/时钟/SRAM 能量；
4. 全 flop-memory OpenROAD 不能替代真实 SRAM 宏 PPA；
5. 当前是 row slice，不是完整 encoder 或应用加速；
6. SVA 尚缺 common/split 原子退休、multiplicity 守恒和无重无漏的因果断言。

### 7.4 已完成整改

1. 修正逐 row speedup 分布曾被错误放大 `1e6` 的报告 bug，并增加无量纲比值单测；
2. 修正 VCD 跨层 alias 归属，重新生成活动代理；
3. 将“随机反压”更正为“确定性周期反压”；
4. 将 Acc32 更正为 synthetic checksum，不再暗示真实 projection 权重；
5. 重新生成并绑定 RTL report 与 OpenROAD report 的 SHA；
6. 明确只有 RQTB post-route setup/hold 闭合，Fixed 仍有一条 2.2 ps setup 违例。

上述整改提高证据可信度，但没有关闭强基线、功耗和系统范围三个主要拒稿原因，因此评分不自动上调。

## 8. 下一轮唯一 Motion 门槛

下一轮先做同硬件带宽强基线，不先扩更多机制：

1. `Fixed-1S`：当前单 slot/cycle 基线；
2. `Fixed-2S`：每拍可处理两个 16-bit slot，支持同类 score 双更新；
3. `RQTB-2S`：使用相同双 slot FIFO/双更新后端；
4. 可选组合：现有 exact temporal-delta reuse，但必须单独列面积和控制开销。

三者必须共享 FIFO bit 容量、K bank 端口、class histogram 带宽、反压、时钟和 P&R 规则。主判据为：

- gated-K 与 synthetic checksum 零失配；
- RQTB 相对 Fixed-2S 仍有可测周期或活动收益；
- 额外双更新逻辑纳入面积和时序；
- 若面积归一收益低于 `1.10x`，RQTB 降级为内部压缩机制，不作 DATE 独立贡献。

## 9. Local5 并行状态

Local5 的同窗全-head profile 和 RelationMemo 联合统计 watcher 仍在等待训练侧 GPU/最终产物，不重复启动。产物到达后优先执行：

1. 冻结 theta、Q7/Q1.7、hardware-order Shiftmax5 和真实 invalid-candidate mask；
2. 完成 12-block 时间复用调度；
3. 闭合 `score/Shiftmax5 -> relation transpose -> source-major term -> TCFM5 -> accumulator` 单顶层；
4. 使用真实 trace、确定性与随机种子反压、Acc32 miter 和同约束开放物理代理签核。

Motion 的强基线工作与 Local5 profile 等待可并行，但不能因 Motion 本轮有正结果而放弃 Local5。

## 10. 关键产物

- RTL 报告：`results/h67_rqtb_physical_flow_t450_20260809/report.{json,md}`
- OpenROAD 代理：`results/h67_rqtb_openroad_proxy_t450_20260809/report.{json,md}`
- 偶奇多播负结果：`results/h67_rqtb_temporal_pair_multicast_20260809/report.{json,md}`
- 一键 RTL 回归：`sim_h67/run_h67_rqtb_physical_flow_checks.sh`
- OpenROAD 入口：`openroad_hifp/run_openroad_rqtb_t450.sh`
- 约束审计：`openroad_hifp/run_check_setup_rqtb_verbose.sh`
- 物理代理汇总：`scripts/summarize_rqtb_openroad_proxy.py`

## 11. 当前允许写入论文的句子

> 面向 T=2 Motion attention，本文实现了一种无损的 post-quantization temporal score quotient：相等的 Q7 score pair 被编码为单个 temporal-mask slot，Shiftmax 通过 mask popcount 保留归一化 multiplicity，并在 gated-K 边界按 active mask 恢复原 token 顺序。

> 在绑定 H67 epoch30 checkpoint 的 sample0/window0、覆盖 12 个 attention block 的 138 个 T450 head-row 上，该 RTL 与固定 TTB 的 gated-K 输出和 synthetic Acc32 checksum 一致；在相同 16-bit x 32 FIFO、同步双 bank K 模型和确定性周期反压下，slot、exp 事务和当前单 slot/cycle row-slice 周期分别减少 45.17%、22.04% 和 32.01%。

不得写“首次 temporal reuse”“RQTB 节能 27.59%”“ASIC PPA”“完整 encoder 加速 1.471x”，也不得把 TESC 与 RQTB 分拆成两项贡献。
