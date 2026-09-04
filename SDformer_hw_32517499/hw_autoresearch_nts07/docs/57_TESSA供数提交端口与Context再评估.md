# TESSA 供数、提交端口与 Context 再评估

**日期**：2026-07-13  
**对象**：H67/H68 百样本真实 workload  
**目的**：关闭独立架构审阅指出的两个周期模型缺口：128-bit/拍供数假设和每个时间对双结果提交冲突

## 1. 结论先行

1. **128-bit 时间对布局仍成立，但 81-cycle 只在 128-bit/拍或等效双 bank 无冲突供数时成立。** 只有 64-bit 单读口时，布局可以减少重复 K 搬运，但不能自动减少前端拍数。
2. **commit 是新的主瓶颈。** H67/H68 的双 K-zero pair 均约 83%，若不合并，同一 pair 往往需要两个 histogram 写；128-bit 前端会把瓶颈从 fetch 转移到 commit。
3. **PCCC 可能是解除瓶颈的必要机制。** H67 分 bank 单写口、2-context、无合并时仅比当前周期代理下降 22.33%；全合并乐观上界可下降 48.85%。
4. **最终 context 数不能只由原两阶段模型决定。** 无合并时 4-context 相对 2-context 只改善 3.74%，全合并上界下可改善 13.49%。
5. **当前实现决策**：fixed-bitmap、可旁路 PCCC、参数化 `NUM_CONTEXTS=1/2/4`；首版运行配置 2，4-context 物理实例化等待 ordered trace 和 SRAM/DC 结果。

## 2. 可从旧 Profile 精确恢复什么

旧 profile 没有逐 pair 同 class 记录，但已有：

- `ttb_tok1_total`：时间对总数；
- `ttb_tok1_kzero`：两时间片 K 都为零的 pair 数；
- `zaf_kzero_token_ratio`：单时间片 K-zero 比例。

定义：

```text
P = pair 总数
B = 双 K-zero pair 数
Z = 两时间片 K-zero token 总数
O = 单 K-zero pair 数 = Z - 2B
A = 双 active pair 数 = P - B - O
```

恢复结果：

| 模型 | pair/行 | 双 K-zero | 单 K-zero | 双 active |
|---|---:|---:|---:|---:|
| H67 | 81 | 67.32，83.11% | 8.98，11.09% | 4.70，5.80% |
| H68 | 81 | 67.47，83.29% | 8.67，10.70% | 4.86，6.00% |

这说明 H67/H68 不仅单时间片 K-zero 高，而且 K-zero 在两个时间片间高度持续。它支持 pair-coalesced commit，但不能推出两个 score 一定属于同一 class。

## 3. Commit 结构边界

### 3.1 理想双写口

active bank 和 histogram bank 各能在一拍接受同一 pair 的两个结果：

```text
Ccommit = P = 81 cycle/row
```

这是面积和能耗较高的理想下界。

### 3.2 Active/Histogram 分 Bank，各单写口，无 PCCC

- 双 K-zero：histogram 需要两拍；
- 单 K-zero：active 与 histogram 可并行，一拍；
- 双 active：active bank 需要两拍。

```text
Ccommit = 2B + O + 2A
```

H67/H68 分别约为 `153.01/153.33 cycle/row`，几乎抵消 81-cycle pair fetch。

### 3.3 分 Bank 单写口，PCCC 全合并上界

乐观假定每个双 K-zero pair 的两个 score 都属于同一 class，可执行一次 `hist[class] += 2`：

```text
Ccommit = B + O + 2A
```

H67/H68 分别约为 `85.70/85.86 cycle/row`，接近理想双写口。

### 3.4 统一单写口

```text
无合并：Ccommit = 2P = 162
全合并：Ccommit = 2P - B
```

即使全合并，统一单写口仍约为 H67 `94.68 cycle/row`，因此第一版应优先分离 active/histogram 写路径，而不是共用一个提交 SRAM 端口。

## 4. 三阶段 Context 重放

模型按行重放：

```text
Pair Fetch -> Commit -> Shared SCS Backend
```

资源约束：

- fetch、commit、SCS 各只有一份；
- context 从 fetch 开始占用，到 SCS/emit 完成后释放；
- 允许不同行在不同阶段重叠；
- 仍未计入逐 pair 内流水、SRAM 同步读延迟、ordered burst 和输出反压。

### 4.1 H67

| 供数 | commit | context | 周期/帧 | 相对当前 |
|---|---|---:|---:|---:|
| 128-bit/拍 | 理想双写口 | 2 | 694793 | -49.93% |
| 128-bit/拍 | 分 bank 单写口，无合并 | 2 | 1077711 | -22.33% |
| 128-bit/拍 | 分 bank 单写口，全合并上界 | 2 | 709702 | -48.85% |
| 128-bit/拍 | 统一单写口，无合并 | 2 | 1120441 | -19.25% |
| 128-bit/拍 | 统一单写口，全合并上界 | 2 | 739919 | -46.67% |
| 64-bit/拍 | 分 bank 单写口，全合并上界 | 2 | 1133739 | -18.29% |
| 128-bit/拍 | 分 bank 单写口，无合并 | 4 | 1037358 | -25.24% |
| 128-bit/拍 | 分 bank 单写口，全合并上界 | 4 | 613965 | -55.75% |

### 4.2 H68

| 供数 | commit | context | 周期/帧 | 相对当前 |
|---|---|---:|---:|---:|
| 128-bit/拍 | 理想双写口 | 2 | 687617 | -49.89% |
| 128-bit/拍 | 分 bank 单写口，无合并 | 2 | 1081856 | -21.16% |
| 128-bit/拍 | 分 bank 单写口，全合并上界 | 2 | 702601 | -48.80% |
| 128-bit/拍 | 统一单写口，无合并 | 2 | 1122693 | -18.19% |
| 128-bit/拍 | 统一单写口，全合并上界 | 2 | 731762 | -46.67% |
| 64-bit/拍 | 分 bank 单写口，全合并上界 | 2 | 1136703 | -17.16% |
| 128-bit/拍 | 分 bank 单写口，无合并 | 4 | 1041900 | -24.07% |
| 128-bit/拍 | 分 bank 单写口，全合并上界 | 4 | 618279 | -54.94% |

## 5. 对架构的直接修改

### 5.1 Pair Source

必须对照两种实现：

1. `1x128-bit` 同字读取；
2. `2x64-bit` 双 bank 同拍读取，并统计 bank conflict。

单个 64-bit 端口两拍 assembler 只作为功能 fallback，不应作为高吞吐主配置。

### 5.2 Commit Fabric

第一版接口冻结为：

```text
pair result 0 ----+--> active commit queue --> active bank 1W
                  |
pair result 1 ----+--> class merge/bypass --> histogram bank 1W
```

要求：

- 两个 2-entry queue，分别接 active 和 histogram；
- mixed pair 可同拍写两个 bank；
- 双 active 或双 class 允许排队；
- 同 class 时可选 `+2` 合并；
- PCCC 关闭时逐 token 提交，与 golden 完全一致；
- queue full 必须反压 pair front，不得丢结果。

### 5.3 Context Pool

- RTL 参数：`NUM_CONTEXTS=1/2/4`；
- 首版验证重点：1 和 2；
- 4-context 只完成 elaboration、状态隔离和 cycle model 接口；
- ordered trace 返回后，若相对 2-context 的真实收益达到 8% 且新增 memory/control 面积低于 15%，才物理晋级；
- 不实现 8-context。

## 6. 对论文创新叙事的影响

新的证据使 PCCC 从“可能减少 histogram 写”的局部优化，提升为 temporal-pair 架构能否兑现吞吐的关键桥梁：

> temporal-pair 把 fetch 压缩为 81 个逻辑输入，但也把两个 token 的结果集中到同一拍；PCCC 和分 bank commit 在保持精确 class/denominator 语义的同时，解决这种由数据流融合主动制造的提交带宽问题。

这一表述比“提出一个 histogram 合并器”更具有架构完整性，但仍必须满足：

- 双 K-zero 同 class 真实比例；
- commit transaction 至少下降 2 倍；
- 相对无 PCCC 的净能耗改善；
- 0 mismatch；
- 同约束 DC/SAIF。

## 7. 证据边界

已完成：

- H67/H68 各 100 样本、12 block 的 pair 类别恢复；
- 64/128-bit 供数、五种 commit 结构、1/2/4-context 的三阶段重放；
- 公式单元测试和真实 profile 分析器回归。

未完成：

- 双 K-zero 同 TTX/H67 class 真实比例；
- 逐 pair 双提交 burst 和有限 queue stall；
- SRAM 端口、bank conflict 和宏取整；
- RTL cycle、DC 面积/时序、SAIF 功耗。

因此本报告用于规格和候选淘汰，不是 RTL 或芯片性能结果。
