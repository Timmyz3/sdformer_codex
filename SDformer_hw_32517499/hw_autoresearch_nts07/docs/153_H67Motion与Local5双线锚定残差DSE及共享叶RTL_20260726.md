# H67 Motion 与 Local5 双线锚定残差 DSE 及共享叶 RTL

> **后续进展**：共享叶已经集成为单实例双模式复合核，并按独立评审修正
> Local5 固定 Q、bias 上界和物理共享问题。见
> `docs/155_TARE4双模式复合核与独立评审修正_20260726.md`。该进展仍不代表
> 完整 H67 双 score row 或 Local5 Shiftmax5 row。

## 0. 本阶段结论

本阶段没有把 Motion 与 Local5 强行合并成同一套完整 attention，而是识别并
实现了两条线真正共享的底层执行原语：

```text
静态锚点 score
  + 32-bit 新旧状态差分 mask
  + ZERO / SPARSE<=4 / DENSE>4 精确分流
  + 4-lane alpha-XNOR residual
  + dense 项回放到原 32-lane direct engine
```

暂称该候选为 **TARE-4（Temporal/Topology-Anchored Residual
Execution）**：

- H67 的锚点是 `Q0/K0`，残差是 `Q0/K0 -> Q1/K1` 的时间差分；
- Local5 的锚点是 `Q/Kself`，残差是
  `Q/Kself -> Q/Kneighbor` 的拓扑方向差分；
- 两者共享 mask 分类、四路 set-bit 提取和 alpha-XNOR lane delta；
- Motion-XOR、Local5 边界/方向控制、最终 Shiftmax 语义仍分别保留。

这是一项可继续做 PPA 的架构候选，不是已经成立的 DATE 主创新，也不是
端到端加速器结果。

## 1. 借鉴机制与改动

| 来源 | 借鉴 | 本项目修改 | 不宣称 |
|---|---|---|---|
| Prosperity | exact/partial residual reuse | 锚点由时间相邻或 stencil 拓扑静态给出，不用 TCAM 搜关系；最终只做一次 RNE | 首次 exact reuse |
| FireFly-T | 每拍多 lane 稀疏解码 | 解码的是 Q/K 更新或 Kself/Kneighbor 翻转，不复制其 FPGA 双引擎 | 首次 multi-lane decoder |
| Bishop | density stratification、TTB | `>4` 不送独立 dense core，而是精确回放到已有 32-lane anchor engine | TTB、stratifier、异构双核 |
| Phi | pattern + residual 对照 | 当前采用固定真实锚点，不加 codebook lookup；Phi 保留为 Local5 强基线 | pattern residual 原创 |
| 复旦 ISSCC 蝶形网络 | 稀疏 compact、局部注意力复用 | 首版使用固定 32->4 extractor；Local5 固定五点优先 line buffer，蝶形只做后续 PPA 对照 | 蝶形 zero-skip 原创 |

核心变化不是给已有名词改名，而是把两个确定关系：

```text
H67: temporal peer
Local5: spatial stencil neighbor
```

变成无需在线搜索的精确锚点，并复用同一 residual 算术。是否足以构成论文
创新仍取决于后续同约束 PPA、联合 trace replay 和系统收益。

## 2. 整数等价

脚本：

- `scripts/dual_line_delta_reference.py`
- `scripts/test_dual_line_delta_dse.py`

### 2.1 H67

每个 alpha-XNOR lane 的 `raw16` 贡献为：

| Q | K | raw16 |
|---:|---:|---:|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 64 |

只在：

```text
M = (Q0 XOR Q1) OR (K0 XOR K1)
```

指示的 lane 上累加新旧贡献差。Motion 项
`16*popcount(K0 XOR K1)` 对两个时间 score 相同，不进入 residual。

### 2.2 Local5

```text
A0 = 65*n11_self + 32 - q1 - k1_self
U  = (~Kself) & Kneighbor
D  = Kself & (~Kneighbor)
Delta = 65*(|Q&U|-|Q&D|) - (|U|-|D|)
score_q7 = RNE((A0 + Delta) / 16)
```

该式与同一 alpha-XNOR lane delta 完全一致。Local5 只需令
`Qold=Qnew=Q`。

### 2.3 验证结果

- H67 单活动 lane 的 `Q0/K0/Q1/K1` 全 16 种 raw16/Q7：PASS；
- Local5 单活动 lane 全 8 种：PASS；
- 两线各 500,000 个 32-bit 随机向量：raw16/Q7 零不一致；
- Python DSE/队列性质测试：7/7 PASS。

证据等级是 `[推导+随机整数]`，尚不是冻结部署 bit-vector trace 回放。

## 3. H67 ordered DSE

输入是 profile100 的 54,432,000 个逐 pair ordered trace。基线定义：

- Direct32：单个 32-lane engine 顺序算 T0、T1，理想周期
  `2*pairs`；
- Direct32x2：两个 32-lane engine 理想并行，周期 `pairs`；
- TARE：32-lane anchor 每拍产生一项，W-lane residual backend 消费
  `ceil(update/W)` 个 quantum；
- dense fallback：`update>T` 时 anchor 追加一拍 direct T1，该拍允许
  residual backend 继续排空一个 quantum。

所有结果都是 cycle model；没有计入 detector、SRAM、RNE、控制、频率和功耗。

### 3.1 关键候选

| 候选 | fallback | vs Direct32 | 吞吐/vs Direct32x2 | lane 面积效率/vs Direct32x2 | FIFO p99/max |
|---|---:|---:|---:|---:|---:|
| W2/T4 | 5.9838% | 1.8871x | 0.9435x | 1.7761x | 58/119 |
| **W4/T4** | **5.9838%** | **1.8871x** | **0.9435x** | **1.6774x** | **0/0** |
| W8/T8 | 1.2120% | 1.9761x | 0.9880x | 1.5808x | 0/0 |

`W4/T4` 晋级叶 RTL 的理由不是绝对吞吐最高，而是：

1. 相对理想双 Direct32 只低约 5.65% 吞吐；
2. lane 数从 64 降到 36；
3. `W=T` 保证每个 sparse 项最多一个 backend quantum；
4. dense 项额外占 anchor 一拍并精确回放；
5. 在同拍旁路、无 SRAM 等待、无下游反压、backend 每拍稳定消费的模型内，
   不需要跨 pair residual work FIFO。

最后一条不等于“硬件不需要缓冲”。RTL 仍需 output register/skid buffer，
并在集成后重新计算反压下的 FIFO。

### 3.2 辅助机制边界

- Motion-zero：83.2087%，只是时钟门控机会；
- score equal：98.6949%，只允许合并 SCS class-count commit；
- SCS transaction-count 减少：49.3475%，不是已实现周期收益；
- TTB4 empty-delta bundle：61.1272%，描述符从 54,432,000 打包成
  14,112,000；不是 payload 或周期减少；
- 以上数字不能与 1.8871x 相乘，必须做联合 ordered replay 消融。

## 4. Local5 pre-G0 DSE

Local5 当前只有 per-record 四方向 histogram，没有 query 内联合顺序。因此：

- 能报告 residual lane work、service quantum 和 fallback edge 数；
- `max(producer, backend)` 是周期守恒下界；
- 用基线周期除以上述下界得到的是**理想 speedup 上界**；
- 不能报告真实 FIFO、burst、p95/p99、SRAM conflict 或 makespan。

| 候选 | fallback edge | serial工作量speedup | decoupled理想speedup上界 |
|---|---:|---:|---:|
| W2/T4 | 5.3990% | 待 ordered 联合复算 | 3.8212x |
| W4/T4 | 5.3990% | 待 ordered 联合复算 | 3.8216x |
| W8/T8 | 1.7801% | 待 ordered 联合复算 | 4.1264x |

这些值只说明 Local5 值得继续 profile，不允许据此冻结 W/T，也不允许与 H67
数字横向比较。

## 5. 共享叶 RTL

### 5.1 `delta_bounded_classifier`

路径：`rtl_delta/delta_bounded_classifier.sv`

接口合同：

```text
in:  tag + 32-bit delta_mask + opaque payload
out: tag + original mask + payload
     + kind {ZERO, SPARSE, DENSE}
     + popcount
     + at most four ascending lane IDs
```

特性：

- 固定 `32->4`，对应当前获准晋级的 W4/T4；
- 每个输入事务都有输出，zero 不静默丢弃；
- dense 保留完整 mask 和 payload，供 exact direct replay；
- 一项输出寄存器支持 ready/valid backpressure；
- stall 时所有输出稳定；
- 不包含 H67/Local5 特有控制，因此两线复用。

### 5.2 `alpha_xnor_delta4`

路径：`rtl_delta/alpha_xnor_delta4.sv`

四个 lane ID 读取新旧 Q/K，输出 signed `delta_raw16`。范围为：

```text
4 * [-64, +64] = [-256, +256]
```

10-bit signed 输出足够。最终 RNE 不在该叶执行，避免 anchor score 先舍入后再
加舍入 delta。

### 5.3 工具结果

入口：

```bash
hw_autoresearch_nts07/sim_delta/run_delta_bounded_classifier_checks.sh
```

| 检查 | classifier | delta4 |
|---|---:|---:|
| Icarus 功能 | 2,326 transactions PASS | 20,016 checks PASS |
| Verilator 功能/断言 | PASS | PASS |
| Verilator lint | 0 warning/error | 0 warning/error |
| Erie RTL lint | 0 warning/error | 0 warning/error |
| Yosys check | 0 problem | 0 problem |

另外完成：

- 独立随机 ready/valid scoreboard：Icarus/Verilator 均为
  `3,000 accepted = 3,000 emitted`；
- classifier 与 delta4 联合等价：H67/Local5 共 2,112 组；
- classifier 覆盖 popcount 0..32、全部 32 个 single-bit、偏置 sparse
  和随机 dense；
- zero-forward、sparse residual、dense replay **路由合同**三类均逐事务检查；
- lane ID 的 mask 归属、前缀 valid、严格递增和两两唯一均有断言。

真实 H67 位级回放：

- 来源：`results/h67_real_bit_trace_20260717`；
- 覆盖 sample0、S0-S3 的 B0、每 stage 一个窗口；
- 共 3,645 个时间 pair；
- Python raw/Q7 不一致：0/0；
- Icarus/Verilator RTL：`ZERO=2168`、`SPARSE=914`、`DENSE=563`，
  逐项 PASS；
- 该小样本 fallback 为 15.4458%，只用于 bit-exact，不替代 profile100
  的 5.9838% 总体 DSE。

开放 Yosys 未映射统计：

| 模块 | generic cells | 说明 |
|---|---:|---|
| classifier | 619 | 含 popcount、32->4 priority extractor、payload output register |
| delta4 | 75 | 含动态 bit select、四路 lane delta 与 signed reduction |

这些 cell 数不等于面积。当前 priority extractor 仍有 339 个 `$mux`，是下一轮
分段 priority、两级 prefix 或蝶形 compactor PPA 消融的主要对象。

## 6. 独立评审

独立 DATE 审稿代理结论为 **CONDITIONAL PASS**：

- 允许 H67 W4/T4 进入共享叶 RTL；
- 不允许把模型数字直接写成 RTL/PPA speedup；
- 指出并已修正 Local5 speedup 上下界方向错误；
- 要求补 Direct32x2 对照，已补；
- 确认 reflected queue 与 dense replay drain 在当前条件下成立；
- 要求把 queue-free 限定为模型条件，已补；
- 要求 H67 16 种单 lane raw 等价和 W=T 性质测试，已补；
- 冻结集成前仍缺真实 hardware-order bit-vector、位宽/溢出和反压合同。

第二轮独立 RTL 评审仍为 **CONDITIONAL PASS**，指出：

- Local5 当时只比较最终 Q7，现已增加 raw16 零不一致；
- 原 TB 没有连续 refill 与独立随机反压，现已增加 3,000 笔 scoreboard；
- 原 sparse 随机覆盖不足，现已补 0..32、single-bit 和偏置 sparse；
- delta4 依赖 lane ID 唯一，现已加入独立及跨叶断言；
- 原来没有跨叶路径，现已补 2,112 组 H67/Local5 联合等价。

因此本地叶 RTL 阻塞项已关闭，真实 hardware-order Q/K bit-vector 也完成了
四 stage B0 单窗口回放。尚未关闭的是多样本、多窗口、全部 12 个 block 的
外推覆盖。

## 7. 下一阶段

按证据优先级执行：

1. 扩大 H67 真实 `Q0/K0/Q1/K1` 回放：当前四 stage B0 单窗口已通过，
   后续补多样本、多窗口和全部 12 个 block；
2. 建立同接口的 Direct32、Direct32x2、TARE-4 composite top，加入
   dense replay arbitration 和真实 backpressure；
3. 比较 linear priority、分段 prefix、蝶形/Benes compactor 的
   Fmax、面积、toggle 和能量，不预设蝶形胜出；
4. Local5 补 G0/G1 后 ordered STT：同 query 四方向联合 delta、边界 halo、
   bank 地址和下游反压；
5. 三个候选在同一 SDC、同一 SRAM macro 规则下做 DC/STA/SAIF；
6. 只有联合 replay 仍保持 bit-exact 且 energy/frame、面积归一吞吐有净收益，
   才把 TARE 写入 DATE 主贡献。
