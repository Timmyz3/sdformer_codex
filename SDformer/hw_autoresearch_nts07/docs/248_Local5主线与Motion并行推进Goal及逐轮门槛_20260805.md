# Local5 主线与 Motion 并行推进 Goal 及逐轮门槛

## 1. 总目标

保持 Local5 和 Motion 两条硬件线并行研究，当前以 Local5 为主线，但不冻结或放弃
Motion。目标不是增加 RTL 数量，而是让两条线分别形成可审计的算法到硬件闭环，并
筛选出至少一条具备 DATE 主文竞争力的系统架构故事。

- **Local5 主线**：优先补系统完整度，闭合定点 attention、关系转置、source-major
  term、TCFM5 和 accumulator 的端到端执行链及 12-block 时间复用。
- **Motion 并行线**：继续补真实多样本证据、维护 SCS/NMF/DCTF 回归，并允许基于
  Motion 特有时间差分和门码统计提出、筛选和实现新机制。
- **共同原则**：新机制必须来自 workload 特征和瓶颈证据，不能因为概念好听直接
  写 RTL；常规 banking、ping-pong、FIFO、定点化不能单独作为 DATE 贡献。

“DATE best-paper 水平”只能作为质量门槛，不能预先保证录用。判断依据必须包括
系统闭环、原创架构抽象、公平强基线、真实 workload、bit-exact 验证和同约束 PPA。

## 2. 每轮执行纪律

每轮只关闭一个最高优先级的**实现或证据缺口**。另一条线可以并行做不改 RTL 的
profiling、文献差分和候选建模，但不同时启动第二个大规模实现，以免形成多个未闭环
机制。

每轮固定输出：

1. 完成了什么；
2. 使用了哪些本机可复现证据；
3. 出现了哪些负结果；
4. DATE 独立评审分数和拒稿理由；
5. 下一轮唯一最高优先级缺口。

证据标签严格使用：

| 标签 | 含义 |
|---|---|
| `[rtl]` | 本机 RTL 仿真、SVA、miter 或综合可读证据 |
| `[prof]` | 真实网络、真实 trace 的统计证据 |
| `[模型]` | 周期、存储、能量或调度模型，尚非 RTL/物理结果 |
| `[open-pnr][代理]` | 开放库 OpenROAD 物理趋势，不是目标工艺签核 |
| `[待验证]` | 仅为候选机制或论文假设 |

不得把未集成的 FCSR 与 TCFM5 描述成端到端闭环，不得把 Yosys/OpenROAD 结果称为
ASIC PPA，不得把外部 agent、远端机器或未落盘结果计入本机完成度。

## 3. Local5 主线闭环

### L0：算法与部署语义冻结

需要固定并审计：

- theta 的来源、每 block/stage 配置和训练/推理一致性；
- score 位宽、符号、截断、饱和和舍入顺序；
- Shiftmax5 的 LUT、invalid-candidate mask 和 hardware-order；
- gate 输出格式及零值语义；
- relation 的五方向定义、边界和坐标合同；
- full-resolution `15x15x2 = 450` token 合同；
- 12 个 attention block 的参数清单和执行图。

算法 agent 的 full-resolution exact 结果未落盘前，相关结论保持 `[待验证]`，但接口
审计和 RTL 参数表可以先完成。

### L1：端到端数据流闭合

目标数据流：

```text
Q/K event
  -> Local5 score
  -> Shiftmax5 + invalid mask
  -> destination-major relation
  -> relation transpose/frontier
  -> source-major descriptor
  -> gate/lane term builder
  -> TCFM5 bank mapping
  -> source-resident or direct accumulator
  -> Acc32 readback
```

闭环标准不是“叶模块都存在”，而是同一顶层中所有 valid/ready、frame/block/window
边界、异常传播和数值格式均已连接，并通过端到端金参考。

### L2：12-block 时间复用

采用 descriptor 驱动的一套物理 attention-to-projection pipeline 服务 12 个 block。
需要定义：

- block/stage/head/window descriptor；
- 参数 SRAM/LUT/权重版本；
- start、seal、drain、commit、done 状态；
- block 间 state reset 与 ATLIF/skip 接口边界；
- 双缓冲是否必要，以及其收益是否超过控制和存储成本。

时间复用本身不是创新，贡献必须来自它与 Local5 关系拓扑、门码低基数或确定性稀疏
调度的协同机制。

### L3：系统验证

最低验证矩阵：

- 真实 trace 多 sample、多 window、多 stage；
- 正常流、输入随机空隙、下游随机反压；
- block/window 边界和 reset/restart；
- invalid mask、全零、边界像素、最大 gate、累加溢出；
- score/Shiftmax5 中间 miter；
- relation transpose multiset miter；
- 端到端 Acc32 miter；
- functional coverage 和 SVA 分类统计。

## 4. Motion 并行线

Motion 不是只维护旧回归。其并行任务分为三层：

1. **证据扩展**：将 SCS/NMF/DCTF 从单窗口扩到多 sample、多窗口、四 stage，报告
   mean/p50/p95/p99、最差样本和 stall 原因；
2. **现有机制闭环**：保持 score/SCS/gated-K/term/accumulator 的 bit-exact、随机
   反压和 Acc32 回归，补 `T=450` 地址、计数和存储合同；
3. **新机制筛选**：允许继续探索 Motion 特有的 temporal quotient、exact delta
   reuse、跨窗口 gate-class reuse、TTB/STT 调度或其他架构，但必须经过准入门槛。

## 5. 新 Idea 准入门槛

Local5 与 Motion 的候选机制均依次通过以下五关：

1. **Workload 关**：真实 trace 中存在稳定结构特征，并报告分布而非单个均值；
2. **差分关**：明确和 Bishop、PHI、Prosperity、FireFly-T、Sanger、CICC 2026
   光流加速器等工作的 borrow/not-borrow/difference；
3. **收益上界关**：先用保守模型证明 cycle、SRAM traffic、NoC traffic 或能量存在
   足够上界，且包含 metadata、build、drain 和 fallback 开销；
4. **最小 RTL 关**：只实现能验证核心因果的一条路径，与等带宽、等 lane、等宏
   基线比较；
5. **物理关**：通过同 SDC、同宏、同 outline 的面积、时序、功耗代理，收益不成立
   就降级或淘汰。

### 5.1 可继续筛选的 Local5 候选

- `[待验证]` **Relation-frontier streaming**：把 Shiftmax5 结果直接转为可消费的
  source frontier，减少完整 destination-major relation 的写后读和转置等待；
- `[待验证]` **Gate-stationary cross-block execution**：评估相同 gate/lane 在相邻
  block 或 tile 中的确定性复用是否足以摊薄目录和权重读取；
- `[待验证]` **Topology-quotient scheduling**：利用五色拓扑等价类而不是普通 token
  顺序调度，目标是减少 bank conflict 或 barrier，而不是重新命名 banking；
- `[待验证]` **FCSR/TCFM5 融合**：只有在同一顶层、同一 trace 和端到端 Acc32
  等价后才能晋级，当前不能写作已闭环贡献。

### 5.2 可继续筛选的 Motion 候选

- `[待验证]` **Exact temporal quotient reuse**：只在数学上能证明 score/gate/term
  不变时复用，禁止以 firing 相似度代替等价；
- `[待验证]` **TTB/STT 自适应发射**：借鉴事件 bundle 和时空 tile，但需将 Motion
  的 empty、K-zero、motion-zero 分类本土化，并计入打包、解包和负载失衡；
- `[待验证]` **Cross-window gate-class persistence**：先 profile 生命周期和失效条件，
  再决定做目录驻留还是完全不做；
- `[待验证]` **Prosperity-style exact product reuse**：必须在真实 hit rate、标签成本、
  容量和端口模型下仍优于直接计算。

## 6. 双线主线选择

当前 Local5 是主线，原因是算法竞争力和 full-resolution 部署评估仍在推进，同时它的
系统 RTL 缺口明确、可关闭。Motion 保持成熟备选，因为其 H67 数值与 RTL-exact
证据更完整，且 SCS/NMF/DCTF 已有较深实现。

最终选择不按概念新颖度拍板，而按同一张表：

- valid825 AEE/AAE 与 spike/energy；
- full-resolution exact 精度损失；
- 端到端周期、吞吐和 tail latency；
- SRAM/NoC/compute 能量分账；
- 面积、频率、功耗、FPS/mm2、FPS/W、EDP；
- RTL/trace/物理证据完整度；
- 与最近工作的可辩护差异。

## 7. 近期逐轮顺序

1. 收口已启动的 OUT32 同宏 Direct/Issue/DS 物理代理并独立评审；
2. 冻结并审计 Local5 theta、定点语义和跨模块接口合同；
3. Motion 多样本、多窗口 profile 与现有回归扩展；
4. Local5 12-block descriptor 调度；
5. Local5 score/Shiftmax5 到 relation transpose 的端到端接入；
6. Motion 新机制从 profile 上界中只晋级一个；
7. Local5 source-major term 到 TCFM5/accumulator 的系统 miter；
8. 两线同口径系统/物理表后再冻结投稿主线。

以上顺序可因 full-resolution 算法结果调整，但任何调整都必须在文档中记录理由和证据。
