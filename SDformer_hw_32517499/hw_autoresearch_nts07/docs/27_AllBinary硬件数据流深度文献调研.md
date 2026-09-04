# AllBinary 硬件数据流深度文献调研

## 1. 总体结论

all-binary NTS/H60 的硬件设计不应从“通用 Transformer accelerator”或“经典 LIF SNN accelerator”直接套模块。更合适的迁移路线是：

```text
软件语义不变
-> 把执行粒度重排成 token x time x head x channel-bitword
-> 用 Token-Time Bundle 做存储/调度/跳过
-> 用 popcount/shift/INT8 gate 实现 H60
-> 把少量非二值算子集中到 scale/descriptor/projection 后端
```

最有价值的设计范式：

- Bishop 的 Token-Time Bundle、stratifier、error-constrained pruning。
- FlashAttention/FLAT 的不物化 attention map、row-wise streaming reduction。
- BSA 的 Shiftmax 和 polarity-aware score。
- Spike-driven Transformer 的 binary spike communication 与 mask/add。
- SNN accelerator 的 time-batching、spike compression、state-local dataflow。
- Binary Transformer accelerator 的 scale folding、binary matmul 和少量非二值算子集中处理。

对当前项目最关键的限制：

- H60 是 token-wise selector，不是完整 `N x N` attention matrix。
- 当前 H60 patched forward 忽略 shifted-window mask。
- 当前 H60 的 `attn_sn` 输出没有进入 projection。
- 当前 ATLIF wrapper 是 PSN temporal mixer + threshold emitter，不是 LIF membrane recurrence。
- `value_mode=threshold` 表示 K path 不是天然纯 1-bit，1-bit event + scale folding 需要验证。

## 2. 文献矩阵

| 论文/工作 | 核心硬件 idea | 可迁移到本项目的方式 | 风险/不适配点 |
|---|---|---|---|
| FlashAttention, 2022, https://arxiv.org/abs/2205.14135 | IO-aware exact attention，分块读 Q/K/V，在线 max/sum softmax，不物化 `N x N` attention。 | 对 H60 做 window-token streaming：score、Shiftmax、gated-K 串流融合，不存完整 gate 中间矩阵。 | H60 不是标准 softmax attention；只能迁移 IO-aware streaming，不迁移 QK^T V 公式。 |
| FlashAttention-2, 2023, https://arxiv.org/abs/2307.08691 | 更细 work partition，减少非矩阵操作和 shared-memory 交换。 | H60 每个 window 只有 162 token，可按 head group/token group 切分，减少 PE 空转。 | GPU thread-block 经验不能直接变 ASIC/FPGA 架构，需要重做 SRAM/PE 比例。 |
| FLAT, 2023, https://dl.acm.org/doi/10.1145/3575693.3575747 | Fused Logit and Attend Tiling，融合 logit 计算和 attend，降低中间矩阵访存。 | H60 可融合 `TX/SC score -> Shiftmax -> K*gate`，只保留 row/token 统计和输出累加。 | H60 是 token-wise gate，不是 matrix attend；数据流更简单，但不能照搬 FLAT 的矩阵 tile。 |
| MAS-Attention, 2025, https://arxiv.org/abs/2411.17720 | 矩阵单元和向量单元双流并行，覆盖 cache latency。 | bit-popcount score engine 和 Shiftmax/vector gate engine 半同步流水。 | 需要合理配置 bit-PE 与 vector-PE 比例，否则向量 gate 可能成为瓶颈。 |
| ITA, 2023, https://github.com/pulp-platform/ITA | 8-bit MHA 专用硬件，整数 softmax，streaming 减少数据移动。 | 参考整数 Shiftmax/归一化和 row-wise 控制，用于 INT8 score/gate 部署。 | ITA 面向普通 MHA，不是事件 H60；binary/event 存储收益需另算。 |
| A3, 2020, https://arxiv.org/abs/2002.10941 | 把 attention 看成 content-based search，用近似候选筛选降低计算。 | 可作为未来 H60 top-k/token pruning 的候选过滤器。 | 会改变 token 覆盖范围；没有重训/校准前不能进主线。 |
| ELSA, 2021, https://dl.acm.org/doi/10.1109/ISCA52012.2021.00060 | 低影响关系过滤 + 候选 attention 精算。 | 可用于事件 token 很稀疏时的两阶段 H60：cheap popcount 先筛，再精算 score。 | H60 当前没有 sparse edge list 语义；动态筛选需要精度验证。 |
| SpAtten, 2021, https://arxiv.org/abs/2012.09852 | 动态 token/head pruning、top-k engine、progressive quantization。 | 可迁移 top-k engine 作为 profiling/可选 token/head 调度器。 | token pruning 会删除信息，不能默认保持语义。 |
| Sanger, 2021, https://dl.acm.org/doi/10.1145/3466752.3480125 | 预测 attention sparsity，并用可重构架构执行 sparse attention。 | 对 binary Q/K 先预测非零 bundle，再重排非零任务提升 PE 利用率。 | 索引与负载均衡开销可能超过 H60 本身收益。 |
| Energon, 2021, https://arxiv.org/abs/2110.09310 | Mix-precision multi-round filtering，低位多轮筛重要 Q-K pair。 | 可借鉴 bitword 多轮 early-reject。 | all-binary 信息量低，误筛风险更高。 |
| DOTA, 2022, https://dl.acm.org/doi/10.1145/3503222.3507738 | 训练 detector 找 sparse attention graph，token-parallel dataflow 和乱序执行。 | 可借鉴 sparse task queue 和 out-of-order reduce。 | detector 是模型改动；当前主线只能借鉴调度。 |
| LeOPArd, 2022, https://arxiv.org/abs/2204.03227 | 学习 runtime pruning threshold，联合优化阈值和权重。 | 如果要硬件化 token pruning，阈值应由训练得到，不应手工设。 | 需要训练闭环，跨场景稳定性不确定。 |
| SPRINT, 2022, https://arxiv.org/abs/2209.00606 | ReRAM 近存近似 score，模拟阈值剪枝，再数字精算。 | 可迁移“存储旁粗筛、数字 PE 精算”的二级结构。 | 模拟 ReRAM 不适合当前最小 RTL；剪枝近似需验证。 |
| CPSAA, 2022/2024, https://arxiv.org/abs/2210.06696 | crossbar PIM 支持 sparse attention SDDMM/SpMM。 | 若后续 H60 引入稀疏 edge list，可把 Q/K score 靠近 SRAM/CIM bank。 | H60 当前不是 SDDMM/SpMM 矩阵 attention。 |
| ASADI, 2024, https://www.comp.nus.edu.sg/~tulika/HPCA24.pdf | 利用 sparse attention diagonal locality，DIA 格式和 in-situ computing。 | 如果 optical-flow window 中 token 关系呈空间局部性，可用 block/diagonal layout 加速。 | 当前 H60 是同 token selector，不存在 token-token diagonal attention。 |
| SADIMM, 2025, https://www.comp.nus.edu.sg/~tulika/TC25.pdf | DIMM-based near-memory sparse attention，处理大模型 sparse attention 随机访存。 | 可借鉴 memory-bank 任务分配和负载均衡。 | 边缘 H60 window 较小，DIMM/NMP 结构过重。 |
| BETA, 2024, https://arxiv.org/html/2401.11851v2 | binary Transformer accelerator，通过 computation-flow abstraction 减少全精度操作。 | 高相关：把 binary matmul、scale、residual、norm 分层调度，集中处理少量非二值算子。 | 本项目有 SNN 时间维、Swin window、U-Net decoder，不能只看 BERT/Vision Transformer。 |
| COBRA, 2025, https://arxiv.org/html/2504.16269v1 | binary Transformer FPGA，支持 `-1/0/+1` attention 优化。 | 可参考 binary/ternary 编码、popcount attention block。 | all-binary 主线不应引入 ternary 主 datapath。 |
| BiBERT, 2022, https://arxiv.org/abs/2203.06390 | fully binarized BERT，Bi-Attention 缓解 MHA 二值化信息损失。 | 给 binary attention 的训练/蒸馏提供参考。 | 算法论文，不解决事件时间维调度。 |
| BiT, 2022, https://arxiv.org/abs/2205.13016 | 二值权重/激活、多阶段蒸馏和 elastic binary activation。 | 若未来把 projection/MLP 也强二值化，可参考蒸馏策略。 | 当前主线只证明了 ATLIF/H60 all-binary，不等于全网络权重二值。 |
| Spike-driven Transformer, 2023, https://arxiv.org/html/2307.01694 | SDSA 使用 spike-form Q/K/V，用 mask 和加法替代乘法，事件驱动零输入不触发。 | 支撑 all-binary event communication 和 sparse add 叙事。 | SDSA 是不同 attention 结构，不能替换 H60。 |
| QKFormer, 2024, https://arxiv.org/abs/2403.16552 | spike-form Q-K attention，线性复杂度，层级 token 表示。 | H60 源自 QK selector 系列，可借鉴 token-wise Q/K gate 解释。 | 当前 H60 已改为 TX/SC no-carrier，不能按原 QKFormer carrier 讲。 |
| BESTformer, 2025, https://arxiv.org/html/2501.05904v1 | Binary Event-Driven Spiking Transformer，二值化和事件驱动结合。 | 支撑 all-binary 作为硬件主线的合理性。 | 偏算法，不提供 SDformerFlow 硬件数据流。 |
| SpinalFlow, 2020, https://ieeexplore.ieee.org/document/9138926/ | SNN 专用数据流，压缩/排序 spike 序列并复用膜电位。 | 可借鉴 timestamp-sorted spike stream 和 state-stationary 思路。 | 当前 ATLIF wrapper 不是膜电位递推，不能直接讲 Vmem stationary。 |
| SATO, 2022, https://dl.acm.org/doi/10.1145/3489517.3530592 | temporal-parallel SNN accelerator，并行累加所有 time-step 膜电位。 | 可借鉴 time-batching，展开 H60 的 `T_window=2` 或全局 `T=10`。 | 完全复制时间维会增加面积。 |
| LoAS, 2024, https://arxiv.org/html/2407.14073v1 | dual-sparse SNN 的 temporal-parallel dataflow，spike 压缩保证连续访存。 | 可迁移 spike bitpack + contiguous memory + sparse inner-join。 | 若权重/中间值不是稀疏二值，控制开销会上升。 |
| SpiDR, 2024, https://arxiv.org/html/2411.02854v1 | 数字 CIM SNN，reconfigurable modes，减少 weight/Vmem 数据移动。 | 可借鉴近存 weight/state 与模式切换。 | CIM 不适合最小 RTL 起步。 |
| Bishop, 2025, https://dl.acm.org/doi/10.1145/3695053.3731063 | Token-Time Bundle，把多个 token 和多个 time-step 打包，异构 core 处理稀疏 workload。 | 最直接：把 H60 window 组织成 `head x token x time x bitword` bundle，做空 bundle fast path 和 dense/sparse 分流。 | Bishop 的 error-constrained pruning 会改语义；第一版只迁移 TTB 和 stratifier。 |
| Spiking Transformer Hardware Accelerators in 3D Integration, 2024, https://dl.acm.org/doi/10.1145/3676536.3676826 | 3D memory-on-logic/logic-on-logic，减少 spiking transformer 数据移动。 | 可借鉴层次划分：权重/descriptor SRAM 靠近 PE，event SRAM 靠近 score engine。 | 3D 工艺门槛高，DATE 主线可作为未来扩展。 |
| Hardware Efficient Accelerator for Spiking Transformer, 2025, https://arxiv.org/html/2503.19643v1 | tick-batching dataflow、time-step reconfigurable neuron、IAND residual。 | tick-batching 对固定 `T=10` 和 `T_window=2` 有参考价值。 | IAND 替代 ADD residual 是模型语义改动，不能迁移到当前主线。 |
| Loihi, 2018, https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf | 异步 manycore、片上 SRAM、spike message routing、可编程学习规则。 | 可借鉴 event packet 和 core-local memory。 | 通用 neuromorphic 路由对 windowed Transformer 可能过重。 |
| SCNN, 2017, https://arxiv.org/abs/1708.04485 | 压缩域稀疏计算，避免先解压再算。 | event SRAM 应保持 bitpack/压缩，H60 score engine 直接消费 bitword。 | CNN 稀疏复用与 Swin/H60 不完全一致。 |
| Eyeriss v2, 2018, https://arxiv.org/abs/1807.07928 | 层级 NoC 和可重构 dataflow 支持稀疏 DNN。 | 可借鉴多级 buffer、NoC 和 PE 映射方法。 | attention score/gate 的向量归一化不是 CNN 卷积。 |

## 3. 对 UniBin-H60 的设计启发

### 3.1 Token-Time Bundle 是主存储粒度

推荐 bundle 定义：

```text
bundle_id = (stage, block, window_id, head_group)
payload:
  Q_event bitpack: [162 tokens, 32 bits/head_dim]
  K_event bitpack: [162 tokens, 32 bits/head_dim]
  K_value descriptor or threshold-valued lane
  valid token/time mask
  density metadata
```

这样同时匹配：

- H60 的 window token 数 `162`；
- head_dim 固定 `32`；
- all-binary Q/K 的 bitword popcount；
- Bishop 式 TTB 的 spatiotemporal reuse。

### 3.2 不物化 gate 矩阵

H60 没有 `N x N` attention matrix，只有 token gate：

```text
score[162] -> Shiftmax[162] -> K[162,32] * gate[162]
```

因此不需要传统 attention map SRAM。只需要：

- score vector buffer；
- max/sum/denom reduction buffer；
- gate vector buffer；
- gated-K output buffer。

### 3.3 稀疏跳过应从“全零事件”开始

推荐跳过层级：

1. `Q_word == 0 && K_word == 0`：走 same_zero constant fast path，而不是直接输出零。
2. `Q_token_empty && K_token_empty`：常数 score 快速生成。
3. `bundle_density < threshold`：只做低功耗 sparse score path。
4. dynamic token pruning：暂不进主线，除非补训练/校准验证。

### 3.4 非二值算子集中化

当前主线虽然叫 all-binary，但严格讲：

- ATLIF 输出是 binary/threshold-valued spike；
- Q/K event 可 bitpack；
- TX/SC 可 popcount；
- score/gate 是 INT8/定点；
- projection、MLP、decoder conv 仍需要定点 MAC；
- norm/scale 需要 descriptor。

因此硬件不应宣称全芯片只做 bit operation。更稳的说法：

```text
event-dominant dataflow with centralized low-bit scalar lanes
```

## 4. DATE 可写贡献点

1. **UniBin-H60 Token-Time Bundle**
   - 针对 windowed event-flow transformer 的 bundle，而非通用 ViT token bundle。

2. **Carrier-Free H60 Score Streaming**
   - 保持软件 `TX + mu*SC -> Shiftmax -> K*gate` 语义，不构造 `N x N` attention。

3. **Same-Zero Aware Sparse Fast Path**
   - 空事件不直接跳零，而是保留 H60 的 same-zero 常数贡献，避免语义错误。

4. **Threshold Descriptor Folding**
   - 用 1-bit event SRAM + threshold/scale descriptor 近似或等价替代 threshold-valued spike 存储。

5. **U-Net Skip Aware Event Buffering**
   - 明确支持 4 级 encoder skip 和 3 级 prediction skip，不把网络误画成纯 encoder。

## 5. 不建议直接迁移的 idea

- IAND residual 替代 ADD residual：会改变当前软件 block 语义。
- shifted-window mask 硬件化：当前 H60 patched forward 忽略 mask。
- 标准 QK^T V attention core：H60 是 token-wise selector。
- dynamic token pruning/top-k：需要训练或校准验证。
- 在线 adaptive LIF membrane：当前 ATLIF wrapper 不是这个语义。

## 6. 后续验证优先级

1. H60 fixed-point golden vector。
2. ATLIF PSN temporal mixer golden vector。
3. threshold-valued K folding 验证。
4. same-zero fast path 验证。
5. decoder skip/prediction buffer profiling。
6. token/bundle density histogram，决定是否需要 stratifier。
