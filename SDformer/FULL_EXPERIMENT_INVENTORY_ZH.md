# SDformerFlow 实验全景清单

生成时间：2026-05-11 | 项目：SDformerFlow 稀疏脉冲神经网络光流估计

---

## 一、项目概览

**目标**：在SDformerFlow（PSN并行脉冲神经元 + Swin Transformer + DSEC事件光流）基础上，通过神经元优化、稀疏门控、体素化改进和注意力替换，实现精度基本不降前提下的SOPs（突触操作数）大幅降低，面向硬件加速器投稿。

**基线指标**：PSN baseline, AEE=1.5848, AAE=7.5012, firing_rate=0.08496, SOPs=3.6219G (valid40, 105个脉冲神经元层)

**当前最佳结果**：G1 部分稀疏门控, AEE=1.6056 (+1.3%), SOPs=2.7134G (-25.1%)

---

## 二、环境搭建与Baseline训练（实验 #1-#18）

| # | 日期 | 类型 | 状态 | 关键结果 |
|---|------|------|------|---------|
| 1 | 04-21 | 单序列smoke | 失败 | CuPy backend缺失 |
| 2 | 04-21 | 单序列smoke | **成功** | torch backend跑通全链路 |
| 3 | 04-21 | 全量探测 | 探测 | 路径验证 |
| 4 | 04-21 | 全量训练 | 失败 | AMP NaN (AddmmBackward0) |
| 5 | 04-21 | 全量训练noAMP | 中止 | 速度太慢(1.28it/s) |
| 6 | 04-21 | 全量AMP+lr5e-5 | 手动停止 | 稳定跑到epoch12, NaN风险降低 |
| 7 | 04-21 | 续训epoch12→ | 失败 | epoch19 NaN (ConvolutionBackward0) |
| 8 | 04-22 | 续训epoch18→ | 失败 | epoch50 NaN (AddmmBackward0, 非固定位置) |
| 9 | 04-23 | 续训epoch45→ | 中断 | epoch54终端断开(非NaN) |
| 10 | 04-23 | 续训epoch54→60 | **成功** | **首次完成全量baseline训练** |
| 11 | 04-23 | 推理评估 | **成功** | AEE=2.39, AAE=12.01 (valid全部样本) |
| 12 | 04-24 | 后端benchmark | **成功** | torch略快于cupy (0.749 vs 0.732 samples/s) |
| 13 | 04-24 | 吞吐调优 | **成功** | bs4/bs8为甜点区 (7.53 samples/s @bs4) |
| 14 | 04-24 | bs4从头训练 | 中止 | loss高于bs1 baseline, 不如继续微调 |
| 15 | 04-24 | bs4续训(关MLflow) | 失败 | 保存逻辑仍调用MLflow |
| 16 | 04-24 | bs4续训(本地ckpt) | **成功** | 20epoch完成, epoch15 ckpt可用 |
| 17 | 04-24 | bs4断点续训→epoch60 | **成功** | **最佳baseline: checkpoint_epoch59.pth** |
| 18 | 04-25 | 推理对比 | **成功** | bs4 epoch59 AEE=1.33 vs 原始AEE=2.39 |

**Baseline最终权重**：`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`
**valid40指标**：AEE=1.5848, SOPs=3.6219G

---

## 三、神经元实验 — 全部已尝试（E/F/G/H系列）

### 3.1 关键发现：核心洞察

1. **G1是唯一成功策略**：25% SOP减少+仅1.3% AEE增加
2. **全替换是死胡同**：15个全替换实验0成功
3. **包装优于替换**：在PSN外面加机制，不破坏PSN内部时间混合
4. **部分插入 > 全替换**：选对节点 + 最小改动是正确的
5. **开源移植 ≠ 直接可用**：所有移植的开源神经元在SDformerFlow上都失败

### 3.2 完全神经元替换（E系列 — 10个实验）

将模型中所有PSN神经元替换为一种新类型。**全部失败，无一能在精度和稀疏性上同时超越PSN基线。**

| ID | 神经元 | 论文来源 | AEE | SOPs | 结论 |
|----|--------|---------|-----|------|------|
| **E0** | PSN | SDFormerFlow原生 | **1.5848** | **3.6219G** | 基线 |
| E1 | Simple SN | 自建(SpikingJelly LIF) | 仅smoke | — | 脚手架，未完整评估 |
| **E2** | ATLIF | Activity Pruning SNN (NeurIPS 2024) | 3.76~2.51 | 2.87G~5.21G | 6个分支探索。低SOP分支(-21%)但AEE×2.4；高精度分支AEE=2.51但SOPs=5.21G。阈值增长过度剪枝或不够 |
| **E3** | LMHT | LM-HT SNN (NeurIPS 2024) | 2.56~2.73 | 9.7G (×2.7) | 多层级输出反增SOPs，缺直接推理重参数化 |
| **E4** | TS-LIF | TS-LIF (ICLR 2025) | 2.18 | 4.01G (+11%) | 全替换中最佳平衡，但仍不如PSN |
| E4b | TS-LIF官方风格 | 同上 | 6.99~7.06 | 2.16~2.36G | 稀疏但精度崩溃 |
| **E5b** | Ternary Spike | Ternary-Spike (AAAI 2024) | 29.77 | 25.89G | 灾难性失败，范式不兼容(ANN→SNN分类 vs 事件流) |
| **E6a** | NASN | NASN (arXiv 2604) | 2.17 | 33.31G (×9.2) | 精度尚可但SOPs爆炸 |

#### E2 ATLIF 详细分支

| 分支 | 设置 | AEE | SOPs | 结论 |
|------|------|-----|------|------|
| E2a 早期 | 错误替代函数 | 4.01 | 16.44G | 破损实现 |
| E2b 修正 | 部分修正但训练规模错 | 8.66 | 16.15G | 仍破损 |
| E2c 官方低SOP | official surrogate + eta=1e-3 | 3.76 | 2.87G | **SOP减少21%但AEE+137%** |
| E2d 全预训练 | 从PSN初始化+弱惩罚 | 2.51 | 5.21G | 精度较可但SOPs反增 |
| E2e Plan A | 保守lr=1e-5 | 5.66 | 6.86G | 阈值未增长 |
| E2f 冻结阈值 | 冻结54.9M仅训105阈值 | 2.58 | 4.93G | 微幅改进但不如PSN |

### 3.3 融合神经元脚手架（F系列 — 5个实验）

**所有仅smoke测试，无完整训练。smoke结果普遍不亮眼。**

| ID | 融合内容 | smoke train_loss | smoke val_loss | 质量评价 |
|----|---------|:---:|:---:|---------|
| F1 | PSN + 自适应阈值 | 8.16 | 6.22 | 中等 |
| F2 | LMHT多层级 + ATLIF递归膜 | 6.96 | 6.42 | 较好 |
| F3 | TS-LIF双室 + 自适应阈值 | 6.04 | 6.35 | **smoke最佳** |
| F4 | LMHT掩码 + TS-LIF双室 | 14.90 | 15.38 | **smoke最差** |
| F5 | 有符号脉冲 + PSN混合 | 9.01 | 6.81 | 中等 |

### 3.4 部分稀疏门控（G/H系列 — 最成功路线，4个实验）

| ID | 机制 | 目标 | AEE | SOPs | 结论 |
|----|------|------|-----|------|------|
| **G1** 🔥 | HardSparseGate (6个layer0节点标量STE门控) | 6个layer0 Swin节点 | **1.61 (+1.3%)** | **2.71G (-25.1%)** | **当前最佳！** 冻结骨干+仅训6个门控+BN保持eval |
| H1 | HardwareSparseNeuron (GTCN)扩展到36个encoder节点 | 全encoder proj节点 | 2.67 | 2.71G (-25%) | SOPs降了但AEE从1.61掉到2.67，扩展失败 |
| H2 | Adaptive Ternary PSN (attn Q/K only) | attention Q/K节点 | 仅smoke: train 1.1, val 0.8 | — | 仅smoke，仅Q/K目标有意思但未完整验证 |
| H3 | Official ATLIF + PSN (attn Q/K only) | attention Q/K节点 | 仅smoke | — | 初步评估，未完整训练 |

#### G1 精确目标节点 (6个)
1. `layers.0.swin_blocks.0.attn.proj_sn`
2. `layers.0.swin_blocks.0.mlp.sn1`
3. `layers.0.swin_blocks.0.mlp.sn2`
4. `layers.0.swin_blocks.1.attn.proj_sn`
5. `layers.0.swin_blocks.1.mlp.sn1`
6. `layers.0.swin_blocks.1.mlp.sn2`

#### G1 关键训练技巧
- 冻结骨干 + BN设为eval模式 (防止BN统计漂移)
- 门控从关闭状态初始化 (init_logit=-2.0 → prob≈0.119 → 6个全关)
- 仅训6个标量gate_logit, lr=0.01 (比正常lr高100倍)
- reg_lambda=0.02 维持关门压力

### 3.5 灵敏度分析

对PSN baseline epoch59做了逐层SOPs贡献排序 + 零化消融，发现：
- **Top10层贡献47.7% SOPs**，Top20贡献70.85%
- **Decoder和prediction head虽SOP高但对精度极其敏感** — 零化后AEE从1.01跳到7.04
- **Layer0 Swin block的6个MLP+Attention proj节点是最佳第一靶点** — 贡献18.5% SOPs，零化后AEE仅+2.5%
- 这直接指导了G1的节点选择

---

## 四、Token Mixing 实验（替代QKV注意力）

**发现**：大量attention层firing rate为0%，QKV三个投影浪费计算。

| Mixer | 机制 | 参数量变化 | AEE (轻量训练后) | SOPs (轻量训练后) | 状态 |
|-------|------|-----------|:---:|:---:|------|
| **Identity** | 无token mixing,仅proj层 | ~-67% | 2.1029 | **2.9495G (-18.6%)** | **已验证可行** — QKV注意力在SNN中可移除 |
| Conv | 深度可分离Conv(spatial)+1×1 Conv(channel) | ~-70% | 待测 | 待测 | 运行中 |
| MLP | Transpose→Token-MLP→Transpose→Channel-MLP | ~-60% | 待测 | 待测 | 排队 |
| Pool | AvgPool(spatial)+proj | ~-80% | 待测 | 待测 | 排队 |

**关键发现**：去掉所有QKV投影和attention计算后，模型仍能有效学习光流(AEE仅从1.58→2.10，且仅训了5epoch)，SOPs立即降低18%。

---

## 五、体素化（Voxelization）调研

**状态**：已完成文献调研和分析报告，未执行任何体素化实验。

调研了10种可移植体素化方案，覆盖EventPillars(AAAI 2025)、EDCFlow(CVPR 2025)、可学习binning(ICLR 2026)、OmniEvent(AAAI 2026)、EventFlash(ICLR 2026)等。建议第一阶段做V1(EventPillars轻量版)、V2(EDCFlow时间差分)、V6(离散计数体素)三组低风险对照实验。

---

## 六、已规划但未执行的实验

### 6.1 A系列 — 配置就绪，随时可跑 (neuron_autoresearch)

| ID | 实验 | 机制 | 预期 | 风险 |
|----|------|------|------|------|
| **A1** | FSN on G1 | G1的6个节点升级为FusedSparseNeuron(2-level signed, 三元脉冲) | SOPs<2.5G, AEE<1.75 | 中 |
| **A5** | Refractory Pruning | 全encoder PSN外挂2步不应期 | SOPs 2.9-3.3G, +2% AEE | 低 |
| **A6** | Bipolar Attention Gate | FSN signed专门用于attention Q/K投影 | SOPs 2.3-2.7G | 中 |
| **A8** | Dual-Sparse Regularizer | 训练时同时惩罚firing rate + weight L1 | 额外10-15% SOP减少 | 低 |
| A9 | ATLIF Threshold | ATLIFThresholdNeuron包装PSN(layer0) | 自适应阈值剪枝 | 中 |

> **A5状态更新**：训练完成60epochs，待profiling！

### 6.2 A系列 — 仅设计构思，需要开发代码

| ID | 实验 | 机制 | 预期 | 风险 |
|----|------|------|------|------|
| A2 | Leakage-as-Gate | PSN现有decay参数推导gate信号(零额外参数) | SOPs 2.8-3.2G | 中 |
| A3 | Hierarchical Shared Gates | 4个stage各1个共享gate(替代36独立gate) | SOPs 2.7-3.0G | 低 |
| A4 | Timing-Dependent Gate | 早期时间步(噪声)门代价高,晚期(信号)代价低 | SOPs 2.5-3.0G | 中 |
| A7 | IMP Gating | 可学习初始膜电位推导gate信号(零额外存储) | SOPs 2.5-3.0G | 中 |
| A9 | Adaptive Timestep | 低事件率区域动态减少时间步 | SOPs 2.5-3.0G | **高** |

### 6.3 V系列 — 体素化实验（未开始）

| ID | 名称 | 机制 | 风险 |
|----|------|------|------|
| V1 | EventPillars轻量版 | 时间范围+极性+密度编码 | 低-中 |
| V2 | EDCFlow时间差分 | 相邻bin特征差分 | 低-中 |
| V3 | 可学习/无偏binning | 端到端可学习体素化 | 中-高 |
| V4 | OmniEvent adapter | 空间/时间解耦+融合 | 中-高 |
| V5 | 自适应时间窗口 | 事件密度驱动窗口聚合 | 中 |
| V6 | 离散计数体素 | RVT风格hard-bin计数(不用插值) | 低 |
| V7 | 多事件表示一致性 | 双分支+一致性约束 | 中 |
| V8 | 自适应密度体素 | 事件密度重加权 | 中 |
| V9 | 多窗口时间堆叠 | 历史窗口级联聚合 | 中 |
| V10 | V2V增强 | 阈值/噪声随机化增广 | 低-中 |

---

## 七、外部论文启发的全新方案（未开发代码）

基于2024-2026顶刊（含NeurIPS/ICLR/ICCV/AAAI/Nature Comms/ISCA等）全面调研，提炼出PSN包装器范式下的新方案：

### 7.1 高优先级（改动小，风险低，收益明确）

| ID | 方案 | 参考论文 | 包装方式 | 预期SOPs | 预期AEE | 改动量 |
|----|------|---------|---------|---------|---------|--------|
| N1 | AHSAR零参数自适应发放率 | AHSAR (arXiv Dec 2025) | PSN输出后乘homeostatic阈值缩放因子, 每层1个标量 | -10~15% | +2~5% | **极低** |
| N2 | RPLIF发方触发不应期(改进A5) | RPLIF (arXiv Sep 2025) | A5基础上加spike-triggered阈值动态 | -15~20% | +3~8% | 低 |
| N3 | AT-LIF NeurIPS改进版阈值 | Activity Pruning (NeurIPS 2025) | 升级ATLIF阈值机制: MPD驱动+鞍点理论eta调度 | -20~30% | +5~15% | 中 |
| N4 | SGP替代梯度改进 | SpQuant-SNN (2024) | 替换SpikeFn: 分离梯度路径+梯度惩罚窗口 | 训练更稳定 | ±0% | 低 |
| N5 | DGN动态门控(升级G1) | DGN (arXiv Sep 2025) | G1固定STE门→电导动态门控 | -25~30% | +2~5% | 中 |

### 7.2 中优先级（需验证兼容性）

| ID | 方案 | 参考论文 | 包装方式 |
|----|------|---------|---------|
| N6 | QB-LIF量化burst | QB-LIF (arXiv Apr 2026) | PSN输出后接可吸收量化器 |
| N7 | LT-Gate软门控 | LT-Gate (arXiv Oct 2025) | PSN时间混合改快慢双通道+γ软混合 |
| N8 | MSF多阈值包装 | MSF Neuron (Nature Comms 2025) | PSN输出→K路不同阈值量化→合并 |
| N9 | OSBC一次性后训练压缩 | OSBC (arXiv Jun 2025) | G1模型上做膜电位损失最小化剪枝 |

### 7.3 外部注意力/Token剪枝方案（可能结合）

| ID | 方案 | 参考论文 | 核心思想 |
|----|------|---------|---------|
| E7 | 免训练Token剪枝 | TP-Spikformer (ICLR 2025) | 数据预处理中时空信息保留token筛选, eval-only |
| E8 | MPD自适应阈值 | DS-ATGO (AAAI 2026) | 膜电位分布驱动ATLIF阈值更新 |
| E9 | 侧抑制注意力 | SpiLiFormer (ICCV 2025) | 在attn.proj_sn处加FF-LiDiff旁路 |
| E10 | BSA稀疏训练 | Bishop (ISCA 2025) | 损失函数加bundle sparsity项 |
| E11 | STAS统一框架 | STAS (arXiv Aug 2025) | 2D(空间+时间)自适应token剪枝 |
| E12 | Q-K注意力替换 | QKFormer (NeurIPS 2024) | Q-K注意力替换Swin窗口注意力 |

---

## 八、总体优先级矩阵

按 **收益×可行性÷风险** 排序

### 立即执行（配置就绪）

| 优先级 | 实验 | 理由 |
|--------|------|------|
| **P0** | Profile A5 Refractory | 已跑完60epochs训练，最快得到答案 |
| **P1** | 跑A1 FSN on G1 | G1已证明选对节点可行，FSN是G1的升级版 |
| **P2** | 跑A8 Dual-Sparse | 仅加损失项，零架构改动，可叠加任何方案 |
| **P3** | 跑A6 Bipolar Attention | 注意力稀疏化，已验证QKV attention中有冗余 |

### 短期开发（低风险，改动小）

| 优先级 | 实验 | 理由 |
|--------|------|------|
| P4 | N1 AHSAR零参数自适应发放率 | 极小改动，零风险 |
| P5 | N5 DGN动态门控(升级G1) | 直接升级当前最佳结果 |
| P6 | G1+Refractory双重叠加 | 两个已验证机制叠加 |

### 中期探索（需验证兼容性）

| 优先级 | 实验 | 理由 |
|--------|------|------|
| P7 | N3 AT-LIF改进版阈值 | 解决E2固定eta难调问题 |
| P8 | E4部分插入(仅高发放层) | 避免全替换精度损失 |
| P9 | Conv/Pool Token Mixer完成训练 | Identity已验证QKV可移除 |
| P10 | V1/V2/V6体素化轻量实验 | 输入端优化，与神经元稀疏互补 |

### 长线探索（复杂度高，潜力大）

| 优先级 | 实验 | 理由 |
|--------|------|------|
| P11 | H4/H5/H6多阶段自适应稀疏 | 阶段感知+目标发放率 |
| P12 | A2/A3/A4/A7 更多门控策略 | 需要开发新的门控机制 |
| P13 | E7免训练Token剪枝 | 与神经元稀疏互补 |
| P14 | V3可学习binning | 研究价值高但工程量大 |

---

## 九、已参考但未移植的开源实现

| 代码 | 论文 | 状态 |
|------|------|------|
| PSN | Parallel Spiking Neuron (NeurIPS 2023) | **已内置SpikingJelly，当前基线** |
| ATLIF | Activity Pruning SNN (NeurIPS 2024) | E2已移植，结果不佳 |
| LMHT | LM-HT SNN (NeurIPS 2024) | E3已移植，SOPs太高 |
| TS-LIF | Temporal Segment LIF (ICLR 2025) | E4已移植，全替换中最好但不如PSN |
| TSN | Ternary Spike (AAAI 2024) | E5b已移植，范式不兼容 |
| TC-LIF | Two-Compartment LIF (AAAI 2024) | 未移植，双室时序参考 |
| RVT | Recurrent Vision Transformers (CVPR 2023) | 未移植，离散计数体素参考 |
| TemporalEventStereo | ECCV 2024 | 已clone，多窗口体素参考 |
| EventDance | CVPR 2024 | 已clone，多表示一致性参考 |
| V2V | NeurIPS 2025 | 未移植，体素增强参考 |

---

## 十、融合方案汇总

### 已验证可行的组合方向

| 方向 | 内容 | 风险 |
|------|------|------|
| G1 + ATLIF阈值 | G1局部插入 + ATLIF自适应阈值替代固定门控 | 中 (E2教训) |
| G1 + Refractory | G1的6关断节点 + 不应期机制双重稀疏 | 中 (可能过度) |
| E4 + 通道感知alpha | TS-LIF改标量alpha为通道级张量(已知特征维度处) | 低 |
| F3 + 全训练 | 融合自适应TS-LIF (smoke最佳)做完整训练 | 高 |

### 有潜力的新融合方向

| 方向 | 内容 |
|------|------|
| PSN + LMHT多级 | PSN并行混合后接多级阈值量化 |
| ATLIF阈值 + PSN外挂 | ATLIF阈值作为PSN外挂自适应剪枝 (同A9) |
| TS-LIF部分 + PSN | 仅在高发放层(>20%)用TS-LIF双室 |
| FSN混合模式 | 不同层用不同FSN模式(浅层2级+深层三值) |
| TP-Spikformer免训练剪枝 + G1 | 两层面互补：输入token筛选 + 神经元门控 |
| DS-ATGO MPD驱动 + ATLIF | MPD驱动阈值替代固定eta |

---

## 十一、实验基础设施总览

### 目录结构
- `third_party/SDformerFlow/` — 上游baseline（只读）
- `src/models/modules/spiking_neurons/` — 神经元基础设施
  - `candidates/` — 候选神经元(SNNode, ATLIFNode, TSLIFNode, LMHNode, TSNNode)
  - `hardware/` — 硬件包装器(HardwareSparseNeuron/GTCN, FusedSparseNeuron, RefractoryNeuron)
- `neuron_experiments/` — 自包含神经元实验（E/F/G/H系列）
- `neuron_autoresearch/` — A系列实验（autoresearch基础设施）
- `autoresearch_sparsity/` — 稀疏预处理autoresearch
- `voxelization_experiments/` — 体素化实验（仅调研）
- `experiments/checkpoints/` — 所有训练权重
- `experiments/reports/` — 所有实验报告
- `docs/literature/` — 文献调研

### 硬件配置
- GPU: A800 80GB
- 甜点配置: bs4, workers4, torch backend, AMP, TF32, lr=5e-5(训练)/1e-5(微调)

### 评估协议
- **valid40**: 40个验证样本，SOPs profiler (tools/profile_sops.py)
- **valid**: 全部825个验证样本 (DSEC valid)
- 指标: AEE (平均终点误差), AAE (平均角度误差), firing_rate (全局发放率), SOPs (突触操作数)

---

## 附录：完整成果时间线

| 时间 | 里程碑 |
|------|--------|
| 04-21 | 跑通单序列smoke，开始baseline全量训练 |
| 04-23 | 首次完成baseline全量训练(epoch60, 经多次NaN中断+恢复) |
| 04-24 | 完成吞吐调优，确定bs4为甜点；开始bs4续训 |
| 04-25 | bs4 epoch59权重确认优于原始full baseline (AEE 1.33 vs 2.39) |
| 04-26 | 完成体素化文献调研报告(10种方案) |
| 04-26~05-01 | E2 ATLIF多分支探索 (6个分支) |
| 05-01~05-02 | E3 LMHT移植训练；E4 TS-LIF移植训练 |
| 05-04 | E4b/E5b官方风格短跑评估 |
| 05-06 | 神经元全面评估报告(21个指标对比)；E6a NASN自建训练 |
| 05-07 | **G1完成 — 首个成功结果**：灵敏度分析驱动节点选择 + 部分门控，SOPs -25% |
| 05-07 | H1扩展到全encoder (失败 — AEE +67%) |
| 05-08 | 神经元全景研究报告(56页)；A系列实验规划 |
| 05-09 | H2 Adaptive Ternary PSN烟雾测试；H3 Official ATLIF+PSN烟雾测试 |
| 05-09 | Token Mixing Identity实验 — 验证QKV注意力在SNN中可移除 |
| 05-10 | H4/H5/H6多阶段自适应稀疏规划 |
| 05-11 | 本清单生成 |

---

*本文件持续更新，每次新实验完成后追加。*
