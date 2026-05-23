# DATE 论文骨架与硬件化路线

更新日期: 2026-05-23

本文档用于把当前 SDformerFlow 神经元/注意力/稀疏替换实验收束成 DATE 风格论文。核心判断是: 当前工作可以往 DATE 投，但不能只写成“事件光流模型替换神经元后 AEE/AAE 变化”。DATE 更关心电子系统、硬件架构、设计方法、低功耗、嵌入式/边缘 AI 和可实现性，因此论文主线需要从模型实验上升为“硬件感知的事件视觉 Spiking Transformer 协同设计”。

官方范围参考:
- DATE 2027 CFP: https://www.date-conference.com/date-2027-call-papers
- DATE topic/TPC 页面: https://www.date-conference.com/tpc

## 1. 当前工作与 DATE 的匹配性

### 1.1 可以匹配的 DATE 方向

当前工作最适合包装到以下方向:

| DATE 方向 | 本工作的对应点 |
|---|---|
| Architectural and microarchitectural design | 为事件光流 Spiking Transformer 设计稀疏神经元、三值注意力和硬件友好算子 |
| Low-power, energy-efficient design | 通过发放率、SOPs、稀疏替换、低乘法注意力降低能耗 |
| Approximate computing | 三值/二值神经元、近似归一化、stage-aware 替换属于精度-能耗折中 |
| Embedded/edge AI and neuromorphic systems | 事件相机光流是典型边缘感知任务，SNN/事件流天然匹配 neuromorphic/edge |
| Design methodologies for ML architectures | 不是单个模型 trick，而是神经元、注意力、替换范围、学习率和稀疏约束的协同搜索方法 |

### 1.2 当前还不够 DATE 的地方

现在实验已经有 AEE、AAE、SOPs、firing rate、threshold、ternary pos/neg 等指标，但这些仍偏模型侧。DATE 审稿人很可能会追问:

1. SOPs 如何对应真实能耗、面积、延迟和访存?
2. shiftmax 的 `2^x`、ATLIF 阈值缩放、三值输出乘阈值是否真的硬件友好?
3. 只替换部分 block 的异构网络会不会让硬件控制流复杂化?
4. 如果要设计加速器，是不是有明确的数据流、buffer、算子单元和调度方式?
5. 这个方法是手工调参，还是可复用的 design-space exploration 方法?

因此后续需要补“硬件代价模型 + 微架构草图 + 固定点/整数化可行性 + 系统级消融”。

## 2. 部分 block 替换是否硬件友好

结论: **可以硬件友好，但要把它解释成编译期静态异构调度，而不是运行时动态乱切换。**

### 2.1 不友好的写法

如果论文里说“我们随便挑一些 block 替换，另一些 block 不替换”，硬件上会显得很乱:

- 每个 block 的算子类型不同，控制逻辑复杂;
- 加速器要同时支持 dense PSN、binary ATLIF、ternary ATLIF、shiftmax、shiftnorm 等很多路径;
- 运行时需要频繁切换 kernel，buffer 格式可能来回转换;
- 难以解释为什么这些 block 被替换，而不是其他 block。

这种写法更像模型调参，不适合 DATE。

### 2.2 友好的写法

应该改写成:

> 本文提出一种 stage-aware static sparsification schedule。替换位置在部署前由 sensitivity-cost analysis 决定，运行时不做动态搜索。加速器只需要按预先编译好的 layer schedule 执行不同 kernel。

也就是说，部分 block 替换不是缺点，而是“静态异构硬件映射”:

- 每个 stage/block 的替换类型在编译期固定;
- 网络结构导出为 layer-wise schedule table;
- 加速器按表调用 dense kernel、binary spike kernel、ternary spike kernel、popcount attention kernel;
- 不需要运行时判断，不需要动态剪枝控制器;
- 避免全网络替换带来的 AAE 崩坏，同时保留高 SOPs 层的节能收益。

### 2.3 推荐替换粒度

从硬件角度，不建议讲“任意 block 子集”。更建议限制为几类结构化模板:

| 模板 | 硬件友好性 | 论文可解释性 | 备注 |
|---|---:|---:|---|
| 整个 stage 替换 | 最高 | 高 | 控制最简单，但精度风险大 |
| stage 内 even/odd block 替换 | 高 | 高 | 结构规则，适合讲 pipeline schedule |
| stage0+stage2 / stage0+1+2 高 SOPs FFN 替换 | 中高 | 高 | 与 SOPs 敏感性强绑定 |
| 任意 block 搜索结果 | 中低 | 中 | 可以作为 DSE 输出，但不适合作为最终硬件故事 |
| token/通道级动态替换 | 低 | 中 | 更像动态剪枝，需要额外硬件控制 |

建议最终论文主方案使用“stage-aware + rule-based block mask”，例如:

- Attention Q/K: 所有 block 使用 ternary PSN+ATLIF，保证注意力范式统一;
- FFN: 只在高 SOPs 且低敏感的 stage/block 使用 binary PSN+ATLIF;
- Downsample: 除非收益明显，否则尽量少动，因为格式转换和精度风险更高;
- 替换 mask 固化为编译期 schedule。

## 3. 是否需要设计完整加速器系统

结论: **不一定需要完整芯片系统，但至少要有可被 DATE 接受的硬件证据链。**

DATE 不是必须要求 tape-out，也不是每篇都要完整 SoC。但是如果主张“硬件友好”，就不能只给模型指标。最低可接受形态应该是:

1. 算子级硬件代价模型;
2. 关键 kernel 微架构;
3. layer-wise schedule 和数据流;
4. 与 baseline 的能耗/延迟/访存 proxy 对比;
5. 最好再有 FPGA/HLS 或 Timeloop/Accelergy 级别验证。

### 3.1 最小硬件设计范围

不用设计完整事件相机到输出光流的一整套系统。建议设计一个 **Spiking Event-Transformer Accelerator Subsystem**，覆盖模型中最核心、最能支撑论文贡献的算子:

| 模块 | 是否必须 | 作用 |
|---|---:|---|
| Spike neuron engine | 必须 | 支持 PSN、binary ATLIF、ternary ATLIF、阈值更新 |
| Ternary attention engine | 必须 | 支持 sign extraction、XNOR/popcount、shiftmax/shiftnorm/L1 norm |
| Sparse FFN engine | 必须 | 支持高 SOPs FFN 的 binary/ternary 稀疏执行 |
| Schedule controller | 必须 | 按静态 layer/block schedule 切换 kernel |
| On-chip buffer model | 必须 | 估算激活、阈值、权重、attention map 访存 |
| Voxelization frontend | 可选 | 如果引入 EDCFlow/体素化优化再加，否则先不作为主贡献 |
| Full optical-flow postprocess | 可选 | 可用软件模拟，不必硬件化 |

### 3.2 DATE 论文里需要画的硬件图

至少需要 3 张图:

1. **Overall Co-Design Flow**
   - Baseline SDformerFlow
   - Sensitivity/SOPs profiling
   - Stage-aware replacement schedule
   - Hardware cost model
   - Final sparse ternary SNN Transformer

2. **Kernel Microarchitecture**
   - comparator + threshold register
   - ternary sign/valid encoding
   - XNOR/sign-match + valid-mask + popcount tree
   - shiftmax/shiftnorm/L1 normalization
   - sparse FFN accumulation

3. **Layer Schedule**
   - stage/block 表格
   - 每层 kernel 类型
   - 数据格式: dense / binary spike / ternary spike
   - buffer reuse 和格式转换位置

## 4. 需要补的硬件指标

### 4.1 不能只报 SOPs

SOPs/firing 是有用的，但 DATE 审稿人会认为它是模型侧 proxy。建议补以下指标:

| 指标 | 说明 |
|---|---|
| Energy proxy | 按 MAC、add、compare、XNOR、popcount、shift、SRAM access 加权 |
| Latency proxy | 每层计算周期估计，尤其 attention 和 FFN |
| Memory traffic | activation、spike map、threshold、attention score、weight 的读写量 |
| Area proxy | 关键单元数量: comparator、adder tree、popcount、LUT/shift unit、buffer |
| Fixed-point sensitivity | 阈值、score、normalization 用 8/12/16-bit 的精度影响 |
| Per-stage breakdown | 每个 stage 的 SOPs/firing/energy 贡献 |
| Format conversion overhead | dense 到 binary/ternary spike 的编码和解码成本 |

### 4.2 推荐 energy model

先用论文级可解释的 analytical model，不必一开始就做完整 RTL:

```
E_total =
  N_mac      * E_mac
+ N_add      * E_add
+ N_cmp      * E_cmp
+ N_xnor     * E_xnor
+ N_popcount * E_popcount
+ N_shift    * E_shift
+ N_sram_rd  * E_sram_rd
+ N_sram_wr  * E_sram_wr
```

其中:

- baseline QKFormer attention 主要消耗 dense add/multiply/activation traffic;
- ternary attention 把 Q/K 相关计算转成 sign、valid-mask、XNOR/popcount、shift/L1 norm;
- sparse FFN 通过 firing rate 减少有效累加次数;
- ATLIF 阈值更新作为训练/部署时可选开销分开统计。推理时阈值固定，训练时阈值更新另算。

### 4.3 shiftmax 与硬件友好问题

当前 shiftmax 如果含 `2^x`，可以用 LUT 或移位近似，但仍要解释成本。建议三种版本都保留:

| 版本 | 作用 |
|---|---|
| shiftmax | 精度优先，贴近 BSA 灵感 |
| shiftnorm | 硬件折中，用 L1/shift 规避指数 |
| popcount-L1 | 硬件极限版本，纯符号计数 + 归一化 |

论文主方案可以是 shiftmax 或 shiftnorm，但必须有 popcount-L1 对照，证明“去指数归一化”时精度/能耗如何变化。

## 5. 方法命名与主贡献建议

临时命名:

**SATFlow: Stage-Aware Ternary Spiking Transformer for Energy-Efficient Event Optical Flow**

或者:

**HASTE-Flow: Hardware-Aware Sparse Ternary Event-Flow Transformer**

建议主贡献写成 4 点:

1. **Adaptive ternary spike primitive**
   - 以 PSN 保留表达能力;
   - 以 ATLIF 自适应阈值控制稀疏;
   - 以对称三值输出保留正负事件方向信息;
   - 支持 binary/ternary 两种部署模式。

2. **Hardware-oriented ternary attention**
   - Q/K 转为 sign/valid 编码;
   - 用 XNOR/popcount 或 signed consensus 替代 dense score;
   - 用 shiftmax/shiftnorm/popcount-L1 做归一化对比;
   - 重点证明乘法减少和整数化可行。

3. **Stage-aware static replacement schedule**
   - 基于 SOPs contribution 和 accuracy sensitivity 选择替换范围;
   - 替换 mask 在编译期固定;
   - 让硬件用静态 schedule 执行，避免动态控制复杂度。

4. **Hardware cost and event-flow validation**
   - 在 SDformerFlow/DSEC 上验证 AEE/AAE;
   - 报告 SOPs、firing、energy proxy、latency proxy、memory traffic;
   - 与 baseline、H9a、纯 binary、纯 ternary、不同 attention 归一化对比。

## 6. DATE 论文骨架

### Title

Hardware-Aware Sparse Ternary Spiking Transformer for Energy-Efficient Event-Based Optical Flow

### Abstract

事件相机适合低功耗边缘视觉，但现有事件光流 Transformer 仍包含大量 dense attention 和 FFN 操作，难以直接部署到资源受限硬件。本文提出一种硬件感知的稀疏三值 Spiking Transformer 协同设计方法，结合 PSN 表达能力、ATLIF 自适应阈值、对称三值脉冲、符号计数型注意力和 stage-aware 静态替换策略。在保持光流 AEE/AAE 接近 baseline 的同时，显著降低 SOPs、发放率和硬件能耗 proxy。我们进一步给出算子级能耗模型和面向 ternary attention/sparse FFN 的微架构分析，证明该设计适合事件视觉边缘加速。

### 1. Introduction

要讲清楚的问题:

- 事件相机低延迟、低冗余，但事件光流网络仍然计算密集;
- SDformerFlow/QKFormer 类结构虽然精度好，但 attention 和 FFN 对硬件不友好;
- SNN 可以稀疏，但简单替换神经元会造成 AEE/AAE 退化;
- 需要同时解决表达能力、稀疏率、注意力归一化和硬件执行;
- 本文提出硬件感知的神经元-注意力-替换范围协同设计。

### 2. Background and Motivation

需要包括:

- SDformerFlow/QKFormer 的基本结构;
- PSN 神经元为何是 baseline;
- ATLIF 的自适应阈值思想;
- BSA/三值注意力/shiftmax 的启发;
- 为什么 SOPs/firing 还不够，需要 hardware cost model;
- 部分 block 替换的动机: 全替换容易精度崩，局部替换能获得 Pareto 折中。

建议图:

- baseline stage/block 结构图;
- per-stage SOPs/firing sensitivity 图;
- 全替换 vs 局部替换的 AAE/SOPs 对比图。

### 3. Method

#### 3.1 Adaptive Ternary PSN-ATLIF Neuron

内容:

- PSN transformation;
- ATLIF threshold update;
- binary 输出和 ternary 输出;
- 对称正负阈值;
- 推理时阈值固定，训练时阈值可学习;
- fixed-point/power-of-two threshold 的部署选项。

需要消融:

- PSN baseline;
- PSN + ATLIF binary;
- PSN + ATLIF ternary;
- 对称阈值 vs 负阈值压制;
- target-rate / activity penalty sweep。

#### 3.2 Ternary Attention Primitive

内容:

- Q/K 三值编码: sign bit + valid bit;
- signed consensus / alpha-XNOR / BSA-like QKV 的区别;
- shiftmax、shiftnorm、popcount-L1 三种归一化;
- 为什么 Q/K 可以全替换，FFN 选择性替换;
- 注意力输出如何接回原 SDformerFlow。

必须避免的问题:

- 不要把所有方案都叫 BSA;
- 如果不是标准 BSA QKV，就叫 BSA-inspired 或 ternary consensus attention;
- 明确 QKFormer 适配与标准 QKV 的差异。

#### 3.3 Stage-Aware Static Replacement Schedule

核心公式:

```
score(layer) = alpha * normalized_SOPs(layer)
             - beta  * sensitivity(layer)
             - gamma * conversion_overhead(layer)
```

选择高 score 的 layer/block 替换。

硬件解释:

- 替换结果固化为 schedule table;
- 运行时按 layer id 调用 kernel;
- 不需要动态 pruning;
- 支持 stage-level、even/odd block、high-SOP FFN 三类规则模板。

#### 3.4 Hardware Cost Model and Accelerator Subsystem

内容:

- 算子能耗模型;
- ternary attention engine;
- spike neuron engine;
- sparse FFN engine;
- schedule controller;
- buffer and memory traffic model;
- fixed-point 配置。

### 4. Experimental Setup

模型和数据:

- Baseline: SDformerFlow / QKFormer-style event optical flow;
- Dataset: DSEC 或当前 baseline 使用的数据集;
- Metrics: AEE、AAE、SOPs、firing rate、energy proxy、latency proxy、memory traffic;
- Training: baseline checkpoint continuation and/or from-scratch 对比;
- Seeds: 关键方案至少 3 seeds。

对比方法:

| 类别 | 对比 |
|---|---|
| Baseline | 原 SDformerFlow PSN |
| Neuron-only | binary ATLIF、ternary ATLIF |
| Attention | compat QK、signed consensus shiftmax、shiftnorm、popcount-L1、strict BSA-like |
| Replacement | QK only、S02、S012、stage-aware selected |
| Hardware | dense attention、binary sparse、ternary sparse、integer normalization |

### 5. Results

建议主表:

| Method | AEE | AAE | SOPs | firing | Energy proxy | Latency proxy |
|---|---:|---:|---:|---:|---:|---:|
| SDformerFlow baseline | | | | | | |
| H9a/H9e historical best | | | | | | |
| binary ATLIF | | | | | | |
| ternary PSN-ATLIF + shiftmax | | | | | | |
| ternary PSN-ATLIF + shiftnorm | | | | | | |
| proposed stage-aware schedule | | | | | | |

重点图:

1. AEE/AAE vs SOPs Pareto curve;
2. per-stage energy breakdown;
3. firing heatmap;
4. threshold evolution;
5. fixed-point bitwidth sensitivity;
6. attention normalization ablation;
7. replacement schedule ablation。

### 6. Hardware Analysis

需要回答:

- 三值神经元如何编码?
- 阈值乘法如何避免?
- shiftmax 和 shiftnorm 的硬件差异?
- 部分 block 替换如何调度?
- 稀疏 FFN 是否真的减少访存和累加?
- 与 dense baseline 相比，energy/latency bottleneck 是否转移?

建议结论:

- 训练时可以使用实数阈值;
- 推理部署时将阈值量化为 fixed-point，或共享 per-stage threshold;
- 输出不一定真的乘任意实数阈值，可以使用 sign/valid 编码参与 attention，必要时在 layer scale 中吸收阈值;
- 因此硬件执行以比较器、bit operation、popcount、shift/add 为主。

### 7. Discussion and Limitations

需要诚实说明:

- 局部替换依赖 sensitivity profiling;
- 如果数据集或模型结构变化，schedule 需要重新搜索;
- shiftmax 仍有 LUT/shift 成本;
- full RTL/ASIC tape-out 尚未完成;
- 从 checkpoint 续训与 from-scratch 训练的差异需要补充。

### 8. Conclusion

强调:

- 不是单个神经元 trick，而是事件光流 Spiking Transformer 的硬件感知协同设计;
- 在精度可控下获得稀疏、低 SOPs、低能耗 proxy;
- 静态 stage-aware schedule 让局部替换也能硬件友好部署。

## 7. 必须补的实验清单

### 7.1 论文主线必做

| 优先级 | 实验 | 目的 |
|---:|---|---|
| P0 | baseline 重新统一推理 | 固定 AEE/AAE/SOPs/firing 参考 |
| P0 | proposed 主方案 full training | 得到最终主结果 |
| P0 | proposed 主方案 valid/test 推理 | AEE/AAE/SOPs/firing/threshold/pos-neg |
| P0 | per-stage SOPs/energy breakdown | 支撑 stage-aware 替换 |
| P0 | energy proxy 计算脚本 | 从 SOPs 升级到硬件能耗 |
| P1 | shiftmax vs shiftnorm vs popcount-L1 | 证明硬件友好归一化 |
| P1 | binary vs ternary FFN | 证明三值是否值得扩到 FFN |
| P1 | QK only vs S02 vs S012 | 替换范围消融 |
| P1 | target-rate/activity_eta sweep | 稀疏强度和精度折中 |
| P1 | differential LR / warmup / slow backbone | 训练稳定性 |
| P2 | fixed-point threshold/score bitwidth | 部署可行性 |
| P2 | 3 seed for final 2-3 configs | 统计稳定性 |
| P2 | from-scratch vs baseline continuation | 排除续训偏差 |

### 7.2 加速器证据必做

| 优先级 | 内容 | 最低实现 |
|---:|---|---|
| P0 | 算子级能耗模型 | Python 脚本统计 op count + energy weight |
| P0 | layer schedule table | 由 config 自动导出每层 kernel 类型 |
| P1 | memory traffic model | activation/spike/weight/threshold 读写估算 |
| P1 | latency proxy | 按 popcount tree、shift/L1、sparse FFN 周期估算 |
| P2 | HLS/FPGA 小 kernel | ternary attention 或 sparse FFN 单 kernel 即可 |
| P2 | roofline/瓶颈分析 | 判断算力受限还是访存受限 |

## 8. 当前实验如何映射到论文

| 当前实验线 | 论文中的角色 |
|---|---|
| H9a/H9e | 历史强 baseline，说明 shiftmax/局部替换有效 |
| H13 | 负脉冲修复和对称三值范式来源 |
| H34/H35/H37 | attention 范式审阅后的对照，包括 strict BSA-like |
| H40 | 大规模短测和 stage-aware 搜索证据 |
| H41/H42 | 候选 full training 和更稳学习率/稀疏参数 |
| 体素化/剪枝 | 暂作为未来扩展，不建议抢主线 |

## 9. 后续推荐路线

### Step 1: 固定最终候选

从已有结果里选 2-3 个 Pareto 候选:

- 精度优先: H9a/H9e 或其修正版;
- 硬件优先: SN/SC + shiftnorm/popcount-L1 + S02/S012;
- 平衡方案: stage-aware schedule + ternary QK + binary FFN。

### Step 2: 做硬件化修正

- 把阈值输出从“乘实数阈值”改写/实现为 sign-valid 编码 + layer scale;
- 推理阈值做 fixed-point;
- attention 里优先保留 shiftnorm/popcount-L1 对照;
- 替换 mask 限制为规则模板。

### Step 3: 跑最终实验

- 每个候选 full training;
- 每个 full checkpoint 做统一推理;
- 统计 AEE/AAE/SOPs/firing/energy/latency;
- 选最终主方案。

### Step 4: 写 DATE 论文

优先写硬件故事，而不是实验流水账:

1. 问题: 事件光流 Transformer 计算密集，不适合边缘硬件;
2. 方法: 自适应三值 SNN + 符号计数注意力 + stage-aware 静态替换;
3. 硬件: kernel 微架构 + schedule + cost model;
4. 结果: 精度接近 baseline，SOPs/energy/latency 明显降低。

