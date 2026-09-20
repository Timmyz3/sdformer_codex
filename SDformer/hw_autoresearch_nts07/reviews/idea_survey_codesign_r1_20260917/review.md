# 算法+硬件一体（co-design）调研 r1（2026-09-17）

口径（用户指定）：**只收同时提出算法（模型/算子/训练方法）与对应硬件（芯片/加速器/微架构）的工作**；产出供 SDformer 做「算法机制 → 硬件对象变化」范式借鉴。
方法说明：本应派 agent 执行，连续 3 次 API 405 失败后改由主会话直查（WebSearch 为主，未取全文的以摘要为限，均已标注）。
证据分档：论文数字为 [论文]（引用源）；本地结论为 [模型]；未本地验证。

---

## 1. 代表作清单

### A. 事件相机视觉/光流加速器（与我们同一应用域）

| 工作 | venue/年 | 算法机制 | 硬件机制 | 耦合点 | 收益 |
|---|---|---|---|---|---|
| **ASNA-Flow** [1] | IEEE TVLSI 2025.08 | 硬件感知算法优化；针对光流空间局部性做稀疏化 | TSMC 28nm 异步事件驱动架构；数据模式分析驱动架构决策 | 光流的**空间局部性**直接作为稀疏计算对象 | 104 FPS @ 7.9 mW；0.3 pJ/SOP；自称首个事件光流专用神经形态方案 |
| **SpiDR** [2] | IEEE conf 2025.09 | 面向输入稀疏的可重构精度 | 65nm 数字 CIM，weight-Vmem 融合宏，zero-skipping，异步流水 | 「输入稀疏率」直接变成宏级 zero-skip 门控 | 5 TOPS/W @95% 稀疏；支持含光流的 event 负载 |
| **EmFlow** [3] | Carleton 硕士论文 2026.08 | 稀疏卷积 SNN + 1-bit spike 特征图 + 有限持久膜状态 | AMD Kria KV260 FPGA + Prophesee GenX320 | 「1-bit 特征图 + 稀疏卷积」把存储对象压到 bit 级 | 40 FPS @3.7–4.0 W（整机）；HFlow320/MVSEC/DSEC |
| Durance RIVER [4] | 商用 2026 | 事件驱动脉冲算法（演示光流应用） | 存算一体 FPGA 开发板 | 事件驱动 ↔ 存内计算 | 商用平台（非论文） |

### B. Spiking Transformer 加速器（与我们同一模型族）

| 工作 | venue/年 | 算法机制 | 硬件机制 | 耦合点 | 收益 |
|---|---|---|---|---|---|
| **FireFly-T** [5] | IEEE TC 2026.03 | 激活稀疏 + 二值注意力双引擎拆分 | 稀疏解码器 + 负载均衡（权重分发、乱序执行）；3D spiking-attention 的 **byte 级 SRAM 写**；LUT6 AND-PopCount | 稀疏率 → 负载均衡调度；注意力数据布局 → 存储写粒度 | 能效 1.39×/2.40× vs FireFly v2/SpikeTA |
| **28nm Spiking ViT 加速器** [6] | IEEE 2025 | EMA-free 自注意力 + 一阶近似（减 46.5% 注意力计算） | 28nm；双路稀疏计算核（−58% 外部访存）；1b/8b 统一加法树阵列 | 注意力近似 ↔ 加法树数据通路（去掉乘法器） | 1.79 mJ/frame；57.7 TOPS/W |
| **Workload-Balanced Sparsity + In-Situ Weight Reconstruction** [7] | IEEE 2026 | 列级结构化剪枝让负载确定；编解码器权重原位重建 | 针对「利用率墙」（时空不规则性）与「内存墙」的数据通路 | 结构化稀疏 → 确定性负载 → 高利用率流水 | 3.14× 吞吐、85.99× 能效 vs SOTA transformer 加速器 |
| Spikformer FPGA [8] | 物理学报 2026.05 | Conv-BN 融合 + QAT 压缩（15.92MB → 1/4） | Xilinx Zynq UltraScale+；多时步并行 | 模型压缩比 ↔ 片上存储预算 | CIFAR-10 端到端 ~53ms@200MHz；开源 |
| Spike-aware training arch [9] | 2026 | 训练期 spike 累积 + FP16 复用 | 统一脉冲 systolic array（前向/反向/权梯度） | 训练稀疏 ↔ 阵列复用 | 2.36 TFLOPS/W @28nm（估算） |

### C. 稀疏/跳过 co-design（通用范式，跨域可迁移）

| 工作 | venue/年 | 算法机制 | 硬件机制 | 耦合点 | 收益 |
|---|---|---|---|---|---|
| **SPARTA** [10] | ICCAD 2025 | RL 动态 token 跳过（空域+时域结构化稀疏）+ token 预测 | 异构 ReRAM-CIM 架构 | 跳过决策 ↔ CIM 阵列的激活粒度 | 543.1×/10.2× 加速 vs GPU/SOTA SNN 加速器 |
| **SpikeX** [11] | IEEE TCAD 2025（arXiv 2505.12292） | 网络侧稀疏化与数据流协同优化 | 面向不规则时空稀疏的数据流 | 网络稀疏分布 ↔ 数据流选择 | EDP 降 15.1×–150.87×，无精度损失 |
| **ASP-DAC'26 zero-skip** [12] | ASP-DAC 2026 | 稀疏编码（spike 数 −88% vs rate coding） | zero-skipping 加速器 | 编码稀疏率 ↔ skip 门控 | 省 88%/89% 能耗；4.5×/26.8× 吞吐；LUT −82% |
| TSA-P [13] | ISCAS 2026 | few-spike 神经元 + 脉冲串压缩/复用 | 三重稀疏感知可重构处理器 | 脉冲串结构 ↔ 访存压缩 | 降计算/访存能耗 |
| TAIL [14] | DATE 2025 | 时间异步执行 + 层间并行 | 数据流设计支持跨层多时步并发 | 时步并发 ↔ PE 利用率 | 6.94× 加速、6.97× 能效 |
| SpikeStream [15] | DATE 2025 | 稀疏计算映射为寄存器映射流 | RISC-V 集群 ISA 扩展（低开销流式稀疏） | 稀疏访存模式 ↔ ISA/寄存器流 | 提升通用多核的 SNN 推理效率 |
| SpAtten [16] | 经典（2020，HPCA 系） | 级联 token/head 剪枝 + 渐进量化 | quick-select top-k 引擎；不取回被剪数据 | 剪枝决策 ↔ 访存/计算绕行 | 范式源头：算法剪枝直接决定硬件"不访问" |
| Sparse attention on Tensor Cores [17] | conf 2025.09 | 注意力矩阵结构化剪枝（精度-稀疏可调） | Tensor Core 微架构 + ISA 扩展 | 结构化稀疏 ↔ 指令级支持 | 速度 +8.7%；能耗 −54.3% |

### D. 神经形态处理器 / 阈值机制（与 ATLIF 问题直接相关）

| 工作 | venue/年 | 算法机制 | 硬件机制 | 耦合点 | 收益 |
|---|---|---|---|---|---|
| **DATE'26 Dynamic Neural Thresholding** [18] | DATE 2026 | 阈值与权重**联合学习**的定制训练法 | 数字逻辑承担阈值控制（省掉高分辨率 DAC），不动模拟核 | 可学习阈值 ↔ 数字控制回路 | 五个分类基准一致增益，延迟/能耗开销极小 |
| SpikeRAM [19] | ISSCC 2026 | 终身片上学习 | 48.1 pW/synapse/bit 事件驱动 compute-near/in-memory + 集成神经形态传感器 | 事件驱动 ↔ 存近算 | 可穿戴/AR 级 mW 功耗 |
| GALSNP [20] | ISSCC 2026 | 混合 ANN-SNN + 免全局梯度片上学习 | 40nm GALS 流形处理器 | 异步域划分 ↔ 学习机制 | 1.01 pJ/SOP |
| DMP-SNN [21] | Nature Machine Intelligence 2026 | 双记忆通路；用时步跳步/dilation 制造时间稀疏 | 算法-硬件协同设计 | 跳步长度 ↔ 硬件更新周期 | 关键警示：**序列视觉可容忍 dilation=10，事件流不行**（跳步需保守） |

---

## 2. 五条「算法 ↔ 硬件耦合范式」+ SDformer 落点草案

**P1 稀疏可预测性 → 确定性负载/固定延迟流水**（来源：B-Workload-Balanced、FireFly-T、SPARTA）
- 范式：先把稀疏做成**可预测/结构化**（列级正则、token 预测），硬件才能做负载均衡与确定性调度；不可预测的稀疏只会让流水空转。
- SDformer 落点：Q/K gate + motion-XOR 的稀疏率目前是不可预测的逐元素稀疏；**时间商记录文件（H1 TLQ-5）的 run-length 结构天然是"可预测稀疏"**——把 run-length 直接作为调度单元（长度已知 → 固定延迟），这是 H1 相对现有三贡献的增量硬件点。最小验证（CPU）：统计 run-length 分布、p95 调度单元数。

**P2 跳过粒度 ↔ 存储对象粒度**（来源：FireFly-T byte 级 SRAM 写、SpikeStream 寄存器流、SpAtten 不取回被剪数据）
- 范式：跳过的**粒度**必须与存储对象**同粒度**，否则跳过的收益被访存吃掉。
- SDformer 落点：父-积捕获通路已是对象级；**跨窗目录（H2 XWIN-RB）把"窗"做成存储对象**（跨窗命中 → 目录指针 → 整对象跳过），对齐 P2。最小验证（CPU）：跨窗命中率与对象跳过量、目录位宽/面积模型。

**P3 门控信号 → 执行域/功耗域切换**（来源：ASNA-Flow 异步、GALSNP GALS、SpiDR zero-skip）
- 范式：稀疏/事件的门控不只省乘法，还直接切时钟/电源域（异步握手、GALS）。
- SDformer 落点：相位解耦神经元服务已有雏形；增量点是 **tile 级异步握手 + 时钟门控**（静默 tile 不进时钟树），与 M935 物理基线（WNS +0.001795）结合做功耗域核算。

**P4 神经元参数（阈值）→ 数字控制的可学习机制**（来源：DATE'26 Dynamic Neural Thresholding、DMP-SNN）
- 范式：把神经元状态量（阈值/膜更新周期）做成**数字控制、可编程、可学习**，用学习法弥补硬件简化。
- SDformer 落点：**直接接当前 ATLIF 死神经元问题**（见 `neuron_autoresearch/CLAUDE_ATLIF_DEAD_NEURON_DIAGNOSIS_20260917.md`）：部署侧阈值寄存器 + 训练侧双向闭环（bidirectional rate feedback）；硬件代价 = 每模块一个小寄存器 + 负发放计数器（原 P1 计划已含）。可引用 DATE'26 作为"阈值数字控制 + 联合学习"的范式先例。

**P5 时步级跳过 → 保守使用**（来源：SPARTA 时域 token 稀疏、TSA-P、DMP-SNN 警示）
- 范式：时域跳过（silent timestep / dilation）收益大，但**事件流对 dilation 敏感**（DMP-SNN 实测：事件流退化早于帧序列）。
- SDformer 落点：只做**"静默时步跳过"**（某 tile 的 T 维全零则整段跳过，用 valid 位向量），不做激进 dilation；时步预算用 T=5 保守档。最小验证（CPU）：统计 T 维全零 tile 比例。

---

## 3. 竞品 / novelty 撞车警示（必读）

1. **ASNA-Flow（TVLSI 2025）是最接近的竞品**：事件光流 + 28nm + co-design + 稀疏计算。差异轴必须讲清：它是**空间局部性**驱动的异步稀疏（非 Transformer）；我们是 **Spiking Transformer + 时间商/跨窗结构**。论文里必须显式引用并对比（它是 28nm 事件光流专用 ASIC 的唯一先例）。
2. **28nm Spiking ViT 加速器（2025）**：spiking ViT 注意力引擎（EMA-free、一阶近似、1b/8b 加法树）——与我们的注意力硬件点可能撞；差异化靠：任务（embodied intelligence 分类 vs 事件光流回归）+ 我们的跨窗/商结构。
3. **FireFly-T（TC 2026）**：spiking transformer 稀疏双引擎 + byte 级 SRAM 写——注意其"3D spiking attention 数据布局"与我们的时间窗结构可能相似，需在 related work 中区分（FPGA overlay vs ASIC；稀疏引擎 vs 商记录）。
4. **SPARTA（ICCAD 2025）**：spiking transformer 的 token 跳过（RL）+ ReRAM-CIM——若我们做 token 跳过必须引用；我们的差异：不依赖 RL/ReRAM，用 run-length/目录的**确定性**跳过（对齐 P1）。
5. **有利空白（2026-09 检索）**：**未发现"事件相机光流专用 Spiking Transformer ASIC"**——SDformerFlow 扩展版（TETCI 2026）只有算法、无芯片；ASNA-Flow 有芯片但非 Transformer。我们的交叉位（事件光流 × spiking transformer × 硅实现）仍空置，这是 DATE 主张的立足点。

## 4. 下一步建议

1. 把 P1/P2 落进 H1 TLQ-5 / H2 XWIN-RB 的 CPU 最小验证（run-length 分布、跨窗命中率、目录面积模型），作为 4.0 候选的证据第一步。
2. ATLIF 修复方案补上 P4 硬件点（阈值寄存器 + 双向闭环 + 负发放计数器），与 H12-G1 同批验证。
3. related work 必引：ASNA-Flow、28nm Spiking ViT、FireFly-T、SPARTA、SDformerFlow-TETCI；论文中给出一张"算法机制→硬件对象"对比表（我们的商记录/跨窗目录 vs 他们的空间局部性/byte 写/token 跳过）。
4. P5 先做静默时步跳过的 CPU 统计，再决定是否进 H12-G2 之后的实验矩阵。

## 5. 来源

[1] ASNA-Flow, IEEE TVLSI 2025, https://ieeexplore.ieee.org/document/11142472
[2] SpiDR 65nm CIM SNN, https://ieeexplore.ieee.org/abstract/document/11214115
[3] EmFlow (Carleton MASc), https://carleton.ca/share/2026/thesis-defense-2/
[4] Durance RIVER (Embedded Vision Summit 2026), https://www.scienceandtechnologywatch.com/article/911156039-durance-ai-to-demonstrate-river-durational-ai-platform-at-embedded-vision-summit-2026
[5] FireFly-T, IEEE TC 2026, https://ieeexplore.ieee.org/document/11432885
[6] 28nm Spiking Vision Transformer Accelerator, https://ieeexplore.ieee.org/abstract/document/11124571
[7] Workload-Balanced Sparsity + In-Situ Weight Reconstruction, https://ieeexplore.ieee.org/document/11670305
[8] Spikformer FPGA（物理学报 2026）, https://wulixb.iphy.ac.cn/article/cstr/32037.14.aps.75.20260085 ；repo: https://github.com/tooddler/FPGA_SpikingTransformer
[9] Spike-aware training architecture for spiking transformers, https://www.sciencedirect.com/science/article/abs/pii/S1879239126000925
[10] SPARTA, ICCAD 2025, https://ieeexplore.ieee.org/document/11240724
[11] SpikeX, arXiv 2505.12292, https://arxiv.org/abs/2505.12292v1
[12] Zero-skipping co-design, ASP-DAC 2026, https://xplorestaging.ieee.org/document/11420466
[13] TSA-P, ISCAS 2026, https://www.semanticscholar.org/paper/TSA-P%3A-A-Triple-Sparsity-Aware-Reconfigurable-SNN-Tang-Shen/e100ac604abc86ae90b7894abf0b26e07b7c5074
[14] TAIL, DATE 2025, DOI 10.23919/DATE64628.2025.10993093
[15] SpikeStream, DATE 2025, DOI 10.23919/DATE64628.2025.10992749
[16] SpAtten, https://www.emergentmind.com/topics/spatten-algorithm-architecture-co-design
[17] Sparse Attention on Tensor Cores, https://ieeexplore.ieee.org/document/11235365
[18] Dynamic Neural Thresholding, DATE 2026, https://ksp.etri.re.kr/ksp/article/read?id=72525
[19] SpikeRAM, ISSCC 2026, https://www.ablesci.com/assist/detail?id=pz1zM8
[20] GALSNP, ISSCC 2026, http://faculty.hust.edu.cn/WangChao/en/lwcg/2220850/content/168358.htm
[21] DMP-SNN, Nature Machine Intelligence 2026, https://www.nature.com/articles/s42256-026-01255-3
[22] Motion-aware Event Suppression (RSS 2026, token pruning 加速 ViT 83%), https://rpg.ifi.uzh.ch/event_suppression/
[23] SDformerFlow 扩展版（TETCI 2026，仅算法无芯片）, https://github.com/yitian97/SDformerFlow
