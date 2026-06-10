# SDformerFlow 硬件加速器设计可借鉴论文清单

**目的**：为 SDformerFlow（事件相机体素化 + Spiking Swin Transformer + QKFormer/SC 注意力 + 三值/二值 spike + 异构引擎）硬件加速器（28nm ASIC 目标，Event Scatter + SparseMAC + Binary PopCount + DenseMAC）提供高相关引用。

**筛选标准**（2024-2026 优先）：
- 直接支持 spiking transformer / spike-driven attention 硬件化（binary/ternary spike, popcount/XNOR, 无或极少乘法器）。
- 稀疏利用（activation sparsity, TTB-like bundling, product/pattern sparsity, zero-skipping）。
- 异构/多引擎架构（sparse + binary/dense 路径）。
- Event-based / 神经形态视觉硬件，特别是 optical flow / scene flow。
- 内存层次与数据流（3-level SRAM/RF, window tiling, prefetch, spike grouping）。
- 3D stacking / 先进集成用于 on-chip memory 压力缓解。
- FPGA/ASIC 实现 + 28nm/类似工艺能效数字，便于对标。
- 避开纯算法或仅 ANN transformer 加速器。

---

## 顶级必引（与本设计高度重叠，2025 顶级会议/期刊）

| 论文 | 会议/期刊 | arXiv / DOI | 核心贡献 | 与 SDformerFlow 硬件的直接借鉴点 | 推荐引用位置 |
|------|-----------|-------------|----------|----------------------------------|-------------|
| **Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-constrained Pruning** (Boxun Xu et al.) | ISCA 2025 | arXiv:2505.12281 | 首次 spiking transformer 专用加速器；提出 **Token-Time Bundle (TTB)** 打包时空 spike；异构 Dense/Sparse cores + ECP 剪枝。 | **TTB 调度完全匹配**（我们的 H41 profile 高度稀疏的 window token + timestep）；异构引擎（Sparse Core 做 1b-spike × INT8，Dense 做 FP）；6.11x 能效 vs prior；直接用于我们的 Sparse MAC + TTB Stratifier 设计。 | Hardware Architecture Overview, Sparse MAC Engine, Comparison Table, TTB 调度图 |
| **FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture** (T. Li, J. Li et al.) | IEEE Trans. Computers 2026 (earlier arXiv ~2505.12771) | — (见 IEEE Xplore) | **Dual-engine overlay**：Sparse Engine (activation sparsity) + **Binary Engine (spiking attention)**；AND-PopCount 注意力；细粒度 sparsity 利用；1.39x / 2.4x 能效提升 vs FireFly v2。 | **与我们 Binary Engine + SparseMAC 架构几乎一模一样**！FireFly-T 的 binary engine 处理 spiking attention (AND-PopCount)，sparse engine 处理 MLP/Conv 的 binary spike。我们 SC gate / QKFormer popcount + Shiftmax LUT 可直接对标/超越其 binary 路径。**核心对标对象**。 | Binary Engine 微架构详述（Figure 对比），Dual-Engine 总览，Energy Efficiency 对比 |
| **Spiking Transformer Hardware Accelerators in 3D Integration** (B. Xu et al.) | ICCAD 2024 | arXiv:2411.07397 | **首个 spiking transformer 3D 加速器**；F2F 3D bonding (memory-on-logic, logic-on-logic)；空间/时间权重复用；专用 spiking trans 架构与物理设计 co-opt。 | 解决我们 **Decoder 中间激活 149MB+** 的片上 SRAM 瓶颈；3D 可极大增加 on-chip memory 容量而不增加面积；权重复用策略可用于我们的 weight prefetch + window SRAM。 | Memory Hierarchy & Storage Architecture, 3D Integration 讨论（未来工作或面积优化），System Integration |
| **Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing** (B.Y. Chen et al.) | 2025 (IEEE?) | arXiv:2503.19643 (html) | 首个低功耗 spiking vision transformer 加速器；解决 non-spike computation 问题；**reconfigurable parallel timestep** 计算。 | 直接针对 spiking transformer（Spikformer 系）；parallel timestep 策略可用于我们 PSN (T=5) 的 temporal mixing 以及多 bin voxel 输入；低功耗设计目标与我们 <5W 边缘目标一致。 | PSN / Temporal 单元，整体能效 claim，Related Work 中的 spiking transformer HW 小节 |

---

## Event-based Optical Flow / 神经形态视觉硬件（应用级对标）

| 论文 | 会议/期刊 | 关键数据 | 借鉴点 | 引用位置 |
|------|-----------|----------|--------|----------|
| **ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow** (J. Wang et al.) | IEEE TVLSI 2025 | 7.9mW, 104 FPS, 0.3 pJ/SOP @28nm；异步 event-driven。 | **完美应用匹配**：事件相机 → 实时光流；异步 scatter-add voxel/grid 更新；我们的 Event Scatter Unit + 增量 VoxelGrid 更新直接从中借鉴（双线性 scatter、乒乓缓冲、只处理新增事件）。我们的目标 30FPS@<5W 可直接对标其 104FPS@7.9mW。 | Event Scatter Unit, Voxelization 优化, 光流专属优化章节, Comparison (application-specific) |
| hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow | ~2022 (Pitt / NSF) | 事件流硬件加速架构，实时性能。 | 早期事件光流 HW 数据流；plane-fitting 或类似局部方法 vs 我们的 learning-based。 | Related Work (event flow HW) |

**FPGA 事件光流参考**（原型验证用）：
- gorchard/FPGA_event_based_optical_flow (plane fitting, ISCAS 2018 及后续)
- 各种 reconfigurable event-based optical flow on FPGA。

---

## 稀疏性利用 SNN 加速器（补充 Bishop/FireFly 的 sparsity 技术）

| 论文 | 会议/期刊 | arXiv | 核心思想 | 借鉴 |
|------|-----------|-------|----------|------|
| **Prosperity: Accelerating Spiking Neural Networks via Product Sparsity** (C. Wei, C. Guo et al.) | HPCA 2025 | arXiv:2503.03379 | **Product Sparsity** (利用先前内积结果复用)；TCAM matching 检测 shortcut；SNN 加速器架构。 | 我们的 SC gate / popcount 注意力天然有大量 "product" 结构（sign 匹配）；可用于进一步优化 Binary Engine 的 score 计算或 gate 复用。 |
| **Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks** (C. Wei et al.) | ISCA 2025 | arXiv:2505.10909 | **Pattern-based hierarchical sparsity** (两级：pattern + unstructured)；离线模式选择 + 运行时动态生成；3.45x speedup, 4.93x energy eff。 | 我们的 spike 激活（尤其是 attn 后几乎全零、sn2 极低 firing）有明显 pattern；可用于结构化 pruning controller 或 TTB 内的 pattern 检测，进一步提升 zero-skipping。 |

---

## 其他高价值 FPGA / SNN 硬件（原型、survey、注意力特定）

- **SeaSNN: Hardware implementation of FPGA-based spiking attention neural network accelerator** (S. Geng et al.), PeerJ Computer Science, 2025.  
  FPGA 上 spiking efficient channel attention (SECA)；轻量 attention 机制 + 并行优化。**借鉴**：FPGA 原型验证 spiking attention（我们的 Binary Engine 可先在 FPGA 上快速验证 popcount + LUT）；资源优化技巧（loop unroll/pipeline）。

- **FireFly 系列**：
  - FireFly: A High-Throughput and Reconfigurable Hardware Accelerator for Spiking Neural Networks, IEEE TVLSI 2023.
  - FireFly v2: Advancing Hardware Support... Spatiotemporal FPGA Accelerator.
  - FireFly-S: Exploiting Dual-Side Sparsity... (2024)。
  **借鉴**：DSP 优化、spatiotemporal 处理、reconfigurable overlay；我们的 FPGA 原型路径可参考。

- **SENECA 系列** (imec)：
  - SENECA: building a fully digital neuromorphic processor... (2023, PMC10326429 等)。
  - 相关：ANN vs SNN comparison on SENECA using event-based vision。
  **借鉴**：**3-level memory hierarchy (RF / local SRAM / shared)** 直接采用；event-driven depth-first processing；spike grouping 减少 memory traffic；RISC-V 灵活控制器 + 专用 NPE。我们 Window SRAM + RF per PE + DRAM prefetch 设计受其启发。用于 event vision 公平对比的实验方法也可引用。

- **SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks** (S. Narayanan et al.), ISCA 2020.  
  经典 SNN 数据流：compressed, time-stamped, sorted spike 序列；高度复用 membrane potential、input spikes、weights。**借鉴**：早期 SNN 专用数据流基础；我们的 spike 流处理（尤其是 decoder 前的 binary spike）可参考其 ordering + 压缩策略。

---

## 推荐补充引用（算法支撑硬件可行性）

- QKFormer / Spike-driven Transformer 系列 (NeurIPS'23, ICLR'24 等)：证明 spike-driven SDSA (mask + add, 线性复杂度，无乘法) 的可行性与精度 → 我们的 Binary Engine (AND-PopCount + 零乘法) 的算法基础。
- 各种体素化 / event representation 论文（已在 voxel_papers 目录）用于 Event Scatter 输入端。

---

## 建议在论文中的使用方式

1. **Related Work > SNN Hardware Accelerators**：
   - 分小节：通用 SNN ASIC/FPGA (SENECA, FireFly, SpinalFlow, ODIN 等)；Spiking Transformer 专用 (Bishop, FireFly-T, 3D SpkTrans, 2025 reconfig one)；Event-based Vision HW (ASNA-Flow, hARMS)。

2. **Hardware Design Section**：
   - Architecture Overview：引用 Bishop (TTB + hetero) + FireFly-T (dual-engine) 说明 "受 XX 启发，我们提出针对 event scene flow 的四引擎异构设计（Event Scatter + Sparse + Binary + Dense）"。
   - Binary Engine：重点对标 FireFly-T 的 AND-PopCount，强调我们 SC (sign consistency) + full ternary + Shiftmax LUT 的创新（三值、零浮点、3-cycle/token）。
   - Sparse Engine：Bishop TTB + Prosperity/Phi 的 sparsity 技术。
   - Memory & Dataflow：SENECA 3-level + 3D paper 的 stacking 讨论。
   - Application-specific：ASNA-Flow 作为 "prior event flow accel，但非 transformer-based；我们的端到端 SNN transformer flow 是首次"。

3. **Evaluation / Comparison**：
   - 必须有 Table：工艺、FPS、功耗、能效(TOPS/W 或 pJ/SOP)、支持 SNN 类型 (binary/ternary)、是否支持 transformer/attention、是否 event flow。
   - 对标：Bishop / FireFly-T (spk trans) + ASNA-Flow (flow) + 28nm 通用 SNN + GPU baseline。
   - 突出独特卖点（已在设计文档中总结）：唯一硬件化 ATLIF 三值 + 全 SC PopCount attention + Event→Voxel→SNN 端到端光流。

4. **Future Work**：3D integration (Xu ICCAD), CIM variants, analog in-memory for PSN 等。

---

## 获取论文方式

- arXiv 直接搜索标题或 ID。
- IEEE Xplore / ACM DL（学校订阅）。
- open-neuromorphic/awesome-neuromorphic-hw GitHub repo 有大量链接和分类（包括 FireFly 系列）。
- 对 DATE / ISCA / HPCA / ICCAD 等会议论文，优先用官方 DOI。

**更新记录**：2026-05-29 初始整理。后续 profile 数据或 RTL 数字出来后，可补充具体能效对比条目。

如需我帮你：
- 为特定论文生成引用 BibTeX
- 扩展某个模块的 microarchitecture 对比图描述
- 把此清单合并进 HARDWARE_ACCELERATOR_DESIGN.md 的参考表
- 搜索特定子方向（e.g. only FPGA prototypes, or only 3D/CIM）

随时说。