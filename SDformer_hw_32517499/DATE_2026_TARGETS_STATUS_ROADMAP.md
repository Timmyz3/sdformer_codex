# DATE 2026 HW/SW Co-Design 项目目标、现状与路线图

**项目定位**：基于 SDformerFlow baseline 的 **事件相机光流 SNN Transformer 软硬件协同设计**。
- 软件：从 SDformerFlow (Swin Spikeformer / QKFormer-style) 改进，引入自适应三值神经元 (PSN+ATLIF ternary)、硬件友好注意力 (TX ternary_alpha_xNOR / signed_consensus + shiftnorm/popcount-L1)、stage-aware FFN 稀疏替换。
- 硬件：静态异构调度 (stage-aware static schedule)、算子级能耗模型、稀疏加速器子系统 (spike neuron engine + ternary attention engine + sparse FFN + controller)。
- 目标会议：DATE 2026（电子设计自动化与测试，强调低功耗、嵌入式、近似计算、架构/微架构、神经形态系统）。
- 量化目标（用户指定）：
  - 准确率：在 baseline 5% 误差内（AEE 相对退化 ≤5%，目标 ~1.66 以内，视 baseline 精确口径）。
  - 稀疏度：提升 20%（SOPs 降低 ≥20%，firing rate 显著下降；当前已超）。
  - 故事结构：清晰、可讲的 HW/SW co-design 叙事（不是单纯模型 trick 替换）。
- 当前实验主力记录：`neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md`（H 系列注意力/FFN/训练策略搜索）。

更新日期：2026-05（基于 redesignmd + DATE skeleton + co-design proposal + hw docs 综合）。

## 1. 当前结果 vs 目标（关键数据摘自 redesignmd）

### Baseline (PSN / upstream style)
- valid40：AEE ≈ 1.585, AAE ≈ 7.501, SOPs ≈ 3.622G, firing ≈ 0.085
- 后期 full-valid816 参考：AEE 1.33 左右（split 差异需注意），SOPs ~3.96G

### 当前最优稀疏候选（H41-TX S02 C slowbb epoch27 等）
- AEE 1.732 (+9.3% vs 1.585)
- AAE 8.404 (+12%)
- SOPs 2.615G (-27.8%)
- firing 0.061

**其他亮点**：
- TX S02 / SC S012 / SN S02 三强在 Phase 短测中反复出现，SOPs 可压到 2.8-3.2G 区间。
- H49 QK-selector (ternary alpha xnor qkselector) 短测有希望：在保留线性复杂度 + K carrier 的前提下做 selector，360-step valid40 最佳 AEE 1.80 / AAE 8.54 / SOPs 2.97G。
- S02 (stage0+2 FFN 替换) 通常优于 S012 或全替换。
- SOPs 目标已**超额**（-20% 轻松达到 -27%+），核心瓶颈是**精度恢复到 5% 误差内**，同时不把 SOPs 推回 3.5G+。

**split 口径警告**（redesignmd 反复强调）：
- 本机常使用 816 样本 valid。
- 推荐论文统一 canonical 825 样本 `valid_split_seq.csv` (sha 7f3dc28...) + 明确报告 samples。
- 所有用于主表的 checkpoint 必须在同一 split + 同一 profile 脚本下重跑（包括 baseline）。

**精度-稀疏 Pareto 现状**：SOPs 已经很好；需要把 AEE 从 1.73 拉回 ≤1.66 左右（或在同口径下证明相对 baseline 退化 <5%）。

## 2. 故事结构（强烈推荐直接沿用并细化）

`docs/DATE_PAPER_SKELETON_CN.md` 已经为 DATE 准备了**极佳的讲故事框架**，核心不是“换了几个神经元”，而是：

**标题建议**（skeleton 给出）：
Hardware-Aware Sparse Ternary Spiking Transformer for Energy-Efficient Event-Based Optical Flow
（或 SATFlow / HASTE-Flow 等简洁缩写）

**核心贡献（4 点，skeleton 推荐）**：
1. Adaptive ternary spike primitive（PSN 保表达 + ATLIF 自适应阈值控稀疏 + 对称三值保正负事件方向）。
2. Hardware-oriented ternary attention（sign/valid 编码、alpha-XNOR 或 signed consensus 替代 dense score、shiftmax/shiftnorm/popcount-L1 三种归一化对比，证明乘法/LUT 减少）。
3. Stage-aware static replacement schedule（基于 SOPs contribution + sensitivity 分析，在编译期固定替换 mask；支持 stage 级 / even-odd / high-SOP FFN 规则模板；运行时纯静态调度，无动态控制开销）。
4. Hardware cost & event-flow validation（AEE/AAE + SOPs/firing + energy proxy / latency proxy / memory traffic；layer schedule table；与 baseline、binary-only、不同归一化消融）。

**为什么这个故事适合 DATE**（skeleton 分析）：
- 匹配 Architectural/microarchitectural design、Low-power/energy-efficient、Approximate computing、Embedded/edge/neuromorphic、Design methodologies。
- 把“部分 block 替换”从缺点变成优点：**静态异构硬件映射**（schedule table + 预编译 kernel 切换）。
- 必须补的硬件证据链：算子能耗模型（MAC/add/cmp/XNOR/popcount/shift/SRAM）、关键 kernel 微架构草图、layer-wise schedule、per-stage breakdown、fixed-point 敏感性。

**避免的坑**：
- 不要只报模型 AEE/AAE/SOPs。
- 不要把替换说成“随便挑几个 block 试试”。
- shiftmax 含 2^x 要说明成本，并保留 shiftnorm / popcount-L1 作为硬件更友好对照。

`PAPER_CO_DESIGN_PROPOSAL.md` 进一步补充了 multi-level sparse scheduling (timestep/window/head/token)、IO-aware kernel、accuracy recovery head、QAT 联合校准等可落地的升级路径。

`hw/docs/` (arch.md, perf_model.md, interfaces.md) 已给出 target accelerator blocks、dataflow、cycle/energy/area model 框架（当前 RTL 是最小可综合 skeleton，attention_unit.v 等仍需实质 datapath）。

## 3. 路线图与优先级（已转为结构化 todo）

P0（必须先做，否则所有数字不可信）：
- Canonical evaluation pipeline：固定 825 split、统一 profile 脚本（AEE/AAE/PE1-3 + SOPs + firing + threshold stats + pos/neg 平衡）、对 baseline + 2-3 个 top candidate（H41-TX-S02、H49 变体、SC S012）+ 关键 epoch 重跑 full valid825 并落盘 sops_summary + 报告。
- 建立 PAPER 实验 master table（明确 split sha、samples、eval 脚本版本）。

P1（精度达标 + 故事闭环）：
- Accuracy recovery 实验：
  - Progressive / gradual sparsity schedule（H45 思路：早 epoch 宽松 target 学特征，后期压）。
  - Feature distillation（H46 思路：用 PSN baseline 做 teacher，保护中间特征，尤其是 stage2 FFN 输出）。
  - Voxelization 改进（voxelization_experiments/VOXELIZATION_NEXT_REVIEW_20260523.md 已分析 EDCFlow temporal diff residual、EventPillars Lite；优先 residual 轻量版，避免硬 mask 破坏光流方向）。
  - 必要时加 lightweight refinement head（FlowFormer / SEA-RAFT 风格，低分辨率复用 PE array）。
- 消融：Q K only vs S02 vs S012、shiftmax vs shiftnorm vs popcount-L1、binary vs ternary FFN、target_rate / activity_eta sweep。
- 至少 2-3 seeds for final configs。

P2（HW 证据 + 可复现代码）：
- 实现/增强 energy/latency/memory traffic proxy（从 SOPs 升级；按 MAC/add/XNOR/popcount/shift/SRAM 加权；导出 layer schedule table）。
- 把实验 harness 里的 winning attention（ternary_alpha_xnor / H49 qkselector）干净 merge 到 `src/models/`（当前很多在 neuron_experiments/H9_bipolar_self_attention/ 独立；baseline 仍用 upstream QK 路径）。
- 推进 hw/rtl：至少实现一个真实 ternary attention / spike datapath（当前 controller/attention_unit 仍 skeleton/passthrough）。
- Fixed-point 敏感性 + 推理时阈值 fixed-point / sign-valid 编码 + layer scale 吸收（避免推理必须乘任意实数阈值）。

P3（论文产出）：
- 按 DATE skeleton 写 Introduction/Method/Results/Hardware Analysis。
- 准备主表（AEE/AAE/SOPs/firing/energy proxy/latency）、Pareto curve、per-stage breakdown、schedule ablation、normalization ablation、fixed-point 敏感性。
- 补充 hardware figures（Co-Design Flow、Kernel Microarch、Layer Schedule）。

## 4. 立即可执行的下一步建议

1. **今天/本周优先**：解决 split 口径 + canonical profile。所有后续决策和论文数字都依赖它。
2. 基于 H41-TX / H49 结果，快速实现 1-2 个 acc recovery（progressive schedule + EDCFlow residual voxel adapter + 简单 distillation）。
3. 写一个 `tools/export_paper_table.py` 或增强 profile 工具，自动生成用于论文的 summary（带 split 指纹）。
4. 决定主线方案后，把 attention 逻辑回迁到 src + 更新 baseline/variant configs。
5. 补 energy model 脚本（参考 hw/docs/perf_model.md），对当前最佳 checkpoint 跑一遍 proxy 数字。

## 5. 参考文档速查

- 实验全景：`neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md`（最详细 H 系列结果、公式、演化史、H49 介绍）。
- DATE 故事骨架：`docs/DATE_PAPER_SKELETON_CN.md`（强烈建议作为论文主线直接扩展）。
- Co-design 升级路径：`PAPER_CO_DESIGN_PROPOSAL.md`。
- HW 架构/模型：`hw/docs/arch.md`、`hw/docs/perf_model.md`。
- 体素化下一步：`voxelization_experiments/VOXELIZATION_NEXT_REVIEW_20260523.md`。
- 运行/训练：`README.md`、`RUNBOOK_AND_RESEARCH_PLAN_2026.md`、`scripts/`、`tools/profile_sops.py`。
- 主代码：`src/models/`（spiking_neurons, sparse_ops, sdformer layers）、`third_party/SDformerFlow/`（upstream baseline）、`neuron_experiments/H9_bipolar_self_attention/`（当前 redesign attention 实验田）。

**状态**：sparsity 已达标；accuracy 接近但未达 5% 误差线；故事框架优秀；HW 证据链和代码主线集成是主要待补环节。

下一步请指定具体任务（例如：“帮我实现 canonical valid825 profile 脚本并对 baseline + H41 最佳 ckpt 跑一遍”、“把 H49 集成到主 backbone”、“实现 EDCFlow residual voxel adapter + 短测”、“起草 paper Method 部分 based on skeleton”、“review 并补 hw/rtl 的 attention datapath” 等）。我可以立即开始编码、分析结果、生成配置或文档。

目标明确：**2026 DATE 可投的、讲得通的、数字过硬的 HW/SW co-design 论文**。
