# full-paper-workflow 状态 — SDformerFlow 事件光流 SNNTransformer 软硬件协同优化（目标 DATE 2026）

**工作流启动时间**：2026-06-07（按照视频推荐的 Route 1 灵活组合 + Route 2 一站式，遵循 full-paper-workflow 规范驱动）
**当前阶段**：Phase 2（框架已锁定） + Phase 3（结果图规划完成，描述待人工确认） + 硬件 Week 1 并行（数据流规格细化）。根据用户“继续”指令推进至 Phase 4（正文撰写启动）。
**目标会议**：DATE 2026（架构与微架构设计、低功耗与能效设计、近似计算、嵌入式/边缘/神经形态系统、机器学习架构设计方法学）
**推荐路线**：Route 1（每阶段灵活选择工具），重点使用 research-paper-writing（DATE 系统协同设计逻辑 + 主张-证据-推理纪律） + nature-figure（高质量结果图规划）。必要时将严谨性检查移交给 academic-pipeline。
**总体目标**：将已完成的软件重设计实验与协同设计规划，转化为清晰的“硬件感知的稀疏三值 Spiking Transformer 事件光流”论文，具备真实硬件证据链（静态阶段感知异构调度、算子级代价模型、层级调度表、能耗/延迟代理、微架构草图）。

## 三个铁律（启动时已确认，永不跳过）
1. **每重大阶段后必须版本保存** —— git commit 或带日期拷贝关键目录（neuron_autoresearch/、hw/、docs/、paper_artifacts/）。
2. **从 Phase 0 开始严格匹配 DATE 风格** —— 故事主线必须是“协同设计 + 可实现性”（静态调度，而非“我们换了更好神经元”）。避免纯 CV 会议的“精度 + SOPs”叙事。
3. **所有生成的图、数据、主张必须人工核对** —— redesign 计划中的每一个数字、未来硬件模型等，在使用前必须显式确认。工作流将在“已人工核对通过”后放行。

## Phase 0 入场总结（已完成）
- **当前材料**：
  - 软件实验扎实：neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md（详细 H 系列表格：SN/SC/TX/HT 注意力、S02/S012 FFN 替换，SOPs 降低至约 2.8G 左右，精度瓶颈约 AEE 1.73 vs baseline 约 1.58-1.66，目标退化 <5%）。
  - 协同设计思路清晰：PAPER_CO_DESIGN_PROPOSAL.md（深度注意力注入、三级结构化稀疏、IO 感知内核）。
  - DATE 专项规划优秀：DATE_2026_TARGETS_STATUS_ROADMAP.md、docs/DATE_PAPER_SKELETON_CN.md（已将“部分 block 替换”重构为“阶段感知静态异构调度”）、HARDWARE_RESEARCH_ROADMAP.md（6 周计划）、HARDWARE_ACCELERATOR_DESIGN.md、HARDWARE_DATAFLOW_SPEC.md。
  - 硬件实现现状：hw/rtl/（部分文件：top、controller、pe_array、spike_unit、token_mixer、attention_unit 等）、hw/docs/、golden 仿真。硬件仍以规划和骨架为主（符合用户描述）。
  - 其他：PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md、性能画像、src/models、third_party/SDformerFlow baseline。
- **进入点**：Phase 2（框架，利用现有高质量骨架锁定） + Phase 3（结果图规划，将现有实验表格转化为高质量可视化描述） + 硬件规划并行启动。软件实验已成熟，硬件需从数据流规格开始系统推进。
- **入场识别的关键风险**：
  - 在保持稀疏优势的同时恢复精度（核心瓶颈）。
  - 将 SOPs/发放率转化为真实硬件指标（能耗、面积、延迟、访存）。
  - 将“部分替换”故事转化为优势（编译期静态调度表），而非劣势。
  - 分割口径一致性（redesignmd 推荐使用 canonical 825 valid_split_seq.csv）。

## 阶段状态与下一步（已根据“继续”更新）
- **Phase 0（设置与版本）**：已完成。状态文件创建，三个铁律已声明。
- **Phase 1（调研）**：基本完成（依赖前期 autoresearch + docs/literature/ 文献）。仅在相关工作或 baseline 补充时再调用。
- **Phase 2（框架锁定）**：已完成。基于 DATE_PAPER_SKELETON_CN.md + PAPER_CO_DESIGN_PROPOSAL.md 作为强先验，锁定单一主故事弧 + 调度表概念。已产出中文框架文档。
- **Phase 3（实验结果图规划）**：规划完成（描述阶段）。已根据 redesign 计划中的关键表格（Pareto、阶段 SOPs 分布、发放率收益）规划 3 张核心图。暂不生成代码，按用户要求先以文字描述和规划为主。后续确认后可由 nature-figure 正式产出。
- **Phase 4（正文撰写启动）**：已启动。使用 research-paper-writing（DATE 协同设计视角）产出 Method 子节草案（中文）。重点强化“静态异构调度为何是 DATE 可发表贡献”的主张-证据链。
- **硬件集成**：显式跟踪。遵循用户 6 周路线图（Week 1：完整数据流规格细化，为最高优先并行任务）。已基于 HARDWARE_DATAFLOW_SPEC.md 产出 Week 1 行动计划（中文）。
- **检查点**：每次重大阶段后必须人工确认 + 版本保存，方可跨越 Phase 3/4 主要部分及所有完整性门。

## 版本历史（本工作流）
- v0.1（2026-06-07）：工作流启动，Phase 0 完成，状态文件 + 初始产物创建。三个铁律已在启动时确认。首次框架与结果图规划已分发。
- v0.2（2026-06-07，用户指令“继续”）：Phase 2 框架已根据 DATE 骨架 + 协同设计提案锁定。Phase 3 结果图规划完成（3 张核心图的文字描述与规划，已基于 redesign 计划表格）。硬件数据流规格（来自 HARDWARE_DATAFLOW_SPEC.md）已作为 Week 1 基础读取并细化。主张-证据链地图已扩展为中文草案。严格遵守用户“先别动代码”要求，所有图规划以描述为主，不生成或修改任何代码。下一个检查点：显式“已保存 + 核对通过” + 确认生成内容。

**本次推进后的用户行动要求（v0.2 检查点）**：
- 立即执行版本保存：例如 `git commit -am "v0.2 full-paper-workflow: 框架已锁定 + 结果图规划完成 + 硬件 Week1 行动计划"`，或带日期拷贝 paper_artifacts/ 及关键文档。
- **结果图规划人工核对**：查看 paper_artifacts/ 中的框架与草案文档，确认基于 EXPERIMENT_REDESIGN_PLAN.md 的关键数据（AEE、SOPs、发放率、阶段分布）。如有数据问题请指出。
- 请回复确认，例如：“已保存 v0.2，框架与图规划人工核对通过，继续 Phase 4 完整章节撰写 + 硬件调度表细化 + 准备完整性检查”。
- 之后将分发：完整 Method/Experiments 章节中文初稿、硬件调度表具体示例、准备调用 academic-pipeline 进行完整性门检查（如需要）。

**工作流运行状态持续维护**。所有子技能（nature-figure、research-paper-writing、academic-pipeline 等）仍可独立调用。

---
**协调器备注**：本状态文件将在每次检查点更新。严格遵循 full-paper-workflow 规范（视频 13 种工具组合 + 铁律）。本次所有 MD 输出均为中文，暂不触碰任何代码。
