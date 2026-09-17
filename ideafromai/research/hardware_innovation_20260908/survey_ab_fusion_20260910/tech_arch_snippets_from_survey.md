# 技术详稿拼装片段（来自 survey_ab_fusion_20260910）

**用途**：供拼进 `TECH_ARCH_SOFT_HARD_FULL`（软件算法摘要 + F1–F7/融合 + idea 候选边界）。  
**作者角色**：调研侧摘录与边界；**不是** Stage B 执行裁决，也不是生产 RTL/PPA。  
**规范落盘**：`sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/`  
**材料源**：`understanding_and_gaps.md`、`idea_synthesis.md`、`ab_fusion_candidates.md`、`ab_fusion_priority.md`、`fusion_status_overnight.md`、`p0_deepread_progress.md`；Stage B 现况数字对齐 `CURRENT_LINES_AND_PLAN_20260910.md` / `CODEX_NEXT_PLAN.md`（以仓内最新为准）。  
**硬纪律**：借入 Gustav / da4ml / 普通压缩 ≠ 自称标题级 X；负结果只停布局不杀全家；禁动 `nts07` / `main.tex` / 生产 RTL；融合与 idea 卡均为**第二队列**；Grok `grok_review_20260911` 仅作对照，不当标题 X。

---

## A. 软件算法栈摘要（主岛在算什么）

### A.1 论文语境与主目标（一句话）

面向 **TCAS-II Express Brief** 的光流 **SNN Transformer 软硬件协同**。主目标：在敏感 **patch 残差链 r1** 上，用**可学习 lifting 结构化 T10 PSN**（执行对象与消费者接口）相对 **ordinary dense-source/raw**，在 **同端口 / 同状态 / 同背压** 下证明**同资源净服务**优势，并守住精度门；过门后再谈成对恢复与隔离 RTL。

### A.2 算法挂点（残差链）

典型消费链（概念顺序，非强制 RTL 模块名）：

1. **Source / T10 时间变换**：ordinary dense 源字 vs lifting `fast_raw_diagonal`（可学习提升因子 + 完整常量编译图）  
2. **sn1 → Conv → sn2**：残差块内卷积与尖峰/门相关中间量  
3. **θg / 门与连续消费者**：真实非单位 θ、判决阈值分离；**双 PED（门 + 连续投影）** 为强消费者  
4. **BN / shortcut / 粗头**：与定点/恢复合同绑定  
5. **有限服务合同**：端口、状态驻留、背压、请求组；不可把独立切片周期相加冒充整链

**为何挂 r1**：历史粗头后活动加权账本中整个 patch ~34.83%、r1 两卷积 ~11.07%（**历史代理 ≠ 新学生周期份额**）。结构已改变源活动后，必须用**新分母**重记账。

### A.3 主候选 vs 支线 vs 底座

| 地位 | 对象 | 含义 |
|---|---|---|
| **第一主候选** | patch r1 上结构化 T10 PSN（lifting） | 待证 X：执行对象与消费者接口差分；非“已替换完整 C1” |
| **条件支线** | shared-Q（同一可逆时间坐标） | 仅消融；无净硬件收益则不留共享标题 |
| **条件支线** | 纯门出口 / 部分跨 RNE 合并 | 全合并布局已负；仅允许有证书/回退的新差分 |
| **共同底座（非标题）** | Gustav NRV/供数、da4ml 整图 CSE、普通低秩/量化/剪枝、fixed-BN、DeepShift 对照等 | 可继承到同分母；**借入 ≠ X** |
| **第二队列** | 敏感 patch 结构剪枝、Prosperity 联合图、完整 Gustav 物理链对齐等 | Stage B 后或明确挂队；不抢主实验 |
| **旁路** | motion / 注意力 K=0 等 | 有限份额，不排主岛 |

### A.4 精度与费用门闩（合同数字）

| 门 | 门槛 | 现况（以 CURRENT_LINES / net_cost 为准） |
|---|---|---|
| 绝对 AEE（valid825） | ≤ **1.259** | lifting raw `fast_raw_diagonal` ≈ **1.233**（过）；ordinary dense/raw ≈ **1.220** |
| 相对同预算强对照 ΔAEE | ≤ **+0.005** | **失败**：lifting vs ordinary ≈ **+0.013**（上限对照约 1.2248） |
| 同资源净服务 | 整段潜力相关判据约 **≥10–15%**（灰区语义见 CURRENT_LINES） | **完整链仍未 PASS**；不得用节点比/十帧代理冒充 |
| CSE 节点线索 | — | dense **260** 加减；lifting raw **159** + **35** 中间 RNE；**≠ 周期** |
| 十帧代理 | — | 加权项/H8 相对旧 dense 约少 ~15%；**NOT schedule-closed** |

**推论**：绝对预算过 ≠ 晋级；相对门红灯下**不晋级**性能 RTL / 主贡献句；保留研究资格与有限执行模型。

### A.5 Stage B 现况快照（拼装时以仓内最新覆盖本节）

> 下列数字来自 2026-09-11 `CURRENT_LINES` / Codex 进度，**覆盖** survey 过夜稿里「Stage B 尚未开工」的旧述。

- **对照**：ordinary dense-source/raw vs lifting40 `fast_raw_diagonal`，same-port / same-state / same-backpressure  
- **路径锚**：`…/fast_temporal_recovery_lifting40/schedule_compare_same_port/`（含 `two_stage_writeback/`）  
- **两级写回资源点（源核）**：ordinary **6938**；lifting 不融合 **6194（−10.7%）**；融合后 **5354（−22.8%）**——其中 6194→5354 多为**通用指令融合**，**不可写成标题级 X**  
- **固定长背压**：四臂均 **8088**（优势被吃掉）  
- **完整链**：仍未 PASS；相对 AEE 门仍红  
- **下一接口（锁定）**：接**真实前驱/后继完成与 ready 依赖**（源图 + FP32 preview + sn2 + 整数 U16/F/BN2 + 双 PED + 原生 proj）；**不要**把源槽与另一 MAC 后端槽直接相加；**不要**扫流水参数；**不要**因此启动 paired recovery / 精度恢复 / 生产 RTL / `main.tex`  
- Codex 额度约锁至 **2026-09-15 ~11:35**；Grok Build `01a08b28` 可接手，但须钉上述主线（勿逐卡长审）

### A.6 已停融合布局（只停布局，不杀家族）— 精简表

| 家族/布局 | 结果要点 | 纪律 |
|---|---|---|
| Prosperity∪APEC-θ（G4） | 比 Prosperity 慢；大缓存仍慢 | 停该融合布局 |
| LoAS 启发 C2 静态共享 | 加法少但周期略慢 | 停静态共享层 |
| Gustav 部分供数+时间类别 | 强对照后优势变薄/反慢 | 类别名不晋级；物理链未闭 |
| s2b3 上 C16/Gram 剪枝打磨 | 挂点不敏感 | 停该块打磨；敏感 patch 仍可研究 |
| lifting 全半步门图合并 | 241 加减 ≥ 原 159+35 有效 | 停全合并；部分合并+证书仍可议 |
| shared-Q 标题化 | 增量薄 | 仅消融 |

完整列表见 `CURRENT_LINES` §3；拼装技术详稿时应保留「停止的是哪一层」。

---

## B. F1–F7 / A+B 融合（第二队列设计）

**性质**：鼎汇/鼎勘式候选；**不抢** Stage B；**不做**本文件内实验。  
**优先**：**F1 → F2**（见 `ab_fusion_priority.md`）。

### B.1 总览

| ID | 一句话 B | 与 Stage B | 排队 |
|---|---|---|---|
| **F1** | 敏感 patch r1 上，按 lifting 改变后的源活动 + 双 PED/门误差，选可共同删的物理源字 | Stage B **后**可挂；依赖净服务结论 | **优先 #1** |
| **F2** | lifting 半步/RNE 检查点上，共享请求组「接受/继续」有损共同完成 | 第二队列；看并集/存活是否瓶颈 | **优先 #2** |
| F3 | Prosperity 联合图/公共虚节点+部分 lane 挂到 lifting **源 DAG** | 第二队列 | 备选 |
| F4 | **仅末5门前推** 部分合并 + 轻量误差证书/回退 | Stage B 后可挂（若 RNE/端口吃掉节点优势） | 备选 |
| F5 | 训练约束 lifting 中间量/广播组依赖生存期 → 有限 RF 周转 | 依赖 Stage B 结论 | 条件 |
| F6 | 有界 clip 严格门完成 × lifting 可取消时间列 | 第二队列 | 备选 |
| F7 | Gustav 物理供数上按 lifting 因子组对齐打包（Gustav 仍是底座 A） | Stage B 后可挂；Stage B 期间 Gustav 只作分母 | 条件 |

### B.2 F1（首选）要点

- **B**：固定广播域内共同删除物理源字（含 H 重排同步系数/阈值），适合共享供数且保住隐藏通道。  
- **最强对照**：ordinary 同预算窄稠密/hidden50；完整 HiNM（含二阶）/VENOM-CRISP；幅值/活动 C16；同 lifting 无剪枝端点。**禁止**只回 s2b3 不敏感块。  
- **一句话 X（待证）**：按 **lifting 改变后的源活动形状 + 非因果 T10×双 PED 联合扰动** 决定可删字，而非只换损失名。  
- **杀门**：不胜 HiNM/窄稠密；只降 W² 不降物理源字/门/流误差；AEE 破 1.259 或相对 +0.005；并集仍读满且无周期余量 → **只停该 r1 剪枝布局**。

### B.3 F2（次选）要点

- **B**：半步写回/必要 RNE 边界检查点；部分状态预测整组 T10 门字；一组接受则发预测 θg 并退休请求，否则继续原算。  
- **对照**：完整 Gustav 式共享同步（无预测）；独立每神经元预测 + **相同**组关闭；静态窄层；lifting 无预测端点。  
- **一句话 X（待证）**：共同完成挂在 **lifting 半步自然边界** 与 **r1 多消费者并集费用**。  
- **RTL 粗验（box，≠ Stage B）**：MP2 group accept/continue **PASS**（接受~1 拍，重算~6–7 拍）——仅证明控制通路可表达。  
- **杀门**：净服务不胜窄层与「独立预测+同组关闭」；预测开销≈再做一次 PSN；接受后 AEE 爆；组尾被最慢消费者钉死。

### B.4 F3–F7 边界（拼装时一句话即可）

- **F3**：产品式共享挂到 lifting **源图节点**；旧「产品 mask 相对活动 −0.09045%」不得抬标题。  
- **F4**：范围缩到末5 + 证书/回退；禁止复活全合并。  
- **F5**：把非因果 T10+PED 延长占用的中间生存期**写入训练目标**（借入 Avalanche≠X）；乐观上界无 5–10% 余量则不训练。  
- **F6**：训练范围后取消是否在 lifting **列结构**上足够集中；与 F1 并行时优先 F1。  
- **F7**：打包/行导航与 lifting 共设计；对齐后不优于 ordinary 序+同 Gustav → 停该对齐，不杀 Gustav。

### B.5 明确不进主线的旧想法

shared-Q 标题化；Prosperity∪APEC / C2 静态共享 / NR4 费用训练旧版 / gate 全合并；纯抄 Gustav / 纯 PoT 化系数；只在 s2b3 再磨 C16/Gram；注意力 K=0 整窗当主岛。

### B.6 RTL 微探针（box only，服务表达力）

| 探针 | 结果 | 含义 |
|---|---|---|
| MP1 same-port credit | PASS；~179 cells | same-port/背压计分母可 RTL 表达 |
| MP2 group accept/continue | PASS | F2 组级控制通路可行 |
| half-step RNE（若有） | PASS | 半步边界可粗验 |

**≠** Stage B service% / TCAS PPA；不因此开训或改生产树。

---

## C. Idea 候选边界（文献 → 卡 → 怎么用）

### C.1 盘点数字（诚实，2026-09-11）

| 项 | 数 |
|---|---:|
| 文献分层库存 | 732（P0 246 / P1 198 / P2 130 / P3 158） |
| idea 卡 md | **245** |
| CSV unique uid | **252** |
| P0 uid 覆盖 | **246 / 246** |
| gap01–08 excerpt 包提卡 | 90 |

来源：`p0_deepread_progress.md`、`idea_extract_per_paper.csv`、`idea_cards/`。

### C.2 覆盖 ≠ 全文精读

覆盖包含：方法摘录窗、开源诚实盘点、`unresolved_no_fulltext`（无全文不虚报）、以及 **uid 别名行**：

- `MAIN-R276` → `ARX-012` / `arxiv_2407.08356.md`（FPGA event vision survey）  
- `MUSHA-SP001` → `MAIN-R002` / `GustavSNN.md`

**禁止**把「有卡」写成「已 PDF 全文级精读并验证可迁移」。

### C.3 对 F / Stage B 的用法

- idea 卡只提供 **A 继承线索 / 对照轴 / 杀门措辞 / 旁路警告**  
- **优先仍 F1→F2**；Transformer/注意力硬件（ConvFormer/Xpikeformer 等）**旁路主岛**  
- Gustav / FlexSpIM / LoopTree → **F5 底座语言**（供数·驻留·占用），不自动变 X  
- **Stage B**：MP1 支持 same-port 信用计数表达；**仍不代替**净服务实验  
- Grok Build 产出的 `grok_review_20260911/`（逐卡打分等）：**第二队列对照**；结论「没有标题级 X」可作参考，**不得**替代 Stage B 接口工作

### C.4 簇印象（中期 idea_synthesis；计数已旧，簇名仍可用）

粗挂：F1 结构稀疏/索引税；F2 时间并行/半步/组完成；F3 Prosperity 产品稀疏；F5 Gustav 供数；F7 注意力硬件旁路；大量条目同时标 stage-b **仅表示「可能服务分母表达」**，不是授权开 Stage B 外实验。

### C.5 网页 ChatGPT / 外部模型使用本片段时

1. 先读本节 A.5 Stage B 锁定接口，**不要**改去逐卡长审或开精度恢复。  
2. 融合设计只写到「对照清单 + 杀门 + 不改生产树」；**等** Stage B 净服务分项表再决定挂 lifting 与否。  
3. 任何「少 XX%」必须标明：**源核 / 长背压 / 完整链 / 代理账** 哪一层；禁止跨层合成。  
4. 需要全文级细节时，向用户索取仓内具体 md（本片段自洽但不是代码替代）。

---

## D. 给 TECH_ARCH_SOFT_HARD_FULL 的建议粘贴顺序

1. **问题与门闩**（A.1 + A.4）  
2. **算法挂点与主候选**（A.2 + A.3）  
3. **Stage B 现况与下一接口**（A.5）— 用管理/Codex 最新数字覆盖  
4. **硬件流水/资源合同**（由管理从模型代码与 lifting/PSN/Gustav 文档撰写；本片段不替代）  
5. **已停布局纪律**（A.6）  
6. **F1–F7 第二队列**（B）  
7. **Idea 边界与文献诚实计数**（C）  
8. **外部接手提示词**（见群内「Grok Build 锁定主线」文本；idea 卡第二队列）

---

## E. 路径速查

```
sdformer_codex/ideafromai/research/hardware_innovation_20260908/
  CURRENT_LINES_AND_PLAN_20260910.md
  …/lifting40/…/net_cost_one_page.md
  …/schedule_compare_same_port/…
  survey_ab_fusion_20260910/
    tech_arch_snippets_from_survey.md   ← 本文件
    understanding_and_gaps.md
    idea_synthesis.md
    ab_fusion_candidates.md
    ab_fusion_priority.md
    fusion_status_overnight.md
    p0_deepread_progress.md
    idea_cards/  idea_extract_per_paper.csv
    CHATGPT_WEB_HANDOFF_FULL.md
    CODEX_NEXT_PLAN.md
```

`/home/zhumd/work/ideafromai` 为软链，指向仓内 `sdformer_codex/ideafromai`。
