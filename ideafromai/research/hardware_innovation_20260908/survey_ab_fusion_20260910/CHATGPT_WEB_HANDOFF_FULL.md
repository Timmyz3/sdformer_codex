# ChatGPT Web 全量交接（无仓库/无代码访问）

> **给 parent / 人类 scp 备注（文件最上方）**  
> - **本文件 BOX 路径：** `/workspace/overnight_20260911/CHATGPT_WEB_HANDOFF_FULL.md`  
> - **拟同步到 ismd 路径：** `sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/CHATGPT_WEB_HANDOFF_FULL.md`  
> - **绝对路径（ismd 用户 home）：** `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/CHATGPT_WEB_HANDOFF_FULL.md`  
> - **受众：** 仅网页版 ChatGPT（**零** repo / 代码 / 本地 md 访问）——继续调研合成 **或** 硬件规划/方法论批判。  
> - **生成时点：** 2026-09-11（Asia/Shanghai，CST=UTC+8）。下文时间均按 CST 标注。  
> - **纪律：** 只使用下方已给出的事实与数字；未知处标 **UNKNOWN / 未测 / 未决**；**禁止编造** service%、PPA、RTL 时钟、全链 PASS、相对 AEE PASS、shared-Q 标题、把 22.8% 写成净全链 X。

---

## 0. 一分钟读完（给 ChatGPT 的系统级摘要）

你在协助一项 **光流 SNN Transformer 软硬件协同设计**，目标 venue 是 **TCAS-II Express Brief（约 5 页）**。生产树与主稿只读。主新颖性候选 **X** 是：残差链上的 **可学习 lifting 结构化 T10 PSN**（真实 θg + 双 PED 消费者），在公平 CSE/编译/调度之后仍成立——**不是** CSE  alone，**不是** PoT alone，**不是** shared-Q 作标题。

精度门：绝对 valid825 AEE **≤ 1.259 PASS**（lifting raw **1.232979** / shared **1.247809**；ordinary dense/raw **1.219801**）；相对 ΔAEE **≤ +0.005 FAIL（+0.013178）**。同资源净服务门 **≥10–15%**（service 灰区）——**切勿**把 10–15% 重解释成 AEE 容差。

**Stage B 已开工（旧文档写“未开工”已过时）：** 目录 `schedule_compare_same_port` **存在**。2026-09-10 单槽源核与 2026-09-11 两段写回均有具体 slots 数字；**全链仍 NOT PASS**；决策 **SOURCE_INTERFACE_ONLY**；**不要**开成对恢复。下一 sole interface：同一合同上接真实前驱/后继 completion/ready 依赖。

Codex session `01a07507` **用量上限**于 **2026-09-11 ≈14:24 CST** 触发；建议重试 **2026-09-15 11:35 CST**。此前网页 ChatGPT 只做策略/批判/融合纸面规划；重 GPU 训练/排程在 ismd，且仅当用户明确要求。

---

## 1. 项目身份与路径锚点

| 项 | 事实 |
|---|---|
| 课题 | Optical-flow **SNN Transformer** soft–hardware co-design |
| Venue | **TCAS-II Express Brief**（约 5 页） |
| 主机 | **ismd**（Tailscale 可达） |
| Repo | `Timmyz3/sdformer_codex` |
| Branch | `autoresearch/neuron-ops-20260507` |
| 想法/调研规范根 | `/home/zhumd/work/sdformer_codex/ideafromai`（**canonical**） |
| Symlink 说明 | `/home/zhumd/work/ideafromai` → 上述路径；**2026-09-11 commit `414bf11f`**（*Move idea workspace into repo*）之后以仓内路径为准 |
| Codex session | **`01a07507`**；jsonl 极大（约 **~180MB**） |
| Codex 配额 | **usage limit hit：2026-09-11 ≈14:24 CST**；**retry：2026-09-15 11:35 CST** |
| 生产树 / 主稿 | `nts07` / `main.tex` = **READ-ONLY** |
| 性能 RTL | **在 gates 通过前禁止**；box 上 fusion **微探针** ≠ 性能 RTL / ≠ Stage B service% |

**相关 commits（已告知）：**

- `670dc834` — Measure lifting40 finite-resource…  
- `fe926311` — Evaluate common two-stage…  
- `414bf11f` — Move idea workspace into repo  

**路径习惯（勿搞错根）：**

- 主实验挂点（概念）：`…/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/`  
- 已备输入名（概念，ChatGPT 无文件）：`implementation_inputs.md`、`net_cost_one_page.md`  
- **禁止**把工作根指到错误的 `SDformer` 旧树或未同步的 `/home/zhumd/work/ideafromai` 外副本而不核对 symlink。

---

## 2. 冻结门闩（Gates）——必须原样遵守

### 2.1 精度

| Gate | 阈值 | 结果 |
|---|---|---|
| Absolute valid825 AEE | **≤ 1.259** | **PASS** |
| lifting raw (`fast_raw_diagonal`) | **1.232979**（报告值） | 在绝对预算内 |
| shared-Q 定点 825 | **1.247809** | 在绝对预算内；**仅消融，非标题** |
| ordinary dense-source/raw | **1.219801** | 强对照 |
| Relative ΔAEE vs ordinary | **≤ +0.005** | **FAIL：+0.013178** |
| 相对上限对照（ordinary+0.005） | ≈ **1.224801** | 当前 lifting 端点未过相对门 |

含义：

- 绝对预算过 **≠** 全部准入通过。  
- **当前端点不晋级**性能 RTL / 论文主贡献句。  
- 相对失败 **不等于** 杀掉整个 lifting **家族**——最多 **stop layouts**。

### 2.2 同资源净服务

- 目标：**same-resource net service ≥ 10–15%**（整段净服务潜力 / service 灰区相关判据）。  
- **CRITICAL：不要把 10–15% 重新解释成 AEE 容差或相对精度灰区。**  
  - AEE 相对门是 **+0.005**。  
  - 10–15% 是 **service% / 净服务** 相关语言。  
- 旧十帧代理（加权项/H8 约少 ~15%）= **NOT schedule-closed**，不可冒充净服务%。

### 2.3 家族纪律

- **Don't kill lifting family lightly；stop layouts only。**  
- Prosperity / Gustav = **第二队列**；负结果停布局，不杀家族。  
- shared-Q：**条件支线 / 消融**；无净硬件收益则不强留共享标题。  
- 借入底座（da4ml CSE、Gustav 供数、普通量化/剪枝、PoT 化）**≠ X**。

### 2.4 生产与 RTL 纪律

- `nts07/main.tex`：**READ-ONLY**。  
- **无性能 RTL** until gates pass。  
- Box RTL microprobes（MP1/MP2/halfstep RNE）= **sim+synth PASS** 的微型适合性探针；**不是** Stage B service%，**不是** PPA。

---

## 3. 主新颖性候选 X（写作时只能这样框）

**X（待证明，非已证明）：**  
在敏感 patch 残差链上，**可学习 lifting 结构化 T10 PSN**，带 **真实 θg**，并服务 **双 PED 消费者**；且在 **公平 CSE / 常量编译 / 有限资源调度** 之后，相对 ordinary dense-source/raw，仍留下可辩护的接口与净服务优势。

**明确不是标题 X：**

- CSE alone / 节点数减少 alone  
- PoT alone / 系数改二次幂 alone  
- shared-Q as title  
- 通用 fusion（例如 2026-09-11 norm24 融合把 lifting 6194→5354 的 **−13.56%**）——**Generic fusion ≠ title X**  
- 把 **−22.831%**（fused vs ordinary，两段写回 always-ready）写成 **净全链 X**

**可继承但非标题的底座：** Gustav 式 NRV/供数、da4ml 整图 CSE、普通低秩/量化/剪枝、fixed-BN、连续 PED/CMVM 对照等。

---

## 4. Stage B 实际进度（CRITICAL — 更新旧“未开工”叙事）

> 旧调研页曾写 `schedule_compare_same_port/` 不存在 / Stage B 未开工。**以本节为准。**

### 4.1 目录状态

- **`schedule_compare_same_port` 目录 EXISTS。**  
- Stage A（一页净费用合同 + 编译 / 825 / 十帧代理账）视为已完成背景。  
- Stage B：**部分执行模型结果已有**；**full-chain NOT PASS**；相对 AEE 仍 FAIL；成对恢复 **尚未授权**。

### 4.2 2026-09-10 — 源核（single-slot model）

| 量 | ordinary | lifting | 备注 |
|---|---:|---:|---|
| always-ready slots | **6914** | **6170** | lifting vs ord **−10.7608%** → **service 灰区** |
| fixed long backpressure | **8088** | **8088** | 两边相同 |
| RF peak live | **72 → 12** words | （lifting 侧峰值 live 降） | **allocated 仍同为 96×48b** |
| backend separate resource K864 | **758777 → 714889** | | **−5.784%**（独立后端资源点，非标题闭环） |

**结论（2026-09-10）：** 有限资源源侧有灰区信号；**full chain NOT PASS**；不得据此宣称 Stage B 通过。

### 4.3 2026-09-11 — 两段写回（two-stage writeback）

**注意：** 这是 **不同资源点**；相对单槽模型有 **+50B pipeline state**、**2-slot latency**。方法论上 **不可与 09-10 单槽数字直接横比冒充同一分母**。

| arm | always-ready slots | blocked |
|---|---:|---:|
| ordinary | **6938** | **8088** |
| lifting unfused | **6194（−10.724% vs ord）** | **8088** |
| lifting fused (norm24) | **5354（−22.831% vs ord）** | **8088** |

派生：

- Generic fusion：lifting **6194 → 5354** = **−13.56%** —— **NOT title X**。  
- Payload：**zero mismatch**；independent review：**OK**。  
- **Decision：`SOURCE_INTERFACE_ONLY`。**  
- **Full-chain + relative AEE 仍 open。**  
- **Do NOT start paired recovery yet。**

### 4.4 当前裁决语言（给 ChatGPT 复述用）

1. Stage B **已有** same-port 风格有限模型数字，但 **未闭合全链净服务门**。  
2. 09-10 −10.76% 与 09-11 unfused −10.72% 落在 **service 灰区叙事**附近，**不是** AEE 灰区，也 **不是** PASS。  
3. fused −22.8% 是 **generic fusion / 不同资源点** 上的 always-ready 改善，**禁止**写成“净全链 X 已证明”。  
4. 下一步 **唯一接口**：在同一合同上连接 **真实 predecessor/successor completion/ready 依赖**（见 §5）。  
5. **不要**再加 source+backend slot 表；**不要**扫 pipeline 超参；**不要**开成对恢复。

---

## 5. 下一 sole interface（纸面可设计，执行等 Codex 配额）

**目标合同（同一 contract，勿拆成两套表）：**

连接真实 **predecessor / successor completion / ready** 依赖，覆盖：

- source  
- FP32 preview  
- sn2  
- integer **U16 / F / BN2**  
- **dual PED**  
- native proj spike  

**明确不要做：**

- 不要新增 source+backend **slot tables**（在现有接口未闭合前）  
- 不要 sweep pipeline params  
- 不要 divert 到 Prosperity 长文 / full-chip RTL / `main.tex`  
- 不要在相对 AEE 仍 FAIL 且 full-chain 未过时启动 **paired recovery**

**过门后才讨论（现在禁止当已完成）：** 成对精度恢复 → 同预算 AEE–费用前沿 → 隔离性能 RTL → 改生产树/主稿。

---

## 6. 调研 / 融合 / 管理状态（2026-09-11 更新）

### 6.1 库存与 P0

| 量 | 数字 / 状态 |
|---|---|
| Inventory | **732** |
| P0 目标 | **246** |
| P0 coverage（宣称） | **246/246 uids** |
| 别名（无重复 deep-read） | `MAIN-R276 → ARX-012`；`MUSHA-SP001 → MAIN-R002` |
| ismd idea_cards | **245** |
| ismd CSV | **253 data rows + header** |
| `p0_deepread_progress.md` | 已刷新（ismd） |
| 历史备注 | 调研侧 box 上曾有 ~227–245 cards 待同步；**以 ismd 245 cards / 246 P0 uids 宣称覆盖为准** |

### 6.2 诚实声明（必须保留）

- **宣称 246/246 uid 覆盖 ≠ 246 篇全文同等深度精读。** 覆盖含摘录/开源盘点/unresolved/别名（调研诚实边界）。  
- 部分卡片可能是 **excerpt / alias / 摘要级**，而非 full-text deep-read。  
- 早期“未 246 全深读 / unresolved ~62 / synthesis mid-draft”类表述：若与最新进度表冲突，**以 ismd 刷新的 `p0_deepread_progress.md` + 上表为准**，但 **仍须在融合合成中区分 excerpt vs full-text 证据强度**。  
- **Unresolved / 合成质量：** 合成仍宜视为 **mid-draft** 级别可用；具体 unresolved 条数若未在本交接重核，标 **UNKNOWN（以 ismd progress 文件为准）**——**不要伪造精确 unresolved 计数。**

### 6.3 队列

- **F1→F2 priority** = 第二队列优先。  
- **Prosperity reopen** = 已关闭布局的第二队列；mask 相对活动控制约 **0.09%**（报告 **0.09045%**）逻辑项差额——**未证明标题 X**。  
- A+B 文献融合候选：可做 **纸面** B/A/X/kill 对齐 F1–F7；**不得**抢 Stage B 执行或编造实验数字。

### 6.4 Box RTL microprobes（独立线）

工具：iverilog 12.0、yosys 0.52（box）。

| Probe | sim | synth | 备注 |
|---|---|---|---|
| MP1 same_port_credit | **PASS** | **PASS** | ~179 cells (MODE=2)；credit/RR 行为探针 |
| MP2 group_accept | **PASS** | **PASS** | ~921/985 cells (W=8/16) |
| lifting halfstep RNE | **PASS** | **PASS** | ~76 cells |

**明确：** 这些 **不是** Stage B service%，**不是** TCAS-II PPA，**不是** 生产 RTL。单元数为 generic techmap 代理（无 liberty / 无 timing / 无 P&R）。

---

## 7. ChatGPT 可以帮什么（无代码访问）

1. **批判 Stage B 方法论**：两段写回 vs 单槽模型的公平性；+50B state / 2-slot latency 是否污染同资源叙事；blocked=8088 同值说明什么、不说明什么。  
2. **纸面设计**下一 full-chain ready/completion 合同（接口信号、握手、背压、双消费者、与 source-only 决策如何衔接）。  
3. **文献 A+B 融合候选**：每个候选写清 B（瓶颈）/ A（继承底座）/ X（差分）/ kill gates，对齐 F1–F7；标证据强度（full-text vs excerpt）。  
4. **起草 TCAS-II 贡献句**：严格不 overclaim；区分“已测端点 / 灰区信号 / 未证 X”。  
5. **规划 Grok Build / 人类下一步**：配额前后分工、ismd vs box、禁止 doom-loop。

---

## 8. ChatGPT 必须 NOT invent（硬禁止）

- 假 **service%**、假 **PPA**、假 **RTL 时钟 / 周期闭环**  
- 声称 **Stage B full-chain PASSED**  
- 声称 **relative AEE PASSED**（事实是 **+0.013178 FAIL**）  
- 声称 **shared-Q 是标题贡献**  
- 声称 **22.8%（或 22.831%）是净全链 X**  
- 声称 **generic fusion −13.56% 就是 title X**  
- 把 **10–15% service 灰区** 说成 **AEE 容差**  
- 编造未给出的 train / quant / VCS / DC 数字  
- 建议改 `nts07/main.tex` 或开性能 RTL “先写进稿子”  
- 建议杀掉整个 lifting 家族（只允许 stop layouts）  
- 在未授权时建议启动 paired recovery  

未知就写 **UNKNOWN / 未测 / 需 ismd 复核**，不要补齐“看起来合理”的数。

---

## 9. 续作计划（管理/调研 vs Grok Build vs Codex）

### 9.1 直到 2026-09-15 Codex 配额恢复

- **管理 / 调研** 可继续：论文级叙事、融合纸面、证据强度标注、box RTL microprobes、写计划。  
- **重 GPU train / 正式 schedule** 仅在 **ismd**，且 **仅当用户明确要求且路径已知**。  
- 网页 ChatGPT：**策略 / 批判 / 融合规划 only**。

### 9.2 配额恢复后首选 resume

- 只跑：**同一合同上的 full-chain ready/completion schedule**（ordinary vs lifting unfused 主对照；fusion 仅作非标题旁注）。  
- 仍遵守：SOURCE_INTERFACE_ONLY 决策精神；不扫 pipeline；不开 paired recovery until 明确授权。

### 9.3 Alternate：Grok Build

- 粘贴更新后的 `CODEX_NEXT_PLAN`（含 Stage B **mid-status** + **next interface only**）。  
- 强制 **python3.12**。  
- **Ban：** `/resume-codex` doom loops；长文献 skill 翻页空转；指到 **SDformer 错根**。

### 9.4 Web ChatGPT 本会话角色

- 不假装能读 repo。  
- 不输出可执行攻击性/越权内容（本项目无此需求）。  
- 输出：批判清单、接口草稿、贡献句候选、融合一页卡、下一步 checklist。

---

## 10. 数字速查表（防幻觉）

| ID | 值 | 用途 |
|---|---|---|
| Abs AEE gate | ≤ **1.259** | PASS 门槛 |
| lifting raw AEE | **1.232979** | Abs PASS |
| shared AEE | **1.247809** | Abs PASS；非标题 |
| ordinary AEE | **1.219801** | 强对照 |
| Rel ΔAEE | **+0.013178** | vs +0.005 → **FAIL** |
| Rel gate | **≤ +0.005** | 未过 |
| Service gate语言 | **≥10–15%** 净服务 | **≠ AEE** |
| 09-10 slots | 6914 → 6170（**−10.7608%**） | 灰区；非全链 PASS |
| 09-10 blocked | 8088 / 8088 | 同背压 |
| 09-10 RF | live 72→12；alloc **96×48b** 同 | 峰值≠配额 |
| 09-10 K864 | 758777→714889（**−5.784%**） | 后端分资源 |
| 09-11 ord slots | **6938** | 不同资源点 |
| 09-11 lift unfused | **6194（−10.724%）** | 灰区级 |
| 09-11 lift fused | **5354（−22.831% vs ord）** | generic fusion；非 X |
| fusion 6194→5354 | **−13.56%** | NOT title X |
| 09-11 extra cost | **+50B** state；**2-slot** latency | 公平性注意 |
| Prosperity mask Δ | **~0.09%** | 第二队列；非 X |
| Inventory / P0 | **732** / **246** | 调研 |
| P0 uids claimed | **246/246** | 含 alias；深度不等价 |
| ismd cards / CSV | **245** / unique **≈252** (+表头) | 调研刷新 `p0_deepread_progress.md` |
| Codex retry | **2026-09-15 11:35 CST** | 配额 |

---

## 11. Pasteable blocks（文末三件套）

### 11.1 English — next Codex order（paste into Codex when quota returns）

```text
LOCKED — Codex session 01a07507 (quota retry ~2026-09-15 11:35 CST)
Repo: Timmyz3/sdformer_codex @ autoresearch/neuron-ops-20260507
Canonical ideas: /home/zhumd/work/sdformer_codex/ideafromai
(symlink /home/zhumd/work/ideafromai after commit 414bf11f)

MAIN LINE ONLY = Stage B continuation on EXISTING
  …/fast_temporal_recovery_lifting40/schedule_compare_same_port/

ALREADY DONE (do not re-litigate as “not started”):
- 2026-09-10 single-slot source model:
  always-ready 6914 → 6170 (−10.7608% grey); blocked both 8088;
  RF live 72→12 but same alloc 96×48b; backend K864 758777→714889 (−5.784%);
  full chain NOT PASS.
- 2026-09-11 two-stage writeback (DIFFERENT resource point; +50B pipe state; 2-slot latency):
  ordinary 6938 / lifting unfused 6194 (−10.724%) / lifting fused norm24 5354 (−22.831% vs ord);
  blocked all 8088; payload zero mismatch; independent review OK.
- Decision: SOURCE_INTERFACE_ONLY. Generic fusion 6194→5354 (−13.56%) is NOT title X.
- Do NOT start paired recovery yet. Relative AEE still FAIL (+0.013178 vs +0.005). Abs AEE OK (lifting 1.232979 ≤ 1.259).

SOLE NEXT INTERFACE (only work):
Connect REAL predecessor/successor completion/ready dependencies on the SAME contract covering:
  source + FP32 preview + sn2 + integer U16/F/BN2 + dual PED + native proj spike.
Do NOT add source+backend slot tables. Do NOT sweep pipeline params.
Do NOT divert to Prosperity essays, full-chip RTL, nts07/main.tex, train/quant.

Gates reminder:
- Abs AEE ≤ 1.259 PASS (numbers above)
- Rel ΔAEE ≤ +0.005 FAIL (+0.013178)
- Same-resource net service ≥10–15% is SERVICE language — never reinterpret as AEE tolerance
- Do not kill lifting family; stop layouts only
- Title X = learnable lifting structured T10 PSN on residual chain with real θg + dual PED after fair CSE/compile/schedule — NOT CSE alone, NOT PoT alone, NOT shared-Q as title, NOT the 22.8% fused always-ready figure as net full-chain X

Deliverable: full-chain ready/completion schedule scorecard on the same contract + explicit PASS/FAIL/grey vs service gate + short decision note. Stay on this interface until closed.
Force python3.12. Ban /resume-codex doom loops. Ban wrong SDformer root.
```

### 11.2 中文 — Grok Build 开场白（可粘贴）

```text
你是 Grok Build，在 ismd / box 上继续 sdformer_codex 硬件协同主线。先读本交接事实，禁止编造数字。

身份：光流 SNN Transformer 软硬件协同；TCAS-II Express Brief；repo Timmyz3/sdformer_codex 分支 autoresearch/neuron-ops-20260507；想法根 /home/zhumd/work/sdformer_codex/ideafromai（414bf11f 后 symlink 已迁入仓）。nts07/main.tex 只读；无性能 RTL until gates。

门闩：Abs AEE≤1.259 PASS（lifting raw 1.232979 / shared 1.247809 / ordinary 1.219801）；Rel ΔAEE≤+0.005 FAIL（+0.013178）；同资源净服务≥10–15% 是 service 语言，绝不是 AEE 容差。不杀 lifting 家族，只停 layouts。

X（待证）：可学习 lifting 结构化 T10 PSN + 真实 θg + 双 PED；不是 CSE alone / PoT alone / shared-Q 标题。

Stage B 中段（目录 schedule_compare_same_port 已存在）：
- 09-10 单槽：6914→6170（−10.7608% 灰区），blocked 8088/8088，RF live 72→12 但 alloc 仍 96×48b，K864 −5.784%；全链未过。
- 09-11 两段写回（不同资源点，+50B，2-slot）：ord 6938 / unfused 6194（−10.724%）/ fused 5354（−22.831%）；generic fusion −13.56% 不是 X；payload 零不一致；裁决 SOURCE_INTERFACE_ONLY；禁止开成对恢复。

唯一任务：在同一合同上接真实前驱/后继 completion/ready（source+FP32 preview+sn2+U16/F/BN2+dual PED+native proj spike）。禁止加 source+backend slot 表、禁止扫 pipeline、禁止 divert Prosperity/全文 RTL/错 SDformer 根。强制 python3.12；禁止 /resume-codex 死循环与长文献 skill 空翻页。

Codex 会话 01a07507 配额 2026-09-11≈14:24 CST 触顶，重试 2026-09-15 11:35 CST；若你替代执行，只做上述 sole interface。
```

### 11.3 ChatGPT system brief（English, paste as custom instructions / first message）

```text
You are advising a TCAS-II Express Brief (~5 pages) project: optical-flow SNN Transformer soft-hardware co-design. You have ZERO access to any repo, code, or markdown files. Use ONLY facts in the handoff the user pastes. Mark unknowns. Never invent metrics.

Identity: host ismd (Tailscale); repo Timmyz3/sdformer_codex branch autoresearch/neuron-ops-20260507; ideas at /home/zhumd/work/sdformer_codex/ideafromai (symlink after commit 414bf11f). Codex session 01a07507 hit usage limit 2026-09-11 ~14:24 CST; retry 2026-09-15 11:35 CST. Production nts07/main.tex READ-ONLY; no performance RTL until gates pass.

Gates: Abs valid825 AEE ≤ 1.259 PASS (lifting raw 1.232979, shared 1.247809, ordinary dense/raw 1.219801). Rel ΔAEE ≤ +0.005 FAIL (+0.013178). Same-resource net service ≥10–15% is SERVICE grey-zone language — NEVER reinterpret as AEE tolerance. Do not kill the lifting family; stop layouts only.

Title X candidate (unproven): learnable lifting structured T10 PSN on residual chain with real θg + dual PED consumers after fair CSE/compile/schedule. NOT CSE alone, NOT PoT alone, NOT shared-Q as title.

Stage B mid-status (UPDATE old “not started”): schedule_compare_same_port EXISTS.
2026-09-10 single-slot: always-ready 6914→6170 (−10.7608% grey); blocked 8088 both; RF live 72→12 but same alloc 96×48b; backend K864 −5.784%; full chain NOT PASS.
2026-09-11 two-stage writeback (different resource point, +50B state, 2-slot latency): ordinary 6938; lifting unfused 6194 (−10.724%); lifting fused norm24 5354 (−22.831% vs ord); blocked 8088; payload zero mismatch; review OK. Decision SOURCE_INTERFACE_ONLY. Generic fusion −13.56% is NOT title X. Do NOT start paired recovery. Do NOT claim 22.8% as net full-chain X.

Next sole interface: real predecessor/successor completion/ready deps on the SAME contract (source + FP32 preview + sn2 + integer U16/F/BN2 + dual PED + native proj spike). No new source+backend slot tables; no pipeline sweeps.

Survey/management: inventory 732; P0 246/246 uids claimed (aliases MAIN-R276→ARX-012, MUSHA-SP001→MAIN-R002, no duplicate deep-reads). ismd idea_cards=245; CSV unique≈252 (+ header; 调研刷新); progress file refreshed. Honesty: coverage ≠ all equal full-text deep-reads; some cards may be excerpt/alias. Prosperity reopen second queue (~0.09% mask delta). Box RTL MP1/MP2/halfstep RNE sim+synth PASS — NOT Stage B service%, NOT PPA.

Help with: Stage B methodology critique; paper design of next ready/completion contract; A+B fusion candidates with B/A/X/kill gates; non-overclaiming TCAS-II sentences; Grok Build / human next steps.
Never invent fake service%, PPA, RTL clocks, full-chain PASS, relative AEE PASS, shared-Q as title, or that 22.8% is net full-chain X.
```

---

## 12. 给 parent 的同步检查清单

- [ ] BOX 文件已写：`/workspace/overnight_20260911/CHATGPT_WEB_HANDOFF_FULL.md`  
- [ ] scp/rsync 到 ismd：`…/survey_ab_fusion_20260910/CHATGPT_WEB_HANDOFF_FULL.md`  
- [ ] 网页 ChatGPT：粘贴 **§0 + §11.3**，需要细节时再贴 §4–§6  
- [ ] Codex 恢复后：只贴 **§11.1**  
- [ ] Grok Build：贴 **§11.2**（可附 §4 表）  
- [ ] 勿用旧 `understanding_and_gaps.md` 里 “Stage B 未开工 / 目录不存在” 覆盖本文件  

---

*End of handoff. No fabrication beyond provided facts; unknowns marked. Generated 2026-09-11 CST for web ChatGPT with zero file access.*
