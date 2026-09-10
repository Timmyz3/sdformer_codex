# 去留复审与后续计划（幕僚长执行线 · 2026-09-09）

独立完成；条目与理由从 seed notes / `hardware_innovation_20260908` / 同负载收据归纳，**未** verbatim 抄 Codex `literature_audit_20260909`。数字均带来源路径；不同分母不横比。

## 0. 冻结约束（写进一切计划）

- 只投 **TCAS-II Express Brief（5 页）**；不并行 ISCAS。
- 身份冻结：Motion C12 ep34；θg=ATLIF（τ≠θ）；Motion-XOR attention。
- 生产 `nts07/main.tex` 只读，除非贡献锁定。
- 未过同资源 **≥10–15% 净收益门 + AEE 门（valid825 ≤1.259，且不劣于最强对照 0.005）**，不扩 8×8/EDA/main.tex。
- 2×4 RTL 仅 **mount-only**。
- 区分：「具体布局/学生失败」vs「整类方法失败」vs「完整先验未迁完」。

---

## 1. 真正应停止的具体版本表

| 具体版本（范围） | 证据摘要 | **没被否定的部分** |
|---|---|---|
| C1 Prosperity 融合 G2/G4+残差共享布局 | 相对 Prosperity 多 81.52%/16.44%；大缓存后仍多 2.964%（`same_workload_c1c2_20260907/net_benefit.md`） | Prosperity/APEC 整类；改变共同构造后可重测 |
| C2 静态共模共享（相对时间并行强基线） | 加法少 8.96%，周期多 +0.23454% | 时间并行底座本身；完整源打包/LoAS/BN2 未闭 |
| 支持码「训练整字相消零响应」标题 X | 创新门停；one-hot 后普通块稀疏 L 可等价；概念 ~4.5/10（`bn_state/response_zero_one_page.md`） | LUT-DLA/LUT-NN 完整 A；普通 INT10 无损格式 |
| 「θg 确定就停算 U」+ 完成序 CSE 改写 | 可由严格界/SPARK/USEFUSE 覆盖；未指出可省物理量（`psn/psn_decision_one_page.md`） | **PSN 费用问题本身**；da4ml 常矩阵编译 A |
| patch 持续检查 vs 一次许可 | 持续检查慢 0.6274%（README 顶部） | 一次许可底座；未实施的参考选择/训练 |
| patch 连续分组/row34 作标题 | 相对优化逐行仅少 2.03% 逻辑取权；电路增量薄 | 固定 BN+row34 精度底座（AEE 过门） |
| H8/C16 Gyro/联合消费者剪枝（s2b3） | 不胜 hidden50；服务仍慢 ~19%（`group_pruning_probe/README.md`） | 完整 HiNM/VENOM/CRISP；昂贵 patch 挂点敏感性 |
| NR4 费用训练版 | 真实组 ≤ 打乱组 | Gustav NRV 执行底座 |
| 类别路径 vs 打包时间（弱对照优势） | 强对照后仅 0.58%–1.70%；三原行下更慢 | `F_live>1` 公平重测（P1-4）仍开放 |
| 支持集投影共享 / 全 T10 严格后缀界具体形态 | 额外 ~1%–1.7% | 其他有损共同完成设计 |
| SVD 低秩 PSN 门 | 普通范数门更快 | 范数门强对照 |

---

## 2. 前提改变后可恢复资格的项

| 项 | 恢复前提（须先满足） |
|---|---|
| Prosperity/APEC 融合族 | 改变共同构造/散播/残差共享交互；同资源重跑 |
| HiNM/Gyro 结构剪枝 | **完整**输入重排+内层 2:4+OBS/Hard Concrete；或换到不可廉价删除的 patch 挂点 |
| 有损共同完成（SparseInfer×BitFair×Gustav 并集） | 完整 A + 独立预测同权限对照；预测开销计入 |
| Avalanche 依赖生存期训练 | 先算乐观上界；上界不足则停，不必先训 |
| 类别表示 | 仅 `F_live>1` 同资源公平探针 |
| 因果运动/TMA/TDE 唤醒 | 可训练小预测器；禁止最终 flow oracle；注意力旁路不恢复为主加速 |
| S3Net 稀疏 stem | 端到端重训 stem；本轮零训练掩码失败≠训练路线失败 |
| 支持码 LUT 新 X | 完整 LUT-DLA+表 INT8 QAT 迁完后，提出**区别于**约束块剪枝的求解 |
| PSN 判决新 X | 完整 da4ml A + CompRRAE 严格界对照齐后，给出可省的具体位宽/扇出/RF |

---

## 3. 后续优先级（≤3 条主线）

### 主线 1 — 昂贵 patch 条件生产（对准 ~34.8% 大头）

- **B**：残差 patch 不可廉价删除；普通 2:4/半宽是强控制；动态 BN 可校准消除后仍剩卷积+T10+halo。
- **最强对照**：固定 BN + 普通 2:4 + 一次严格生产许可 + Graham/SBNet 式零列跳过底座。
- **A 完整迁移**：TermiNETor/DynConv 条件生产；子流形索引/halo（S3Net 族）若主张稀疏 stem。
- **待检 X**：消费者完成反馈下的**少生产**（非重新编码已有零）；须同资源净减且 AEE 过门。
- **固定杀门**：不胜一次许可/2:4；或只省便宜工作；或 valid825 破 1.259 / 劣于对照 >0.005。

### 主线 2 — 完整 Gustav A 上的 PSN/广播消费洞

- **B**：旧学生链 PSN/输出占服务 73.81%（MAC 发射 68.517%）；共享源字被最慢消费者钉死（严格界共享后 ~1%）。**注意**：该占比属旧 S0 学生模型，非当前 patch 份额。
- **最强对照**：完整 da4ml 常矩阵编译；MSB 严格界早停；hidden50 / row2:4；完整 Gustav（含多输出上下文）。
- **A 完整迁移**：GustavSNN GP+CPTB+NRV+局部 S；da4ml §2–4；CompRRAE 严格界；MFPSN 若有损。
- **待检 X**：仅当指出强控制做不到的**具体物理差异**（位宽/扇出/RF/反馈）后，才允许一次预预算试验（净服务 ≥15% 等，见 PSN 一页）。
- **固定杀门**：无差异的「换名早停」；失败不扫参续命；无物理闭环不称 RTL 加速。

### 主线 3 — 结构稀疏完整迁移到敏感挂点（非 s2b3 磨参）

- **B**：s2b3 甚至可全删隐藏而 full825 AEE 1.183003，在廉价块上磨分组无意义。
- **最强对照**：hidden50（1.164732）与 row2:4（1.163570）；完整 HiNM 二阶。
- **A 完整迁移**：HiNM/VENOM/CRISP 全链路；广播域以真实 T10 门+FC2 损失选可共同删除的物理字。
- **待检 X**：仅当挂点敏感性证明「不可廉价删除」且完整 A 仍留洞。
- **固定杀门**：不胜 hidden50/完整 HiNM；并集仍读满且无周期余量。

**并行纪律**：任一时刻一条主实现 + 至多一个轻量挑战探针（建议挑战：`F_live>1` 或运动唤醒上界，各 ≤1 日）。

---

## 4. Grilling 对抗自审（不问用户；磁盘证据）

引用技能：**grilling**（对主结论追问完整 A / 最强简单控制 / 局部≠整法 / 是否只换网络换名）。

| 主结论 | 追问与纠正 |
|---|---|
| 「支持码零响应可做标题」 | **完整 A 未迁完**（缺表 INT8 QAT 等）。最强控制（块剪枝+稀疏 L）已可解释目录行为 → 纠正为：停标题 X，保留 LUT 底座问题。 |
| 「PSN 占 73.81% 故当前 X 必要」 | 该数字来自**旧 S0 forced-support 学生**服务模型（`support_service_result.json`），不是 patch/整网份额；且未给同学生常矩阵编译对照 → 纠正：保留问题，停当前 X，先补 A。 |
| 「C2 共享失败 ⇒ 共享思想死」 | 局部布局失败；五项思想复审明确失败布局≠证伪（`five_ideas_reassessment.md`）。 |
| 「Gyro 五轴负结果 ⇒ HiNM 无用」 | 迁移边界明确：未复现输入重排/2:4/OBS/Hard Concrete → **完整先验未迁完**，非整法失败。 |
| 「类别路径曾有 14–20%」 | 弱对照；强 NR4 后撤回 → 禁止再用旧百分比作文。 |
| 「固定 BN / INT8 是创新」 | 明确**不作标题**；仅数值底座。 |
| 「2×4 RTL 已证明加速」 | mount-only 功能切片；字节口在真实 2:4/C16 上可变慢；无 PPA 准入。 |
| 新颖性是否只换名 | 当前停掉的 X 多数是「先验早停/块稀疏换挂点」；grilling 结论：须先完整 A + 强对照，再谈 X。 |

---

## 5. 与 Codex 审计的差异（结构对照，非抄表）

对照过对方目录字段（`id,category,name,...` 等）与规模（对方 CSV **347** 行）。本审计：

| 维度 | 本线（幕僚长） | 相对 Codex |
|---|---|---|
| 规模 | **127** 条有出处 | 更少；拒绝灌水 name_only 堆叠 |
| 本地实验条目 | 显式 `LX*`（C1/C2/零响应/PSN X/patch/NR4…）并写入杀表 | 对方更偏外部文献枚举；本线更强调**已跑学生的版本级停止** |
| 处置口径 | 强制区分变体失败 / 标题停 / 开放 / 完整 A 未迁完 | 同意对方「完整迁移≠新机制」；**不同意**把局部负结果写成整类死刑（本线在表中显式保留未否定部分） |
| 主线排序 | patch 条件生产 → PSN/Gustav 洞 → 完整结构稀疏迁敏感挂点 | 与 AB_STACK/README 顶部一致；可能比对方更强调 **da4ml/LUT 完整 A 前置** |
| 可能漏补 | 部分 CIM/MoE/视频算法 venue 标「待核」；未扩到对方 347 的每一别名 | 对方可能多收了二手别名与方法论卡片；本线有意不收录无证据路径的条目 |
| 多挖 | grilling 纠正表；前提恢复资格表；三视角 scientific-brainstorming（见 `ideas_from_skills.md`） | — |

---

## 6. Idea 短名单（≤5；详见 `ideas_from_skills.md`）

1. Patch 少生产（完成反馈 + 固定 BN 对照）  
2. Gustav 完整 A 上的 PSN 常矩阵/严格界对照后的真洞检验  
3. 敏感挂点上的完整 HiNM/VENOM 迁移（非 s2b3）  
4. 有损共同完成（并集损失 vs 独立预测）  
5. （挑战探针）可训练运动唤醒上界 / `F_live>1`  

---

## 7. 证据锚点（核对用）

- `hardware_innovation_20260908/README.md`
- `hardware_innovation_20260908/bn_state/response_zero_one_page.md`
- `hardware_innovation_20260908/psn/psn_decision_one_page.md`
- `same_workload_c1c2_20260907/net_benefit.md`
- `hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md`
- `hardware_innovation_20260908/algorithm/group_pruning_probe/README.md`
- `hardware_innovation_20260908/algorithm/patch_probe/README.md`
- `idea_reassessment_four_questions_20260907.md`
- `hardware_innovation_execution_plan_20260908.md`
