# TCAS-II Independent Review + Score (Grok Bot)

**Document type:** Independent TCAS-II-style letter review (circuits & systems / HW–SW co-design for ML)  
**Subject:** Timothee Z — optical-flow SNN-Transformer hardware co-design, **Fork A / C1\*+C2\*** under Grok Bot tree  
**Tree:** `/workspace/sdformer_c1c2star_grokbot/` · **Branch:** `tcasii/c1c2star-oss`  
**Reviewer posture:** Hostile-but-fair TCAS-II Express Briefs reviewer; evidence from on-disk RTL/docs only  
**Date:** 2026-09-06 (Asia/Shanghai) · **Author tag:** GROKBOT NEW FILE  
**Hard negatives baked in:** OpenROAD = best-effort sky130hd, **18/18 DRC=0**, **NO PDN**, **NO SPEF**, **NOT signoff** — do **not** quote routed u² as silicon.

---

## 1. Scope & framing / 范围与框定

### 1.1 What is being reviewed / 评什么

| CN | EN |
|---|---|
| **Fork A** 下的 **C1\*/C2\*** 整树：定向 RTL（`rtl_c1star/` + `rtl_c2star/`）、定向 TB、Yosys(+abc) 综合、C1/C2 消融梯子、sky130hd OpenROAD 最佳努力 P&R 记分板。 | **Fork A** C1\*/C2\* stack: directed RTL, TB, Yosys(+abc), ablation ladders, best-effort sky130hd OpenROAD P&R scoreboard. |
| 证据锚点：`out/REGRESSION_REPORT.md` **PASS**（19 sim scripts）、`out/ABLATION_C1_LADDER.md` / `ABLATION_C2_LADDER.md`、`docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md`（**18/18 DRC=0**）。 | Anchors: regress PASS, C1/C2 ablation MD, OpenROAD scoreboard **18/18 DRC=0**. |
| **不**评 Codex/`nts07`/`sdformer_codex` 硬件源；本评仅读 Grok Bot 树 + `ideafromai/research` 合成包。 | Does **not** review Codex/nts07 hardware sources. |

### 1.2 Proposal vs frozen capture / 提案 vs 冻结捕获（必须分清）

| Layer | Status on disk | Safe letter language |
|---|---|---|
| **ATLIF int8 non-absorbable payload** (`docs/ATLIF_contract_r1_grokbot.md`) | **DEFAULT DRAFT — NEW co-design only** | “Proposed datapath contract for HBG-RP / SMAM-RP / ADP-MAC” |
| **Binary ATLIF fire** (ep35 P0 profile; see `docs/16_ATLIF_IDENTITY_VERDICT_GROKBOT.md`) | **Supported by profile file** (93/93 binary, ternary=0, samples=1, checkpoint epoch35) | “Deployed H67 ep35 capture is binary today” |
| Conflating the two | **Forbidden** | Do **not** write “ep34/ep35 already has int8 amplitudes” |

**CN 一句话：** 今天部署捕获更像二值（有 ep35 profile）；本树 HBG 实值载荷是**另开的新合同**；论文必须写成 co-design proposal，不能写成冻结身份已是 int8。  
**EN one-liner:** Binary capture (ep35) ≠ int8 proposal (this RTL). Keep them in separate sentences forever.

### 1.3 What is NOT claimed / 明确不声称

- **Signoff PPA：** 无 PDN、无 OpenRCX SPEF、无多角 STA；`report_design_area` 的 routed u² **不是**硅片面积。
- **Full SoC / full SDSA PE array：** 仅有 thin `front_pipe` / `back_pipe` + `c1s_top` / `c2s_top` 侧挂集成。
- **Same-resource vs old C1/C2：** 本树是叙事重写（retire zero-skip / multiply-reorder），**没有**与旧 ISCAS C1/C2 同资源对照表。
- **AEE / system speedup / mW：** 无算法–RTL 联合仿真、无 liberty 功耗；`power_proxy.py` 仅为 activity×cells 启发式。
- **FACT / FlightVGM / ERAFT / Bishop AAC / SMAM 端到端精度或“发明 Mask-Add”：** 仅可作 related-work knife-edge，不可吞并。

---

## 2. Claimed innovation points — DETAILED / 声称创新点细评

> 将 ~18 具名模块收成 **7 根支柱**。TCAS-II 信件只应主打 2–3 根；其余降为消融层或 roadmap。

### Pillar A — Optical-flow / motion-gated front-end sparsity  
**(Motion-TTB, OGEC, OP-STW; + ECP/MW 见 Pillar C)**

**机制（CN）。** OP-STW 用逐 tile 的光流差分 `|flow_cur−flow_prev|` 与事件密度 `event_cnt` 产生 `wake_bitmap`，调度“哪些 tile 做精确工作”，替代平坦配额 / 裸 zero-skip。OGEC 将 match_ok 分成 ExactMatch vs Propagate。Motion-TTB 把 wake tile 打成 `(tile, dt_bin, hyp)` 时间包，供 MFBD 投递。

**Novelty vs typical sparse Transformer / SNN / OF accel.**  
OF/event 驱动 wake 在 ERAFT / FlowAcc / event-camera 栈中有近亲；**刀刃**是把 OF-Δ+event **绑定到 exact-work 调度**，并写进可综合 RTL+消融计数，而非再讲一遍 empty-tile。相对旧 C1“跳零/重排乘法”，叙事替换成立。相对纯 attention prune（SpAtten/HeatViT），领域绑定更强。

**HW embodiment.**  
- RTL: `c1s_op_stw_predictor.sv`, `c1s_ogec_gate.sv`, `c2s_motion_ttb_packer.sv`  
- TB + `sim_op_stw` / `sim_ogec` / `sim_motion_ttb` → **PASS**  
- OpenROAD: OP-STW / OGEC / Motion-TTB **DRC=0**（记分板行 4/2/10）  
- C1 ablation: ALWAYS wake_pop=64 → OPSTW=32（同刺激）

**Hostile attack.** “这只是阈值比较器 + 位图；新颖度在算法侧已被 ERAFT/GMFlow 吃掉；RTL 没有证明 AEE 不掉。” Motion-TTB 扫描打包偏薄，易被说成 glue。

**Evidence grade:** **Medium**（OP-STW / OGEC）；Motion-TTB **Weak–Medium**.

---

### Pillar B — Hierarchical / residual / HBG-style ATLIF int8 co-design (HBG-RP)

**机制（CN）。** HBG-RP 将事件拆成二元门 `g=(|amp|>EPS)` 与 **不可吸收** 的 int8 载荷 `p`：`g` 作 PE/时钟使能，`p` 仅在门开时进入 MAC。合同文件明确 **禁止** 把幅度吞进下一层 W（否则降级声称）。这是相对 Bishop/AAC“纯二值 / 吸收 spike”的差异化。

**Novelty.**  
双轨 gate+payload 在 FireFly-v2 / SMAM 文献邻域；**真正可辩的新意**是：(1) 显式 **non-absorbable ATLIF** 合同；(2) 与 OF 前端共设计的系统故事。若审稿人接受“新合同=贡献”，则 Medium；若要求“已是部署身份”，则被 ep35 二值 profile **直接打穿**。

**HW embodiment.**  
- RTL: `c2s_hbg_rp_packetizer.sv`（极小：Yosys generic **24** cells；mapped ~121 u² floorplan-scale）  
- TB `sim_hbg_rp` **PASS**；C2 ablation ALWAYS mac_en=16 → HBG=8  
- OpenROAD HBG-RP **DRC=0**  
- Contract: `docs/ATLIF_contract_r1_grokbot.md` + verdict `16_ATLIF_IDENTITY_VERDICT_GROKBOT.md`

**Hostile attack.** “ep35 捕获全是 binary —— 你们在加速一个还不存在的神经元合同；没有 int8 vs binary 的 AEE/Pareto，HBG 只是阈值比较器。” Grok46 Card C 路径会把 HBG 标成与冻结身份冲突。

**Evidence grade:** **Medium** as **proposal co-design**; **Weak** if mis-sold as frozen-capture fact. **Do not conflate.**

---

### Pillar C — Exact/approx capture + MW-ΔBuf temporal reuse

**机制（CN）。** ECP-QKV 在 QKV 投影 MAC **之前**用廉价 `corr_score`（可 OR OP-STW wake）决定 `proj_en`。MW-ΔBuf 计算 `cur−ref` 残差并出 `delta_nz`，残差非零可并入 wake。`exact_capture` **仅当** `OGEC.exact_en ∧ PRRC.allow_exact` 且未超有限 `CAPACITY` 才入队 —— 不是平坦 exact 配额。

**Novelty.**  
ECP 明确挂 FACT-style eager skip，但落点是 OF/corr-tile 前端而非 FACT 全栈。MW 近亲 MotionDeltaCNN / DiffFrame。OGEC×PRRC→exact_capture 的 **预算∩匹配** 组合是本树较干净的电路故事，优于“又一个 skip 信号”。

**HW embodiment.**  
- RTL: `c1s_ecp_qkv_predictor.sv`, `c1s_mw_delta_buf.sv`, `c1s_prrc_ledger.sv`, `c1s_exact_capture_wrap.sv`  
- 全部定向 sim **PASS**；front_pipe 集成 OP-STW→ECP←MW **PASS**  
- C1 ladder 有 ALWAYS/OPSTW/ECP/MW 四档；**缺**独立 “+OGEC/PRRC” 测量档（计划有、梯子未合入）  
- OpenROAD: ECP / MW / PRRC / ExactCapt / front_pipe **DRC=0**

**Hostile attack.** “ECP 是阈值阈值；MW 无真正 warp 插值，只是 tile 级减法；PRRC 是小计数器账本；组合故事大于电路深度。”

**Evidence grade:** **Medium**（组合）；单块 ECP/MW **Weak–Medium**；exact_capture 协议 **Medium**.

---

### Pillar D — Attention-side sparse control (ECP-QKV, STH-Gate, SMAM-RP, SP-Gate)

**机制（CN）。** SMAM-RP：gate 轨做 Mask-Add(+1)，payload 轨在 `spike_gate=1` 时使能实值 MAC。STH-Gate：按 spat/temp score 出门使能/双使能/skip。SP-Gate：attention-mass 阈值门 `run_en`。ECP 见 Pillar C。

**Novelty.**  
SMAM 文献已有 Mask-Add；本树 delta 是 **gate×ATLIF payload 绑定**（须承认 inspired-by）。STH 近亲 Sparse VideoGen dual-head；SP 近亲 SpAtten。作为“注意力稀疏控制织物”可支撑消融层，**不宜**当 TCAS-II 唯一主贡献。

**HW embodiment.**  
- RTL + TB 均 **PASS**；back_pipe = HBG→SMAM + STH 并行 **PASS**、OpenROAD **DRC=0**  
- C2 ladder：HBG 与 HBG+SMAM 计数相同（mac_en=8）—— **SMAM 在该刺激下未显示额外稀疏收益**（对齐/透传为主）

**Hostile attack.** “命名通胀；STH/SP 是阈值比较器阵列；SMAM 消融无增量；不要把 SpAtten/SVG/SMAM 的刀占为己有。”

**Evidence grade:** **Weak–Medium**（系统胶水强于单点新颖）。

---

### Pillar E — Resource / priority / ADP-MAC / PRRC / ARM / MFBD control-plane

**机制（CN）。** ADP-MAC：双侧 `skip_a/skip_b` + `mac_en` 门控累加，参数 `FORBID_REORDER=1` 明确禁止 A↔B 乘法重排叙事。ARM-Acc：多假设孔径累加。MFBD：按 bundle hyp 投递到 one-hot lane。PRRC 见 Pillar C。

**Novelty.**  
ADP 相对 FireFly/BBS 比特分解文献 **偏薄**——实现上仍是条件乘加，`FORBID_REORDER` 是声明式卫生而非新微架构。ARM/MFBD 是小状态机/选择器，偏 context 模块。

**HW embodiment.**  
- 全部 sim **PASS**、OpenROAD **DRC=0**  
- C2 FULL 档：mac_en 8→4（ADP skip 生效）—— 有计数证据，但是合成刺激

**Hostile attack.** “把 hygiene 参数包装成创新；ARM/MFBD 是 demo 级；控制平面模块数推高‘贡献列表’但不增加信件信息密度。”

**Evidence grade:** ADP **Weak–Medium**；ARM/MFBD **Weak**；PRRC（归 C）**Medium**.

---

### Pillar F — System composition: C1\* front_pipe + C2\* back_pipe

**机制（CN）。** `c1s_front_pipe`：OP-STW ∥ MW → ECP（wake|delta_nz），合并 `tile_active`。`c2s_back_pipe`：HBG→SMAM，STH 并行。`c1s_top` / `c2s_top` 侧挂 OGEC/PRRC/exact/stats 与 Motion-TTB/ADP/ARM/MFBD/SP/stats。叙事脊柱：**运动调度的精确捕获（C1\*）× 二元门控实幅载荷（C2\*）**。

**Novelty.**  
组合叙事是本树相对“散装加速器模块 zoo”的主要字母价值；但 thin pipe ≠ 全加速器，也无端到端延迟/能效数字。

**HW embodiment.**  
- front_pipe / back_pipe sim **PASS**；flatten 后 OpenROAD **DRC=0**（记分板 #17/#18）  
- 回归全绿；里程碑文档 `HW_STACK_MILESTONE_GROKBOT.md` 关闭骨架

**Hostile attack.** “集成是 wrapper；侧挂模块未进入关键 datapath 闭环；缺少 cycle-accurate 系统模型与 workload trace。”

**Evidence grade:** **Medium**（叙事与可复现集成）；系统完整性相对全芯片 **Weak**.

---

### Pillar G — Measurement story (ablations + OpenROAD as completeness)

**机制（CN）。** C1/C2 消融梯子给 wake_pop / proj_skip / delta_nz / exact_hit 与 mac_en / skipped / gate_fire。OpenROAD 记分板证明 **实现可布线性**（18/18 DRC=0），**不是** PPA 声称。OSS 脚本可回归。

**Novelty.**  
测量本身不是算法创新，但是 TCAS-II 审稿人最需要的 **可证伪骨架**；在 OSS 工具链上做到 leaf+pipe 全路由，对信件“硬件真实性”有加分——前提是作者自己不把 routed u² 写成硅片结果。

**Evidence grade:** **Strong** as *implementation completeness / reproducibility*; **Speculative** if used as silicon PPA.

| Ablation (on-disk) | Key deltas |
|---|---|
| C1 ALWAYS→OPSTW | wake_pop 64→32；exact_hit 64→32 |
| C1 +ECP | proj_skip 32→28；exact_hit 32→36（刺激相关） |
| C1 +MW | wake_pop→49；delta_nz=22 |
| C2 ALWAYS→HBG | mac_en 16→8 |
| C2 +SMAM | 无额外计数变化（该刺激） |
| C2 +ADP FULL | mac_en 8→4 |

**Gap：** C1 计划中的 `+OGEC/PRRC` 档未出现在 `ABLATION_C1_LADDER.md` 实测表。

---

## 3. Independent TCAS-II-style scoring / 独立评分

**Scale:** 1–5（允许一位小数）。校准：5=该信件轨顶尖；3=可发表边缘；2=需大修；1=拒。

| # | Category | Score | Rationale (short) |
|---|---|---:|---|
| 1 | Novelty / originality | **3.2** | 脊柱（OP-STW + HBG 合同 + exact 预算门）有合成新意；大量叶模块增量/近亲。 |
| 2 | Technical depth / HW story correctness | **3.0** | RTL 真实可综合；多数块电路深度浅（比较器/账本/条件 MAC）；ADP 尤薄。 |
| 3 | Experimental / verification completeness | **3.5** | 19/19 sim PASS、双消融、18/18 P&R DRC=0 —— 强；但无 SPEF/PDN/AEE/signoff，封顶。 |
| 4 | Clarity vs prior art / prior ISCAS C1–C3 | **2.8** | 已有“retire zero-skip/reorder”与 cousin 表，好；proposal vs capture、模块清单仍易糊。 |
| 5 | Significance / impact (edge OF+Transformer) | **3.0** | 问题重要；缺 workload AEE/系统占比 → 影响力未证实。 |
| 6 | Presentation readiness | **3.2** | 中英 claim draft / innovation review 已有；AS-IS 投稿仍易 overclaim。 |
| 7 | Reproducibility (OSS flow) | **4.0** | 脚本、日志、DEF、消融、回归 —— 信件轨少见的 OSS 完整度。 |
| | **Mean (unweighted)** | **3.24** | |
| | **Letter-weighted\*** | **≈3.1** | *↑ novelty/depth/exp/clarity；repro 降权* |

### Overall recommendation / 总评

**Borderline**（偏 **Weak Reject** 若按“要硅基 PPA 的 TCAS-II 电路信”；偏 **Weak Accept** 若接受“以可复现 RTL+消融为主的架构/协同设计短文”，且作者把贡献压到 2–3 刀并写清 proposal vs capture）。

**ISCAS / TCAS-II letter calibration:**
- 相对作者先前 ISCAS ~Weak Accept 的旧 C1/C2 叙事：本树 **叙事更干净、证据链更长（OSS+P&R）**，但 **仍缺** 同资源对照与精度数字。
- TCAS-II Express Briefs 惯例更吃 **可测电路结果**；AS-IS 更像“强 ISCAS 扩展稿”，尚未稳坐 TCAS-II。

### Estimated chance / 投稿概率（主观）

| Package | Est. accept-class chance | Note |
|---|---|---|
| **AS-IS**（模块 zoo + 易混 int8/binary + 无 AEE） | **~15–25%** Weak Accept；其余 Borderline/WR | 高拒因 overclaim / 深度 |
| **After ~4–6 weeks** focused fixes（见下） | **~40–55%** Weak Accept | 仍难 Strong Accept（无 signoff） |
| **After AEE Pareto + 1 个闭环 pipe 指标 + 严格 3-knife 信件** | **~55–65%** WA | 上限受无硅签核约束 |

### Top 5 must-fix before submission / 投稿前必修

1. **AEE / 精度–稀疏 Pareto：** int8 非吸收载荷 vs 二值基线（ep34/ep35 真迹或等价 harness）；没有则 HBG 主刀无法防守。  
2. **Abstract/Intro 强制分段：** “Proposed ATLIF int8 contract” vs “Frozen ep35 binary capture” —— 各一句，永不合并。  
3. **信件只保留 2–3 刀：** OP-STW（+ECP/MW 一层）、HBG-RP（+SMAM 一层）、OGEC×PRRC→exact_capture；其余进附录/消融。  
4. **补全 C1 消融 `+OGEC/PRRC` 实测档**，并给 front_pipe 级联合计数（非仅叶模块）。  
5. **删除/软化一切 routed u²→硅片、mW、系统加速比、FACT/ERAFT 精度、18 模块=18 创新** 的句子；OpenROAD 只当 *implementation completeness*。

### Top 3 overclaims to delete or soften / 三大过声称

1. **把 OpenROAD routed u² / util 写成硅基面积或 PPA**（记分板已警告；投稿稿仍是高危）。  
2. **把 int8 HBG 写成冻结 ep34/ep35 身份**（与 `16_ATLIF_IDENTITY_VERDICT` 证据冲突）。  
3. **模块计数通胀**（“我们提出 OP-STW, HBG, ECP, MW, OGEC, PRRC, SMAM, ADP, STH, SP, ARM, MFBD, TTB…”）—— 审稿人会读成 *laundry list*，反而削弱脊柱。

---

## 4. Honest comparison / 诚实对照

### 4.1 vs prior Timmyz3 / `sdformer_codex` ISCAS ~Weak Accept narrative

| | Old C1/C2-ish narrative | This Fork A C1\*/C2\* |
|---|---|---|
| Knife | zero-skip / multiply-reorder（已宣布退役） | OP-STW scheduling + HBG non-absorbable payload |
| Evidence depth | 偏算法/早期 HW 叙事 | **更强** OSS regress + ablation + 18× P&R DRC=0 |
| Risk then | 新颖度被 Prosperity/TSBG/稀疏 Transformer 近亲压 | 新颖度被 FACT/SMAM/SVG/FireFly 近亲压 + **ATLIF 身份分叉** |
| Net | WA 级故事但刀钝 | 刀更利、工程更实，**但 TCAS-II 仍差 AEE/PPA** |

**结论：** 相对旧稿是 **升级而非换题成功**；不要把“已 WA 过 ISCAS”自动平移为 TCAS-II WA。

### 4.2 vs Grok46 binary Motion-XOR / Card C（仅竞品叙事）

| | Fork A（本评对象） | Grok46 Card C / MX3P |
|---|---|---|
| ATLIF | int8 proposal | binary frozen-path |
| Knife | OF wake + dual-rail payload | Motion-XOR triple popcount + dirty-lane |
| HW status here | RTL+P&R 已落地 | 本树 **未**实现；勿切换 fork |
| Conflict | Card B HBG vs 冻结二值 | Grok46 明确勿与 A+B 混岛 |

**本评继续 Fork A**；MX3P 仅作“若走冻结二值路线的替代主刀”。系统占比若如 Grok46 所忧（attention 极小），则 **两岛都可能变成 leaf paper** —— 这是共同威胁，不是站队理由。

### 4.3 Module-count inflation risk / 模块数通胀

| Deep-ish contributions (defend) | Named blocks that are mostly glue / thin |
|---|---|
| OP-STW；HBG-RP contract；OGEC×PRRC→exact_capture；front_pipe 组合 | ECP（阈值）；MW（无 warp）；SMAM（透传+计数）；STH/SP（阈值门）；ADP（条件乘）；ARM（多槽累加）；MFBD（hyp 选择）；stats（计量） |

**审稿人可读公式：** *Few deep contributions + many named blocks ≈ weak letter.*  
建议 Table I 只出现脊柱与消融 rung，不出现 18 行模块名片。

---

## 5. Final one-page CN executive summary / 一页中文执行摘要

**对象：** Fork A / C1\*+C2\* 光流–脉冲 Transformer 协同设计 RTL + OSS EDA 证据（非签核）。  
**身份纪律：** ATLIF **int8 不可吸收载荷 = 提案合同**；ep35 捕获 **二值 = 冻结事实** —— 严禁混写。

**评分（1–5）：** 新颖 3.2 · 深度 3.0 · 实验 3.5 · 相对先前工作清晰度 2.8 · 影响力 3.0 · 呈现 3.2 · 可复现 4.0 → **均分 ≈3.24 / 信件加权 ≈3.1**。  
**推荐：** **Borderline**（AS-IS 更近 Weak Reject–Borderline；收束声称并补 AEE 后可冲 Weak Accept）。  
**概率：** AS-IS ~15–25% WA；专注返工 4–6 周 ~40–55% WA。

**值得防守的 3–4 个真创新（不要摊成 18 个）：**
1. **OP-STW**：光流差分 + 事件密度调度 tile 唤醒 / 精确工作（相对旧 zero-skip 叙事成立；有消融 wake_pop 64→32）。  
2. **HBG-RP**：二元门控 × **不可吸收** int8 ATLIF 载荷的协同设计合同（电路小但故事尖；必须标 proposal）。  
3. **OGEC × PRRC → exact_capture**：匹配门控 ∩ 金字塔残差预算 ∩ 有限容量的精确入队（比平坦配额干净）。  
4. **（系统层）C1\* front_pipe + C2\* back_pipe 组合与可复现测量链**（消融 + 18/18 DRC=0）—— 作为 *实现完整度*，不是硅基 PPA。

**过声称（删/软）：** 路由 u²当硅片；int8=已部署；模块清单当贡献列表；ADP/STH/SP 等深创新口吻；无联合仿真的 AEE/加速比。

**必修 Top：** AEE Pareto；proposal/capture 分句；三刀信件；补 OGEC/PRRC 消融档；OpenROAD 只报完整度。

**一句话给作者：**  
工程完整度与 OSS 可复现性已经超过多数同阶段 ISCAS/TCAS 初稿；**信件命运取决于你是否忍痛丢掉模块动物园、并诚实区分“我们想训练的 int8 合同”与“今天捕到的二值脉冲”。**

---


---

## Post-review remediation (2026-09-06)

**Trigger:** Independent review novelty ≈ **3.2** (Borderline). Author asked: (1) apply review fixes, (2) deepen innovation with new mechanisms.

**What changed (docs + RTL) — do NOT fake higher scores yet:**

| Item | Change |
|---|---|
| Claim draft | Compressed to **exactly 3 knives**: OP-STW(+TDE3-Prior), HBG-RP (proposal vs ep35 binary), OGEC×PRRC→exact_capture; TMA-Agg = support |
| Laundry list | Demoted STH/SP/ARM/MFBD/stats/OpenROAD zoo from contribution rank |
| Proposal vs capture | Re-stated in claim draft + Card G: int8 = proposal; ep35 = binary |
| OpenROAD language | Softened to *implementation completeness only*; no µ²/mW |
| New RTL Card G | `c1s_tde3_prior` + `c1s_wake_merge` (C1*); `c2s_tma_agg` (C2*) |
| C1 ablation | New **`EXACT`** / OGEC-PRRC rung: budget-gated `exact_hit` capped vs MW |
| Regress | `sim_tde3_prior` + `sim_tma_agg` wired into `run_all.sh` |

**Re-score status:** **Pending** after RTL lands and full regress PASS. Prior scores (novelty 3.2 / letter-weighted ≈3.1) remain the last independent numbers until a fresh hostile pass.

**Expected direction (non-binding):** Knife hygiene + TDE3/TMA depth should *help* novelty/clarity if evidence stays honest; AEE still missing → do not claim Strong Accept trajectory yet.

## Appendix — Evidence index (on-disk)

| Item | Path |
|---|---|
| Innovation review | `docs/TCASII_INNOVATION_REVIEW_GROKBOT.md` |
| Claim draft | `docs/TCASII_LETTER_CLAIM_DRAFT_GROKBOT.md` |
| Design plan | `docs/TCASII_design_plan_grokbot.md` |
| OpenROAD scoreboard | `docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md` |
| ATLIF verdict | `docs/16_ATLIF_IDENTITY_VERDICT_GROKBOT.md` |
| ATLIF contract | `docs/ATLIF_contract_r1_grokbot.md` |
| Milestone | `docs/HW_STACK_MILESTONE_GROKBOT.md` |
| C1/C2 ablations | `out/ABLATION_C1_LADDER.md`, `out/ABLATION_C2_LADDER.md` |
| Regress / summary | `out/REGRESSION_REPORT.md`, `out/SUMMARY_TCASII_OSS.md` |
| Cross-AI fork note | `docs/15_CROSS_AI_READ_GROK46_CODEX_VS_GROKBOT.md` |
| Idea pack R2 | `/workspace/ideafromai/research/09_ROUND2_SYNTHESIS_C1_C2_UPGRADE.md` |

---

*GROKBOT NEW FILE — independent review for parent agent / Timothee Z. Local tree only; no nts07; no push.*

---

## Rescore pointer (2026-09-06 ~12:50+08 Asia/Shanghai)

**Card G/H + OpenROAD 25/25 landed.** Fresh hostile-but-fair rescore:

→ **`docs/TCASII_RESCORE_AFTER_CARD_GH_GROKBOT.md`**

| | Prior (this file) | After Card G/H |
|---|---|---|
| Mean | ≈3.24 | ≈**3.43** |
| Letter-weighted | ≈3.1 | ≈**3.30** |
| Recommendation | Borderline | **Borderline (WA-leaning)** — still not WA |
| Tip | (pre G/H P&R) | ≈`334e8fd` |

Do **not** treat this file's §3 scores as current; use the rescore. Caps unchanged: no AEE; SCI WNS −0.019 not closed; int8 = proposal vs ep35 binary; OpenROAD ≠ PPA.
