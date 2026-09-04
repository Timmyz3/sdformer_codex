# 第二轮创新攻击：Motion/H67 跨窗持久 RQTB quotient（方向 B）——数据裁决

日期：2026-08-18。本轮红线遵守：不写 RTL、不碰 GPU（H86 训练占卡）、不改 `docs/359`/194436Z/任何现有文件，脚本与输出全部在 `/tmp`。证据分级沿用项目惯例：`[rtl]` / `[prof]` / `[模型]`；模型数字不冒充周期。

---

## 0. 总裁决

| 候选 | 裁决 | 创新分 |
|---|---|---|
| 方向 B：跨窗持久 RQTB quotient 目录 + delta 更新 | **NO-GO（维持 round-1 M1）**，理由本轮重写 | 2.8–3.0，不是 4.0 门 |
| 幸存候选：Motion 自身 profile 三腿全过的 quotient-file 侧车（窗内对象） | CONDITIONAL_PROFILE_GATE_SUPPORT_ONLY_NO_RTL | 3.0–3.2 →（+RTL/PPA）3.5–3.7 |
| 4.0 | NO（解锁条件见 §5） | — |

**一句话**：方向 B 的"重叠区保持 destination identity"机制在部署模型上无对象（SWIN 窗口是**非重叠 tile**，代码证据），其数据支撑（相邻窗口类集相似）经修正后的 null model 检验有正信号但 73% 由同帧效应解释、位于 <1% 的目录写活动预算内，且无新算法算子合同——三条腿都不站。

**诚实修正**：上一轮攻击文档中我的 null model（token 级边际 `Σp²/(2−Σp²)` 得 0.8926 > 实测 0.72，据此称"无空间结构"）是**错误 null**，本轮作废。正确 null 是窗口级 permutation（全局 iid 0.316 / 同记录同 head 0.476），实测相邻 0.650，邻接特有增量 +0.174——空间平滑结构**存在**。裁决不变，但论据改为：身份基板缺失、活动预算太小、无新算子合同。

---

## 1. 数据：冻结 ep35 全量跨窗类集统计（新测量，`[prof]`）

来源：`results/h67_fullres_ep35_postconvergence_t450_20260805_profile100/nts11_hardware_p0_profile.json`（571MB，冻结 ep35，100 samples × 1200 records，672,000 个 (window, head) 行，151.2M pairs）。CPU-only。

方法（脚本 `/tmp/motion_lag_null_v2_20260818.py`、`/tmp/motion_adj_delta_v3_20260818.py`）：

- 分数用冻结 RTL 公式精确重建：`score_q7 = 4×overlap + motion + RNE((32−q−k+overlap)/16)`，网格 163 bins（MAX_SCORE=162）。
- 窗口类集 = 该窗口两个时间平面 225 个 token 的 class 占用**并集**（`[W,H,163]` bool）。逐窗口循环 ground-truth 校验（record 0 完全一致）。
- 相邻 = scan order 相邻窗口、同 head、双方类集非空；Jaccard 双口径：pooled（Σinter/Σunion）与 mean-of-ratios。
- null：A=全局 iid 窗口对（3M 对）；B=同记录同 head 随机对；C=同记录任意 head 随机对。
- **勘误**：上一轮的 `motion_swqd_stats_20260818.py` 用 `np.stack([s0,s1]).reshape(2W,H,225)` 后按 `[0::2]/[1::2]` 拆时间平面——该 reshape 是 C-order 先把全部 t0 行放前，拆出的是**相邻窗口的并集**而非时间平面，C/J 均偏高（C p50=4/p95=17 应为 3/16）。该文件仅留档，不作证据。

### 1.1 核心数字

| 指标 | 值 |
|---|---:|
| 真窗口 C：p50 / p95 / p99 / max / mean | 3 / 16 / 19 / 46 / 5.116 |
| 相邻窗类集 J（lag1，pooled / mean-of-ratios） | 0.650 / 0.730 |
| lag 衰减（pooled）：2 / 5 / 10 | 0.599 / 0.531 / 0.505（衰减主要在 lag1→3：0.650→0.566，之后趋平） |
| null A：全局 iid 窗口对 | 0.316 |
| null B：同记录、同 head 随机对 | 0.476 |
| null C：同记录、任意 head 随机对 | 0.459 |
| **邻接特有增量（lag1 − null B，pooled / mean-of-ratios）** | **+0.174 / +0.151** |
| 同帧解释比例（null B / lag1 pooled） | 73% |
| 相邻对 delta：survive / insert / delete | 3.988 / 1.050 / 1.098 |
| edits / 450 tokens | 0.00477 |
| retention（survive / prev C） | 0.856 |
| 全局类占用率：class 2 / 3 / 4 | 98.2% / 66.5% / 53.7% |

### 1.2 两个结论

**结论 1（修正上一轮）**：相邻窗口类集相似性**高于**同帧随机对（+0.174），且随 lag 平缓衰减——运动场空间平滑确实让相邻 tile 的 score 类集更相似。上一轮"无空间结构"的判断作废；类似地，H82 rank-1 的 p50(J_win)=1.0 只需约半数窗口为单类且共享主导类即可解释，不能当作跨窗结构证据 `[推理]`。

**结论 2（对硬件的意义）**：相似性中 73% 由"同一帧/同一 head"解释（null B 0.476），全局边际（top 3 类出现在 54–98% 的窗口）解释大部分剩余，**邻接特有增量只有 +0.174**。这意味着"跨窗持久 + delta"能捕获的**独有**结构不到总相似性的三成，且集中在目录写活动这一微小区间（见 §2）。

---

## 2. 方向 B 逐项过 433 合同

433 门槛原文："必须是新的算法算子合同，且要同时改变硬件存储/执行对象；不能再从现有 RTL 中拆分工程模块计数。"

| 合同项 | 方向 B 的声称 | 裁决 |
|---|---|---|
| 新算法算子（一句话） | 无。跨窗持久 = RQTB 内部 class-file 的存储生命周期扩展，语义与窗内完全一致 | **FAIL**：属 score-front CSE 类别，389 §5 已封（MSSB5 仅作 RQTB 支撑，不可单列） |
| 新存储对象（bit 级） | 163×9-bit 目录写端口复用：每窗口省 survive≈3.99 条 class 项 ≈ 35.9 bit 写活动 | **FAIL**：不新增容量/端口对象，只是把已有目录少写几次；35.9 bit vs 窗口 450 token 流（数千 bit），活动占比 <1% |
| 新执行对象 | 无（ST_CLASSIFY/ST_SHIFTMAX/ST_EXPAND 全部不变） | **FAIL** |
| bit/cycle 对照 | 基线 450 带分名单 / fused pair-gather：目录写活动 <1%；周期腿持平（450+C+desc ≈ 696 vs 450+C+PAIRS ≈ 691，+0.7%） | **FAIL**（周期腿略负） |
| 文献差分 | PADE/SpAtten/Transitive Array/FuseMax/TeAAL/CSR 类均无"跨窗 quotient 目录持久"先例 | 文献空白≠创新：同帧基线已捕获 73%，静态 top-class 驻留（class 2 占 98% 窗口）不需任何跨窗状态机即可捕获大部分其余 |
| **身份基板** | "destination identity 通过重叠区保持" | **FAIL（致命）**：`window_partition_v2`（`third_party/SDformerFlow/.../Spiking_swin_transformer3D.py` 100–113 行）把特征图按 window_size 直接 `.view` 切块 + cyclic shift 轮转，**相邻窗口不共享任何 token**（时间维窗口 (t0,t1)/(t2,t3) 也不重叠）。round-1 M1 的"滑窗"前提（docs/264 §9 预告）与部署模型不符 |

**预闸门（可证伪，若将来要重开）**：真实 T450 ordered trace 中 score-front（相等测试+packetize）占 head-row 周期/能量 ≥10%，且 band 方案在相同 FIFO/端口下净赢 ≥15% 能量。按 docs/262 现有 `[prof]`（RQTB 主目标是 descriptor/SCS/K-store 流量，不是 score 峰值吞吐）该门不可能过。

**裁决：NO-GO，创新 2.8–3.0，不得作为 4.0 门**（与 round-1 M1 一致，论据更新）。

---

## 3. 幸存候选：Motion 自身数据的 quotient-file 侧车（窗内对象）

方向 B 被否后，Motion 线唯一可辩护的剩余候选是 H82 侧车合同在 Motion 语义下的迁移（513/163-bin class bitmap + quotient descriptor，denom-only 变体）：

| 三腿（Motion 冻结 ep35 全量 `[prof]`） | 门限 | Motion 实测 |
|---|---:|---:|
| occupied p95 C ≤ 192 | ≤192 | **16** ✓ |
| descriptor/token p95 ≤ 0.60 | ≤0.60 | 0.578 ✓（mean 0.511） |
| state 相对 fused pair-gather 降 ≥20% | ≥20% | quotient-gate（p95 desc=260）：513+9×16+12×260=**3777 vs 4932 → −23.4%** ✓；denom-only：**3647 → −26.1%** ✓；mean desc=230 时 −30.7%/−33.4% |

周期腿（同端口下界 `[模型]`）：450+C+desc = 450+16+230 = 696 vs 基线 450+C+PAIRS = 691 → **+0.7%，持平/略负**；最坏点（C=192, desc=238）880 vs 867 亦为负。侧车唯一可辩护的生产腿是 energy（目录/位宽活动），需要 RTL + SAIF/PPA 才能落地，本轮无 RTL。

**裁决**：CONDITIONAL_PROFILE_GATE_SUPPORT_ONLY_NO_RTL（角色同 445 对 H82 的判定：数据流支撑对象，不是独立贡献）。过门 + sidecar exact → 3.0–3.2；+SAIF/PPA 同宏 → 3.5–3.7；4.0 NO。

---

## 4. 三大死穴（对 4.0 而言）

1. **身份死穴**：SWIN 非重叠 tile（`window_partition_v2` 代码证据 `[代码]`）。跨窗不存在共享 token，"重叠区保持 destination identity"的机制基板缺失；M1 的滑窗假设在部署模型中不存在。
2. **合同死穴**：无新算法算子。跨窗持久是已有 RQTB class-file 的生命周期扩展（换名不换对象），433 的"新算子 + 同时改存储/执行对象"双腿均不满足；389 §5 已将 score-front CSE 类别整体封死。
3. **收益死穴**：目录写活动每窗口 ≈ 36 bit（9-bit class 项 × 3.99 条），占窗口总活动 <1%；邻接特有增量只有 +0.174（73% 由同帧效应解释，class 2 全局占用 98% 让静态驻留近乎免费）；周期腿持平。即使 100% 消除目录写，端到端 EDP 收益也在噪声级，过不了 docs/262 的 `[prof]` 能量门。

---

## 5. 4.0 解锁条件（诚实列明）

Motion 要打 4.0，必须（缺一不可）：

1. **新算法算子合同**：任何现有对象（score、目录、descriptor、K-store、FCSR）的换名、换边界、生命周期扩展都不算。可证伪的类别：T>2 关系（433 已断言与 Motion T=2 身份冲突，需算法侧先出合同）；或模型层新增跨窗语义（非硬件自造）。
2. **在线构建 + 不可解释收益**：445 要求 class file/quotient 由 score 流原生在线构建，且端到端 EDP 收益不能被普通融合/CSR/RQTB 解释。
3. **工程闭环**：score→projection 周期 ≥10% 或组件动态能量 ≥15%（同端口）；逻辑+宏面积 ≤+10%、Fmax ≥−5%；bit-exact 随机反压 0 mismatch。

若未来要重开任何"跨窗/tile 持久"类候选，锚定证据必须是：同帧随机对 vs 相邻对的边际增益 ≥0.3（当前 **+0.17**）；score-front 占 head-row 活动 ≥10%；目录写活动占窗口 ≥5%（当前 **<1%**）。当前全部不满足。

---

## 6. 证据清单与可复现性

| 项目 | 等级 |
|---|---|
| 冻结 ep35 全量跨窗类集统计（C/J/lag/null A-C/delta，672,000 行） | `[prof]`（脚本 `/tmp/motion_lag_null_v2_20260818.py`、`/tmp/motion_adj_delta_v3_20260818.py`，输出 JSON 同目录，CPU-only） |
| 时间平面拆分 bug 勘误（旧脚本 `/tmp/motion_swqd_stats_20260818.py` 仅留档） | `[prof]` 勘误 |
| SWIN 非重叠 tile（`window_partition_v2` + cyclic shift） | `[代码]`（模型源码） |
| H82 rank-1 类文件统计（rank1_stats.json，5,544,000 窗口） | `[prof]`（已有，引用） |
| 侧车 bit/cycle 模型（pair-gather/quotient-gate/denom-only） | `[模型]`（`scripts/h82_multiplicity_free_quotient_model.py` 公式） |
| 侧车 RTL、SAIF、PPA、Fmax | `[待验证]`（未写 RTL） |

本轮未修改 `docs/359`、selector、生产 RTL 或 194436Z；未抢 GPU。
