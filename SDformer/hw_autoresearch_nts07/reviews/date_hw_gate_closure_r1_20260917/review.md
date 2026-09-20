# DATE 硬件门槛闭合缺口分析 + 解阻计划 + 新机制候选（R1，2026-09-17）

- 评审身份：只读分析（`READ_ONLY_ANALYSIS__NO_EDA__NO_GPU__ADVISORY`）。本轮**未运行**任何 GPU 训练/推理、未运行任何 EDA（VCS/DC/Formality/PT/PTPX）、未创建 attempt/result/lock、未修改本目录外任何文件。
- 授权边界：**本文不授权任何 launch**。所有 launch 仍必须走既有 source → 独立 source hammer → release → launch 前 live gate 流程。
- 证据分档：`[rtl]` / `[prof]` / `[模型]` / `[代码]` / `[待验证]`，模型与代理数字不冒充周期、能量或 PPA。
- 输入：`docs/524`、`reviews/m936_m931_m912_...hammer`、`m1006`、`m1597`、`m1666`、`m2044/m2045`、`m628`、`GROK_REVIEW_20260821_H1`、`CLAUDE_INNOVATION_ATTACK_ROUND2_MOTION_20260818`、`docs/433`、`CLAUDE_MOTION_SIDECAR_CAPACITY_PORTS_20260818`、`dc_handoff/CLAUDE_SIDECAR_SAIF_PPA_RUNBOOK_20260818`、`dc_handoff/SERVER_RUN.md`、`claude_score_20260917`、contracts 树（m1465/m1504/m1630–M1680/m2045/m1652–m1663 等）、`dc_handoff/runs/` 隔离目录与 attempt marker、`neuron_autoresearch/CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md`。

---

## 0. 结论摘要

1. **门槛缺口结论**：DATE 硬门槛（同资源、decoder-complete、全网络总周期 ≥1.10x，首选 ≥1.15x）**未闭合**，且缺口是**结构性的**：全树不存在 decoder-complete 的 exact 全网络 trace/总周期数字；`m628` 的 Table A 仍为 0 行。截至本轮，**唯一在 9 月新增的"网络级"闭合件是精度腿**（M2045 ep34 valid825 attention+8-op QDQ，精度门 PASS），但 `paper_accuracy_result=false`——**尚无独立 result hammer，不可引用**。
2. **三处派单假设修正**：① `m931` 不需要"恢复"——它已被 M935 三阶 exact 父匹配链取代并隔离，且 M935 已在 3.000 ns 拿到 setup WNS `+0.001795 ns` 的 99/100 独立 hammer（`m1006`）；② `M912` 被 M935 取代，`M946` **不是服务器绑定**（decoder bounded-prefix/仿真侧）；③ C2 SAIF/PTPX 腿的真实状态不是"被 UID 卡住"，而是**工具前置缺陷反复消耗 attempt**（≥7 次尝试、0 SAIF/PTPX 产物）。
3. **解阻要点**：把 EDA 服务器上的"已授权但一次都没开跑"的 release 清账（M1465/M1504/M1663/M1680），顺序 **CPU 先行（补 M1466/M1505/M1685+1686 hammer 与产物对账）→ C2 三轴 DC（M1663）→ C1 Formality+PT-STA（M1680）→ C2 SAIF/PTPX（tiny-UCLI 演练后才准花生产 attempt）**；UID 争用用"窗口化采样 + 与 fangyl 排时"，**不要换服务器**（库/宏/license/工具身份全部钉死在该机）。
4. **m931 vs C3 rank3 裁决**：两者都不选——采用 **M935 系（M993 canonical → M1665 hold-closed 后继）** 为 C1 物理基线，`m931` 永久留隔离，C3 rank3 不承担 C1 物理（0 宏、口径不可比）。
5. **侧车裁决**：**SHELVE（搁置）**，按 count 文件口径重述收益（state 腿实为 `+13.5%/+10.3%` ✗），并预注册 4 条 revive 条件。
6. **新机制候选（2 个，均未被 ROUND2 否决清单否决）**：**H1 `TLQ-5`**（T=5 时间商记录文件 + 时间维 run-length 广播执行器，源自算法侧 D1）与 **H2 `XWIN-RB`**（stride-12/window-15 重叠滑窗 + 滚动分母增量执行器 + 跨窗 quotient 目录，源自算法侧 D2）；H2 正面回应 round2 的"身份死穴"但工程腿（≥10% 周期或 ≥15% 能量）当前数字不过，H1 是首选。

---

## 1. 门槛闭合缺口分析

### 1.1 门槛与最小证据集（引用）

- DATE 硬门槛（`m628`/scoreboard）：**至少一个同资源、非重叠、decoder-complete 的全网络总周期结果 ≥1.10x（首选 ≥1.15x）**；核心 RTL 需 VCS/SVA + DC/STA；能量分列 logic/SRAM/DRAM；≥3 条 DSEC 序列或密度分层。
- `m628` 最小证据集（4 项）：① decoder-complete exact 全网络 trace；② 一个统一 CPU cycle+memory replay（Dense96/B1/K1/K1x8/K8/Ours 六配置，固定分子 + completion receipt）；③ matched logic+macro 面积/STA 与 logic/SRAM/DRAM 系统能量闭合；④ ≥3 序列或密度分层 + 盲独立 hammer。**Table A 当前 0 行。**
- 冻结 Amdahl 机会包络 `620,302,905`：patch embed `32.1489%`、ATLIF `20.6384%`、FC1 `19.0826%`、4 层 bottleneck Conv `12.8374%`、FC2 `6.6764%`、attention `0.5894%`。
- decoder 补齐后的**分析**包络（`docs/524` §：corrected analytical envelope）：Fixed dense `1,442,206,883` vs bit-sparse `790.920–803.774M` cycle ≈ `1.794–1.823x`；**仅作分析敏感性**，exact decoder trace + memory stall 未闭合前不得进摘要/主表。

### 1.2 现有证据覆盖矩阵

| 证据 | 数值/状态 | 分档 | 可否用于 DATE 门槛 |
|---|---|---|---|
| C1 slice PTPX（M448R4） | slice logic `24,548.7 µm²`；`6.2538 mW`；`18.7614 pJ/cyc`；M416 setup `+0.7636`/hold `+0.025 ns` | `[rtl]` | 组件能量点，**不是**全网络 |
| C1 生产周期（M1597，ep34 live93，同账本） | `382,848,700 cyc` vs strong zero `648,741,051`（`1.694510x`）vs same-coordinate bit `646,619,098`（`1.688968x`）；账本 `51,840,000` 行 / `78,668,732` active bits；容量更正 `214,912 B` logical / `215,040 B` mapped（旧 `213,376 B` 作废，预算 `245,760 B` 余 `30,720 B`） | `[cycle model]` | 组件级 CPU 周期模型；`27,160,940,160 B` 仅 parent-scratch 流量，**不是** SRAM/DRAM 总量 |
| C1 物理（m931） | **准入 FAIL**：setup WNS `−4.9058 ns`、TNS `−15,026.33 ns`、3,128 违例路径（top-100 全为 `match_bank_q_reg`→`directory_q_reg`，511 logic levels）；hold WNS `−0.0894`（诊断）；85,396 cells；logic `80,149.86` + macro `78,825.24` + total `158,975.10 µm²`；容量 `18,432 B` 物理 vs `213,376 B` 账本；PWR-428 阻塞功率主张 | 已隔离 | 否 |
| C1 物理（M935→M993→M1006） | 3.000 ns、setup WNS `+0.001795 ns`、TNS 0、0 setup 违例、100/100 paths MET、9 个 `TS1N28HPCPHVTB128X128M4S` 宏、DC cell area `147,246.39209 µm²`；hammer 99/100（P0/P1/P2=0/0/1） | `[rtl]` | **当前 C1 物理点**；非 hold 签核 / 非功率 / 非全 storage / 非系统 PPA |
| C1 残余 hold + 签核链（M1630→M1680） | M1649 DC 已消耗隔离（PID 519344）→ M1659/M1664 copy-only canonical 恢复 → M1665 结果 + M1667 hammer → M1674/M1676/M1678/M1680 Formality×2 + PT×1 授权（`launch_now=true`） | 契约层在、**本地无 run/review 产物** | 待闭合（见 G6/G9） |
| C2 matched 三轴 logic-only DC（旧网表，M872/M903） | areas k1 `124,620.17` / k8 `131,086.24` / k1x8 `585,479.15 µm²`；directed cycles k8 `1913` vs k1x8 `1945` → `1.0167x`（五条 directed 合法组件负载，**DC 不产生周期证据**） | `[rtl]` | 组件级；非 decoder-complete |
| C2 registered-fault 功能（M1627） | 99/100：compactor-local 合法终端 no-false-pulse + sticky illegal header/raw 已证；外圈 OR-chain、全 K8/K1x8 周期同一性**未证** | `[rtl]` | 部分 |
| C2 decoder 前缀诊断（M1666，M1656 结果） | dense / equal-service `1.9931x`（时间 −49.83%、字节 −35.68%）；dense / typed-K8 `2.1501x`；typed-K8 / equal-service `1.0787x`；1,034,451/519,007/481,123 cyc | `[rtl]` | **prefix-only diagnostic，明示 no L3/paper claim**（42 destination、4 output block） |
| C2 matched K1/K8/K1x8 新物理（M1663） | 授权 3 次 dc_shell，**未开跑**（无 attempt marker / result dir） | `[待验证]` | 否 |
| C2 SAIF/PTPX（多链） | 本地可见 5 个隔离目录：m1001 链 r2 `COMPILE_k1 rc=1`、r3 `COMPILE_k1 rc=255`、r4 `RUN_k1_CASE0 rc=1`、r5 `RUN_k1_CASE0 rc=1`；M1432 `SIM_k8_0`、`UCLI-117`、ptpx 0 / saif 0；后继 M1467(`-debug_access+r`)→M1493(`-lca`)→M1502/M1504（**需 fresh M1505，未 launch**）；M1684 生产能量源契约（M1685 评审 + M1686 release **均未创建**） | 隔离 | 否；**零产物** |
| C2 FC2 端点（M519 r1–r15） | r5/r11/r12/r15 全部消耗并隔离；**r15（M789 授权的 atomic-artifact-gate）已跑**：rc=36、k1 RUN_COMPLETE、k8 `FAIL_PRECOMPILE_LOOP__EXPLICIT_EXIT36`（`TIM-209=1`、`OPT-150=0`）；无 canonical result | 隔离 | 否 |
| C3 rank3（M289） | `102,852.29 µm²`、133,263 cells、9,639 FF、setup/hold 0、**0 宏** | `[rtl]` | 组件；非 C1 物理 |
| 网络级精度（M2045，新增） | ep34 valid825 attention-hw-order + 8-op weight QDQ：candidate AEE `1.197367` vs baseline `1.199514`，Δ `−0.002147`（门 `0.02`）→ **PASS**；825 samples / 18 seq / 8 targets 各 825 calls；claim boundary：`paired_valid825_subset_deployment_result=true`、`paper_accuracy_result=false`（**须独立 result hammer**）、hardware cycles/speedup/energy/PPA 全 false | `[prof]`（待 hammer） | 精度腿，非门槛数字 |

### 1.3 缺口清单（按闭合优先级）

| # | 缺口 | 现状证据 | 严重度 | 闭合路径 |
|---|---|---|---|---|
| G1 | decoder-complete exact 全网络 trace + 总周期 ≥1.10x | 不存在；仅 prefix 诊断与组件周期模型 | **P0（唯一真正卡门槛的）** | tsbg/deployment-complete capture 链（M1707→M1765 等）+ 统一 replay；或按 §5 的新合同换对象后重定门槛口径 |
| G2 | 统一六配置 CPU cycle+memory replay（固定分子） | 不存在 | **P0** | CPU-only，可先做（六配置中 K1/K8/K1x8/Dense96/B1 已有部分模型件） |
| G3 | PPA_ADMISSION=0 | Grok 8/21 冻结；`SERVER_RUN.md` 要求 `MACRO_DBS`+`EXPECTED_MACRO_REFS`+adapter 才能置 1 | **P0（任何 PPA 主张前置）** | C1 顶已产出 foundry 宏身份（`tsmc28_128x128_1rw_20260822.json`：`TS1N28HPCPHVTB128X128M4S`、128×128 1RW、`8,758.36055 µm²`/instance、slow `ssg0p9v125c` 0.616 ns / fast、slow/fast `.db` SHA 齐、`PRIVATE_ASSET_DB_VALIDATED`）→ **flag 与 adapter 未翻**；m931 的 PWR-428 仍阻塞功率 |
| G4 | matched K1/K8/K1x8 新物理（M1609 registered-fault 源锥） | M1663 授权未跑 | P1 | launch M1663（3 次 dc_shell，一次 attempt） |
| G5 | C2 matched SAIF/PTPX（K8 vs K1x8 五类负载） | ≥7 次尝试、0 产物 | P1 | 先做 tiny-DUT UCLI 演练（见 §2.4），再花生产 attempt |
| G6 | C1 hold 签核 + Formality + PT STA | M1680 授权未跑；M1665/M1667 本地缺 | P1 | 服务器侧物证对账 → launch M1680（2 Formality + 1 PT） |
| G7 | logic/SRAM/DRAM 能量分解、全 encoder 系统表 | 0；spike energy 仅代理，禁止顶替 | P1 | 依赖 G3 + G5 完成后 |
| G8 | ≥3 条 DSEC 序列或密度分层 | M1597 仅 `zurich_city_09_a` 10 样本；M1666 单前缀 | P1 | CPU 账本侧可扩样本（成本低） |
| G9 | M1630–M1680 链本地物证缺失 | 5 个 review（m1641/m1653/m1662/m1675/m1679）+ run 目录（m1630/m1649/m1651/m1665）**本地均不存在**，仅契约引用其 SHA | P1（审计/同步） | 从 EDA 主机对账回同步；在补齐前，任何本地 hammer 都无法核验 C1 hold/签核身份 |
| G10 | M2045 结果无独立 hammer | `reviews/` 无 m2045 hammer；M2044 失败件已保留（`FAILED_DO_NOT_CITE`，环境性：`ModuleNotFoundError: spikingjelly`），M2045 env 后继 preflight PASS | P1 | CPU-only 独立结果 hammer（读 `results/m2045_.../`） |
| G11 | M1468/M1495 两次 C2 失败**本地无 namespace**（仅 M1432 有隔离目录） | 无 attempt/失败目录 | P2（审计） | 补隔离目录或明确记录"仅远端"，否则失败谱系不完整 |
| G12 | `+0.737%` 归属错挂 | 该值 = C2 FC2 K1→K8 logic 增量 `(20,587.39208−20,436.696076)/20,436.696076`（M532/M535 hammer 已核 `1.007373794836484x`），`CLAUDE_SCORE_20260917` 却记在 C3 rank3 行 | P2（口径） | 更正为 C2；C3 rank3 需另立增量来源 |

### 1.4 PPA_ADMISSION=0 的准确现状（本轮勘定）

- 技术前置**部分已具备**：C1 顶的 9 个 foundry SRAM 宏已在 DC 流程中被消费（m931、M993/M1006 均为宏感知 DC，`macro_count=9`），宏清单与 slow/fast `.db` 已私有校验通过。
- 但**准入 flag 仍为 0**：无 adapter/`EXPECTED_MACRO_REFS` 的正式绑定记录；无 C1 宏感知网表的 PTPX/功率数字（m931 侧 `PWR-428` 阻塞）；C2/C3 侧 matched 宏面积与能量腿全空。
- 结论：**G3 是"可攻但未攻"**，不是"技术上不可行"。建议把它列为高于任何新增 RTL 的第一优先（与 8/19 评分文档建议一致）。

---

## 2. 解阻计划

### 2.1 阻塞面重构（先纠三件事）

1. **M912/M935/M946 与"服务器 UID 争用"的关系**：M912 已被 M935 取代；M946 属 decoder bounded-prefix/CPU 仿真侧（其门是 fresh hammer 与 100K→全行授权，不是 EDA）；服务器侧真正被卡的只有"一次都没开跑"的 release：M1465（C1 runtime-witness VCS，因缺 M1466 hammer 而 `inert_until_m1466`）、M1504（C2 source-chain successor，需 fresh M1505）、M1663（C2 三轴 DC）、M1680（C1 Formality+PT-STA）。
2. **C2 FC2 matched DC 的真实失败模式不是资源**：M519 r15（M789 授权）**已消耗**，死于 k8 `TIM-209=1`（`TIM_EXPLICIT_FAILURE`，rc=36），`docs/524` 的"result/attempt 均未产生"已过期。后继路线（M1609 registered-fault 结构修正 + M1634→M1661/M1663）正是针对该前端环路与 M1652 runner 断言缺陷的修复。
3. **外 UID 不是同 UID 碰撞**：M519 R15 runner 的门只统计"**与 runner 同 UID** 的 EDA 进程"（`UID 1909`（owner `fangyl`）的长期 `simv` 被记录并披露、不阻断，M789 原文为证）；M496 时代的容忍策略为 foreign simv 仅当 `state∈{S,I}`、`%CPU ≤0.5`、`RSS ≤262,144 KiB`、3 采样 ×10 s、且每轴点前复检。**绝不许 signal/kill 外部进程或抢 license seat。**

### 2.2 CPU 侧先行件（不占 EDA、不占 GPU；本轮可立即做）

| 项 | 产出 | 依赖 | 价值 |
|---|---|---|---|
| U1 | **M1466 hammer**（C1 runtime-witness VCS release 的前置盲评） | CPU-only，读 M1464/M1465 | 解阻 M1465，使 C1 runtime-witness VCS 可 launch |
| U2 | **M1505 fresh hammer**（C2 M1502/M1504 链） | CPU-only | 解阻 C2 SAIF/PTPX 第 4 次尝试 |
| U3 | **M1685 评审 + M1686 release**（M1684 生产能量源） | CPU-only | 给出 K8 vs K1x8 的 VCS→direct-SAIF→PTPX 正式一次尝试 |
| U4 | **M2045 结果独立 hammer** | CPU-only，读 `results/m2045_.../` | 把 ep34 网络级精度腿变成可引用（`[prof]`） |
| U5 | **物证对账清单**：列出 G9 的 5 review + 4 run 目录，从 EDA 主机（`/home/zhumd/work/...`）回同步或书面记录"仅远端" | CPU-only | 否则 M1674/M1680 的身份在本地不可核验 |
| U6 | **launch 前置"存在性检查"**：逐个 release 校验其 pinned runner 与输入在**目标主机**上存在且 SHA 相等（本地缺 `run_dc_m1661_...`、`run_m1678_...`、`run_m1674_...`、`run_formality_m1674_*.tcl`、`run_ptsta_m1674_*.tcl`） | CPU-only | 避免把 host 资源当替罪羊、避免白跑 |
| U7 | **exact 匹配/miter 设计规格**（C1/C2 桥接）：把"exact 父-积捕获"与 K1/K8/K1x8 的 miter 目标、端口契约、向量来源写成规格文本 | CPU-only | 后续 EDA 队列只需 compile+simulate+report，最少化机时占用 |
| U8 | **窗口采样器**（只读 `/proc`、`/proc/meminfo`、`free`）：按 §2.3 门限持续采样并在窗口合格时提示 | CPU-only（在目标主机运行） | 让"排时"从等待变成可执行 |

### 2.3 Synopsys 服务器 UID 争用：可执行排时协议

**门限（后继契约统一口径，单位 KiB）**：commit headroom `≥50,331,648`（48 GiB，旧的 64 GiB 已由 M1649/M1652/M1661 后继退役）；MemAvailable `≥100,663,296`（96 GiB）；SwapFree `≥16,777,216`（16 GiB）；**同 UID EDA 碰撞 = 0**；license status-only gate 在 attempt 消耗前执行；`cgroup oom_kill` 预启动必须为 0；三轴串行 `k1 → k8 → k1x8`，禁并行；**attempt 只在首次 DC launch 时消耗**，预检失败按 prelaunch 隔离、不消耗 attempt → **窗口挑选是免费的，launch 后的重试是被禁止的**。

**结论与建议**：
1. **不要换服务器**。库/宏/license（`SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo`、`LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat`）与工具身份已钉进每个契约；换机 = 全部 tool/input identity 失效 = 重走 hammer。
2. **两条腿并行**：① 与占用方（`fangyl`）对表排时，锁定低内存窗口；② 在目标主机跑只读采样器，遇合格窗口立即按 release 流程 launch（launch 仍需先做 U6 存在性检查）。
3. 历史证据表明资源门是**间歇性**的（M555：170 快照中 16 次 <64 GiB、最低 `54.186 GiB`；M789 8/28 22:43 实测 commit headroom min `106,485,836 KiB`、MemAvailable `416,279,372 KiB`、SwapFree `56,632,060 KiB`，全过）——**48 GiB 后继门把可命中窗口显著放宽**，因此"排时 + 采样"是可行且低成本的主路径。

**launch 优先序（一次一个，前一个出 hammer 再下一个）**：
`M1663（C2 三轴 DC，3 dc_shell）` → `M1680（C1 Formality×2 + PT×1）` → `M1465（C1 runtime-witness VCS，先补 M1466）` → `M1504`（C2 source-chain successor，先补 M1505）→ `M1686`（C2 生产 SAIF/PTPX，先补 M1685）→ 其它。

### 2.4 C2 SAIF/PTPX quarantine 处置与重跑前置

**处置：原样保留，不删、不改、不重命名**（`results/m1432_..._FAILED_OR_INCOMPLETE.quarantine` 与 m1001 链 r2–r5 四个隔离目录），它们是不可引用的失败谱系；同时**补记** m1467/m1493 两次失败的隔离目录或"仅远端"声明（G11）。

**重跑前置（硬性，缺一不可）**：
1. **tiny-DUT UCLI 演练先行**：用冻结的极小 DUT/TB 走一遍真实 UCLI power 协议（`-debug_access+r` + `-lca` → `power -enable` / `-disable` / `-report`），验证 SAIF 非空且层级名正确；这是 M1043 审计自己提出的处方，**该处方在 M1046 r5 上未被证明执行**（r5 仍死于 `RUN_k1_CASE0`）。该演练本身是 EDA 运行，需自己的 source+release（成本：1 次 compile + 1 次短 simv，无 PTPX），但必须排在生产 attempt 之前。
2. compile 前置检查必须**冻结检查编译命令行里含 `-debug_access+r` 与 `-lca`**（两次真实失败分别是漏前者 `UCLI-117`、漏后者 `Error-[LCA_FEATURES_NEED_OPTION]`）。
3. 生产 attempt 仍须满足 §2.3 的全部门限，且 `ptpx 仅在 10/10 correctness+SAIF 门后`（`E_case_pJ = P_case_mW × duration_ns`）。
4. 三次同因失败后**停止**该协议，转入协议重设计（禁止第四、第五次同样探针）。

---

## 3. C1 物理准入：m931 恢复 vs C3 rank3 换基 vs 采用 M935（裁决）

### 3.1 三选项事实对比

| 维度 | (a) 恢复 m931 | (b) C3 rank3 换基 | (c) 采用 M935 系（**推荐**） |
|---|---|---|---|
| 面积 | logic `80,149.86` + macro `78,825.24` = `158,975.10 µm²` | `102,852.29 µm²`、**0 宏** | DC cell area `147,246.39209 µm²`（含 9 宏；logic 拆分须以 M993 area report 复核） |
| setup | WNS `−4.9058 ns`、TNS `−15,026.33 ns`、3,128 违例 | WNS `0` | WNS `+0.001795 ns`、TNS 0、100/100 MET |
| hold | WNS `−0.0894`（诊断） | `0` | 未签核（走 M1630/M1649/M1665 残余 hold 链） |
| 宏/存储对象 | 9 宏，但物理容量 `18,432 B` vs 账本 `213,376 B` 严重不符 | 无宏（无法代表 C1 的存储/宏账） | 9 宏（与 C1 存储对象一致） |
| 攻击面/代价 | 需对 3,128 条违例（top-100 全在 `match_bank_q_reg`→`directory_q_reg`）做目录路径流水割切/重构 → 新一次一次性全流程（compile+STA+Formality+PTPX）并重走全部 hammer；无证据比重走已完成的 M935 便宜 | 直接采用会把 C1 与 C3 两个不同执行织物混为一谈，且 0 宏点无法支撑 C1 的 SRAM/宏 PPA，破坏"同资源可比" | **已完成**：M963/M964/M965→M973→M975/M976 copy-only 恢复→M989/M990/M992 提升→M993 canonical→M1006 hammer 99/100；CPU-ledger `1.746753x`（**不是** RTL 周期加速，其原文注明） |
| 周期账 | 436,917,659 cyc 目标（M935） | 无 | M935 目标 `436,917,659 cyc` = `1.740259560x` / `1.734758869x`，硬上限 `445,851,049`（≥1.70x 门）；新状态 283 bit（<512）、+2 cyc/task |

### 3.2 裁决

- **不恢复 m931**：它与自己的账本容量不自洽（`18,432 B` vs `213,376 B`），且修复代价是"重做一次全流程"，收益为零（M935 已达标）。**m931 永久留隔离**，其 `6.25 mW` 归属须继续按更正版归到 C1 slice、不得挂 m931。
- **不以 C3 rank3 换基**：rank3 是 C3（相位解耦神经元服务）的执行织物，0 宏、面积口径不同源；用它当 C1 物理基线会同时破坏"同资源可比"和"贡献不混线"两条 DATE 评审底线。rank3 的 `WNS 0/0` 与其自身面积留在 C3 行（并更正 §1.3 G12 的口径错挂）。
- **采用 M935 系为 C1 物理基线**，剩余闭合项：① hold 签核（M1630/M1649→M1665 链，本地缺产物需对账）；② Formality（RTL→M993 与 M993→M1665 门到门）；③ PT slow-max/fast-min；④ C1 宏感知网表的 PTPX（现仅有 slice 点 `6.2538 mW`/`18.7614 pJ/cyc`）；⑤ PPA_ADMISSION 与宏账本绑定（G3）；⑥ 9 条宏与 `214,912 B` / `215,040 B` 容量账本的物理-逻辑一致性复核（`18,432 B` 的历史出入必须在 C1 报告里显式说明其口径）。

---

## 4. 侧车 revive-or-shelve 裁决

### 4.1 修正口径下的净账

| 腿 | 修正后数字 | 判定 |
|---|---|---|
| state（存储位） | 相对 fused pair-gather：`+13.5% / +10.3%`（合法域最坏行 `+47.5% / +17.8%`）——**不是** −23.4%/−26.5% | ✗ FAIL |
| 诚实可辩护口径 | 仅"相对现网 C7 物化 −41.1%"（上下文列）+ 写活动位宽差（energy 腿，**待 SAIF**） | `[模型]` |
| 周期（同端口下界） | `696 vs 691` → `+0.7%`，持平/略负 | ✗ |
| RTL/SAIF/PPA | 从未写 RTL；SAIF 机器被 C2 腿占用且 C2 自身 SAIF 已多次失败 | `[待验证]` |
| ROUND2 定位 | CONDITIONAL_PROFILE_GATE_SUPPORT_ONLY_NO_RTL；过门 +SAIF/PPA 才 3.5–3.7，**4.0 NO** | — |

### 4.2 裁决：**SHELVE（搁置）**

理由：① 唯一生产腿（energy）依赖 SAIF/PTPX，而该机器通道已被 C2 与 C1 签核队列占满，且 C2 自身还没跑通一次 SAIF；② 对 DATE 硬门槛（G1 网络级 ≥1.10x）**零贡献**；③ 周期腿为负、state 腿在合法域为负——revive 会以"负收益 + 抢 EDA 队列"的代价换取一个 4.0 也上不去的对象；④ 与"新机制"路线（§5）争夺同一批工时。**数据保留、不下架、不 revoke**；任何后续引用只能使用 count 文件口径（`163×9=1,467 bit`），**禁止再用 −23.4%**。

### 4.3 预注册 revive 条件（全过才复活，否则保持 SHELVE）

1. 以 count 文件口径重述：相对 fused pair-gather 的最坏合法行不劣于 `−20%`（当前 `+47.5%`）；
2. C2/C1 的 matched SAIF/PTPX 至少一条跑通并出 hammer（证明 SAIF 通道可用）；
3. 与 C1 宏账本同宏下的能量增量出现 `≥15%` 组件动态能量收益（runbook 门）；
4. 逻辑+宏面积 `≤+10%`、Fmax `≥−5%`、bit-exact 反压 0 mismatch。

---

## 5. 新机制 idea 搜索（不停）

### 5.1 方法与筛选口径

- **不许再从现有 RTL 拆模块计数**（`docs/433` 原文："必须是新的算法算子合同，且要同时改变硬件存储/执行对象"）。
- 也不许重复已封死的方向：patch/ATLIF/FC1/Conv 第四 matcher/跨帧 warp tile cache/AER-NoC bridge/reorder adapter（`date_open_rtl_gap_mining` 8 候选 NO-GO 表）。
- 因此候选来源限定为**算法侧已存在的算子合同草案**（`neuron_autoresearch/CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md` 的 D1/D2；D3=A3S 已 NO-GO 不用），并做**硬件实例化**：新存储对象 + 新执行对象必须落地为可合成的接口/时序对象，而不是已有流水的一次改写。

### 5.2 候选 H1：`TLQ-5`（T=5 时间商记录文件 + run-length 广播执行器）

- **合同一句话**：时间商从 pair 升为 T=5 五元组；每槽规范融合式 `s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)`（融合式为唯一规范，拆分式在 RNE 平局商奇偶处差 1 档、全域 2.74%——**硬件与部署必须同式**）。
- **新存储对象**：每位置 5 槽 `(o_t, sz_t, m_t)` 的**时间商记录文件**（记录→分数双向可逆，反解 `(s−m̄)→(o,r)` 在物理域唯一；`r_t∈{0,1,2}`，无 `s%4==3`）。
- **新执行对象**：**时间维 run-length 广播执行器**——同 Q7 类码沿 T 广播，替代逐槽位门执行（`eq=0.979` 下每位置独立门 `1.084/5`）。
- **收益（算法侧 `[模型]`，待硬件侧重算）**：门/exp-add 流量 `T=4 −73.4%`、`T=5 −78.3%`；时间边覆盖 `5/9=55.6% → 8/9=88.9%`；存储净增约 `<2%`（4 条运动边 × 5 bit）。
- **落地形态（硬件侧）**：新增"run-length 判定 + 广播"两条流水 + 5 槽记录文件接口；Q7 163 档网格与现网相同，`RNE16` 融合式已存在于 `analyze_binary_temporal_pair_arch` 的 RTL 语义中（`[rtl]` 现状），因此**风险集中在记录文件端口/带宽与广播调度**，不在算术。

### 5.3 候选 H2：`XWIN-RB`（重叠滑窗 + 滚动分母增量执行器 + 跨窗 quotient 目录）

- **合同一句话**：SWIN 非重叠 tile 改为 stride-12 / window-15 重叠滑动窗（36% token 重数 2、每窗边 90 token 共享带），归一化分母用滚动恒等式 `Z_{i+1} = Z_i − Σ_leave + Σ_enter`（整数幂和逐位精确）。
- **新存储对象**：**跨窗 quotient 目录**（共享带类码 + `Δcatalog`），共享带身份码由合同保证（`J(A,B) ≥ |classes(shared band)|/|A∪B|`；合成 Motion C 分布 300 窗链 mean `J=0.948` vs 现网 lag1 pooled `0.650`）。
- **新执行对象**：**滚动分母增量执行器**（每新窗只重算 `2×15×3=90` 个进入带 exp 项 = 270/窗，而非 450 全量；`J6` 净 exp 流量 `−4.8%`）。
- **与前次否决的关键区别**：ROUND2 的"身份死穴"是**部署模型里 SWIN 不重叠 → 无共享 token**；H2 不是"假设有共享"，而是**把重叠写进模型层语义**（正对 ROUND2 §5.1 允许的"模型层新增跨窗语义"分支），并**新建存储对象**（跨窗目录）与**新建执行对象**（滚动分母），不再是"已有 class-file 的生命周期扩展"。
- **已知代价（诚实）**：窗数 `520→825（+58.7%）`、净 exp 流量仅 `−4.8%` → **当前数字过不了 ≥10% 周期或 ≥15% 能量的工程腿**；且 baseline 对比口径要重建（需同时报 dense 与 overlap 两版）。

### 5.4 逐条过 ROUND2 否决清单

| 否决/门槛条目（原文口径） | H1 `TLQ-5` | H2 `XWIN-RB` |
|---|---|---|
| V1 方向 B"跨窗持久 RQTB quotient 目录 + delta"五项 FAIL：无新算子 / 无新存储对象 / 无新执行对象 / 周期 `+0.7%` / 身份死穴 | **不适用**（H1 不是跨窗对象） | **部分适用→已回应 3/5**：新存储对象（跨窗目录）✓、新执行对象（滚动分母增量器）✓、身份死穴（把重叠写进模型层）✓；仍**未解决**"周期/能量工程腿"（`−4.8%` < 门）与"在线构建"（目录需由 score 流在线构建） |
| V2 侧车 quotient-file CONDITIONAL、4.0 NO | 不适用 | 不适用（H2 不是侧车的换名） |
| V3 Local5 系数融合 NO-GO（`108,726` product terms `2.229x`、0/100 组下降、`0.869x` 同端口） | 不适用（非 Local5 线） | 不适用 |
| V4 A3S NO-GO（`+31.10/32.73/51.32%` @δ=2/4/8） | 不适用 | 不适用 |
| V5 第四个通用 sparse matcher / bitmap decoder / event router / delta cache / NoC bridge / reorder adapter 禁令 | **通过**：H1 新增的是"时间商记录文件 + 广播调度"，不是第四种 matcher，也不做事件路由/桶缓存 | **需澄清**：跨窗目录**不能**实现成第四个通用类码旁路——它必须由共享带身份码合同驱动，且不得新增 snapshot/delta cache 形态 |
| V6 `docs/433` 门槛：新算法算子合同 + 同时改存储/执行对象；不得拆分现有 RTL 计数 | **通过**：算子 = T>2 时间商（ROUND2 §5.1 明确点名"T>2 关系"为可证伪的解锁类别）；存储 = 5 槽记录文件；执行 = 广播执行器 | **通过（带条件）**：算子 = 重叠滑窗跨窗语义（模型层）；存储 = 跨窗目录；执行 = 滚动分母；但**必须保持"不拆分现有 RTL"**——窗口分区改动属模型语义，不是从 `window_partition_v2` 拆模块 |
| V7 4.0 解锁三条：在线构建 + 不可解释收益 + 工程闭环（≥10% 周期 或 ≥15% 组件动态能量；面积 ≤+10%；Fmax ≥−5%；反压 0 mismatch） | **未满足**：`−78.3%` 是 **gate/exp 流量**（`[模型]`），必须先在真实 trace 上换算为 ATLIF 相位占比（Amdahl ATLIF `20.6384%`）才知能否过 ≥10%/≥15%；在线构建同样待证；面积/Fmax/反压未测 | **未满足**：`−4.8%` 明显不过工程腿；在线构建待证 |
| V8 重开锚：邻接边际 ≥`0.3`（现 `+0.17`）；score-front ≥`10%`；目录写活动 ≥`5%`（现 `<1%`） | 不适用（非邻接/目录类） | **未满足**：目录写活动必须先在重叠窗下重新定价并跨过 `<1%` 的死区（`J5` 的"55.0% 交集由共享带携带"是唯一可用杠杆，属 `[模型]`，待真实 trace 复核） |
| V9 红线：不重启 H82/H86、不动 `docs/359/362/366` 与冻结合同、`[rtl]/[prof]/[模型]` 分档 | **遵守** | **遵守** |

**结论**：两者**均未被 V1–V5 的否决条目直接命中**（H1 完全不适用；H2 已回应其中 3 项），但**都还没满足 V7（4.0 工程闭环）与 V8（H2 特有）**。因此本轮定位为"**通过否决清单、待最小验证**"的候选，而不是"已达 4.0 门"。

### 5.5 各自最小验证路径（CPU/模型级）+ 证据分档

**H1 `TLQ-5`**
1. **S1（CPU-only，现在可做，无需 GPU/EDA）**：用冻结 `[prof]`（`nts11_hardware_p0_profile.json`，571 MB，672,000 `(window,head)` 行 + 已有 T=2 记录）与既有 ordered capture 复算：① `eq` 率与 5 槽 RLE 广播率（对照 Bernoulli 界 `1.084/5`）；② 4 条运动边的存储追加位账；③ 把 gate 流量降幅折算到 ATLIF 相位（用冻结 Amdahl `20.6384%`）→ 得出"周期腿/能量腿"的**区间上界**。产出 `[prof]/[模型]` JSON（脚本与输出写在**本评审目录**内或既有 `results/` 只读引用，不新增 RTL）。
2. **S2（GPU 队列，排在 t49 之后）**：`(5,15,15)` 短训 probe（seed 0，判据：loss 不塌、形态同 T=2 基线）→ 槽位 RLE 实测 dump；若 probe 过 → fullres valid825（判据 `AEE ≤ 1.3297·1.01`，锚点 Motion ep35）。
3. **S3（CPU）**：通过后才写 RTL 草图 + 位账 + 同端口周期/能量模型（`[模型]`），并按 V7 逐项预注册。
- **门槛信息**：S1 的判定必须先看"ATLIF 相位占比 × 流量降幅"能否碰到 `≥10%`；若上界 `<10%`，H1 直接降级为附录级，不必进 S2。

**H2 `XWIN-RB`**
1. **S1（CPU-only）**：在冻结 profile 上重算重叠窗分区：① 目录写活动占比（目标跨过 `<1%` 死区，锚 `≥5%`）；② exp-term 净流量（现 `−4.8%`，需在真实 token 分布下复核；若不能接近 `≥10%` 则直接触发 V7 不通过）；③ 真实数据上的 mean `J` 与共享带携带比（复核合成值 `0.948` / `55.0%`）→ `[prof]/[模型]`。
2. **S2（GPU 队列）**：dense vs overlap 两版有效训练对比（判据 `overlap AEE ≤ dense × 1.02`）——**成本最高、放最后**。
3. **S3（CPU）**：目录/滚动分母的接口规格与同端口模型；须同时说明"为什么不是第四个 delta cache/旁路"。
- **门槛信息**：S1 若不能在真实 trace 上把目录写活动或 exp 流量推进到 V7/V8 门内，**立即冻结**，只留作附录的"跨窗语义演进"讨论。

### 5.6 与现有线的共存与占用纪律

- 两个候选都**不动 C1/C2/C3 的现有 RTL**，也不要求 EDA 机时（S1 纯 CPU）；S2 走 GPU 队列排在 `t49_ternary_ws4_ep5` 之后（本评审未接触该进程）。
- 与 §2 的 EDA 队列**不冲突**：H1/H2 的 S1 是 CPU-only，正好填"等窗口"的空档；且它们回答的是 DATE 门槛的另一条腿（新机制），与 G1–G6 的收口并行。

---

## 6. 本轮未做 / 不做（红线声明）

- 未运行 GPU（`t49_ternary_ws4_ep5`，PID 721081，全程未触碰）、未运行 VCS/DC/Formality/PT/PTPX、未执行任何 launch、未创建 attempt/result/lock。
- 未修改 `docs/359`/`362`/`366` 与任何冻结合同或编号文档；未删除任何文件；`H82`/`H86` 保持停止。
- 本目录外零写入；本目录内仅 `review.md` + `review.json` + 两个封文件。
- 本文所有数字均带来源与分档；凡未独立复算者一律标注为引用（不冒充本轮实测）。
