# M531｜H67 分层硬件覆盖与 G-line 有损状态独立审计 r1

日期：2026-08-27  
范围：Motion H67 ep35；只读审计，不运行 CPU/GPU/VCS/DC/PTPX，不访问远端，不修改生产 RTL。  
裁决：**当前不是“每一层都有独立、已准入的加速”，而是一个共享 signed-source fabric 已覆盖多类算子，外加若干层级支撑模块。Conv 有最强局部 exact candidate，FC2/attention/decoder/prediction 有硬件证据，FC1/patch/dynamic-BN 尚无可升主表的正结果；当前有损线没有 accuracy-admitted 性能点。**

## 1. 证据等级

| 等级 | 含义 | 能否进性能主表 |
|---|---|---|
| E0 | idea / 规格 / 未测 | 否 |
| E1 | 分析上界、opportunity 或 Amdahl sensitivity | 否，只能动机/附录 |
| E2 | 冻结真实 trace 的 CPU/cycle ledger，守恒与身份闭合 | 可作为局部 candidate，必须标注非 RTL/PPA/system |
| E3 | exact-SHA VCS/SVA 或 RTL-handshake recurrence | 可写功能/协议或局部周期；不能外推系统/PPA |
| E4 | 新思 DC/STA/PTPX 已封存 | 可写限定范围的逻辑 PPA；0-macro 不等于 paper PPA |
| E5 | 同资源、memory-inclusive、decoder-complete、多序列系统测量 | 可进系统 headline |

本项目目前最高到 E4 的**切片证据**，尚无 E5。

## 2. 各层/算子覆盖表

| 层/算子 | 当前最强可核数字 | 最高等级 | 论文资格 | 仍缺什么 |
|---|---|---:|---|---|
| Bottleneck Conv | M528 单口 parent/product capture：`435,293,339` cycles；相对 M468 strong-zero `1.746753x`、相对 same-coordinate bit `1.741232x`；10/10 samples 均 `>1.50x`；宏取整 `213,376 B < 240 KiB` | E2 | **主贡献 C1/C2 的局部候选**；只能写 ep35 单序列、四层、exact CPU same-ledger | M530 repaired RTL 的独立 admission、VCS/SVA、1R1W macro/端口、DC/STA、memory-inclusive cycle/energy、decoder-complete 系统合成；当前 `rtl/vcs/dc/energy/system=false` |
| FC1 | M482 full-width RTL-handshake recurrence `1.359896673x`；冻结门 `1.50x`，结论 NO-GO；理想旧 envelope sensitivity `1.044983x` | E3（负结果） | **消融/负结果**，不是正贡献；不必再补“真实 trace”，因为 100-record recurrence 和协议已闭合 | 只有全新、同资源架构能重开；不得把 M229 早期 `2.59x` island 当真实 FC1 点 |
| FC2 | C2 K8/单 K1 `4.7642x`，逻辑面积 `20,436.7/20,587.4 um^2`（`+0.737%`）；但等带宽 K8/K1x8 仅约 `1.01--1.04x` | E3；旧局部面积为 E4 边界 | **主贡献 C2 的支撑算子**；合法卖点是 shared state、throughput/mm2、energy/work，不是等峰值稀疏倍率 | M519 clean 三点同 top DC/STA；r5 runtime gate 已假杀，必须以新身份 r6；再补 SAIF/PTPX 和公共 SRAM 宏 |
| ATLIF | M518 Fixed-T10 VCS compile/sim `0/0`，51 assertions、25/25 covers，`17 cycles/tile`；rank3 旧分析 isolated `3.399935x`，旧 logic-only DC `102,852.3 um^2`、WNS 0/0 | Fixed E3；rank3 各自 E1/E4 | **C3 条件候选**，现阶段只写 Fixed 行为和 rank3 单独逻辑 PPA，不能把二者拼成 speedup/PPA 点 | matched Fixed/rank3 DC、同资源分母、rank identity/accuracy、公共状态 SRAM、系统 schedule |
| Attention / RQTB | local Fixed `112,589` vs RQTB `94,891` cycles，`1.18651x`；面积 `134,076.60` vs `135,760.46 um^2`（`+1.2559%`），area-normalized `1.17179x`；旧 included-scope 系统仅 `1.000911x` | E3+E4 的 bounded component | **网络完整性/局部 exact 支撑**，不作 headline；attention 旧份额仅 `0.5889%`，无限快上限约 `1.0059x` | power、macro、完整 attention dataflow；但低 Amdahl，不应占 P0 队列 |
| Patch embed | MRU exact ordered trace 最好约 `1.017588x`；whole-temporal zero site `156/4,032,000 = 0.00387%`；极端 scan/commit 上界约 `1.0617x` | E1/E2（负） | **不作为加速贡献**；只说明共享 fabric 可执行，并把负结果放消融 | 没有值得新开 RTL 的同资源机会；不应从输入事件稀疏外推中间层空 site |
| Dynamic BN | Q24→bit-tight 16/24 的分析 traffic opportunity `1.446701x`，512-bit/32-lane schedule `1.446835x`；旧 envelope sensitivity 约 `1.054233x` | E1 | **可能成为 memory-support 子机制**，不能单列 novelty/性能 | exact integer/raw/address capture、consumer schedule、RTL/VCS、宏/DRAM 能量；current-batch BN 不能静态 fold |
| Decoder ConvTranspose | M522 mapper：`383.670001 um^2`、442 cells、setup/hold `+1.4266/+0.0106 ns`；M523 bundler VCS 43 taps/8 bundles/0 assertion，10/10 covers | Mapper E4；bundler E3 | **C2 的网络完整性支撑**；当前没有 decoder speedup | 四层×10 samples exact trace；same-resource A1 vs TDR/PGPR cycle+traffic；decoder 约占修正 envelope `21.57--22.83%`，这是当前最大未闭口径 |
| Prediction head | M60 bounded signed event+commit opportunity `3.086389x`；M62 directed VCS 通过；观察到的未封存 DC 为 `35,459.1717 um^2`、setup/hold `+0.6523/+0.0101 ns` | opportunity E1，VCS E3，DC 不准入 | **shared signed-source fabric 的支撑模块**；不能当系统贡献 | M60 不是 RTL cycle；M62 DC/Formality 身份未封；该 kernel 份额约 `2.6%`，即使局部 `3.086x`，Amdahl 也只有约 `1.018x` |

结论不是“缺哪个层就再造一个 accelerator”。DATE 可接受的写法是三项组合贡献：

1. signed-source execution fabric 覆盖 Conv/FC/decoder/prediction；
2. exact parent/product capture 解释 Prosperity/Phi opportunity 如何在 240 KiB、单口、commit 约束下被捕获；
3. phase-decoupled ATLIF service，前提是 M518 matched 公平门闭合。

Patch、BN、RQTB、decoder mapper/bundler 是覆盖或支撑，不应拆成七八个“创新点”。

## 3. G-line / 有损状态

| 线 | 结论 | 证据与边界 | 是否重开 |
|---|---|---|---|
| G7 activation magnitude gate | **真死（当前对象）** | 105 个 ATLIF 与四层 bottleneck Conv 的非零值近 `1.0`；冻结 θ 网格移除 0，跨过幅值即 0→100% cliff，没有微有损 Pareto | 否；新 checkpoint/新算子 trace 才是新研究身份 |
| G8 whole-FFN token/site skip | **当前机制真死，不是数据缺口** | M460 capture 已完成；M462r2 冻结 τ 网格的 accounted savings 全为 0；`tau=0` 无周期收益。事后 oracle `tau>0.8713` 的 1.15x 没有 accuracy/预注册资格 | 否；不应再以“补一次 capture”名义重跑 |
| G10 empty tile/site skip | **周期线真死** | bottleneck output-site empty 约 `0.1117%`，patch whole-temporal empty `0.00387%`，bit baseline 已跳 zero-source work | 只可留 traffic/energy 敏感性一句，不开 RTL |
| G11 bounded source drop | **真死** | 静态 beta 点要么收益小，要么 S10 `Delta AEE=0.110531` 超 `0.02`；动态累计预算 B 在收费 schedule 下最好反而 `0.625315x` | 否 |
| G12 ATLIF remaining-budget early stop | **真死** | term skip `6.577%`，但 32-lane aligned issue reduction 仅 `0.0676136%`，conditional speedup `1.0000797x` | 否；可作“term sparsity不等于周期”消融 |
| epsilon-RQTB / mass truncation | **附录概念，不是性能主线** | 未形成预注册 accuracy/cycle/RTL 点；attention 份额过低。即使局部 4x，系统仍约 `1.004x` | 仅当附录 completeness；不得占 RTL/DC 队列 |
| drift-bounded cross-frame | **未来条件候选** | 当前自然 local-vs-temporal source-work 机会仅约 `2.7%`；前帧状态 SRAM、warp/mask/refresh 未收费；与 DeltaCNN/MotionDeltaCNN 邻近 | 只准多序列离线 DSE；本轮不写 RTL |
| PAFT positive-distance near-match | **accuracy gate 失败，当前身份永久关闭** | 早期十帧 S10 `Delta AEE=0.014362` 通过，但 paired valid825 primary `Delta AEE=0.0293279696 > 0.02`；18/18 sequences mean delta 均为正，实际执行 `245,630,707` 次 positive-distance replacement | 不得搜新 τ/layer subset；只有新 checkpoint、train-only 预注册和独立 holdout 才能作为新研究 |

因此，有损方向当前大体进度是：**机制空间已系统筛选，但没有一个可写入 DATE 主性能表的 accuracy-admitted 点。** 这不是实现停滞，而是验证已经排除了虚假的局部高倍率。论文应以 exact 主线为主，有损只在负结果/未来工作中说明。

## 4. M528 之后唯一最高 ROI 增量（建议，不授权运行）

### 选择：完成 M511/M513 四层 ConvTranspose exact trace 与同资源 CPU gate

不选 G8：已有真实 capture，冻结点 savings 为 0。  
不选 FC1：M482 已在 100-record frozen workload 上闭合 RTL-handshake recurrence，`1.3599x` 是已判定负点。  
选择 decoder：旧 `620M` ledger 明确漏四层 ConvTranspose；decoder 占修正 envelope `21.57--22.83%`，M522/M523 支撑前端已经存在，exact trace 同时修正系统分母和决定 TDR 是否值得继续。

严格 GO gate：

1. 身份与守恒：四层×十样本共 40 records；checkpoint/source/payload SHA 全闭合；layer 名称、K3/S2/P1/OP1、tap、边界、event 和 destination update 全守恒。
2. 公平基线：A1 与候选使用相同 96 product lanes、weight/psum SRAM bank/port、descriptor、destination commit；候选额外 state/bitmap/refresh 全收费。
3. 周期：ratio-of-summed-runtimes 局部 `>=1.20x`，且每个 sample `>=1.10x`；计 scan、bundle、bank conflict、commit、stall 和 SRAM latency。
4. 载荷：不得增加总 DRAM；若局部周期不足 `1.20x`，只有 measured DRAM traffic `>=30%` reduction 才保留为能量支撑，否则 NO-GO。
5. Amdahl：以 decoder 份额 `21.57--22.83%` 计算，局部 `1.20x/1.30x/1.50x` 仅对应系统约 `1.0373--1.0396x / 1.0524--1.0556x / 1.0775--1.0824x`。这些是决策敏感性，不是测量结果。
6. 在 CPU gate 通过前，**不授权新增/修改 decoder 性能 RTL**；本审计也不授权任何 GPU、VCS、DC、PTPX 或远端运行。

## 5. DATE Accept readiness

| 维度 | 分数 / 5 | 判断 |
|---|---:|---|
| Novelty | 3.5 | 三项组合贡献可成立，但相邻 prior art 强，必须以对象/协议/资源差异而非改名包装 |
| Soundness | 4.1 | fail-closed、双 seal、负结果纪律强；主动修正 decoder 分母加分 |
| Implementation | 3.6 | C2/RQTB/ATLIF/decoder/pred-head 有 VCS/DC 切片；M528 仍未过 RTL/物理门 |
| Evaluation | 2.5 | 无 E5：decoder 未补、memory-inclusive system/energy 与多序列缺失 |
| Paper convergence | 3.0 | claim 边界已清楚，但主表还不能封 |
| **综合** | **3.3 / 5** | **Borderline Reject；完成 P0 后可进入 Weak Accept 讨论** |

### P0

1. 无 decoder-complete、same-resource、non-overlap 的 full-network cycle ledger；旧 `620M` 不能再叫全网。
2. M528 `1.7468x/1.7412x` 仍缺 admitted RTL/VCS/DC/1R1W macro 与 memory-inclusive closure。
3. 无统一 logic/SRAM/DRAM energy 与公共 28 nm/3 ns 资源主表。

### P1

1. M519 FC2 三点 clean DC/SAIF/PTPX 未闭；r5 中间 QoR 不可引用。
2. M518 Fixed/rank3 matched DC、公平资源和 accuracy/rank identity 未闭。
3. 至少三条 DSEC sequence 或低/中/高 event-density 分层未闭。
4. Dynamic BN exact address/traffic capture 缺失；只能先作支撑。

### P2

1. Prediction-head M62 DC/Formality 身份未封。
2. RQTB power/macro 未闭，但 Amdahl 低，不升 P0。
3. FC1/patch/G-line 负结果需要在消融表统一口径，避免被误读为遗漏。

## 6. 证据锚点与边界

主要锚点：

- `docs/524_DATEAccept当前硬件贡献与机制迁移收口表_20260827.md`
- `docs/500_新RTL封门与DATE硬件收口路线_20260827.md`
- `results/m528_h67_single_port_same_ledger_recompute_r4_20260827/`
- `reviews/m528_r4_result_hammer_r1_20260827/`
- `reviews/m482` 对应 exact runner/RTL 与 `reviews/m528_false_kill_audit_r1_20260827/`
- `results/m518_matched_fixed_t10_atlif_vcs_r11_exact_20260827/`
- `reviews/m519_r5_channel_local_fault_vcs_receipt_blind_hammer_r1_20260827/`
- `reviews/m522_m514_c2d_logic_only_dc_receipt_blind_hammer_r1_20260827/`
- `reviews/m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_blind_hammer_r1_20260827/`
- `reviews/m487_positive_distance_correction_independent_hammer_r1_20260827/`
- `reviews/m462r2_independent_hammer_r1_20260826/`
- `reviews/topwork_rtl_lastcall_scout_r1_20260827/`

本报告没有运行任何生产 workload、EDA、GPU 或远端任务，没有修改生产 RTL，也没有修改 `docs/359_DATE终局冻结_20260813.md`。冻结 SHA256 复核值必须保持：

`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

