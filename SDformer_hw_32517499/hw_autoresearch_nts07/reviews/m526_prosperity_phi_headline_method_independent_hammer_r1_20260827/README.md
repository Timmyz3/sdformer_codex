# M526 Prosperity / Phi headline 方法独立打铁 r1

日期：2026-08-27  
评审边界：只读检查 M526 方法文档、审计脚本、封存输出、官方 Prosperity workbook/代码与三篇公开论文；独立重算聚合指标。未修改被审文件，未运行 EDA，未修改 `docs/359_DATE终局冻结_20260813.md`。

## 结论

评分 **91/100**，`P0=0 / P1=5 / P2=2`，裁决为 `CONDITIONAL_GO__METHOD_SOUND_AFTER_SCOPE_AND_BASELINE_CLARIFICATION`。

M526 的主方向是对的：它没有把外部 `7.4x/3.45x/2.459487x` 冒充 H67 结果，也明确禁止局部倍率相乘、理论工作减少冒充周期、隐藏等带宽强基线和 selected-slice 功耗冒充 energy/frame。官方 workbook 的 16 workload 重算、输入身份和双 seal 均通过。因此这份材料可作为 H67 统一系统表的实验设计基础。

但 **Prosperity 的 `7.4x` 不应无限定地称为“全网 7.4x”**。它是 PTB/Prosperity 在 16 个模型/数据集上的平均 runtime 比值 headline；对 CNN，作者模拟了支持的 Conv/FC/LIF 路径，而对 spiking transformer，论文明确说 PTB/SATO/MINT 只运行其支持的 linear layers，官方代码也让这些 baseline 跳过 attention/LN。与 A100 的 `1.79x` 才被论文明确描述为 spiking-transformer end-to-end。最安全的表述是：

> Prosperity reports a 7.4x average speedup over PTB across the evaluated workload suite under the paper's supported-operator comparison; this is not a strict all-operator, iso-area, full-network comparison for every transformer workload.

H67 可以合法仿照其**呈现结构**：完整候选对 structured/dense baseline 给 headline，同页给 strongest exact iso-service baseline，并用同一 simulator 的直接重跑 waterfall 解释增量。不能仿照成“只保留弱 denominator”；H67 的 K1x8 等服务对照必须留在主表。

## 独立重算：数字成立，但聚合口径仍不能反推

官方仓库 commit 为 `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`，`time_reference.xlsx` SHA256 为 `47a05d06a0e762b9a67490875803441eac2bcec9a24a14576896f945452ba563`。逐 workload 以 `PTB runtime / Prosperity runtime` 独立重算：

| 聚合 | 全部 16 | CNN 4 | Transformer 12 |
|---|---:|---:|---:|
| arithmetic mean of ratios | `7.461106560x` | `7.840703124x` | `7.334574372x` |
| geometric mean of ratios | `7.313884876x` | `7.765144699x` | `7.169369695x` |
| ratio of summed runtimes | `6.731836241x` | `8.152991553x` | `6.457129839x` |
| min / max per-workload | `4.975778700x / 11.466771759x` | `6.785623226x / 9.438124911x` | `4.975778700x / 11.466771759x` |

M526 的 arithmetic/geomean/min/max 与独立重算逐位一致，官方 workbook 的 Eyeriss geomean `R22=14.134711225914344` 也一致。结果目录 `SHA256SUMS` 与外层 seal 均通过，审计脚本 `py_compile` 与 `git diff --check` 通过。

但是论文的 `7.4x` 距 arithmetic mean 为 `0.0611x`，距 geomean 为 `0.0861x`；两者按通常保留一位小数分别会成为 `7.5x` 与 `7.3x`。所以只能说 artifact 的 arithmetic 值**数值上更接近** `7.4x`，不能据此断言论文 headline 采用 arithmetic mean。尤其 ratio-of-sums 只有 `6.7318x`，说明 H67 必须冻结聚合函数，不能在写摘要时再选择。

## Prosperity 大倍率的合法构成

[Prosperity 论文](https://arxiv.org/html/2503.03379)给出的消融是：PTB 到 unstructured bit-sparse `2.28x`、再加高开销 product-sparse dispatch `2.16x`、再用 overhead-free dispatch `1.49x`。三项标称平均的乘积为 `7.337952x`，接近 headline，但“平均的乘积”等式并不等价于“逐 workload 最终倍率的平均”；它适合做叙事 waterfall，不应当作重新推导最终结果的算式。

同架构更强且更干净的机制口径是：product sparsity 相对 bit sparsity 平均 runtime `3.2x`，product density 理论减少平均 `5.0x`。headline `7.4x` 之所以更大，是因为 denominator 同时包含 PTB 的 structured-sparsity/dataflow 劣势，而不是 product reuse 单项就给 `7.4x`。

论文/官方 artifact 的其他范围边界：

- 28 nm、500 MHz；128 PE；8/32/96 KiB spike/weight/output buffers；64 GB/s DDR4；
- 逻辑用 RTL/DC、SRAM 用 CACTI、DRAM 用 DRAMsim3、性能用 cycle simulator；不是流片全芯片测量；
- Eyeriss 与 Stellar 比候选多 31% PE，因此不是严格 iso-area；
- 论文声称 spiking GeMM 占工作量 `>98%`，这使 kernel-focused architecture 在其模型上接近系统主导，但该先验不能迁移到 H67，必须用 H67 自己的 operator ledger；
- 官方 simulator 的 `num_ops` 在 product 模式按 product-reduced activation accumulation 计数，在 bit 模式按原始非零 accumulation 计数；它不是天然固定的 dense-equivalent numerator，不能直接据此构造 H67 headline GOP/s。

## Phi / FireFly-T 的呈现方式

[Phi 论文](https://arxiv.org/html/2505.10909)的 `3.45x` 是 Phi 对作者选定的最强 baseline Stellar 的 VGG16/CIFAR100 系统级模拟比较；Stellar 使用原论文结果，并非同一开源 simulator 的所有原始 trace 重放。Phi 同时用 `3.2--6.1x` theoretical speedup over bit sparsity 说明机会，用 RTL/DC + CACTI + DRAMsim3 说明实现成本，再把 PAFT 的 `1.26x` 增量和精度单列。这支持 M526 的“theoretical / captured / lossy 分列”，但不支持把理论密度倍率写成系统倍率。

[FireFly-T 论文](https://arxiv.org/html/2505.12771)的 `3.48x` 是 fixed-total-bandwidth、comparable-resource 的 load-balance 微基准；论文主表更强调 GOP/s/W 与 GOP/s/DSP，且网络、FPGA、资源和频率各不相同。这支持 C2 用 throughput/mm2、energy/source 和 equal-service area 去体现价值，而不是强行从 K8/K1x8 的 `~1.01--1.04x` cycle 差制造系统 headline。

## Open findings

### M526-P1-01 — 顶部“工作负载平均加速”仍需显式限定 supported-operator scope

M526 第 9 行随后在第 31 行说明 transformer baseline 不含 attention/LN，但用户最关心的“是否全网”答案仍容易在只读摘要时被理解成完整网络所有算子。论文自己把第 VII-C 节命名为 End-to-End Performance Analysis，同时又明确限定 PTB/SATO/MINT 只跑 transformer linear layers；这两句必须一起呈现。

修复：把首句改成“16-workload supported-operator-scope average runtime speedup”；单独列 `PTB vs Prosperity` 和 `A100 vs Prosperity` 的 operator scope。H67 的 headline 只有在 decoder、attention、ATLIF、BN、patch/conv/FC 和 memory 全部进统一账本后才能使用 full-network/frame。

### M526-P1-02 — 不能从“更接近”断言 `7.4x` 的 averaging convention

算术平均 `7.4611x` 与几何平均 `7.3139x` 都在 `7.4x` 附近，且两者通常一位小数舍入都不是 `7.4x`。官方 workbook 明确保留了 Eyeriss geomean，但没有给 PTB headline 的可验证公式单元格。

修复：保留两种均值，并新增 ratio-of-sums `6.731836x`；把 JSON boolean 改成三种聚合与 `7.4x` 的距离，不输出暗示 convention 已识别的布尔结论。H67 摘要默认 geomean；若用 arithmetic，正文显式定义，ratio-of-sums 进入敏感性。

### M526-P1-03 — B3 将 K8 与 K1x8 合成一行，破坏 C2 的 strongest-baseline 定义

K8 是 typed shared-state candidate/机制路径，K1x8 是 replicated equal-service baseline；二者不是一个 baseline。合并后既无法计算 C2 的增量，也可能让 Ours 同自身比较。

修复：拆成 `B3=exact K1x8 equal-service replicated baseline`、`C2=exact K8 typed shared-state candidate`。分别冻结 peak source/cycle、ports、queue depth、Acc24、frequency、SRAM/BW、area 与 power。headline 主表必须有 Ours/B3；Ours/B2 `4.76x` 只标为 bandwidth-scaling。

### M526-P1-04 — effective GOP/s 的 numerator 必须固定，不能沿用 architecture-dependent `num_ops`

官方 Prosperity code 在 product 和 bit 模式对 `num_ops` 使用不同 activation population；M526 第 49 行把它描述为 effective GOP/s 的依据，证据不足且可能使 numerator 随 candidate 改变。

修复：H67 主表只使用固定 checkpoint/trace 的 `dense-equivalent OP` 和固定 `original useful-nonzero OP` 两个 numerator；另报 physical retired source/cycle。任何 architecture-reduced operation count 只能叫 executed additions，不能作为跨配置 throughput numerator。

### M526-P1-05 — B0/B1 的“相同 96 lane”不是 iso-area，headline 必须双口径

Prosperity 的论文比较也不是严格同面积：Eyeriss/Stellar 有更多 PE。H67 可以合法用 B0/B1 做 architecture-stack headline，但只锁 lanes/SRAM/BW 不能证明资源公平，尤其 C1/C2 matcher/scoreboard 增量明显。

修复：Table A 同时给 `iso-lane latency` 与 `area-normalized throughput`，并始终保留 `iso-service K1x8`。若完整 Ours 面积显著高于 B1，摘要 speedup 必须标注 iso-lane，而不能写 iso-resource。

### M526-P2-01 — 审计脚本记录 commit 字符串但没有动态验证仓库 HEAD

本评审现场确认 local official repo HEAD 与常量一致，workbook SHA 也正确；但脚本只验证 workbook SHA，无法证明输出时该 commit 对应的代码树仍在。

建议：冻结 `git rev-parse HEAD`、dirty status、paper version/date 和 workbook relative path 到 input receipt；若 repo dirty 则 fail closed 或显式记录。

### M526-P2-02 — 外部 paper claim 与 workbook audit 证据层尚未分机器状态

当前 JSON 只审计 Prosperity workbook aggregation，Phi/FireFly-T 数字来自论文人工审阅，正文边界清楚但机器状态只有一个总标签。

建议：将 `prosperity_artifact_recomputed`、`paper_text_verified`、`h67_not_run` 分字段；避免后续自动生成表格时把人工 paper facts 当 artifact-replayed 数字。

## H67 的合法仿照模板

H67 应采用以下四层，而不是寻找一个可以任意放大的数字：

1. **Headline system table**：同 checkpoint/ordered trace/precision 下，报告 Ours 对 Dense96 Fixed-T10 与 PTB-like structured baseline 的 latency/frame、energy/frame 和固定 numerator effective GOP/s；明确 `iso-lane`。
2. **Strongest-baseline row**：同页报告 Ours 对 exact K1x8 equal-service 的 cycle、area、power、throughput/mm2、energy/source；这是防 reviewer 反除的核心。
3. **Measured waterfall**：B1 -> unstructured exact -> C1 -> C2 -> C3 每一步都重跑统一 simulator；只报告直接测得 incremental/cumulative，不乘离线局部倍率。
4. **Opportunity/capture graph**：理论 source-work、无限 parent/PWP、240 KiB + 64 GB/s captured、最终 decoder/memory 四点并列，解释为何外部 2.46x 机会在 H67 物理约束下收缩。

如果最终 Ours/B0 或 Ours/B1 直接重跑超过 `2x`，摘要可以明确写 `up to/geo-mean Xx over Dense96/PTB-like at equal lanes`，同句或下一句给 Ours/K1x8。若只有约 `1.8x`，用 `1.8x system + C2 throughput/mm2/energy + C1 capture-gap + C3 exact ATLIF` 仍是完整、可信的 DATE Accept 叙事；不应为追求 `2x` 更换聚合或删掉强 baseline。

## 证据与封存边界

| artifact | observed SHA256 / result |
|---|---|
| M526 文档 | `7abd0b289216ea5f6e27c5cf4eeb977d3120e86355e95e3f2f1798054d6af5e5` |
| M526 审计脚本 | `38a1cdf7c524081bd516812b60e5580c45749afb6b7b113c5429ff58777d3aa4` |
| M526 REPORT | `3d42bf60b8f90b0aa959474aedba79304550772bbd1f60c7cc4272f542c8b686` |
| M526 JSON | `905ed9c59ad5d3ac157d0e80cb74fcbc60f8c9120518526a9fde0ec62bdd3c30` |
| M526 inner manifest / outer seal | `2302f2fe...ce03` / `8ce77bf6...b0f`，均校验通过 |
| official workbook | `47a05d06a0e762b9a67490875803441eac2bcec9a24a14576896f945452ba563` |
| official repo HEAD | `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`，工作树 clean |
| `docs/359` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改 |

机器结论见 `m526_prosperity_phi_headline_method_independent_hammer_r1.json`，独立重算见 `independent_recompute_r1.json`。本评审不准入任何 H67 system speedup、energy/frame、paper PPA 或外部方法的 ours claim。
