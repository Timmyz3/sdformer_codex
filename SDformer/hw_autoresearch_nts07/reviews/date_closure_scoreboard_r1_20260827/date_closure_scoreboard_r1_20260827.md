# DATE/BP 硬件收口证据矩阵 r1

审计日期：2026-08-27（Asia/Shanghai）  
审计对象：`hw_autoresearch_nts07` 当前已封证据  
审计方式：只读、fail-closed；未运行 VCS/DC/Formality/PTPX，未修改生产 RTL，未修改 `docs/359`。

## 0. 结论先行

当前硬件工作**足以组织成一篇有明确贡献的 DATE 投稿，但尚未达到可稳收、也未达到 Best Paper 证据完整度**。强项是：已有多个 exact-SHA VCS/SVA 回归，C1 选定切片已有 DC、Formality、PrimeTime STA、SAIF/PTPX 闭环，A1 也有形式等价与局部功耗；弱项集中在共同分母：新 Conv/FC2 顶层还没有匹配的物理与能量闭环，没有目标 SRAM macro，没有跨多条 DSEC sequence 的分布，也没有一个不重叠的全网周期/能量表。

截至本审计截点，模拟评分为：

| 口径 | 分数 | 裁决 |
|---|---:|---|
| 硬件机制与实现证据 | **3.35/5** | 有 DATE 竞争力，但主性能句仍受同资源和系统口径约束 |
| 按当前完整投稿推断 | **3.2/5** | Borderline Reject，约 25--35% 接收概率 |
| BP shortlist readiness | **1.9/5** | 未就绪；当前约 0--5%，不能以继续叠加局部 RTL 解决 |

最短上升路径不是再开一种稀疏机制，而是封住两个物理端点并建立共同分母：

1. M498 Conv 最终 DC 判门，并对最终 admitted top 做 Formality、PTPX 与 144 B 1R1W scratch/psum 存储模型；
2. M496 对 K1/K8/K1x8 做同 top、同 SDC、同端口的三点 DC，随后只以吞吐/mm²、能量/完成 token 描述 8 倍带宽 Pareto；
3. 生成一张不重叠的全网 cycle、SRAM、DRAM、energy/frame 表，并至少覆盖 3 条真实 DSEC sequence；
4. 外部 Prosperity 重放 `2.4595x` 只作为 iso-workload opportunity，不得写成“本加速器 2.46x”。

## 1. 审计规则与符号

本报告把“有文件”与“可投稿准入”分开：

- **PASS**：冻结身份、测量对象和结论边界均明确，可直接支撑相同范围的论文句子；
- **PARTIAL**：有结果，但缺 macro、同资源、公平基线或只覆盖局部切片；
- **OPEN**：没有可引用的最终结果，或任务仍在运行；
- **FAIL/NO-GO**：已有反例或约束失败，不得当作正结果；
- “系统倍速”仅指同一完整网络、同一 workload、共同资源/内存模型下的端到端周期比。局部 source-work、算子周期、不同银行数的吞吐比均不属于系统倍速。

命名存在一组容易误读的重叠：C2 是旧的 FC2 K8 grouped-source 贡献，而“FC2 closure”是 M499/M496 用 K1/K8/K1x8 重建公平物理端点；C1 是已物理化的 selected Conv/PWP slice，而“Conv closure”是 M467/M473/M498 新执行岛路线。后两者分别是对 C2、C1 的资格补强，**不是两个额外独立贡献**。

## 2. 总证据矩阵

| 线 | exact VCS/SVA | DC/STA | Formality | SAIF/PTPX | SRAM/macro | 多序列 | 统一周期/能量 | Claim boundary |
|---|---|---|---|---|---|---|---|---|
| **C1 selected Conv slice** | **PASS**：M414，17,280 configs、51.84M source rows、0 mismatch | **PASS/PARTIAL**：M416 24,548.71 µm²；3 ns setup +0.7636 ns、hold +0.025 ns；M422 PT setup +0.7598 ns、hold +0.0178 ns；均 pre-layout/0 macro | **PASS**：M420，5,368 compare points；0 fail/abort/unmatched | **PASS/PARTIAL**：M448R4 6.2538 mW、18.7614 pJ/measured cycle，22,800/22,800 nets annotated；selected slice only | **OPEN**：PWP/scratch/psum 未用目标 macro | **OPEN**：当前 H67 S10 是 Zurich 的 10 个窗口，不是 10 条 sequence | **OPEN**：没有 full Conv/full network energy | **CLEAR**：可报 exact selected-slice 与 prelayout logic；不得外推 full Conv 或 mJ/frame |
| **C2 legacy FC2 K8** | **PASS**：M216 exact VCS；局部 K8/K1 `4.764209x` | **PARTIAL**：恢复的合法子运行 K1 20,436.70、K8 20,587.39 µm²，+0.737%；3 ns setup/hold 非负；父目录失败，且 logic-only | **OPEN** | **OPEN** | **OPEN**：权重 SRAM、bank conflict、BN2/SN2 不在边界 | **OPEN**：120 records 不是多 sequence | **OPEN** | **CLEAR after correction**：`4.764x` 是 K8 对低带宽 K1 的局部比；同峰值 K1x8/K8 为 `1.000x`，不能称同资源稀疏或系统倍速 |
| **C3 ATLIF** | **PASS**：M273，225 result beats、7 protocol attacks、0 mismatch；压力测 1,618 cycles、FIFO occupancy 16 | **PARTIAL**：M289 protocol-repaired DC，102,852.29 µm²、133,263 cells、9,639 FF、3 ns setup/hold 0；0 macro | **OPEN** | **OPEN** | **OPEN**：状态存储未物理化 | **OPEN** | **OPEN** | **CLEAR**：可报 rank-0 局部机制；ep35 全 rank-0，rank-3 未被训练/精度准入；分析 `~2.000x` 不等于同面积系统倍速 |
| **A1 RQTB attention** | **PASS**：冻结 sample0 与 S10 路径均有 Synopsys VCS 证据 | **PARTIAL**：Fixed 94,663.67、RQTB 96,080.92 µm²；0 macro；ideal-clock hold WNS -0.01 ns，非 signoff | **PASS**：Fixed 26,580、RQTB 26,713 compare points；0 fail/abort/unverified | **PASS/PARTIAL**：80.50→68.39 nJ/head-row，`1.177x`；100% SAIF，但只含 pre-macro logic | **PARTIAL**：CACTI 32 nm proxy；无 foundry SRAM/DRAM macro | **OPEN**：S10 仍是同一 Zurich sequence 的窗口 | **PARTIAL**：activity envelope Fixed 620,868,243→RQTB 620,302,905，`1.000911x`；是模型而非完整 schedule/FPS | **CLEAR**：局部 `1.1865x`、S10 `1.1764x`；attention 仅 0.5894%，只能作局部能量/完整性贡献 |
| **FC2 closure M499/M496** | **PASS**：M492 K8/K1x8 各 B1/B2/B4/B8 周期完全相同；M497 K1x8/K1 geomean `5.8634x`、aggregate `6.2343x`，均 0 mismatch；M499 修正同拍 reuse | **OPEN**：M496 三点 matched DC 尚无运行回执 | **OPEN** | **OPEN** | **OPEN** | **OPEN** | **OPEN** | **CLEAR**：`5.863x/6.234x` 来自 8× bank/service 资源，是带宽 Pareto；论文主轴应为 throughput/mm² 与 energy/completed-token |
| **Conv closure M467/M473/M498** | **PASS**：M467R4、M478 wrapper；M498 full+stale-RAW targeted exact VCS，0 assertion fail | **OPEN at cutoff**：M475 子切片 37,316.29 µm²、3 ns clean，但不含 matcher/CAM/macro；M477 42,370.65 µm² 且 max transition/cap/fanout 失败；M498 DC 仍在运行 | **OPEN** | **OPEN** | **OPEN**：144 B 1R1W scratch、resident psum 未物理化 | **OPEN** | **OPEN** | **CLEAR**：M473 fused `1.9436x` 是理想机会；unfused `~1.015x`；M472 `2.4595x` 是外部官方模拟器结果，均非本 RTL/system speedup |

### 2.1 矩阵的直接含义

- **验证最完整的是 C1 selected slice，而不是最新 Conv top。** M416/M420/M422/M448R4 的证据不能整体转移给 M498；顶层、状态和存储边界已经变化。
- **A1 是闭环较完整但 Amdahl 最弱的点。** 即使 attention 无限快，冻结 envelope 的上限也只有约 `1.0059x`；当前模型实际为 `1.000911x`。
- **C2/FC2 的创新应从“4.76x 加速”改写成共享状态的资源效率。** 同峰值 K8 与 K1x8 周期是 `1.000x`，这个负结果反而给出了公平表格应采用的横轴。
- **C3 的局部 2x 仍缺公平 fixed 边界。** M289 自己明确 `fixed_same_boundary_rtl=false`、`area_matched_fixed=false`，因此不能与已综合面积相绑定写成物理 2x。
- **Conv 的 capture gap 是可以写的科学结果。** 官方 product opportunity `2.4595x`、fused ideal `1.9436x`，但真实 unfused 仅约 `1.015x`，说明 parent read、completion bubble 和存储物理代价决定可捕获收益；这比继续堆 matcher 更可信。

## 3. 各线可直接写入与禁止写入的数字

### 3.1 可直接写入，但必须带范围标签

| 数字 | 合法表述 | 必须同句出现的限定 |
|---:|---|---|
| C1 `24,548.71 µm²`, 3 ns | selected execution slice 的 28 nm pre-macro DC 面积与时序 | 0 macro、ideal clock/pre-layout，不是全 Conv/chip |
| C1 `6.2538 mW`, `18.7614 pJ/cycle` | selected slice 的 SAIF-annotated prelayout std-cell power/energy | measured cycle、非 frame、无 SRAM/DRAM/CTS |
| C2 `4.764209x` | H67 FC2 局部 K8 对 canonical low-bandwidth K1 的 cycle ratio | FC2 share 6.6764%；不是 equal-peak、不是 system |
| FC2 `5.8634x` geomean / `6.2343x` aggregate | K1x8 对 K1 的 8-bank service Pareto | 8× bank/port/service 资源；不是稀疏算法同资源收益 |
| C3 `102,852.29 µm²`, 3 ns | protocol-repaired ATLIF top 的 logic-only DC | 不含 state macro；rank3 accuracy 和 fixed-area baseline open |
| A1 `1.186509x` / `1.176421x` | sample0 / S10 attention local cycle reduction | S10 是选定窗口，不是多 sequence |
| A1 `1.000911x` | 把 RQTB 放进 620M activity envelope 后的模型比 | model-only，不是 end-to-end measured system speedup |
| A1 `80.50→68.39 nJ/head-row` | pre-macro logic 的 SAIF energy | memory/clock-tree excluded |
| Conv `1.9436x` fused / `~1.015x` unfused | model opportunity 与可执行同步代价之间的 capture gap | 不是 M498 admission，不是系统倍速 |
| M472 `2.459487x` | 官方 Prosperity 周期框架内、同配置 product-vs-bit、H67 四层 support-tile 聚合 | external artifact、51.84M source rows、432 run calls、非本 RTL/非 monolithic Conv/非 system |

### 3.2 禁止混填或相乘

1. 不得把 `4.764x × 1.9436x × 1.1865x` 相乘；三者作用范围、基线与资源均不同。
2. 不得把 M472 `2.4595x` 放入“ours speedup”列；它只能位于 external iso-workload audit/opportunity 列。
3. 不得用 C1 `18.76 pJ/cycle` 或 A1 `68.39 nJ/head-row` 推导 mJ/frame；缺少各模块实际调用次数、SRAM、DRAM 与 overlap。
4. 不得把 C1、C3、A1、M475/M498 的面积直接相加当 chip area；它们不是同一综合 top，且共享/重复硬件未消歧。
5. 不得把 CACTI 32 nm proxy 的 SRAM 面积直接加到 TSMC 28 nm std-cell area 后称 post-layout PPA。
6. 不得把 source-work、product terms 或空 tile 比例称为周期；只有可执行调度器/周期模型产生的计数才可进入 latency/FPS。
7. 不得把 FireFly-T 的 FPGA GOP/s/W、DSP efficiency 与本项目 ASIC µm²/mW 做直接优劣排序。
8. 不得把 Zurich 10 个窗口写成 10 条 DSEC sequences。

## 4. 对照公开工作的论文表项

### 4.1 对标工作的证据习惯

| 工作 | 常见主表项 | 证据栈 | 对本项目最相关的要求 |
|---|---|---|---|
| [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v2) | 28 nm、500 MHz、0.529 mm²、390.10 GOP/s、299.80 GOP/J、737.17 GOP/s/mm²；跨网络 speedup/energy | 官方 cycle-accurate simulator；DC 28 nm；CACTI buffer；DRAMsim3 64 GB/s；统一 tile | 必须给完整 tile 的 buffer 容量、带宽、周期、面积、功耗与 end-to-end 工作负载，不只 matcher/source-work |
| [Phi, ISCA 2025](https://arxiv.org/html/2505.10909v1) | 28 nm、500 MHz、0.662 mm²、242.80 GOP/s、285.81 GOP/J、366.70 GOP/s/mm²；240 KB buffer 分解；speed/energy vs baseline | DC + CACTI 7 + DRAMsim3；共同 OP 定义；面积/功率模块分解；多模型准确率 | 本项目必须冻结 OP 定义和 240 KiB/带宽资源点；无损与 PAFT/有损分表，不得把不同 bank 数放同列 |
| [FireFly-T](https://arxiv.org/html/2505.12771v1) | network/dataset/accuracy/FPS/GOP/s/GOP/s/W/DSP efficiency/LUT/BRAM/URAM/DSP/frequency/device；资源分解 | FPGA implementation 与多个 SNN/数据集 | 可借用 FPS、利用率、资源分解和 network-accuracy 联表；不能跨工艺直接比较 ASIC area efficiency |
| [FEATHER, ISCA 2024](https://arxiv.org/abs/2405.13170) / [artifact](https://github.com/maeri-project/FEATHER) | 多网络 latency、energy efficiency、utilization、area/power/timing/PnR；FPGA end-to-end throughput | 开放 RTL；cycle/layout DSE；DC/PnR；两级 numeric/cycle verification | 最应借鉴“同一配置清单 + cycle + pJ compute + utilization + PPA + 多网络”的可复跑表格，而不是复制其机制 |
| 本地 CICC'26 光流芯片 | MVSEC 子集 AEE、ops/EMA、energy、latency；28 nm silicon voltage/frequency/area/power/throughput/energy efficiency | 真实流片；逐数据子集；LPDDR 3.7 pJ/bit 敏感性 | 可借表结构：逐 sequence 精度/周期/能量与总芯片表；不能把 post-silicon 数字与本项目 pre-macro 直接排名 |

### 4.2 本项目今天能填、还缺和禁止填的表格

| 论文表项 | 今天能否填 | 可填内容 | 缺失/禁用内容 |
|---|---|---|---|
| Process / target clock | **PARTIAL** | TSMC 28 nm HPC+ target；3.0 ns（333.3 MHz）模块级 | 不是一个完成 chip 的 achieved post-layout Fmax |
| Arithmetic / OP definition | **PARTIAL** | INT8/signed source、模块局部 cycle 定义可列 | 缺统一“1 OP”定义；不能把 bit accumulation、source issue、MAC 混为 GOP |
| On-chip memory | **OPEN** | 可列逻辑合同中的容量需求与端口 | 无 foundry macro 面积/延迟/能量；不能把寄存器或 CACTI proxy称实际 SRAM |
| Off-chip bandwidth/energy | **OPEN** | 可做 32/64 B/cycle、LPDDR 3.7 pJ/bit 敏感性列 | 无 DRAMsim3 address-timed schedule；不能声称实测 DRAM energy |
| Full-chip area / power | **OPEN** | 各 selected module 可各自列 pre-macro logic | 不得相加成 chip；没有 clock tree、macro、NoC/controller 完整面积/功耗 |
| Module latency/speedup | **YES with labels** | C1/C2/C3/A1/Conv opportunity 分表 | 不得放在同一 “system speedup”列 |
| Throughput/mm² | **OPEN for final design** | M496 后可给 FC2 三点 Pareto | 当前 K1/K8/K1x8 资源未综合匹配；无全 top mm² |
| Energy/op or energy/frame | **PARTIAL/OPEN** | C1 pJ/measured-cycle、A1 nJ/head-row | 无统一 OP、无 full frame、无 memory/DRAM |
| Accuracy/quality | **PARTIAL** | exact 路径不改 checkpoint/arithmetic 的身份声明 | 仍需 end-to-end RTL/cycle replay 与每 sequence AEE/Fl；rank3/有损不得借 ep35 |
| Dataset breadth | **OPEN** | Zurich 窗口统计 | 至少两条额外真实 DSEC sequence，报告 density/均值/最差值 |
| External comparison | **PARTIAL** | M472 官方 Prosperity iso-workload audit；公开论文原表背景 | 自设计网络无法直接用论文网络的 GOP/s 排名；需要三层对标方法 |

### 4.3 自设计网络的正确三层比较

由于网络由本项目设计，不能要求其他论文加速器已直接跑过相同 checkpoint。建议三层分开：

1. **Iso-workload artifact replay（最强可比层）**：把冻结 H67 trace 映射到 Prosperity/Phi 等公开周期模型，在各自原配置与 OP 定义下运行；结果明确标为 external simulator，不冒充本项目 RTL。M472 已是这一层的有效样例。
2. **Iso-resource analytical/model comparison**：统一 28 nm、频率、buffer 容量、DRAM 带宽、精度与算子边界，比较 latency、energy、area efficiency；所有缩放和 CACTI/DRAMsim3 假设单列。
3. **Published-point context only**：Prosperity/Phi/FireFly-T/CICC 原论文数字只说明领域量级，表中用不同底色或脚注标记不同网络/平台，不计算“ours is X× faster”。

主表必须优先回答本项目自身 baseline：同一 H67 checkpoint、同一 DSEC sequence、同一 buffer/带宽、相同 arithmetic 下，Fixed/bit-sparse/本项目各配置的完整周期、能量、AEE 与资源。这样即使网络不同，也仍是可审计的 architecture comparison。

## 5. 当前模拟评审

### 5.1 分项评分

| 维度 | 分数 / 5 | 依据 |
|---|---:|---|
| Novelty | **3.5** | analog ATLIF、event-flow workload 与 cross-operator shared sparse execution 有差异化；但 product/pattern reuse、banked sparse reduction 与公开工作重叠高 |
| Significance | **3.0** | FC/Conv/ATLIF 命中较大工作份额；目前可准入的系统收益尚未闭合 |
| Soundness | **4.2** | exact-SHA、fail-closed、攻击向量、污染反例、负结果封存很强；口径纪律优于一般 artifact |
| Evaluation | **2.7** | 有 DC/Formality/PTPX 局部闭环，但 macro、共同资源、full-network cycle/energy 和多 sequence 缺失 |
| Reproducibility | **4.1** | 合同、receipt、seal、官方 artifact replay 完整；部分新物理任务仍未完成 |
| Presentation/readiness | **1.8** | 证据分散，尚无一张共同分母的最终 paper table；贡献命名有重叠风险 |
| **综合** | **3.2** | Borderline Reject |

### 5.2 模拟审稿意见

**支持接收的理由**：机制不是简单做零跳过；对局部 RTL 的数值、协议、形式等价和功耗证据严谨；对无效方向有可复现反例；外部 Prosperity 重放提供了可信 opportunity anchor。

**反对接收的理由**：所有漂亮倍率都可被质疑为局部、不同资源或外部模拟器；没有一个最终共同硬件 top；SRAM/DRAM 是稀疏加速器的主成本却仍未物理化；只用 Zurich 窗口无法证明事件密度变化下收益稳定；多项面积/能量不能相加形成 chip-level 指标。

当前最可能的审稿人一句话是：**“机制和验证令人信服，但评测尚未证明这些局部模块在相同资源、真实存储和完整光流 workload 下形成有意义的端到端收益。”**

## 6. 提升到 Weak Accept 的硬门

以下门是合取关系；缺任何一项都很容易继续 Borderline Reject：

1. **最终 top 门**：M498 DC 五类约束 clean 且面积不超过冻结上限，或明确 kill；M496 K1/K8/K1x8 同 top/SDC/IO 三点 DC 完成。不得以 M475/M216 代替最终 top。
2. **等价门**：对论文真正引用的最终 Conv 与 FC2 top 各完成 RTL↔netlist Formality；旧 M420/A1 等价不能继承。
3. **能量门**：同一 activity 协议下做 matched SAIF/PTPX；加入 CACTI/目标 SRAM macro 与 DRAMsim3/address-timed traffic，至少报告 logic/SRAM/DRAM 三项分解。
4. **系统门**：构建一个不重叠的可执行全网 schedule，逐算子列 cycles、overlap、bytes、energy；以同资源 Fixed/bit-sparse baseline 得到**实测模型级系统收益**。建议保留门为 `>=1.10x`，达到 `>=1.15x` 更有把握；低于门则主卖能效/带宽而非 speedup。
5. **数据门**：至少 3 条 DSEC sequence，按 event density 分层，报告 mean、P10/P90 或 min/max；同时给 AEE/Fl 与原 checkpoint 对齐。
6. **口径门**：M472 只出现于 external artifact 列；所有局部倍率带 operator、baseline、resource 三标签；无损/有损分表。
7. **artifact 门**：一条命令可从冻结 trace 复现主周期表，另一条可校验所有 SHA/receipt；给最终论文数字到文件的索引。

这些门全部完成后，预计可到 **3.7--3.9/5，Weak Accept 45--65%**。这不是承诺；结果若显示系统收益小，应及时改为“resource/energy efficiency + capture-gap”叙事。

## 7. 进入 BP shortlist 的附加硬门

BP 不是把局部倍率做得更夸张，而是要求贡献、证据和影响同时没有明显短板。在 Weak Accept 门之上还需：

1. **一个不可替代的主机制**：在同资源端到端 workload 上达到约 `>=1.20x` 系统周期收益，或给出同等强度的 throughput/mm² 与 energy/frame Pareto；不能靠多个局部倍率相乘。
2. **物理可信度**：关键顶层至少 macro-aware post-synthesis + PT/SAIF/PTPX，最好有 P&R/SPEF；buffer、clock、controller、NoC/adapter 都计入。
3. **完整能量与质量 Pareto**：逐 sequence mJ/frame、FPS、AEE/Fl；logic/SRAM/DRAM 分解；任何 PAFT/有损点独立 checkpoint 身份和误差预算。
4. **外部同 workload 基线**：至少一个公开 simulator/RTL 在相同 trace、容量和带宽下复现；M472 是起点，但还需把本项目最终 top 放到同一横轴。
5. **泛化与最坏点**：低/中/高 event density、多个场景与最差序列均不崩；解释收益来自何种光流稀疏性，而不是 Zurich 的偶然窗口。
6. **可复跑 artifact**：公开或可匿名复现的 trace→cycle→energy→table 流水线，审稿人可在合理时间内重建主要图表。

即使全部完成，BP shortlist 仍取决于投稿池和叙事，估计也只有 **15--25%**；当前状态不得称 BP ready。

## 8. 72 小时收口顺序

| 优先级 | 动作 | 通过门 | 失败处置 |
|---:|---|---|---|
| P0 | 等 M498 DC 自然结束并独立 hammer | 五类约束 clean；面积 `<=44,779.2 µm²`；结果身份一致 | 永久关闭 dual-slot 物理 claim，保留 C1 selected slice/capture-gap，不开新 matcher |
| P0 | 运行 M496 matched K1/K8/K1x8 DC | 同 top/SDC/pins；给 area/FF/Fmax/throughput/mm² | 若 K8 不优于 K1x8 资源效率，只保留 C2 共享上下文/能量假设，禁用 speedup headline |
| P0 | 构建 paper metric registry | 每个数字有 numerator/denominator、scope、resource、source、admission 标签 | 无标签数字不进表 |
| P1 | 最终 Conv/FC2 Formality + matched SAIF/PTPX | 0 fail；相同 activity window；energy/token | 只写功能/面积，不写能效 |
| P1 | SRAM/DRAM 闭环 | 目标 macro 或保守 CACTI；address-timed DRAMsim3；容量/带宽 sweep | 只能称 pre-macro logic study |
| P1 | 3+ DSEC sequences 与统一 schedule | 每序列 cycles/bytes/mJ/AEE；全网表无重复计数 | 降为单序列 case study，难以争 BP |

在这些项完成前，继续开发新的 generic sparse RTL 会增加 prior-art 暴露与验证债务，不能提高 DATE 分数。

## 9. 证据索引

核心本地证据：

- C1 exact VCS：`results/m414_q32_balanced16_vcs_r1_20260826/m414_q32_balanced16_zero_stop_vcs_receipt_r1.json`
- C1 DC/Formality/PT/Power：`dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/`、`m420_m414_dual_formality_r1_20260826/`、`m422_m416_selected_slice_prelayout_ptsta_r1_20260826/`、`m448r4_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r4_20260826/`
- C2 recovered DC：`results/m216_fc2_scope_matched_k1_k8_logic_only_dc_recovery_r2_20260825/m216_fc2_scope_matched_k1_k8_logic_only_dc_recovery_r2.json`
- C3 exact VCS/DC：`results/m273_integrated_rank3_atlif_author_r1_20260825/m273_author_result_r1.json`、`dc_handoff/runs/m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825/`
- A1 full-network model：`dc_handoff/runs/h67_full_network_ledger_v2_multisample_vcs_20260821/REPORT.md`
- A1 system tables：`dc_handoff/runs/date_system_tables/DATE_TABLES.md`、`hierarchy_energy.md`
- FC2 exact equal-bandwidth：`results/m492_fc2_cutthrough_8bank_equal_bandwidth_vcs_r1_exact_20260827/`
- FC2 canonical Pareto：`results/m497_fc2_canonical_k1_vs_k1x8_vcs_r1_exact_20260827/`
- M496 contract：`contracts/m496_fc2_three_axis_matched_logic_only_dc_contract_r1_20260827.json`
- Conv VCS/DC：`results/m467r4_row_shared_live_scoreboard_vcs_r3_20260826/`、`results/m478_m476r2_full_wrapper_regression_vcs_r1_20260826/`、`results/m498_segmented_enable_vcs_r1_exact_20260827/`、`dc_handoff/runs/m475_m474_fused_parent_dual_update_dc_3p000ns_r1_20260826/`
- M477 NO-GO：`dc_handoff/runs/m477_m476r2_backpressure_safe_parent_queue_dc_3p000ns_r1_20260826/m477_dc_failure_receipt_r1.json`
- External M472：`results/m472_official_prosperity_admission_r1_20260826/m472_official_prosperity_admission_r1.json`
- 收口路线：`docs/500_新RTL封门与DATE硬件收口路线_20260827.md`
- 新 RTL 独立审计：`reviews/date_open_rtl_gap_mining_r1_20260827/date_open_rtl_gap_mining_r1_20260827.md`

## 10. 身份与截点

- `docs/359_DATE终局冻结_20260813.md` 审计前 SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- M498 DC 在审计截点仍运行，因此矩阵标为 OPEN；任何中间面积/时序均未作为最终值。
- M496 在审计截点没有最终 DC 回执。
- 本报告只评估硬件证据完整度，不把 source-work opportunity、分析 Amdahl 或外部 simulator 结果升级为本项目系统性能。
