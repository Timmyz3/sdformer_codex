# DATE / Best Paper 硬件独立评审 r5

日期：2026-08-27（Asia/Shanghai）  
证据截点：2026-08-27 11:05  
对象：Motion/H67 ep35，重点审阅 `docs/508/510/512`、M472--M514、
M495/M496 与 C1/C2/C3/A1。  
方法：只读、receipt-blind、fail-closed；没有启动 GPU、VCS、DC、DSE，
没有修改 `docs/359`。

## 技术结论：可以开始写 DATE，但还不是稳收，更不是 BP-ready

当前最准确的裁决是：
`CONDITIONAL_DATE_SUBMISSION__BORDERLINE_ACCEPTANCE__NO_BP`。

| 口径 | 评分 | 裁决 |
|---|---:|---|
| 可投稿 readiness | **3.5/5** | 现有硬件和证据足以写一篇诚实的 DATE 稿，但最终系统表仍有空项 |
| 可录用 readiness | **3.2/5** | Borderline Reject，当前约 **25--35%** |
| Best Paper readiness | **1.8/5** | 未就绪，当前 shortlist 约 **0--5%** |

这三个评分不矛盾。“可投稿”表示已经有可写的实现和验证对象；“可录用”要求
审稿人相信它们在共同资源、真实存储和完整 workload 下仍有意义；“BP”还要求
一个不可替代的端到端主结果。当前强项是 soundness 和 reproducibility，短板是
evaluation：旧系统分母漏 decoder、SRAM/DRAM 未共同收费、只有一条 Zurich
sequence、FC2 matched PPA 正在跑且没有最终回执。

按维度评估：Novelty 3.4、Significance 3.0、Soundness 4.3、Evaluation 2.5、
Reproducibility 4.2、paper convergence 2.0。M510 主动发现并封住自己的分母遗漏，
提高了 soundness，但在 exact decoder 重建完成前会暂时降低 evaluation。

## 当前证据能支持什么

| 对象 | 当前最强证据 | 审稿人可以接受的句子 | 不能写的句子 |
|---|---|---|---|
| C1 selected Conv slice | 17,280 configs、51.84M source rows、0 mismatch；24,548.71 µm²；3 ns PT setup/hold +0.7598/+0.0178 ns；5,368 Formality compare points 全过；6.2538 mW、18.7614 pJ/measured-cycle | 一个 exact、balanced、prelayout 的 sparse-Conv selected execution slice 已完成 VCS/DC/Formality/PT/PTPX 闭环 | full Conv、chip、SRAM-inclusive、mJ/frame 或 system speedup |
| C2 FC2 K8 | M216 的 120 records、143.895M events；K8/K1 局部 4.764209x；K8/K1x8 同峰值周期 1.000x | shared-state K8 相对低带宽 K1 的 standalone frontend 周期收益；同峰值下应比较面积和能量 | 4.764x equal-resource 或 full-network speedup |
| C3 ATLIF | 225 beats、7 次协议攻击、0 mismatch；logic-only 102,852.29 µm²、3 ns | rank-0 局部 state/update 机制与 protocol-repaired logic-only top | 把分析性约 2x 与该面积绑定成物理倍速；rank-3 精度或系统倍速 |
| A1 RQTB | 局部 1.1865x；pre-macro energy 80.50→68.39 nJ/head-row；模型 included-scope 1.000911x | lossless attention quotienting 的局部周期/能量消融 | headline 系统加速；attention 只占约 0.589% |
| M472 | 官方 Prosperity product-vs-bit 为 2.459487x，四 Conv、51.84M source-row、432 个 K=16 support-tile 聚合 | 外部 iso-workload opportunity 与 capture-gap 动机 | “our accelerator achieves 2.46x”、同资源 monolithic Conv 或 system speedup |

最大的 P0 是共同分母。旧 `620,302,905 cycle/frame` 漏了四层
`ConvTranspose2d`，从现在起只准叫 included-scope envelope。M510 的无 trace
分析界说明 decoder 约占修正 envelope 的 21.57--22.83%，但该界来自 S100
aggregate activity；不能和正在抓取的 S10 decoder cohort 拼成 headline 数字。

## 最终只保留两到三个硬件贡献

### 1. 必留：C1 exact balanced sparse-Conv selected slice

这是目前唯一同时有大规模 trace、DC、Formality、PrimeTime 和 PTPX 的实现锚点。
论文应把它写成“exact signed-source / product-pattern execution slice”，并以
capture gap 解释设计取舍：官方 Prosperity 在相同 H67 Conv source rows 上有
2.459x product-vs-bit opportunity，而真实 SRAM/完成依赖使很多理想收益无法捕获。

它的贡献不是宣称 2.46x，而是给出一个经过物理和能量验证的可执行点，并说明
为什么事件光流上的理论稀疏性不会免费转化为周期。

### 2. M496 过门才主留：C2 FC2 shared-state eight-bank coissue

如果 M496 三点通过，C2 应成为论文第一硬件 headline。正确表述不是 4.764x，
而是：在 K8 与 K1x8 同八 bank 峰值、吞吐基本相同的前提下，K8 共享 Acc24
状态，显著降低面积、时序寄存器和 energy/completed-token。

这一贡献与 Prosperity/Phi 的 product/pattern reuse 有清晰距离：卖点是
context/state 共享和 bank coissue 的资源效率，不是重新定义一类稀疏。

### 3. 有条件保留：C3 exact rank-0 ATLIF state/update engine

C3 只有在补齐同接口 Fixed baseline、state macro 和 matched SAIF/PTPX 后才配当
第三硬件贡献。当前 102,852.29 µm² 是 0-macro logic-only top，且 setup/hold
报告零裕量；rank-3 也未被 ep35 训练身份准入。若这些门在截稿前关不掉，C3
应降为 unified fabric 的 neuron adapter，不占 contribution bullet。

A1 与 M514 都不应成为独立 headline。A1 适合作为 lossless attention
完整性和局部能量消融；M514 只负责证明统一 source protocol 覆盖 decoder。

因此推荐的最终组合是：

- **M496 通过：C2 + C1 为两条主贡献；C3 通过公平能量门才作为第三条。**
- **M496 物理门失败：C1 为唯一硬锚；C3 必须闭合才能成为第二条。** A1/M514
  只作完整性。若 C3 也不闭合，硬件贡献数量虽多但主证据只有一个，DATE 录用率
  会明显低于当前 25--35% 区间。

## 必须完成的实验表与精确门槛

### 表 A：同 top 的 FC2 三点物理与能量 Pareto

三行固定为 K1、shared K8、replicated K1x8；列必须包括 cycles/completed-token、
area、sequential cells、Fmax/五类约束、throughput/mm²、logic/SRAM energy/token。

M496 的准入门是合取关系：

- 三点全部完成，端口数相同，3.0 ns setup/hold/max-cap/max-transition/max-fanout
  全 clean；任何 partial 子结果不得引用。
- K8/K1 area `<=1.25x`，冻结局部周期 `>=3.0x`，throughput/area `>=2.4x`。
- K8/K1x8 throughput `>=0.98x`；area 和 sequential cells 均 `<=0.50x`；
  throughput/area `>=2.0x`。
- 三点 Formality 的 fail/abort/unmatched 全为 0。
- matched SAIF annotation `>=95%`，K8 energy/completed-token
  `<=0.70x K1x8`。

### 表 B：不重叠的完整 H67 系统表

必须以同一个 S10 cohort 重新覆盖每个实际执行的 Linear、Conv2d、
ConvTranspose2d、ATLIF 和 attention，一次且仅一次。门槛为：

- M511 bitpack `40/40` 校验通过，四层 decoder `4/4`，record/product/transition
  conservation mismatch 为 0；
- 不得把 S10 decoder 与 S100 included-scope 字段拼接；
- 表中 Fixed、bit-sparse、ours 使用同一 28 nm、3 ns、算术、lane、SRAM 容量、
  端口和 DRAM 带宽；
- cycles、SRAM bytes、DRAM bytes、logic/SRAM/DRAM energy 均按算子列出并可求和，
  overlap 单列，不能双计；
- DATE 主 speedup 门为三条真实 sequence 几何均值 `>=1.10x`、最差序列
  `>=1.00x`；达到 `>=1.15x` 才较有把握；BP 门为几何均值 `>=1.20x`、
  最差序列 `>=1.10x`。

### 表 C：macro-aware PPA 与能量分解

表中至少有 logic、SRAM、DRAM 三项，不允许将 0-macro logic DC 写成 chip PPA。

- 每个 headline memory 必须有目标 macro，或明确标成 CACTI sensitivity；
- DRAM 必须来自 address-timed transaction，而不是只用 source-work 乘 pJ/bit；
- DATE 门：相对 strongest iso-resource bit-sparse baseline，energy/frame 至少下降
  **20%**；
- BP 门：energy/frame 至少下降 **33%**，且至少一个最终 top 完成 macro-aware
  PT/SAIF/PTPX 与 P&R/SPEF；
- 若系统周期达不到 1.20x，BP 的替代 headline 必须是完整 top 的
  throughput/mm² `>=2.0x` 且 energy/frame `<=0.70x`，不能拿 FC2 slice 代替。

### 表 D：sequence、质量与公开基线

- 至少 **3 条真实 DSEC sequence**，覆盖低/中/高 event density；报告 mean 与
  worst，或 P10/P90；十个 Zurich 窗口不能写成十条 sequence。
- exact 模式 output mismatch、ΔAEE、ΔFl 均为 0；任何有损点必须另 checkpoint、
  另表、另误差预算。
- DATE 至少保留 M472 一个官方 external iso-workload anchor；BP 至少需要两个
  公开基线，建议 Prosperity 与 Phi-like，并统一 trace、capacity、bandwidth、
  arithmetic。
- 主表必须定义“1 OP”，不能混用 source issue、bit accumulation 和 MAC 计算 GOP/s。

## M514 只能定位为 C2-D 完整性适配器

M514 合法的论文句只有：它消费 binary ATLIF source descriptor，按
K3/S2/P1/output-padding1 精确生成合法 destination tap，不物化插零张量。

它不能叫新的 EPD scheduler，也不能声称相对 strong bit-sparse polyphase A1 有
周期收益。M512 已证明四层 Cout 都是 96 的整数倍，A1 每个合法 tap 恰好填满
96 lanes，product-issue 轴上的 EPD/A1 上限是 1.0x。4.48--4.81x 是 dense 到
activation-bit-sparse 的标准机会，不是 M514 的 RTL 倍率。

证据身份还需特别谨慎：M514 r2 评审审的是 RTL/TB SHA
`7543a25c.../10392f18...`，当前磁盘已变为
`90c44fc.../6c283bf9...`。当前代码看起来已加入 widened extent assertion 与
size-32 测试，但在 r3 独立 seal 和 exact-SHA Synopsys VCS 前，状态仍是
`UNREVIEWED_CURRENT_SHA__NO_VCS`。

因此只允许一次最小收口：r3 静态打铁 + exact-SHA VCS 的边界、stall、fault、
same-edge replacement 与 upper-bound 测试。之后把它并入统一 decoder coverage
段落，不单独做 DC/PPA headline。

## 哪些新 RTL 会稀释论文

以下方向继续写 RTL 会降低而不是提高 r5 分数：

- EPD phase-balanced scheduler：M512 已从理论上 kill；
- PGPR 作为 cycle speedup：强 1R1W output-stationary A1 的周期上界为 1.0x，
  最多保留 energy/dataflow 消融；
- 第四种 Conv queue、matcher、CAM、parent slot 或 M479/M498 结构变体；
- FC1 再扩宽、G-series 有损 skip、epsilon-RQTB、APEC-G2、Local5/H81；
- 在 exact trace 之前写 dynamic-BN 通用 packer；它即使通过也只是 memory
  support，门槛仍为 traffic `>=1.35x`、local schedule `>=1.20x`、
  control `<=15,000 µm²`、净 raw-path energy 至少下降 20%。

唯一仍可能获得新 RTL 名额的是 TDR，但必须先等 M513 生产结果。写 RTL 前至少
要求：ideal speedup `>=1.30x`、`P_delta/P_A1 < 0.7692`、numeric miter 0 mismatch、
收取 previous-input/output state 后总周期仍 `>=1.20x`、净能量至少下降 20%。
任一项失败立即关闭；它不能与正在进行的 M496 收口竞争资源。

## M496 的两种投稿策略

### M496 全过门

立即冻结 C2，不再开发新结构。顺序只能是三点 Formality、matched SAIF/PTPX、
公共 SRAM macro 收费，然后进入统一系统表。摘要主句写成：K8 在 equal-peak
K1x8 吞吐下，以不超过一半面积/状态和不超过 0.70x energy/token 完成 FC2；
`4.764209x` 只放低带宽 endpoint 的 Pareto 边。

最终贡献用 C2+C1；C3 只有闭合公平能量门才加第三条。完成系统、多序列和宏门后，
DATE 预计可升到 3.7--3.9/5、Weak Accept 45--65%。要冲 BP，还必须出现
`>=1.20x` 整网周期或同等强度的完整 top 面积/能量 Pareto。

### M496 完成但设计门不过

把 C2 降为 M216 standalone frontend ablation，不做 K8/K1x8 重构，不用 recovered
子运行或 partial 点救结论。论文以 C1 为物理锚，C3 必须补同边界 Fixed、state
macro 和能量才能成为第二硬件贡献；A1/M514 只作完整性。若统一系统能量表也不强，
建议把论文定位为 exact sparse-execution/capture-gap，而不是性能领先。

如果 M496 只是 host resource failure，则它不是设计失败，但也不是结果。只允许在
隔离主机上以相同 RTL/TCL/SDC/library/effort/order 新建一次独立审阅的 exact replay；
不允许降低资源门、换 effort 或引用三点中的一部分。若时间不够，宁可不把 C2
matched PPA 放 headline。

## 最终模拟审稿意见

**支持接收：** C1 的 exact-SHA VCS、Formality、PT、PTPX 链很扎实；项目对污染、
协议攻击、资源失败和理论 NO-GO 的封存纪律优于一般 artifact；M472 给出了可由外人
复现的 workload opportunity anchor；M510 主动纠正 full-network 口径显示 soundness。

**反对接收：** 最漂亮的数字仍分别属于低带宽局部 baseline、外部官方 simulator、
分析机会或小份额 attention；最终 SRAM/DRAM 和多序列没有闭合；当前还没有一张
Prosperity/Phi 风格的“同资源、完整 workload、cycles+area+energy+quality”主表。

审稿人当前最可能写：

> The implementation discipline is unusually strong, but the paper still lacks
> a common, memory-inclusive, end-to-end denominator showing that the proposed
> modules deliver meaningful gains on diverse optical-flow sequences.

所以 r5 的最终裁决不是“继续堆 RTL 直到看起来像 BP”，而是：**只完成 M496、
decoder exact denominator、macro/energy、多 sequence 和统一表；M514 做一次最小
功能闭合。** 这些门未过之前，任何新模块都会稀释贡献并推迟真正能提高录用分数的
证据。

## 方法与局限

本评审只使用磁盘上截至截点的封存文件和只读进程/文件状态；没有将正在运行的
M496 日志当作结果，也没有读取未来 M511/M513 payload。外部论文评价沿用已封存的
本项目 source inventory 和官方 artifact 审计，没有重新运行公开模拟器。

定量段落采用表格而非图：本轮任务是 admission gate 与口径审计，精确阈值和条件
分支比趋势图更适合核查。机器可读的同内容裁决见同目录 JSON。
