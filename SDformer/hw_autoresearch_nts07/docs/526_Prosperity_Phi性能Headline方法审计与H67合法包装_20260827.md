# M526｜Prosperity / Phi 性能 headline 方法审计与 H67 合法包装

日期：2026-08-27  
状态：`METHOD_AUDITED__H67_EXECUTION_CONTRACT_OPEN`  
边界：本文解释公开论文的大倍率如何形成，并规定 H67 的合法复现方法；文中的外部论文数字不是本项目性能结果。

## 1. 先给结论

Prosperity 的 `7.4x` 是 16-workload、论文所支持算子范围内的 PTB/Prosperity 平均 runtime headline；它不能无限定地称为每个 Transformer 所有算子、同面积意义上的严格全网 `7.4x`，也不是“product sparsity 这一个模块相对同资源最强 bit-sparse 基线的 7.4x”。公开论文与官方 artifact 显示，该数字包含三类收益：较弱 PTB 基线到 unstructured bit sparsity 的架构收益、product sparsity 的额外复用收益，以及低开销 dispatch 对机会捕获率的改善。其消融约为 `2.28 x 2.16 x 1.49 = 7.34x`；这是叙事一致性检查，不是平均值之间的数学恒等式。Prosperity 同一架构中 product 相对 bit execution 的平均运行时间收益约为 `3.2x`，product density 的理论机会约为 `5.0x`，二者也不能混为一谈。

官方 16-workload `time_reference.xlsx` 的独立重算结果为：PTB/Prosperity 比值的算术平均 `7.461107x`、几何平均 `7.313885x`、总 runtime 之比 `6.731836x`、范围 `4.975779--11.466772x`。`7.4x` 数值上更靠近算术平均，但仅凭接近程度不能判定论文实际采用的聚合公式。可复跑 r2 证据位于 `results/m526_prosperity_headline_method_audit_r2_20260827/`；它只审计外部 artifact 的聚合方法，不是 H67 性能。

Phi 的 `3.45x` 是相对所选 SOTA accelerator（Stellar）的系统级模拟结果，不是 L2 residual/PAFT 单一机制相对同资源强 baseline 的倍率。其 theoretical speedup 相对 bit sparsity 约为 `3.2--6.1x`，而 PAFT 在 Phi 无损主结构之上的实测平均增量约 `1.26x`。因此，大 headline 的常见合法来源是“完整架构相对一个定义明确的旧系统”，不是每个新增小机制都能独立提供相同倍率。

H67 应仿照的是这套**实验结构**，不能把外部 `2.46x/7.4x/3.45x` 改名为 ours：

1. 用统一 simulator 建立从 Fixed/structured baseline 到 strongest exact baseline 再到 ours 的 baseline ladder；
2. 用完整 workload 的 latency/frame、energy/frame 和 effective throughput 形成 headline；
3. 用同架构 waterfall 解释 headline 从哪里来；
4. 同时保留 replicated K1x8 等服务强 baseline，并将 typed-K8 candidate 单独成行，防止审稿人指出只挑弱基线；
5. 理论机会、局部 RTL、外部模拟器、全系统结果分表，不相乘。

## 2. Prosperity 的 7.4x 究竟是什么

来源：[Prosperity 论文](https://arxiv.org/html/2503.03379)与[官方 artifact](https://github.com/dubcyfor3/Prosperity)。

### 2.1 比较对象与范围

- 论文在 28 nm、500 MHz 下建模 128 PE，片上 buffer 为 8 KiB spike、32 KiB weight、96 KiB output，外存带宽为 DDR4 64 GB/s。
- 核心逻辑来自 RTL/DC，buffer 用 CACTI，DRAM 用 DRAMsim3，最终性能来自 cycle simulator；它不是全芯片流片测量。
- PTB、SATO、MINT 等 baseline 在作者统一框架中建模。对 transformer workload，它们只运行所支持的 linear layers，attention/LN 不进入这组 baseline 对比；所以 `7.4x` 是 supported-operator-scope suite average，而不是所有 Transformer 算子均同框架执行的严格全网倍率。
- A100 是另一类系统对照；论文明确将约 `1.79x vs A100` 描述为 spiking-transformer end-to-end。它不能与 `7.4x vs PTB` 视为同一 denominator。

### 2.2 大倍率的构成

Prosperity 的论文消融可以解释为：

| 台阶 | 公开口径 | 作用 |
|---|---:|---|
| PTB -> unstructured bit sparsity | `2.28x` | baseline 架构/稀疏粒度升级 |
| bit -> product sparsity（有调度开销） | `2.16x` | 乘积关系带来的新增复用 |
| 高开销 -> overhead-free dispatch | `1.49x` | 把理论机会转成可执行周期 |
| 三项乘积 | `7.34x` | 接近摘要 `7.4x` |

因此论文的关键写法不是造假，而是把一个完整 architecture stack 相对旧 baseline 的增量作为 headline，再用消融说明组成。对 H67 的启示是：C1/C2/C3 必须在同一周期模型中按顺序开启，得到直接测得的最终 latency；不得把三个独立局部倍率直接相乘。

### 2.3 有效吞吐的口径

Prosperity 的 artifact 中 `num_ops` 会随 bit/product architecture 的 activation population 改变，因此不能直接拿来构造跨配置固定分子的 H67 effective GOP/s。architecture-reduced 计数只能叫 executed additions，不是物理阵列峰值，也不是跨候选固定的 workload throughput 分子。

H67 后续必须并列给出：

- `dense-equivalent effective GOP/s`：所有配置都以同一冻结 Fixed 稠密工作为分子；
- `useful-nonzero GOP/s`：所有配置都以同一原始 trace 的非零/有效 accumulation 为分子；
- `physical issue rate`：实际每周期发射/退休的 source 数；
- latency/frame 与 energy/frame：避免只靠可变分子的 GOP/s 包装。

## 3. Phi 的 3.45x 如何形成

来源：[Phi 论文](https://arxiv.org/html/2505.10909)。

- headline `3.45x` speedup、`4.93x` energy efficiency 是相对所选 SOTA Stellar 的完整架构比较。
- 配置为 28 nm、500 MHz、`m256/k16/n32`，L1/L2 各 8 channel x 32 SIMD，总片上 buffer 240 KiB，DDR4 64 GB/s。
- 性能来自 activation/profile 驱动的 simulator；preprocessor、L1/L2、neuron 等组件由 RTL/DC 建模，buffer/DRAM 分别由 CACTI/DRAMsim3建模。
- 理论 pattern speedup 相对 bit sparsity 可达约 `3.2--6.1x`，但它是理论工作减少，不等于含 metadata、PWP、prefetch、bank conflict 和 memory stall 后的系统倍率。
- PAFT 只在 Phi 主结构上再给约 `1.26x` runtime 与 `1.1x` energy 增益，且允许小幅精度变化；所以 PAFT 应是分列的 lossy Pareto 点，不应反向装饰 exact 主结果。
- 论文报告 PWP 预取后仍存在明显的外存代价，也说明“算术工作量大幅下降”与“系统能量同幅下降”不是同一件事。

对 H67 的直接借鉴是：将 C1 parent/PWP capture、C2 typed signed-source service、C3 ATLIF phase/rank 放入一套 240 KiB/64 GB/s 的周期和能量模型；理论 opportunity 与 captured speedup 同图展示，形成 capture-gap 分析。

## 4. FireFly-T 给 C2 的正确借鉴

来源：[FireFly-T 论文](https://arxiv.org/html/2505.12771)。

FireFly-T 的强结果主要体现为相对 FPGA baseline 的 energy efficiency 与 DSP efficiency；其 load-balance 微基准在可比资源和固定总 memory bandwidth 下报告约 `3.48x`。这说明对 C2，等带宽 K8 对 K1x8 的 cycle 只高 `~1.01--1.04x` 并不等于机制无价值；可投稿对象应是：在相同吞吐服务能力下，signed analog source 的共享 scoreboard/partial state/atomic completion 是否显著降低面积、顺序状态和动态功耗。

因此 C2 的合法 headline 候选不是“4.76x 稀疏加速”，而是三联指标：

1. 相对单 K1 低带宽 baseline 的 throughput scaling（标注 denominator）；
2. 相对 K1x8 等带宽 baseline 的 throughput/mm²；
3. 相对 K1x8 的 energy/source 或 energy/token。

tag-elision 的 `27.5346%` metadata bit movement 减少不是差结果；它只是目前还没有证明 cycle speedup。若 matched PTPX 证明动态功耗下降 `>=20%` 或 matched DC 证明局部面积下降 `>=15%`，它应升为 C2 的实现子贡献。

## 5. H67 baseline ladder：合法得到较强 headline 的方法

同一 H67 checkpoint、同一 ordered trace、同一精度、相同 96-lane 预算、240 KiB SRAM、64 GB/s DRAM 下，必须至少运行以下五档：

| ID | baseline / candidate | 用途 | 允许的 headline |
|---|---|---|---|
| B0 | Dense96 Fixed-T10 | dense-equivalent 系统基线 | 允许做主 headline，但必须同时给 B3 |
| B1 | PTB-like structured time-group skip | 对齐 Prosperity 的旧结构化稀疏 denominator | 允许做主 headline，机制定义必须公开 |
| B2 | exact bit-sparse K1 | 低面积、低带宽 exact baseline | 只作 scaling 对照 |
| B3 | exact bit-sparse K1x8 | replicated equal-service strongest exact baseline | 必须进主表和主消融 |
| C2 | exact typed K8 | shared-state signed-source candidate | 与 B3 分开，报告增量和物理成本 |
| Ours | C1 + typed-K8 C2 + C3 exact | 本文完整配置 | 只能报告 simulator 直接测得结果 |
| Ours-PAFT | exact + admitted PAFT | 有损可选档 | 与 exact 分列，绑定 checkpoint/精度 |

PTB-like baseline 的操作定义不能只写名字。建议定义为：一个时间组中若存在任一活动 bit/source，则该组固定扫描所有 lane；若整个时间组为空才跳过。它是现实存在的 structured sparsity 设计点，且能说明 typed K8 fine-grained service 的收益。B0/B1 与 ours 锁定相同 lane/SRAM/BW 只能称 `iso-lane`，不能自动称 `iso-area`；主表必须同时补 area-normalized throughput，并保留 B3 的 iso-service 对照。

### 5.1 允许的“论文技巧”

- 摘要 headline 可以采用 Ours/B0 或 Ours/B1，只要明确 baseline；
- 正文同一页必须给 Ours/B3，证明不是隐藏强基线；
- 多序列汇总同时报告 arithmetic mean、geomean、ratio-of-summed-runtimes、min/max；摘要默认 geomean，若用 arithmetic mean需直说；
- 用 waterfall 报告逐级开启 C1/C2/C3 后的**直接重跑值**；
- 用 opportunity/captured 两条曲线展示理论稀疏与有限 SRAM/DRAM 后的实现收益；
- effective GOP/s、FPS、energy/frame、area efficiency 并列；
- cross-paper 表按原论文网络与原论文口径呈现，不伪造“同网络直接加速”。

### 5.2 禁止的包装

- 把 Prosperity 官方 artifact 在 H67 Conv 上的 `2.459487x product-vs-bit` 写成 ours；
- 把 C1/C2/C3 独立倍率相乘；
- 把 theoretical source-work reduction 写成 cycle speedup；
- 只报 Ours/K1 的 `4.76x` 而隐藏等带宽 K1x8；
- 把 kernel-only 结果写成完整光流网络 FPS；
- 用 PAFT/lossy checkpoint 的精度和 exact checkpoint 的周期交叉配对；
- 用 selected-slice mW 冒充 energy/frame。

## 6. 论文性能表的建议结构

### 表 A：同一 H67 workload 的系统 headline

| Configuration | Exact/lossy | Fairness | Latency (cycle/frame) | Speedup vs B0 | Speedup vs B1 | Speedup vs B3 | FPS | Energy/frame | Area | SRAM | Fixed-numerator Eff. GOP/s | GOP/s/mm² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

当前任何未闭合 decoder/memory 的分析值都不得填入 admitted 行。现有 corrected envelope 中 Fixed 约 `1.442B`、candidate 约 `790.9--803.8M`，对应分析敏感性 `1.794--1.823x`；这是待 exact decoder/memory 闭合的目标区间，不是当前可投稿实测值。相对 B1 有机会超过 `2x`，但必须由统一 simulator 实跑，不预写。

### 表 B：同架构消融 waterfall

| Step | C1 | C2 | C3 | Cycles | Incremental gain | Cumulative gain | DRAM bytes | SRAM energy | Logic energy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|

增量必须按 `Step_i / Step_{i+1}` 直接计算；cumulative 只用 B0 与最终行之比。该表是 H67 对 Prosperity `2.28 x 2.16 x 1.49` 消融写法的合规对应物。

### 表 C：跨论文规格与原报指标

| Work | Tech/freq | Workload | Precision | PE/lane | Area | SRAM | DRAM BW | Throughput definition | Reported speedup/efficiency | Evidence type |
|---|---|---|---|---:|---:|---:|---:|---|---|---|

Prosperity/Phi/FireFly-T 与 H67 网络不同，不能直接用 FPS 或 raw GOP/s 排名。跨论文表负责说明量级和完整度；真正的 apples-to-apples 对比来自同 H67 trace 下的 B0--B3、官方 Prosperity adapter 和 Phi-like adapter。

## 7. 立即执行合同

P0：

1. 闭合 M518 Fixed-T10 RTL 与 M519 K1/K8/K1x8 matched physical baseline；
2. exact decoder trace 到齐后，运行 B0--B3 与 Ours 的统一周期模型；
3. 每个配置输出相同 schema：cycles、issued source、retired destination、SRAM/DRAM bytes、stall breakdown、logic/SRAM/DRAM energy；
4. 按 sequence 与 event-density bin 输出 arithmetic/geomean/ratio-of-summed-runtimes/min/max；
5. 生成 Table A/B 的机器可读 CSV/JSON，表格不得手填数字。

P1：

1. 给 C2 做 tag-elision matched A/B DC+SAIF+PTPX；
2. 在相同 H67 trace 上加入官方 Prosperity 与 Phi-like adapter，标签为 external-method mapping；
3. 给 C1 画 theoretical opportunity、unbounded fused、240 KiB captured 三点 capture-gap 图；
4. 将 CICC 的 DRAM pJ/bit 只作敏感性列，主能量仍用统一 CACTI/DRAMsim3 条件。

准入门：完整 Ours 若相对 B0/B1 获得可重复 `>=2x`，可作为摘要主倍率；若只得到约 `1.8x`，仍可凭 C2 的 throughput/mm²/energy 与 C1 capture-gap、C3 exact ATLIF 形成 DATE Accept 论文，但不得包装成 `2x+`。

## 8. 证据身份

- 官方 Prosperity 仓库 commit：`6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`
- 官方 workbook SHA256：`47a05d06a0e762b9a67490875803441eac2bcec9a24a14576896f945452ba563`
- M526 r2 状态：`PASS_OFFICIAL_ARTIFACT_AGGREGATION_AUDIT__NOT_H67_PERFORMANCE`
- PTB/Prosperity：arithmetic `7.461107x`；geomean `7.313885x`；ratio-of-sums `6.731836x`
- 本文不得修改或重新解释 `docs/359_DATE终局冻结_20260813.md` 的封存数字。
