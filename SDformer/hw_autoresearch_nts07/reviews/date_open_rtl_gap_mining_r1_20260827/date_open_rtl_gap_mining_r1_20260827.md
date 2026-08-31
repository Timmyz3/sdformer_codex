# DATE 新 RTL 缺口与顶会开源机制独立审计 r1

日期：2026-08-27  
范围：Motion/H67；只读审计本地冻结证据、论文原文与作者/机构官方仓库。  
动作边界：未修改生产 RTL、合同、结果或 `docs/359_DATE终局冻结_20260813.md`；未启动 VCS、DC、PT、GPU 或远端任务。

## 0. 独立裁决

**本轮没有一个“全新 RTL 模块”达到立即开发门槛，裁决为 `NO_GO_NEW_RTL__CLOSE_EXISTING_MODULES`。**

这不是因为找不到可抄的技巧，而是因为真正相关的技巧已经落入现有模块或被真实 H67 数据否决：

- Prosperity/Phi 的 product/pattern reuse 已与 C1、M473/M498 的 parent/PWP reuse 同族；
- FireFly-T、LoAS 的多 lane decode、bank dispatch、temporal packing 已与 C2、FC2 K8/M499/M496 同族；
- SNE/ActiveN/ELSA 的 event stream、resident state、bundled AER 已与现有 typed descriptor、ATLIF state 和 M221 phase contract 同族；
- FEATHER 的 reorder-in-reduction 可启发适配器收口，但“在 reduction 中隐藏 layout reorder”本身已有强 prior art；
- ESDA、DeltaCNN、MotionDeltaCNN 的空间/跨帧 sparsity 需要相应网络结构或大规模前帧状态，当前 H67 的空 output-site 和自然 temporal delta 不支持立即开 RTL。

因此，接下来允许的 RTL 仅是**既有贡献的收口**：M498 物理报告驱动的 enable-tree 修复、M499 canonical K1 endpoint、M496 FC2 三点 matched top，以及已有 C1/C2/C3/A1 的接口/物理闭合。这些不应包装成第四个新贡献。

## 1. 新 RTL 准入门

一个新模块必须同时满足以下五项，才值得在 DATE 收口窗口继续开发：

1. 作用于冻结 envelope 至少约 10% 的算子，或由同资源模型证明理想全网灵敏度至少 `1.05x`；
2. 冻结 trace 已显示局部周期收益至少 `1.20x`，或 exposed SRAM/DRAM traffic 至少下降 30%；
3. 使用同 lane、同 bank/port、同 SRAM 容量/延迟的强 baseline；
4. 新增状态、SRAM、matcher、NoC 和控制税已显式收费，不能靠免费前帧缓存或无限端口；
5. 机制与 C1/C2/C3/A1/M498/M499 不重复，且面对公开 prior art 仍有清晰差异。

“论文上可能有收益”“只在理想 opportunity 上超过 2x”或“可做一个 RTL demo”均不足以过门。

## 2. 冻结 Amdahl 地图

冻结机会 envelope 为 `620,302,905` cycles。它只用于优先级和 Amdahl 灵敏度，不是已经包含 SRAM/DRAM overlap、FPS 和整帧能量的系统周期。

| 作用域 | cycles | 份额 | 当前最强相关证据 | 独立含义 |
|---|---:|---:|---|---|
| Patch embed Conv | 199,420,620 | 32.1489% | whole-temporal zero site 极低；强 bit-sparse baseline 已跳零 work | Amdahl 大，但没有新 RTL 的真实机会证据 |
| 全网 ATLIF | 128,020,500 | 20.6384% | G12 issue reduction 仅 0.0676%；rank-3 integrated RTL 已存在 | 收旧线，不再造 arithmetic/state engine |
| FC1 | 118,370,114 | 19.0826% | M482 真 RTL recurrence `1.3599x`，未过冻结 `1.50x` 门 | 不扩宽；收已有证据/负结果 |
| 四层 bottleneck Conv | 79,630,957 | 12.8374% | M473 fused opportunity `1.9436x`，unfused 仅 `1.0147x` | M498 只解决 capture path 的物理可执行性 |
| FC2 | 41,413,997 | 6.6764% | K8/K1 局部 `4.7642x`；K8/K1x8 同峰值为 `1.000x` | 卖 shared-state area/energy，不卖 4.76x 系统倍速 |
| Attention core | 3,656,069 | 0.5894% | RQTB 全网约 `1.0009x` | 即使无限快也只有约 `1.0059x` |

几个重要的理想上界：

- M473 fused Conv 若完整捕获，四层 Conv 理想 envelope 灵敏度约 `1.06647x`，可少约 38.66M 机会 cycles；这仍不是已准入系统倍速。
- 官方 Prosperity artifact 的四层 product-vs-bit `2.4595x` 若被自己的同资源实现完整捕获，理想灵敏度约 `1.08246x`；M472 不是本项目 RTL。
- FC2 局部 `4.7642x` 映射到 6.6764% 份额时，理想灵敏度约 `1.05569x`；K1 是低带宽 endpoint，不能冒充等峰值 baseline。
- Patch 即使局部 `1.50x`，理想灵敏度可到约 `1.1200x`，但当前没有通过真实同资源门的 patch 机制。

## 3. 顶会/开源机制映射

### 3.1 Prosperity（HPCA 2025）

官方仓库公开 cycle-accurate simulator、baseline、CACTI 和论文参考结果；其核心是 product sparsity 与 online redundancy identification。仓库也明确指出 DC power/area DSE 脚本没有公开。

- 可借：cycle/energy 同一框架、product-vs-bit ablation、CACTI buffer 定价。
- 已有对应：C1、M472/M473、PWP/parent reuse。
- 新 RTL 风险：再造 product matcher 是高度 incremental；项目真正缺口是 fused opportunity 如何经有限 port/SRAM 变成可执行 commit。
- 官方链接：<https://github.com/dubcyfor3/Prosperity>；论文 <https://arxiv.org/abs/2503.03379>。

### 3.2 Phi（ISCA 2025）

Phi 使用两级 pattern hierarchy：L1 以预定义 pattern 做 weight-side precomputation，L2 以稀疏 residual 补偿，并配合 pattern-aware fine-tuning。论文报告相对 SOTA SNN accelerator 的 `3.45x` speedup 与 `4.93x` energy efficiency。

- 可借：exact pattern 与 lossy residual 分层报告、PAFT Pareto、只物化被引用 pattern/PWP。
- 已有对应：C1 exact parent/PWP、PAFT/near-match 线。
- 新 RTL 风险：新的 pattern cache/residual suppressor 必须先通过 checkpoint/valid825；否则是对 Phi 的复刻。
- 官方论文：<https://arxiv.org/abs/2505.10909>。

### 3.3 FireFly-T（2025 arXiv 预印本，正式 venue/官方 RTL 未核实）

FireFly-T 提出 multi-lane sparse decoder、bank-conflict-aware load balance、OOO workers、binary attention engine 和 implicit data-layout transformation。论文显示多 lane decode 与统一 weight memory 可减轻 bank conflict，但其公开指标主要是 FPGA DSP/energy efficiency。

- 可借：decoder lane 数作为 DSE 轴、equal-throughput resource baseline、push/pop bank rotation、按模块列 resource breakdown。
- 已有对应：C2 signed source decode、FC2 K8 bank-coissue、M490/M499 adapter。
- 新 RTL 风险：仅增加多 lane decoder/bank dispatcher 没有 novelty；H67 还是 signed analog source，不是文中的 binary spike path。
- 原文：<https://arxiv.org/abs/2505.12771>。

### 3.4 LoAS（2024 原文/公开研究 artifact）

LoAS 以 fully temporal-parallel dataflow、timestep-packed spike compression 和低成本 inner-join 加速 dual-sparse binary SNN，报告相对 prior dual-sparse accelerators 最高 `8.51x` speedup。

- 可借：把 temporal packing、compression、inner-join 分开消融；显式报告 prefix-sum/metadata tax。
- 已有对应：T10 packing、source bitmap/bundle、C2 decode 与 ATLIF temporal path。
- 新 RTL 风险：H67 是 signed analog activation，且没有冻结 dual-sparse weight identity；不能把 LoAS 的 binary FTP 倍率移植过来。
- 原文：<https://arxiv.org/abs/2407.14073>。

### 3.5 FEATHER（ISCA 2024，官方 ASIC RTL）

FEATHER 官方仓库包含 NEST/BIRRD ASIC RTL、Synopsys DC 与 Cadence PnR 报告。其 Reorder-In-Reduction 在 reduction critical path 内隐藏 layout reordering，并提供 LayoutLoop/周期验证。

- 可借：所有 adapter/reorder 必须与 reduction overlap；提供同一 top 的面积、功耗、时序和多规模 DSE；两级 numeric/cycle verification。
- 已有对应：M490/M499 canonical bank adapter 与 C2/FC2 reduction service。
- 新 RTL 风险：单独的 layout reorder/transpose adapter 不构成新贡献；除非冻结 trace 能证明跨层 reformat 是 exposed bottleneck，而目前 ledger 没有该项。
- 官方仓库：<https://github.com/maeri-project/FEATHER>。

### 3.6 SNE（DATE 2022，官方 RTL）与 ActiveN（MICRO 2024，官方 RTL）

SNE 公开 sparse event Conv RTL、event router、resident neuron state；ActiveN 公开 RISC-V many-core RTL、active messages、稀疏识别与直接 forwarding 的 memory path。

- 可借：event packet 的 typed protocol、state locality、请求/响应 exactly-once、稀疏 metadata 与 payload 同步收费。
- 已有对应：M221 typed transaction、ATLIF resident state、C2 source descriptor。
- 新 RTL 风险：event routing、state residency、active-message forwarding 本身均已有强 prior art；ActiveN 还会把本项目带到不需要的 many-core/global-scheduler 范围。
- 官方仓库：SNE <https://github.com/pulp-platform/sne>；ActiveN <https://github.com/CRAFT-THU/ActiveN>。

### 3.7 ELSA（2026 arXiv 预印本，正式 venue 未核实）

ELSA 使用 spine/token-wise elastic pipeline、bundled AER 和 mini-batch spiking Gustavson product，目标是减少 first-response latency、NoC traffic 和 memory access。

- 可借：bundle 的收益必须同时报告 compute、NoC 与 memory，而不能只数 descriptor；报告 first-response latency 与 complete latency。
- 已有对应：C2 bundled signed source、M221 phase contract。
- 新 RTL 风险：项目当前没有 NoC/first-response ledger，且 current-batch BN barrier 阻止完整 FFN 的自由 token streaming。
- 原文：<https://arxiv.org/abs/2605.20802>。

### 3.8 ESDA（FPGA 2024，官方 HLS/toolflow）

ESDA 通过 submanifold sparse Conv 保持 event input 的空间稀疏，并用统一 token-feature interface 连接全网络 dataflow modules。

- 可借：必须同时 co-design 网络和硬件，证明中间层仍稀疏；统一稀疏接口只是系统 glue。
- 本地否决：H67 空 output-site 约 `0.1117%`；patch whole-temporal zero site 也极低，强 baseline 已跳过零 source work。
- 新 RTL 风险：不改网络便复制 ESDA token engine不会得到其收益；改网络则需要新 checkpoint/accuracy 身份，超出当前收口窗口。
- 官方仓库：<https://github.com/CASR-HKU/ESDA>。

### 3.9 DeltaCNN / MotionDeltaCNN（CVPR 2022 / ICCV 2023）

DeltaCNN 传播 delta tensor 与 update mask，并为每个非线性层缓存 feature state；MotionDeltaCNN 再引入 spherical buffer、padded Conv 与 moving-camera initialization。

- 可借：跨帧 skip 必须收费 feature cache、mask dilation、refresh/crop 和 BN/nonlinearity state；精度阈值逐层报告。
- 本地否决：当前 local-vs-temporal 逐行选择仅约 2.7% source-work；如果缓存上一帧大张量，SRAM 税远大于这个自然收益的可信度。
- 新 RTL 风险：阈值化 delta、update mask、moving-camera alignment 已被直接占位；没有多序列准确率与 SRAM 模型时不应开 RTL。
- 官方来源：DeltaCNN <https://github.com/facebookresearch/DeltaCNN>；MotionDeltaCNN <https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html>。

## 4. 候选新模块 fast-kill

| 候选 | 命中份额 | 冻结机会 | 状态/SRAM 税 | prior-art/重复风险 | 裁决 |
|---|---:|---:|---|---|---|
| Patch event-token/submanifold engine | 32.15% | 空 output-site `0.1117%`；whole-temporal zero-site 极低 | token map、line buffer、coordinate metadata | ESDA/SNE；强 baseline 已跳零 work | **NO-GO** |
| ATLIF delta/early-stop engine | 20.64% | G12 issue reduction `0.0676%` | membrane state 本就需要，delta 还需旧值/threshold | SNE/DeltaRNN；M273 已有 arithmetic | **NO-GO** |
| FC1 更宽 context factor engine | 19.08% | M482 真 RTL `1.3599x`，低于 `1.50x` 门 | 更多 lane、port、held contexts | Prosperity/Phi/LoAS 邻近 | **NO-GO** |
| Conv 第四种 matcher/cache | 12.84% | fused `1.9436x`，unfused `1.0147x` | matcher/PWP/scratch 已是瓶颈 | Prosperity/Phi 直接 prior art | **禁止；收 M498** |
| 跨帧 warp/delta tile cache | 可跨大份额 | 自然 source-work 仅约 `2.7%` | 前帧 feature SRAM、warp metadata、refresh | DeltaCNN/MotionDeltaCNN 直接 prior art | **NO-GO RTL；仅可离线多序列** |
| Bundled AER/elastic NoC bridge | 多算子潜在 | 没有 exposed NoC/first-response ledger | FIFO、router、packet state | ELSA/SNE/ActiveN；C2 已 bundle | **NO-GO** |
| Reorder-in-reduction adapter | 边界潜在 | 没有 layout-reorder cycle/traffic 账 | switch、buffer、bank rotation | FEATHER；M490/M499 已是 adapter | **NO-GO** |
| Attention prune/score bin | 0.5894% | 无限快约 `1.0059x` | score/mass state | Bishop/Phi；A1 已有 RQTB | **附录，不开主线 RTL** |

## 5. 唯一可能的“未来新模块”为什么仍不准立即做

若 DATE 截止之后保留一个 future-work 候选，最合理的是**drift-bounded cross-frame tile cache**：对 patch/Conv/FC 中间 feature 做 delta/update mask，`tau=0` 为 exact 子集，周期重同步限制 drift，并用事件光流 warp 处理相机运动。

它理论上可命中 patch+Conv 的大份额，也可与 C1/C2/C3/A1 共存；但当前没有通过三项最小门：

1. 自然 temporal 选择仅约 `2.7%` source-work，远低于新 RTL 门；
2. 上一帧 feature cache、mask、warp、refresh 的 SRAM/带宽未定价；
3. DeltaCNN/MotionDeltaCNN 已覆盖阈值 delta、update mask 与 moving-camera buffer，novelty 必须来自 analog ATLIF state 的可验证 drift budget 和事件光流 accuracy，而这些证据尚不存在。

所以它只能先做多序列离线 DSE；不列为本轮建议立即开发的 RTL。

## 6. 与现有贡献的共存判断

| 现有线 | 作用域 | 应做的收口 | 是否需要新架构模块 |
|---|---|---|---|
| C1 / M473 / M498 | Conv parent/PWP reuse 与 dual destination commit | M498 exact VCS、3 ns DC/STA、scratch macro、SAIF/PTPX、capture-gap 图 | 否 |
| C2 | signed source decode/dispatch | full critical top、SRAM latency、SAIF/PTPX、多序列 source density | 否 |
| C3 / M273/M289 | rank/phase-decoupled ATLIF | checkpoint/accuracy identity、同资源 fixed-vs-rank physical table、state SRAM | 否 |
| A1 | exact attention RQTB | K-zero/empty-Q exact skips、能量分表；不做 headline | 否 |
| M499/M496 | FC2 K1/K8/K1x8 shared-state Pareto | exact replay、三点 matched DC、macro/energy/token | 否 |
| M221 | layer-island coexistence | cycle simulator 中 typed phase/epoch/bank ownership；不造复杂 scheduler | 否 |

这些模块可以统一收口：它们按层/phase 复用算术与 SRAM 接口，不要求同时活跃，也不需要一个大型全局调度器。论文贡献数应保持 2--3 个，M498/M499 是强证据/实现支撑，不各自独立宣称创新。

## 7. 对 DATE novelty 的风险排序

| 主张 | novelty 风险 | 必须如何写 |
|---|---|---|
| “product/pattern sparsity” | 高 | Prosperity/Phi 是直接邻居；本项目只能主张 signed analog source 下的 capture path 与 exact dual commit |
| “multi-lane sparse decoder / bank balance” | 高 | FireFly-T/LoAS 已占位；本项目写 shared Acc24 state collapse 与 matched K1x8 税 |
| “resident neuron state / event routing” | 高 | SNE/ActiveN 已占位；不得把基本 state residency 写成新颖 |
| “temporal/delta reuse” | 高 | DeltaCNN/MotionDeltaCNN；必须有 state tax、moving-camera policy、accuracy 与 refresh |
| “layout reorder hidden by reduction” | 高 | FEATHER 直接占位；只作 adapter 实现借鉴 |
| “event input 所以空 tile 很多” | 极高且事实风险 | H67 真实中间 output-site 不空；禁止从传感器稀疏推断网络中间稀疏 |
| “FC2 4.76x” | 口径风险 | 是 K8 对低带宽 K1；同峰值 K1x8 是 1.000x，应主张 area/energy efficiency |

## 8. 收口优先级（代替新 RTL）

1. **M498 最终物理门**：exact VCS/SVA；3 ns 五类约束 clean；area `<=1.20x M475`；失败即永久关闭 Conv dual-slot 物理线，不再写 M500 matcher。
2. **M499/M496 三点 Pareto**：K1、K8、K1x8 统一功能 top；同 bank/response schedule；DC/STA 后主报 area、energy/token、throughput/mm²。
3. **SRAM/macro 与 SAIF/PTPX**：为 C1/C2/C3/FC2 分别收费 scratch/state/weight/acc SRAM；logic-only 数字不得冒充 paper PPA。
4. **统一周期/能量表**：同一 ep35、同一 sequence/trace、同一频率与内存假设；局部倍率不相乘。
5. **多序列**：至少两条额外 DSEC sequence，报告密度、周期、带宽、最差点；Zurich 单序列不能支撑 BP。
6. **正文与图**：capture-gap、三点 Pareto、Amdahl waterfall、负结果消融、三层对标表；这是当前比第四个 RTL 更能抬升 DATE 分数的工作。

## 9. 最终判断

**建议立即开发的全新 RTL 模块数：0。**

当前硬件创新已经足以组织成 DATE 论文，但尚未达到 Best Paper 证据完整度。缺口不在“再多一个 idea”，而在同资源、物理、能量、SRAM、多序列和统一分母。继续横向加一个通用 sparse engine 会增加 prior-art 暴露和验证债务；把 M498、M499/M496、C2、C3 和统一系统表闭合，才是当前最短的上升路径。

## 10. 身份与局限

- 冻结 `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；本轮未修改。
- 本地数值来自 M486 已封证据与列出的冻结结果；本报告未产生新的 cycle、energy、accuracy 或 PPA admission。
- ELSA 与 FireFly-T 在本报告中按 arXiv 预印本处理；未将其描述成已核实正式顶会论文或官方 RTL。
- “官方 artifact 未定位”不等于作者声明不存在；本报告只使用已定位的作者/机构官方来源。
- 本报告是研究方向/准入审计，不是 paper-ready PPA 或系统 speedup 结果。
