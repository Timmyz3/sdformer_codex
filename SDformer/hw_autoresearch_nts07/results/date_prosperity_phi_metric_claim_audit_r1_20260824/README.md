# Prosperity / Phi 指标口径与本地 DATE 证据门审计

日期：2026-08-24

结论先行：目前不能安全声称 2× 加速器、全网络或系统加速。最强的本地性能数字是 M147 在 Motion/H67 四个 Conv3x3 层、20 条 heldout 记录上的 **1.805434× 同时钟周期模型机会**（相对 M143 B4）；M154 已用 VCS 和 3 ns 逻辑-only DC 证明四路独立向量供应接口可实现，但尚未把这条 RTL 路径与 M147 周期数闭环。因此这两个证据不能合并写成“硬件实测 1.81×”。

这不要求先做复杂的全网调度器。按 Prosperity/Phi 的方法，最短可行路线是把主张明确限定为 **Motion Conv3x3 加速模块**，用真实 trace 驱动的周期仿真器、完整模块数据通路、同资源 baseline、SRAM/DRAM 与新思功耗证据做实。只要不把它叫全网络，模块级论文结果可以成立。

## 1. 论文里的高倍数究竟是什么

页码均为 PDF 的一基页码；详细来源和哈希见 [source_ledger.json](source_ledger.json)。

| 论文数字 | 实际粒度 | baseline | 算子/模型范围 | 存储与实现 | 精度 | 裁决 |
|---|---|---|---|---|---|---|
| Prosperity 13.27× | 单一 VGG-16/CIFAR100 加速器吞吐表 | dense Eyeriss | 表中模型；未称端到端 | 28 nm、500 MHz；CACTI SRAM、DRAMsim3；Prosperity 128 PE，Eyeriss 168 PE | lossless / iso-accuracy | 不是通用全网倍数；PDF p9 Table IV |
| Prosperity 14.2× / 7.4× | 周期仿真器上的 ASIC baseline 比较 | Eyeriss / PTB | CNN 与 Transformer 集合；但 PTB/SATO/MINT 在 Transformer 上只比较线性层 | DC + CACTI + DRAMsim3 | lossless | accelerator aggregate，算子覆盖不统一；PDF pp9–10 |
| Prosperity 3.2× avg、7.4× max | 同架构消融 | bit-sparsity-only Prosperity | spiking GeMM | 同一 Prosperity 数据通路 | lossless | product sparsity 的增量，不是全系统；PDF p10 |
| Prosperity 1.79× / 193× | 明确的端到端 Transformer 性能/能效 | A100 PyTorch + SpikingJelly | 各种 spiking Transformer，包含端到端推理 | 模拟 ASIC 对测量 GPU；不同资源、面积和执行栈 | lossless | 唯一明确标成 end-to-end 的高层性能比较；PDF p10 |
| Phi 38× / 4.5× | 理论计算量 | dense / bit sparsity | Phi 稀疏矩阵计算 | 不含硬件周期和停顿 | PAFT 与否需另分 | 不能当周期或芯片倍数；PDF p12 Table 4 |
| Phi 26.70× | 单一 VGG16/CIFAR100 加速器吞吐表 | dense Eyeriss | 表中模型；未称端到端 | 28 nm、500 MHz；DC + CACTI + DRAMsim3 | Phi without PAFT lossless | 模拟/综合加速器表；PDF p9 Table 2 |
| Phi 3.45× / 4.93× | 模型/数据集集合上的性能/能效 aggregate | Stellar | Phi 支持的 sparse processor + LIF；论文没有证明 attention/LN 全算子端到端 | cycle simulator；能耗含 core/buffer/DRAM | without PAFT lossless | 最值得对标的 Phi 硬件级口径，但不是已证明的 all-operator E2E；PDF p10 |
| Phi 1.26× / 1.1× | 算法-硬件增量消融 | Phi without PAFT | 同一 Phi 数据通路 | 相同架构资源 | 有轻微精度下降 | PAFT 的额外收益，不能与 3.45×重复相乘；PDF p11 |

官方来源：[Prosperity arXiv v2](https://arxiv.org/html/2503.03379v2)、[Prosperity 官方仓库](https://github.com/dubcyfor3/Prosperity)、[Phi arXiv](https://arxiv.org/html/2505.10909)、[Phi ISCA DOI](https://doi.org/10.1145/3695053.3731035)。Prosperity 官方仓库固定 HEAD 为 `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`。本次没有从论文/DOI/作者入口找到 Phi 官方 simulator 仓库；这只是截至审计日期的检索结论，今后本地实现必须叫 `Phi-like clean-room`，不能叫官方 Phi。

## 2. 外部硬件指标及不可直接横比项

| 项目 | Prosperity | Phi |
|---|---:|---:|
| 工艺 / 频率 | 28 nm / 500 MHz | 28 nm / 500 MHz |
| 权重/ALU | 8-bit add；128 PE | sparse accumulation；L1/L2 各 8 channel、32 SIMD adder tree |
| 片上 buffer | 8 KB spike + 32 KB weight + 96 KB output，另有 detector/dispatcher 存储 | 240 KB 合计 buffer |
| DRAM | DDR4，64 GB/s；DRAMsim3 | DDR4，64 GB/s；DRAMsim3 |
| 单模型吞吐 | 390.10 GOP/s | 242.80 GOP/s |
| 单模型能效 | 299.80 GOP/J | 285.81 GOP/J |
| 单模型面积效率 | 737.17 GOP/s/mm² | 366.70 GOP/s/mm² |
| 总片上面积 | 0.529 mm² | 0.662 mm² |
| 报告功耗 | 915 mW（Fig.10 案例，含 DRAM breakdown） | 346.6 mW（Table 3 片上部件合计）；性能能耗图另含 DRAM |
| cycle 证据 | cycle-accurate simulator；正文主表报告吞吐/归一化延迟，未给每工作负载绝对 cycle 表 | trace-driven simulator；正文图报告归一化周期/性能，未给可直接移植的绝对 cycle 表 |

以下项目不能直接横比：

- 我们的 3 ns 是 333.333 MHz 约束，不是测得 Fmax；对方表是 500 MHz。先报绝对 cycle，再按各自已承认时钟换算延迟。
- Prosperity/Phi 的 GOP 把一次稀疏累加/加法定义为 OP；我们的 source event、product、descriptor、vector group、commit 不是同一个 OP。
- Eyeriss、Stellar 与稀疏架构的 PE 数、算子能力不同；“同工艺同频率”不等于“同资源”。
- Prosperity 的 PTB/SATO/MINT Transformer 行只覆盖 linear，Phi 没有证明 attention/LN 的端到端覆盖。未覆盖工作不能默认免费。
- 两篇论文是 cycle model + DC + CACTI/DRAMsim3，不是流片或 post-layout；与我们的 logic-only DC 同样要标清 physical boundary。
- 算法无损与 PAFT 有损必须分表。任何 PAFT 收益都要带 checkpoint、train-only 规则和精度差。
- 面积是片上面积，不含 DRAM die；能耗可以含 DRAM transaction。面积、片上功耗和 DRAM 能量不能混成一个来源不明的数字。

## 3. 我们现在可以说什么

本地数值由 [local_metric_audit.py](local_metric_audit.py) 对固定 SHA 输入重新计算，输出见 [local_metric_recompute.json](local_metric_recompute.json)。

| 线 | 可安全声称 | 当前不能声称 |
|---|---|---|
| M143 | Motion/H67 ep35、20 records、四个 Conv3x3 层的同钟 recurrence：B4 135,461,009 cycles；相对 compact256 2.594690×、dualrow512 1.812226× | RTL throughput、matched macro/frequency、physical/full-network/system speedup |
| M147 | 同一 extent 的 ideal recurrence：75,029,590 cycles；相对 M143 B4 1.805434× | 集成硬件速度。若无 same-destination combine，反而为 137,150,654 cycles / 0.987680×；75.95% descriptor 有重复 destination |
| M154 | VCS/SVA：55 descriptors、208 independent vector groups、19,968 lane checks、40 个 II=1 对、3 个 protocol attacks；DC：3 ns、13,282.668059 µm²、setup/hold +1.6514/+0.0002 ns | 98,304-bit SRAM、load、checkpoint replay、accumulator、M147/M152 cycle ratio、paper PPA |
| ATLIF | exact-SHA standalone logic-only DC/STA/Formality：63,114.407654 µm²，3 ns，setup/hold +0.4173/+0.0104 ns，0 mapped multiplier，5,276 pass/0 fail | SRAM、power/energy、集成 cycle、系统 speedup |
| RQTB | attention-core 模型 3,656,069 → 3,090,731 cycles，1.182914×；代入当前 activity-weighted envelope 为 1.000911× sensitivity | 测量 full-network speedup；与 M147/ATLIF 倍数相乘 |

M147 是当前最值得冲性能的主线；M154 是其必要的可实现性修复。RQTB 对当前 envelope 影响约千分之一，不适合当性能 headline。ATLIF 面积/形式验证完整，但缺 integrated cycle/energy，因此目前更适合当硬件实现子模块证据。

## 4. Motion 模块级 cycle simulator 的最低 DATE 完整度

不做总体调度模块也可以，但必须在一个清晰的 `Motion Conv3x3 accelerator boundary` 内完成以下闭环：

1. 冻结主张和 OP：INT8、一次 sparse add/accumulate 如何计 OP、四个目标层、哪些前后处理不在模块内；正文表中明确写 `module accelerator`，不写 full-network。
2. 冻结真实输入：Motion 至少跨 sequence、event-density/equal-rate strata 的 train/calibration/test 隔离；ordered tuple 必须含 source、destination、sign/negate、partition/sequence identity。Local5 最好作外部泛化，但不应阻塞 Motion-only 主张。
3. 做完整模块链：raw ingress/packer → PWP/weight load → M154 四向量供应 → destination combine → 四 bank accumulator/writeback → barrier/commit；有限 buffer、ready/valid、stall、tail、zero-work、stale/reorder 都要收费。
4. 用 RTL 校准 cycle simulator：每一阶段 latency、II、端口数、bank conflict、load/flush/commit 来自 VCS receipt；同一 ordered trace 在 RTL 小样与 simulator 必须逐周期/逐事务 miter。
5. 同资源 baseline：Fixed/bit-sparse、M143、M147/M154 共用相同 lane、SRAM 容量/端口、DRAM 带宽、clock。Prosperity 使用官方 repo 固定 commit 的 adapter；Phi 只能 `Phi-like clean-room`。不能把论文中的 7.4×/3.45×借来当本地结果。
6. 存储和能耗：目标 SRAM macro 或 CACTI 明确 geometry/ports；ordered trace 生成 address-timed SRAM/DRAM transaction 并跑 DRAMsim3；SAIF/PTPX 给 logic，SRAM/DRAM 分项相加。
7. 新思闭环：综合/STA/VCS/Formality 绑定 exact SHA；逻辑面积、SRAM 面积、总面积分列；3 ns 若不是最终 Fmax 就只叫约束。
8. 结果必须给绝对值：每个 workload 的 cycles、latency/FPS、GOP/s、energy/frame、GOP/J、area、GOP/s/mm²、accuracy，再给合法 geomean/p50/p95/worst 和增量消融。

最低需要四张表，结构已经固定在 [date_metric_claim_audit_r1.json](date_metric_claim_audit_r1.json)：T1 指标粒度/来源，T2 资源/PPA，T3 每 workload 绝对性能/能量/精度，T4 消融与 cycle/memory/area breakdown。

## 5. 缺口优先级与 8 月底证据门

当前评分 **43/100，P0=2、P1=6、P2=3**。分数低的原因是 cycle/memory/energy/coverage 未闭环，不是 M154 功能失败。

P0：

- M147 packing、M154 vector supplier、signed combine、accumulator/writeback、barrier 尚未成为一条 RTL-calibrated cycle path。
- 没有同资源 baseline 加上真实 SRAM/DRAM/energy/area，无法形成 Prosperity/Phi 等级的 accelerator table。

8 月 31 日前的硬门：G1 Motion（最好加 Local5）冻结 trace/accuracy；G2 集成模块 RTL + VCS/SVA/Formality；G3 RTL 校准周期仿真器；G4 macro/CACTI + address-timed DRAMsim3 + SAIF/PTPX；G5 同资源 baselines。详细 P1/P2 和 G0–G6 验收条件见结构化审计文件。

若 G1–G5 未全部关闭，最终只能报 standalone M154/ATLIF 证据，以及 M143/M147/RQTB opportunity/sensitivity；不得报 2× accelerator、full-network 或 system speedup。

## 6. 完整性

- 未修改 production、contracts 或 `docs/359`；`docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- 审计使用严格 JSON、重复键/非有限数拒绝、输入 SHA 固定和 fail-closed admission 检查。
- 本目录的最终文件哈希见 `manifest.sha256`。
