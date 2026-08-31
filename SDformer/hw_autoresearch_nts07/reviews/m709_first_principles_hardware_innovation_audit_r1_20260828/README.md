# M709｜H67 Motion 硬件第一性原理审计与新机制快杀合同

日期：2026-08-28  
性质：独立、只读、fail-closed 审阅；未运行 EDA、GPU 或训练；未修改任何既有文件。  
目标：审阅当前已采纳 idea，并只保留能够突破真实服务下界、可在一天内被 CPU 数据杀死或升格的硬件候选。

## 1. 结论先行

当前硬件包已经具备 DATE 投稿所需的两个可辨识核心：

1. **C1：单口、dead-write-only 的 product-capture 路径。**M528 在同一账本、240 KiB 约束下得到 435,293,339 cycle；相对 M468 strongest-zero 为 **1.746753×**，相对同坐标 bit 为 **1.741232×**。这是目前最像主性能表的 Conv 结果，但它仍缺真实 SRAM 宏、完整 RTL recurrence 和 PPA，不能写成最终硅后 headline。
2. **C2：typed signed K8 source service。**M519 已用 VCS 证明 K1/K8/K1×8 的 exact service。K8 相对单 K1 的 4.89–6.32×主要是八路服务并行度；相对等带宽 K1×8 只有约 **1.01–1.04×**。因此 C2 的论文主张必须落在共享 Acc24/端口、throughput/mm²、energy/source，而不是把 4.9–6.3×包装成稀疏收益。

当前包还不是 DATE Strong Accept。原因不是“层数不够”，而是三个物理闭环尚缺：M528 宏/PPA、M519 等带宽 PPA/PTPX、decoder 22–23% 工作份额仍只有 payload 而没有可执行同资源周期。第一性原理审阅后，建议只新开下面三个快杀候选；其中最多升格一个为第三贡献，其余必须被杀掉或降为支撑。

## 2. 第一性原理：先写不可逃避的成本

### 2.1 稀疏线性层

对精确输出，活动 source 与非零权重形成的有效乘加集合仍必须被消费。硬件只能消掉四类附加成本：

- 无效 source/weight 取数；
- descriptor、tag 和路由开销；
- partial-sum 的 spill/reload/RMW；
- 因端口冲突、完成检测和背压产生的泡。

因此“发现更多相似项”若没有消掉以上任一项，就不会转化为 cycle/energy；这正是若干 Conv matcher、lazy-PWP 和多播方案先前失效的根因。

### 2.2 ConvTranspose K3/S2

精确模式下，活动输入经合法 K3/S2 tap 对目标输出的贡献不能凭空消失。source-scatter 与 destination-pull 的算术工作下界相同；可变的是：谁拥有输出、何时关闭输出、psum 是否跨片外/跨 bank 存活、地址和 descriptor 是否需要物化。

M705 已证明 decoder 输入不是“近乎全空”：三条序列加权密度稳定在 **23.2701%–23.3830%**，D3 约 28%。所以新 decoder 机制不能靠 N=0 空块神话；必须直接消掉 destination rendezvous、descriptor 或 psum RMW。

### 2.3 ATLIF T10

当前 ep35 的 45 个 T10 矩阵全部数值满秩，因而 rank-3 不是当前 checkpoint 的精确子集。M518 Fixed 的 10×10 常数矩阵服务需要每 tile 1600 个乘积；96 multiplier 下的真实 issue denominator 是 **17 cycle/tile**。任何新 C3 必须在同一输入/配置/结果端口上，要么降低常数矩阵服务成本，要么证明更好的 throughput/mm²；不能使用未准入 rank-3 精度替换分母。

### 2.4 current-batch dynamic BN

统计量 `mu,sigma` 未完成前，精确 affine 输出无法定稿。因此系统至少必须：

- 保存每个 pre-BN 原值的无损表示，统计完成后 replay；或
- 在统计完成后精确重算 producer。

M480 的 1.4999× 是相对 materialized baseline；相对已经 fused raw-replay 的强基线为 **1.0×**。任何 BN 新 idea 若没有改变“store 或 recompute”二选一，只是在弱基线上重复记账。

## 3. 对已采纳机制的裁决

| 机制 | 裁决 | 理由与论文位置 |
|---|---|---|
| M528 dead-write-only 1RW product capture | **KEEP / C1 主线** | 同账本 1.746753×、240 KiB；必须补真实宏、RTL recurrence、VCS/DC/PTPX 后才 paper-ready。 |
| M519 typed signed K8 | **KEEP / C2 主线** | exact VCS 已闭；主指标应为等带宽 K1×8 的 area、power、throughput/mm² 与 energy/source。 |
| M518 Fixed T10 | **KEEP / C3 公平分母** | 当前 checkpoint 的 exact 基线和网络完整性；本身暂不足以作为 novelty。 |
| 官方 Prosperity/Phi adapter | **SUPPORT / 外部对标** | Prosperity 2.459× 等数字只能写“官方框架中 H67 workload 的机会”；不能当 ours，也不能与本 RTL ratio 相乘。 |
| M523 decoder polyphase mapper / bundler | **SUPPORT** | 是网络完整性和下一个 decoder slice 的基础；没有独立性能 headline。 |
| M705 decoder S3×10 payload | **SUPPORT / 必须使用的数据合同** | 证明真实密度及跨序列稳定性；不是 cycle。 |
| M480 fused dynamic BN | **KEEP AS BASELINE** | 强 baseline 必须保留；禁止把弱 materialized 对比的 1.5×写成创新。 |
| M502 16/24-bit raw container | **SUPPORT** | 只有整数 bridge 与 consumer 端口闭合后，才能作为 memory-energy 消融；系统敏感度约 1.02。 |
| FC1 context factorization / spatial delta | **SUPPORT** | 约 1.16–1.22×局部机会或约 1.36× recurrence，只适合作为 C2 同族消融，不独立成第四贡献。 |
| RQTB attention exact reuse | **SUPPORT** | attention 仅约 0.46–0.59% envelope；适合作为局部能量/完整性，不是系统性能主线。 |
| tag elision 27.53% movement | **SUPPORT** | 27% 并不“差”；只是需要 PTPX/事务能量证明它是能耗收益，而不能无条件换算成周期。 |
| PBR4 全局 decoder FSM | **KILL AS CURRENT IMPLEMENTATION** | M596 仅 56/100，未执行共享 FSM、存在 false-pass 和隐含资源；不再扩系统调度器。用候选 A 的最小 destination-owner slice 快杀核心假设。 |
| ep35 rank-3 / PAFT headline | **KILL FOR CURRENT CHECKPOINT** | 45 个 T10 全部数值满秩；没有 admitted rank-3 accuracy，不能使用约 2.05×/3.40×理想算术数。 |
| BN materialization-elision 1.5× novelty | **KILL** | 相对强 fused replay baseline 为 1.0×。 |
| TDR 跨 timestep/frame 主性能 | **KILL/DEFER** | 既有 source-work 仅约 2.7%，却要 0.87 MiB input bitmap 加 21.2–31.8 MiB output state。 |
| G10/patch 空 tile、G7/G11/G12、attention 有损主线 | **KILL** | 空率、量化值域、端口对齐或 Amdahl 已证伪；不得再开 RTL。 |
| generic Conv matcher、FC1 F4/F8、更多 K-bank | **KILL** | 已有负结果足够；继续横向扩结构会稀释论文。 |

## 4. 候选 A｜PIDP：decoder parity-indexed destination pull

### 4.1 借鉴来源与对象差

- Transposed convolution decomposition 提供 polyphase / parity 分解；
- SCNN、ELSA 提供 sparse accumulator / Gustavson 式执行；
- OpenEye 提供稀疏流构造与解码器完整数据流的工程模板。

引用：

- [SCNN, ISCA 2017](https://research.nvidia.com/publication/2017-06_scnn-accelerator-compressed-sparse-convolutional-neural-networks)
- [ELSA, ISCA 2026](https://arxiv.org/abs/2605.20802)
- [OpenEye official repository](https://github.com/Learning-Chips-Lab/OpenEye)
- [A decomposition method for transposed convolution](https://arxiv.org/abs/2205.02103)

本项目不应宣称发明 polyphase 或 Gustavson。可主张的对象/协议差是：**针对 H67 K3/S2、binary/scaled-binary ATLIF decoder，使用 parity 派生的 bitmap pull，直接喂给 typed signed K8，并让单一 destination owner 完成 Acc24 与一次 final commit。**

### 4.2 它消掉什么、不能消掉什么

不能消掉：每个活动 source、合法 tap、目标 channel 的精确算术贡献。

希望消掉：

- source-scatter descriptor 的物化/搬运；
- 全局 frontier/directory 与复杂 close 检测；
- psum spill/reload/RMW；
- PBR4 多上下文 phase bank 的状态税。

引入的税：输入 bitmap tile/line buffer、每 phase 1–4 个坐标探针、set-bit decoder、weight refill，以及可能损失的 source multicast。

### 4.3 最强同资源基线与一天快杀门

基线：A1-SC8、A1-ISO8、A1-OSG 三者最强者；同 M705 payload、96 lanes、8 banks、240 KiB、同外部带宽和同 output commit 语义。

先仅做 D0/D2/D3 exact binary。D1 `{0,theta}` 在 theta 权重折叠的整数 miter 之前只作诊断，不准进入 headline。

CPU gate：

1. contributor multiset 与 Acc24 output 必须 0 mismatch；
2. S3×10 全样本 ratio-of-sums ≥ **1.20×**，且每序列 ≥ **1.05×**；或 total cycle 不回退超过 5% 且 psum+descriptor bytes 降低 ≥ **30%**；
3. 若与 A1-OSG 的 sequence 等价，或 bitmap probe/weight refill 吃掉收益，立即 KILL。

最小 RTL：parity/address mapper、按实际端口收费的 bitmap-word reader、set-bit decoder、单 destination owner/close、复用 C2 group command。禁止加全局 scheduler、DRAM controller 或大型 directory。

## 5. 候选 B｜TDA：ATLIF exact temporal distributed arithmetic

### 5.1 借鉴来源与对象差

Distributed Arithmetic (DA) 对常数矩阵乘法用 LUT subset-sum 与 bit-serial accumulation 替代显式乘法。近期开放实现可参考 [da4ml](https://arxiv.org/abs/2507.04535) 及其[官方仓库](https://github.com/calad0i/da4ml)。本项目不得宣称发明 DA。

可主张的对象差是：**将 DA 映射到 H67 的 T=10 full-rank、layer-static signed INT8 temporal PSN 矩阵，保持 Q24 bias/threshold 与 analog ATLIF 事件输出完全一致。**这条线不改 checkpoint，不依赖 PAFT，也不假设低秩。

### 5.2 第一性原理映射

把 10 个输入分成两个 5-input group。每个输入 bitplane 用 5-bit 地址查询一个包含 10 个输出 subset-sum 的向量；两组相加并按 bitplane shift-accumulate，signed MSB 作减法，最后加 Q24 bias 并 threshold。

一个 T10 context 的逻辑表量约为：

`2 groups × 32 entries × 10 outputs × 11 bits = 7040 bits = 880 B`。

若物理槽按 16 bit，则 unique table 约 1280 B；若为 16 spatial lanes 提供每拍 32 个 vector read，简单复制可膨胀到约 14,080 B logical / 20,480 B physical active-context storage。另需 160 个 Acc24 DA accumulator、bitplane transpose 与 table build/load。

这说明“8 bitplane 所以 8 cycle”不是免费结论：只有真的提供 32 个并发读端口才成立，表复制和路由必须进入面积/能量。

### 5.3 同资源门和最小 RTL

基线：M518 Fixed T10，完全相同的 input/config/result ports、相同 3 ns 目标、96 multiplier、同 signed integer oracle。

一天 gate：对 group size 2–5、read banks 8/16/32 枚举真实 table bits、端口和配置成本；先做 M518 directed vectors 的整数 miter，再接 checkpoint-bound S10 trace（若现成可用）。只有同时满足以下条件才开 RTL：

- exact 0 mismatch；
- issue ≤ **10 cycle/tile**；
- active table+acc state ≤ **24 KiB**；
- 估算的 throughput/mm² ≥ **1.25× M518**。

若 16 banks 下仍 ≥17 cycle，或表/扇出使预估吞吐面积比不达门，KILL。

最小 RTL 只做 1 个 two-5-input-group vector subset-ROM lane、bitplane accumulator、bias/threshold，与 M518 做 miter；小切片过门后才复制，不先造完整 16-lane engine。

## 6. 候选 C｜RS-BN：dynamic-BN recompute-or-store

### 6.1 借鉴来源与对象差

[Fused-Layer CNN Accelerators, MICRO 2016](https://compas.cs.stonybrook.edu/~mferdman/downloads.php/MICRO16_Fused_Layer_CNN_Accelerators.pdf) 明确讨论了跨层存储与重算的交换；[DeltaCNN official repository](https://github.com/facebookresearch/DeltaCNN) 提供 delta/update-mask 的系统模板。这里不宣称发明 recompute。

H67 的对象差是：**current-batch BN 的 barrier 前 producer 已经是确定性的 bit-sparse FC/source stream。第一遍计算 producer 并积累 moments，不保存 Q24 raw；barrier 后 rewind 紧凑 source bitmap/descriptor，精确重算 producer 并直接送 fused BN+consumer。**

### 6.2 下界、消除项与税

它不能消掉 BN barrier，也不能把一次 producer 计算变成零。它尝试消掉 M480 的 raw Q24 write+read 与 140.625–281.25 MiB peak retention，代价是第二遍 sparse FC、source replay、weight read 与控制 epoch。

最强基线必须同时包括 M480 strong fused Q24 replay 和 M502 conditional 16/24-bit container；同 K8/lane/banks/BW 与 consumer。

一天 CPU gate：用 M262 FC1 和 M519 FC2 的 exact trace/address schedule，逐 phase 报 raw bytes saved、额外 descriptor/weight/psum bytes、cycle、peak state、memory-energy sensitivity。只有满足局部 cycle ≥ **1.15×**，或 memory energy 降低 ≥ **20%** 且 peak external retention ≥ **8×**，并且不增加隐含端口时，才可作为 support mechanism；否则 KILL。

最小 RTL：epoch replay controller、source-bitmap rewind 和 second-pass mode bit，复用现有 C2；moments 与 fused affine 不变。它最多是 C2 的 memory support，不应与 C1/C2/C3 并列为第四 novelty。

## 7. 为什么没有第四个候选

- **有损稀疏**：没有同 checkpoint、同 validation protocol 的 AEE/EPE Pareto，就不允许借论文机制直接写收益。Bishop/DeltaCNN 可以作为 future work 或方法学引用，不能绕过精度验证。
- **更多 attention trick**：即使无限加速，旧 envelope 的系统上限也约 1.004–1.006×。
- **空 tile / event mask**：decoder 与 Conv 的真实密度已经否定“99.4% 可跳”一类叙事。
- **generic matcher**：已有实验说明机会率不等于可执行周期；再做 matcher 只会重复端口、完成和存储税。
- **rank-3**：不是当前 ep35 的 exact subset；若换 checkpoint，所有 headline 身份必须重跑。

## 8. DATE 模拟评审

### 8.1 当前证据包

| 维度 | 分数 / 5 | 判断 |
|---|---:|---|
| Novelty | 3.4 | C1/C2 的对象和协议差清楚，但 decoder/C3 还没有新物理点。 |
| Soundness | 4.2 | fail-closed、同资源、VCS 与负结果纪律强。 |
| Significance | 3.2 | C1 1.75×可信；C2 headline 尚待等带宽 PPA；全网直接数未闭。 |
| Implementation | 3.2 | 多个 RTL/VCS slice 存在，宏与全链 PTPX 不足。 |
| Evaluation | 3.0 | 多序列 payload 有了，统一 direct simulator / Table A 尚缺。 |
| Reproducibility | 4.2 | 合同、receipt、SHA 和外部 artifact 标注较强。 |

硬件 readiness 综合约 **3.4/5，Borderline/Weak Accept**。强接收概率仍低；不能靠增加 idea 数量改善。

### 8.2 最小闭环路径

1. M519：完成等带宽 K8 vs K1×8 的 DC/PTPX/area，同频同端口，输出 throughput/mm² 与 energy/source。
2. M528：完成真实 1RW macro 账、bounded RTL recurrence、VCS/DC/PTPX，并把 1.746753×接回同一 full-trace simulator。
3. PIDP：先跑一天 CPU gate；GO 才做最小 RTL/VCS/DC，KILL 则把 decoder 作为完整性，不继续系统调度。
4. TDA：先跑静态端口/表复制 DSE；只有 exact 且 throughput/mm² 预计 ≥1.25×才开一个 lane 的 RTL。
5. 统一表：Dense96 Fixed、PTB-like、K1、K1×8、K8、Ours-direct，统一 S3 多序列、周期、流量、面积、能量与 accuracy 身份。外部 Prosperity/Phi 只占第二层对标。

若上述 1、2 闭环，且 PIDP/TDA 中至少一个形成 exact、同资源的第三支撑点，硬件评审可上升到约 **4.0/5 的 DATE Accept 竞争区间**。不要求三个候选全做；两个主贡献闭环比六个半成品更强。

## 9. 论文可写的三条贡献

1. **C1 — Constrained product capture：**在 240 KiB、单口 parent scratch 与真实 completion 约束下，把外部 product-sparsity 机会转化为 H67 signed Conv 的可执行 1.74×同账本收益；外部 Prosperity 只作机会参照。
2. **C2 — Typed signed-source service：**把 binary multi-NZ service 迁移为带符号、带类型、Acc24 原子更新的 K8 source protocol，并以等带宽 K1×8 的物理效率衡量，而非用单 K1 夸大并行收益。
3. **C3 — Full-network exact closure：**以 Fixed T10、dynamic-BN strong replay、decoder exact mapping 构成完整网络；若 TDA 或 PIDP 过门，将其中一个写成 exact operator specialization，另一个留作消融/完整性。

这三条足够形成 DATE Accept 的结构；不需要为 FC1、BN、ATLIF、decoder 每层各造一个“贡献”。层级优化应共享 C2/C3 的协议与同资源表，避免贡献碎片化。

## 10. 冻结与红线

- `docs/359` SHA256 复核为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，本审阅未修改。
- 禁止把 Prosperity/Phi/FireFly-T 官方 artifact 数字写成 ours。
- 禁止把 K8 vs 单 K1 的并行比写成 product-sparsity 比。
- 禁止把 payload、logical movement、analytical lower bound 写成 cycle、energy 或 silicon PPA。
- 禁止把 D1 scaled-binary fold 纳入 decoder headline，直到整数 miter 0 mismatch。
- 禁止在没有同 protocol accuracy/AEE 的情况下升格任何有损机制。
- 禁止新增第四条 generic Conv matcher 或全局 decoder scheduler。

