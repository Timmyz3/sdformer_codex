# M589｜开源机制迁移独立审计（r1，2026-08-28）

## 1. 裁决先行

**最终裁决：`NO_NEW_RTL`。**

Prosperity、Phi、FireFly-T、SNE 与 ELSA 中对 H67 真正相关的机制，要么已经落在现有 C1/M528-M533、C2/M519、C3/M518 的收口范围内，要么已经被同资源 H67 trace（M484、M482 及既有 G/H 系列 NO-GO）证伪，要么不满足 H67 ATLIF-PSN 的数学前提。现在另开模块会增加验证与物理收口面，却没有一条尚未覆盖、且一天内能过同资源门的新假设。

这不等于停止硬件推进。正确动作是完成三个已有端点：

1. **C1**：M533/M528 exact signed product-capture 的 VCS 收口，随后把 144 B parent scratch 放入宏/CACTI 与物理账；
2. **C2**：M519 K8、K1、K1×8 在相同 8-bank、相同 Acc24、相同频率约束下完成 DC/吞吐面积/能量比较；
3. **C3**：M518 Fixed 与 rank-3 的匹配 DC，以及 checkpoint/精度身份收口。

本审计没有运行 EDA、GPU 或远程任务，没有制造新性能数字，也没有修改 `docs/359_DATE终局冻结_20260813.md`。

## 2. 用途、粒度与证据纪律

- **用途**：为 DATE Accept 决定哪些公开机制可合法迁移，哪些只能作为相关工作或负消融，避免把“改名”误写成 novelty。
- **分析粒度**：机制/协议/资源边界；不是整机性能预测。
- **迁移成立的最低条件**：原工作必须明引；H67 必须改变计算对象、协议边界或资源约束；必须在同容量、同端口、同带宽/Acc24 下设门；局部数字不得改写成系统数字。
- **当前证据强度**：公开论文/官方仓库用于机制身份；本地 sealed result/review 用于 H67 准入状态。没有把未执行的 VCS/DC 当成结果。
- **固定红线**：不复活 G10、G7、G11、G12、H1/M470、G15、lazy-PWP、FC1 M482、combined PVRF；不把 M528 CPU 周期代理写成 RTL/PPA/能量；不把 PAFT 与 exact 混列。

## 3. 公开机制到 H67 的可包装贡献表

| 原工作 / 原机制 | H67 上改变的对象、协议或资源边界 | 最接近的现有 M 点 | 一天内可执行的同资源门 | 论文归属 | novelty 风险与裁决 |
|---|---|---|---|---|---|
| **Prosperity（HPCA'25）**：subset/prefix product sparsity；按 popcount+index 稳定排序；prefix 输出作 psum、只补 residual；双缓冲预处理/执行 | 从 binary spike × weight 的 tile 内 subset，迁到 **signed ATLIF source** 的 exact parent/residual；约束为 64-row directory、240 KiB、144 B 单口 1RW scratch、dead-only write 与原子完成；不是照搬其 TCAM+processor | **M528 CPU / M533 RTL** | 仅在资源冲突解除后运行已有 exact-SHA M533 VCS：reference 0 mismatch、SVA/协议攻击/覆盖全过；随后宏化 scratch 后再谈面积/能量。不得新开 matcher | **C1** | **高**：product sparsity 与 stable dispatch 本身属于 Prosperity。可写对象差只限 signed residual、单口 scratch、dead-write/completion 协议；`NO_NEW_RTL` |
| **Phi（ISCA'25）**：L1 pattern/PWP + exact signed L2 residual；PAFT 另作有损档 | H67 exact 模式是 parent/PWP-like reuse 加 signed correction，并由 Acc24 收口；有损只允许 PAFT checkpoint 的独立 Pareto，不得改变 exact 身份 | **M528 / M579** | M579 在同 240 KiB、同端口、同 trace 下与 control 成对 replay；exact 必须 0 mismatch。有损只有在完整 sequence 披露、`ΔAEE≤0.02` 且额外局部 cycle 收益 `≥1.15×` 时才可进附表；未过门不写 RTL | **C1 有损消融** | **很高**：pattern/residual+PAFT 由 Phi 直接占位。当前 PAFT 完整 sequence 已出现回退，不能升主轴；`NO_NEW_RTL` |
| **Phi（ISCA'25）**：L2 row-packing、8-unit compact pack、PWP 预取、可重构加法树 | 改成 H67 的 typed signed descriptor（幅值/标签/时间/terminal）一次 K8 发射、单一共享 Acc24 状态；M528 不存 PWP payload，所以 Phi 的 PWP prefetch 不能直接声称适用 | **M519；tag-elision；M484** | 先完成 M519 K8 vs **等总带宽 K1×8** 的匹配 DC；只接受 throughput/mm² 或 energy/source 的同资源优势，门 `≥1.15×`。tag-elision 只在物理证据显示 metadata 显著时作为子机制 | **C2** | **中高**：packing/prefetch 不是新意；typed signed atomic descriptor + shared Acc24 可构成协议差，但须物理数据支持；`NO_NEW_RTL` |
| **FireFly-T（arXiv'25）**：每拍解 K 个非零的 multi-lane bitmap decoder；wide-bank 广播；额外 worker 维度做 OOO 均衡 | 从 binary spike index 改成 K8 signed source，复用 index 于目的通道，并保持同 8-bank 吞吐和单 Acc24；比较对象必须是 K1×8，不是单 K1 | **M519**；**M482 禁止复活** | M519 匹配 DC + frozen trace：频率/面积/顺序单元数/能量逐项对齐 K1×8；cycle 若只有约等价，不得写“稀疏加速”。FC1 M482 的 100-record 只有 `1.359896673×`，低于其冻结门，不扩宽 RTL | **C2** | **高**：multi-lane decoder/wide-bank/OOO 是直接 prior。只能写 signed typed source、atomic release、shared-state 约束的对象差；`NO_NEW_RTL` |
| **FireFly-T（arXiv'25）**：SRAM bank/address/byte 布局隐式转置；QK/V 与 QKV projection latency hiding | 若 H67 address trace 中确有独立 transpose 取数/停顿，可迁为 RQTB/attention 的布局消除；但 attention 只占旧 envelope 约 0.59%，且 FPGA LUT6 AND-popcount 不是 ASIC 论据 | **A1/RQTB；M520 registry** | 先做只读 address-trace 审计；必须观察到独立 transpose traffic/ stall `≥30%` 或同端口局部 `≥1.20×` 才允许 RTL。当前无这项证据 | **不进 C1/C2/C3；仅 A1 支撑** | **高**：布局和 overlap 直接来自 FireFly-T，且 Amdahl 很小；当前 `NO_GO_RTL` |
| **SNE（DATE'22，官方开源）**：显式 RST/UPDATE/FIRE event；本地双缓冲 neuron state；time-of-last-update 延迟施加 leak | H67 可借鉴 typed event 与状态驻留，但 H67 `ATLIFTernaryPSN` 是 dense temporal matrix/factorized matrix 后阈值，不是常规标量 recurrent LIF；TLU leak skip 不具等价前提 | **M519 typed source / M518** | 只有先给出 H67 PSN recurrence 等价式，且 CPU 0 mismatch、同资源局部 `≥1.15×`，才可讨论 TLU。当前数学前提失败，因此不跑 RTL | **C3 相关工作/边界说明** | **很高**：event router/resident state/TLU 均有直接 prior；对当前网络语义不成立，`NO_GO` |
| **ELSA（ISCA'26）**：bundled AER；mini-batch spiking Gustavson；一行 membrane 只读写一次；spine/token elastic pipeline | H67 对应的是 row-coherent signed bundle，但 dense optical flow 没有 first-correct classification response；且 M484 已把同资源 K8-resident 与 bundle-stationary 路径直接比较 | **M484 / M519** | **门已失败**：M484 同资源 cycle ratio 为 `1.0×`，header/padding 使 traffic 略增；不重跑、不写新 RTL | **C2 负消融/相关工作** | **很高**：BAER/Gustavson 是直接 prior；H67 数据不支持收益，维持 `NO_GO_PERFORMANCE_OR_RTL` |

## 4. 为什么论文里的大倍率不能靠照搬一个模块获得

Prosperity、Phi 与 FireFly-T 的 headline 来自一整套被共同计入的评测边界：稀疏表示、预处理与计算重叠、片上缓冲、外存模型、相应 baseline，以及跨层/端到端周期模拟。公开实现中 Prosperity 的周期模型明确把 tile 延迟按 `max(compute, preprocess)` 处理，并加入 memory stall；Phi 还把 PWP 预取、L2 packing 与 K-first schedule 放在同一系统模型中。因此合法的“论文技巧”是明确分母梯度和测量边界，不是把局部机会乘起来：

1. **机会层**：官方 Prosperity framework 上的 product-vs-bit，只写外部机会；
2. **可执行层**：M528 同一 240 KiB/1RW 账本的 CPU cycle，写 ours 的捕获结果；
3. **RTL/PPA 层**：M533/M519/M518 经 VCS/DC/宏/能量后才进硬件表；
4. **系统层**：只有完整 decoder、ConvTranspose、FFN、attention、存储事务都在统一 simulator 中，才报 frame/system speedup。

这四层并列，反而能形成 DATE 可接受的 capture-gap 叙事；跨层相乘会破坏可信度。

## 5. 防止复活既有死线的逐项核对

| 既有 NO-GO | 本审计处理 |
|---|---|
| G10 空 tile | 不映射为 AER/command-mask headline；真实空 output-site 仅 0.1117% |
| G7 幅度门 | 不以 Phi/FireFly 稀疏名义复活；bottleneck 输入近二值、无中间 Pareto |
| G11 source drop / 累计预算 | 不以 L2 residual 名义复活；端口收费后的执行收益已失败 |
| G12 ATLIF 早停 | SNE TLU 与 H67 PSN 数学不等价，且原 issue 减少仅约 0.0676% |
| H1/M470 payload resident | 不以 PWP prefetch 名义复活；完整 DRAM/psum 交换下整体 0.831× |
| G15 / lazy-PWP | 只作负 DSE；不包装为 Phi prefetch |
| FC1 M482 | FireFly wide-bank 不扩成新 FC1 RTL；冻结 100-record 结果未过 1.50 门 |
| combined PVRF | 不以 dual-parent/row-pack 名义复活；对 dead-only 已是 0 cycle 增益 |
| M484 row bundle | ELSA 映射直接落在此处；同资源 1.0× 且 traffic 更差，维持关闭 |

## 6. 至多两个候选的裁决

**没有新增候选；结论为 `NO_NEW_RTL`。**

若把“候选”放宽到已有 RTL 的收口动作，优先顺序仅为：

1. M533（C1）已有 exact-SHA VCS/宏化收口；
2. M519（C2）已有匹配三轴 DC/能量收口。

这两项不是新结构，也不得重新编号包装成第四项贡献。M518（C3）继续按既有 Fixed/rank-3 计划收口。

## 7. 可写进论文的保守措辞

- **C1**：引用 Prosperity/Phi；声称的是“在 H67 signed ATLIF source、单口 parent scratch 与 dead-write/completion 约束下实现 exact product capture”，不是发明 product sparsity。
- **C2**：引用 Phi/FireFly-T/ELSA；声称的是“typed signed K8 release 在共享 Acc24 状态下的面积—吞吐实现”，主对照为等总带宽 K1×8，不写相对单 K1 的夸张稀疏倍速。
- **C3**：引用 SNE 与动态 SNN 工作；声称的是 H67 PSN 的 phase/rank 实现，不借用不成立的 TLU recurrence。
- **有损**：PAFT 只作独立 checkpoint 的 Pareto 消融。当前完整 sequence 的回退必须披露，不能用 valid825 单一正点替代。

## 8. 来源与本地身份

### 公开一手来源

- Prosperity paper：<https://arxiv.org/html/2503.03379>
- Prosperity official repository：<https://github.com/dubcyfor3/Prosperity>；本地只读副本 commit `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`
- Phi paper：<https://arxiv.org/html/2505.10909>
- FireFly-T paper：<https://arxiv.org/html/2505.12771>
- SNE paper（DATE 2022 archive）：<https://past.date-conference.com/proceedings-archive/2022/pdf/0908.pdf>
- SNE official repository：<https://github.com/pulp-platform/sne>；本地只读副本 commit `92449df7a49f485f331dc785522b82acd33759ae`
- ELSA paper：<https://arxiv.org/html/2605.20802>

### 关键本地证据

- `docs/524_DATEAccept当前硬件贡献与机制迁移收口表_20260827.md`
- `results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json`
- `reviews/m484_independent_hammer_r1_20260827/m484_independent_hammer_r1.json`
- `results/m482_fc1_l96_f2_c16_b2_receipt_blind_hammer_r1_20260827/m482_fc1_l96_f2_c16_b2_receipt_blind_hammer_primary_r1.json`
- `reviews/m519_registered_release_vcs_receipt_hammer_r2_20260827/`
- `reviews/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827/`

## 9. 局限与下一步

- Phi 未使用官方代码 artifact 做新重放；本审计只从论文提取机制，不推断其未公开实现细节。
- FireFly-T 与 ELSA 为公开预印本版本；最终出版版本若改变数据，应在投稿前复核。
- 本次没有取得宏库/CACTI 新数，也没有运行 Synopsys；所以只规定准入门，不报告新面积、功耗或频率。
- M579 PAFT r2 仍应按现有成对 replay 合同完成；结果若未过门，归入有损负消融，不再派生 RTL。

**收口判断**：现在最短的 Accept 路径不是“再借一个机制”，而是让 C1/C2/C3 的现有机制在公平分母下形成 CPU/VCS/DC/宏/能量连续证据链。
