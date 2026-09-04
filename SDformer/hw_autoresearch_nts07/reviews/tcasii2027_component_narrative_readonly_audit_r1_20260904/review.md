# TCAS-II 组件稿叙事与增量机制独立审阅（r1）

- 审阅日期：2026-09-04（Asia/Shanghai）
- 审阅对象：当前 `paper/iscas2027/main.tex`、已封存 C1/C2/TSBG/C3/ep34 证据与最终 ISCAS r6 审阅
- 审阅模式：只读；未运行 EDA、VCS/simv、GPU、许可查询或网络实验，未修改论文、RTL、结果与 `docs/359`

## 裁决

**普通 TCAS-II regular submission 可行；不需要为了准入先做 FPGA。** 当前材料更适合改成一篇“精确 typed-source 数据复用电路 brief”，而不是完整光流 SoC 或逐层硬件大全。现稿直接换模板投稿，我给 **3.8/5，Weak Accept/Reject 边界，约 55--65% 接收倾向**；完成 matched hold、TSBG 动态功耗/存储读能量和 FC2 continuation 后，预计可到 **4.2--4.35/5，约 75--88% 接收倾向**。这些是审稿倾向估计，不是保证。

ISCAS r6 的 4.4/5 不能原样搬到 TCAS-II。TCAS-II 官方要求结果在首次投稿时基本完整、决定为 accept/reject，并明确提醒：电路稿若没有证明相对既有工作的性能优势（通常伴随 measured results）或实际系统意义，可能不送外审。与此同时，普通投稿指南没有 FPGA 或流片的硬性门槛；要求 FPGA 的是 ISICAS 等特定 journal-track 通道。一个已发表 TCAS-II SNN brief 也采用 65-nm 综合、250 MHz 和数据集能量/吞吐作为主要实现证据，说明商业 ASIC 综合/功耗链是合法路线。

官方依据：

- [TCAS-II manuscript submission guide](https://ieee-cas.org/publication/TCAS-II/tcas-ii-manuscript-submission-guide)：5 页，正文 4.5 页、最后半页仅参考文献；快速 binary decision；稿件应完整；电路性能优势与应用意义是送审门。
- [TCAS-II scope](https://ieee-cas.org/publication/TCAS-II)：数字电路、系统和信号处理应用均在范围内。
- [A 24.3-uJ/Image SNN Accelerator for DVS-Gesture With WS-LOS Dataflow and Sparse Methods](https://ieeexplore.ieee.org/document/10143982/)：TCAS-II 2023 的综合型 SNN accelerator 先例。

## 当前真正能支撑 TCAS-II 的两条贡献

### C1：有限容量、单 1RW 端口下的精确 product capture

合法主张不是“发现了 product sparsity”，而是：重复乘积只有在 parent 存活、容量、单口读写冲突和 atomic completion 都被计入后才可执行；C1 用 exact residual reconstruction 与 dead-write suppression 把该机会落到九颗 128x128 1RW 宏的电路边界。

当前可用数字：

| 指标 | 数值 | 证据边界 |
|---|---:|---|
| 同账本周期模型 | 648,741,051 -> 382,848,700，1.694510x，时间 -40.9859% | 单序列 `zurich_city_09_a`、10 samples、四层 bottleneck Conv、51.84M source rows；不是 RTL 周期 |
| mapped component | 166,514.312 um2，九 SRAM | 独立 mapped identity；不是完整 240-KiB ledger 集成 |
| PT/Formality | setup/hold +27.871/+1.827 ps；16,549 mapped-to-mapped compare points | 3 ns、prelayout；不是 RTL-to-gate proof |
| bounded energy window | 29.0763 mW，22.0689 nJ；parent scratch 约 36.1% | 253-cycle directed window，mixed-corner、ideal clock/ZeroWireload、无 SPEF；不是 energy/frame |

TCAS-II 写法应把 **1.6945x 标成 cycle-accurate model**，把 mapped PPA 与单 real-mask VCS tile 作为独立校准轴。若无法做全 51.84M-row RTL replay，不能把两轴合成“mapped RTL achieves 1.6945x”。

### C2：typed K8 共享执行底座，内嵌 context-safe TSBG

C2 与 TSBG 应合并为一条贡献。K8 的价值是等带宽面积效率；TSBG 的价值是跨 token 只复用 weight delivery，同时保持 sign、destination 与 Acc24 context 私有。两者组合后才像 TCAS-II 的“数据通路 + 存储访问控制”电路贡献。

当前可用数字：

| 指标 | 数值 | 证据边界 |
|---|---:|---|
| K8 vs equal-bandwidth K1x8 | 1,913 vs 1,945 VCS cycles，1.016728x | 五个冻结 directed component workloads；周期优势很小，应与面积同时报告 |
| 吞吐/logic-area | 4.541078x；logic area -77.6104% | 131,086.241 vs 585,479.154 um2，3 ns logic-only DC；hold/power/macro open |
| TSBG post-load execution | 12,522,876 -> 5,124,365 VCS cycles，2.443791x，时间 -59.07997% | 1,920 fixed workloads、40 samples、四序列；不是 full FC/network |
| weight requests | 8,774,304 -> 3,136,608，-64.25234% | 同一 TSBG population；公共端口/row cache 相同 |
| matched schedule-mode area | 249,710.452 vs 249,739.810 um2，+0.0117568% | logic-only；两轴 diagnostic hold 均 -16.4 ps；power open |

这条是稿件最强的电路结果。摘要排序建议改成 **TSBG direct VCS result -> K8 equal-bandwidth area efficiency -> C1 cycle-model/mapped anchor**；不要继续让 model-only C1 占摘要第一结果位。

## 五页 TCAS-II 应如何重组

建议题目：

> **Exact Product Capture and Context-Safe Weight Broadcast for Event-Driven Spiking Optical Flow**

当前题目可以保留；不要加“full-network accelerator”“processor”或“system”字样。

建议只列两个 contribution bullets：

1. C1：在有限容量和单 1RW parent lifetime 下，将 product opportunity 变成 exact executable capture；
2. C2/TSBG：在等带宽 typed-K8 底座上，用 context-safe broadcast 抑制 weight delivery，并给出相同端口/cache 的 RTL 与 matched physical ablation。

页面预算（TCAS-II Transactions 双栏）：

| 位置 | 内容 | 取舍 |
|---|---|---|
| P1 | 100--220 词摘要、问题、两条贡献、总览图 | 摘要只放 admitted 数字，不写 C3 |
| P2 | execution contract + C1 数据通路/1RW timing | 用一张 parent lifetime/读写时序图替代证据谱系文字 |
| P3 | K8/TSBG 数据通路、普通/广播模式、exact fallback | 把 K8 与 TSBG 合成一个 subsection |
| P4 | 方法、同资源基线、C1/C2/TSBG 主结果 | 一张主表 + 一张 TSBG 序列/能量图 |
| P5 左栏 | 消融、相关工作、限制、结论 | 正文必须在左栏结束 |
| P5 右栏 | references only | 官方硬格式门 |

现稿中 M2053 的 1,917+3 lineage、失败 attempt 的 provenance、seal 细节适合 artifact README，不应继续占 TCAS-II 正文。正文只需一句“all logs and source identities were independently sealed and rechecked”。把省下空间用于 circuit timing、SRAM read suppression 和 matched energy。

## 应删除或降级的内容

| 内容 | TCAS-II 位置 | 原因 |
|---|---|---|
| C3 Fixed-T10 | 从摘要、贡献 bullet、主表和结论删除；最多 implementation 一句 | 只有 exact 17 cycles/tile 与 PPA，没有 baseline speedup/energy；会稀释两条主线 |
| RQTB/Shiftmax/attention | 只在 workload 定义中一两句 | attention 份额小且不是本文电路结果；算法候选与冻结路径身份复杂 |
| decoder replay/完整系统表 | 不等、不进主叙事 | 组件 brief 不需声称完整 SoC；decoder 结果不能与 C1/C2 相乘 |
| 四指标 accuracy 表 | 压成一句 AEE compatibility + 明示 backend mismatch，或放补充材料 | 当前 candidate 只证明部署兼容，不是因果精度提升；四行表消耗过多正文 |
| 3.27B operator-local element 细节 | 压成 Acc24 bound、1,280 integer probes、825-frame gate 三项 | 细节可转 artifact；保留位宽可信度即可 |
| CICC/Prosperity/Phi 大段方法学 | 缩成 related-work 一段 | TCAS-II 更在意本电路相对 prior 的对象差和 PPA，而非证据等级教程 |
| S2/有损剪枝、空 tile、RQTB 变体 | 删除 | 会成为第四条薄贡献，并要求新的 accuracy Pareto；与 exact brief 冲突 |

## TCAS-II 首投前的缺口

### P0：决定是否容易被 desk reject

1. **TSBG matched hold 必须收口。** 当前同一 physical ablation 两轴均为 -16.4 ps。小负 hold 可以工程修复，但 TCAS-II 首投不应在摘要主动写“hold remains open”。修复后须用同一 netlist identity 重报面积、setup/hold；若网表变更，旧 power 不可拼接。
2. **TSBG matched dynamic energy 必须得到正结果。** 目前 -64.25% weight requests 尚未落成 SRAM/logic energy。至少需要 same-workload ordinary/TSBG SAIF + PTPX，且把 logic、weight-store read 与 leakage 分列。若暂时没有目标 SRAM macro，必须明确写 macro model/Liberty、容量、端口、读能量和 corner；不能只用 request count 代替能量。
3. **换为 TCAS-II Transactions 格式并满足 4.5+0.5 页。** 当前是 `IEEEtran[conference]` 四页 ISCAS 版；还缺 journal author footnote/affiliation、5 个以上 index terms，以及最后一栏 references-only 的版面验证。

### P1：显著提高外审接受率

1. **完成 G>48 FC2 exact continuation。** 它把最强 TSBG 结果从 4/12 FC2 层扩到 12/12，而不增加第四个机制，也不换 checkpoint。
2. **把 C1 的模型/RTL/physical 三轴画成一张 evidence ladder。** 让审稿人一眼看出 1.6945x 是模型、单 tile 是功能校准、166.5k um2 是 mapped anchor；消除“把不同证据拼成同一 silicon result”的疑虑。
3. **补至少一篇相关 TCAS-II/TCAS-I SNN accelerator。** 官方明确要求适当引用本刊及 IEEE 近期文献。WS-LOS TCAS-II 2023 可以作为“数据流 + hierarchical memory + sparse methods”的直接对比对象。

### P2：有则加分，不挡投稿

- C1 多序列 cycle-model 复放；当前只有一个序列。
- C2/TSBG SRAM macro 替代 stdcell state 后的面积/功耗；比 FPGA 更直接。
- full-token TSBG 分布；当前固定 first/middle/last B4 quartets 已够组件稿，但不是完整 token population。

## 最多两个低成本、嵌入已有贡献的改动候选

### 候选 A：Continuation-Safe TSBG（优先级 1，正在做的方向正确）

- **prior：** ELSA 的 bundled/Gustavson event delivery、SpikeX 的跨时空 weight reuse，以及常规 tiled FC accumulation。
- **对象差：** 对 G>48 的 H67 FC2，不扩大 G48 engine；把 source groups 精确切成不超过 48 的 chunks，在 chunks 间保留每 token 的 Acc24、sign/destination ownership 和 terminal 状态，只在最后 chunk commit。TSBG 仍只复用 weight delivery，不复用 signed product。
- **嵌入位置：** C2/TSBG 的 coverage/continuation 子段；不是第三或第四条贡献。
- **1--2 天可测门：** 固定新增 960 workloads（8 FC2 layers x 40 samples x 3 quartets）；ordinary 与 TSBG 使用相同 chunk/preload/cache/public-port charge；integer oracle mismatch=0，Acc24 overflow=0，960/960 完整；新 FC2 population ratio-of-sums >=1.20x，合并 2,880 workload 后 aggregate 不退；VCS 通过后 matched DC 面积增量 <=2%，3-ns setup/hold 均非负。
- **评分影响：** 若通过，Evaluation 约 +0.2、Implementation +0.1、Novelty +0.05；总分约 +0.15--0.25。若仅 CPU premodel 或小于 1.20x，只作为范围消融，评分基本不变。

### 候选 B：Reuse-Hit-Qualified Weight-Read Suppression（优先级 2）

- **prior：** operand isolation/clock gating 是标准低功耗手段；WS-LOS 用 hierarchical memory/data reuse 降低访问，ELSA/SpikeX 已建立 bundle/weight-reuse prior。
- **对象差：** 不把 gating 本身包装成新发明。本文可主张的是：TSBG 的 shared-row hit 在进入 weight-store address/read-enable 之前产生，直接阻断该次 SRAM read、地址译码和 downstream row-register toggle；四个 typed contexts 的 sign/destination/Acc24 仍独立。也就是说，已验证的 -64.25% request reduction 被物理化为可测的 read-energy suppression，而非仅减少一个后端计数器。
- **嵌入位置：** C2/TSBG 的 physical implementation 段和 energy ablation；不是新 contribution bullet。
- **1--2 天可测门：** 同一 1,920（或 continuation 后 2,880）workload population，ordinary/TSBG 相同 SRAM macro/capacity/ports/corner/clock；VCS exact 与所有 stale/backpressure attacks 仍通过；SAIF 100% annotation 且 X/Z=0；weight-store dynamic energy >=30% 降低或 whole-component dynamic energy >=10% 降低；logic area <=1% 增量，3-ns setup/hold 均非负。若只有 request count、没有 SRAM activity/energy，则不能升格。
- **评分影响：** 这是最符合 TCAS-II 的增量。通过后 Implementation/Significance 各约 +0.2--0.3，整体约 +0.2--0.35，并显著降低 desk-reject 风险。若总能量改善 <10%，仍可保留为 memory ablation，但不进摘要。

## 不建议开启的新线

- **不做 FPGA 作为当前 P0。** 它会另开 BRAM、时钟、板级功耗和 host-I/O 口径；对 regular TCAS-II 不构成硬门。只有 ASIC SAIF/PTPX 无法获得可信功耗时，才把 FPGA 当后备验证。
- **不做 S2/有损块剪枝。** 需要新 AEE Pareto 和新 checkpoint 身份，且与 exact 主线竞争篇幅。
- **不做新的 Conv matcher/C1 bank scheduler。** C1 已有明确 single-1RW 物理对象；短期再追 concurrent ceiling 容易重新打开功能与物理闭环。
- **不做 decoder 第四种 sparse engine。** 它不会增强当前两条电路贡献，且会把组件 brief 拖回未闭合系统稿。

## 推荐执行顺序

1. 先让当前 FC2 continuation VCS 收口并独立审阅，决定候选 A 是否进入正文。
2. 在不改变 RTL identity 的前提下规划 TSBG hold repair 与 request-qualified read-enable；若 hold 修复改变 netlist，先修 hold，再用最终 netlist重跑 power。
3. 只开一次 matched SAIF/PTPX campaign，报告 ordinary/TSBG logic + weight-store energy，失败则 fail-closed，不复用旧 partial。
4. 新建 `paper/tcasii2027/`，保留现有 ISCAS 工件不覆盖；转换 Transactions 模板、压缩 C3/accuracy/provenance、增加 circuit timing 图和 TCAS-II prior。
5. 形成首投前一张唯一主表：C1 model/physical 分列；K8 equal-bandwidth；TSBG RTL cycle + matched area/hold/energy。任何空字段不得由跨工件乘法推导。

## 独立评分

| 维度 | 当前 /5 | 完成 P0 后 /5 | 判断 |
|---|---:|---:|---|
| Novelty | 3.6 | 3.7--3.8 | C1 对象差较强；TSBG 是合法 specialization，不是新 reuse 原理 |
| Circuit/system fit | 4.3 | 4.5 | 数据通路、SRAM端口、低功耗控制与事件视觉应用均契合 |
| Soundness | 4.6 | 4.7 | 证据边界非常强；需继续避免 model/VCS/DC 拼接升级 |
| Implementation | 4.1 | 4.5 | 当前缺 TSBG hold/power/macro；C1 已有较强 commercial-flow anchor |
| Evaluation | 4.1 | 4.4 | 四序列 1,920 workloads 强；continuation 可补 FC2 scope hole |
| Presentation | 3.3 | 4.2 | 当前尚非 TCAS-II 格式且过多 provenance；重排后可明显改善 |
| **Overall** | **3.8** | **4.2--4.35** | **当前边缘；P0 完成后是可信 Accept 路径** |

## 最终建议

TCAS-II 版本不要再扩成“所有层都有专用加速”。最稳的稿件是两条 exact reuse circuit：**C1 解决有限单口 parent/product reuse，C2/TSBG 解决多 context weight delivery，并用 K8 共享数据通路提供等带宽面积效率。** C3、attention、decoder 和有损剪枝全部降级。新增工作只做 FC2 continuation 与 read-energy suppression：前者补范围，后者补 TCAS-II 最看重的电路功耗闭环。

## 证据与限制

- 本审阅绑定的论文源 SHA256：`6885a1e7e650b0be15a2a61b42c275f24d68a10ed990500d6ffa85fe9a154ded`。
- 当前 PDF SHA256：`7e8c6350658bedc9df6f5475a37d349d3c50c888d256c913fc7c001f732f2391`。
- 冻结 `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。
- 本审阅没有把正在运行/准备中的 R9/R10 FC2 continuation、decoder tail 或任何未封工件计入正结果。
- 接收概率是基于当前证据完整度、官方编辑政策和已发表同类 brief 的审稿倾向估计，不是统计预测。
